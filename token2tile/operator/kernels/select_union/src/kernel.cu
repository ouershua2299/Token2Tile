#include "kernel.h"
#include <cuda_fp16.h>

#define TPB 128
#define EPT 16
#define NC 2048
#define MARGIN 128
#define SHF 12
#define PARTS 6
#define ROWS_PER_PART 128

__device__ __forceinline__ int warpInclScanP(int v) {
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        asm volatile("{ .reg .s32 t; .reg .pred p;\n\t"
                     " shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff;\n\t"
                     " @p add.s32 %0, t, %0;\n\t"
                     "}" : "+r"(v) : "r"(d));
    }
    return v;
}

__device__ __forceinline__ int addWarpPrefix4(int inc, const int* sred, int w) {
    const int4 A = *(const int4*)(sred);
    int add = 0;
    if (w > 0) add += A.x;
    if (w > 1) add += A.y;
    if (w > 2) add += A.z;
    return inc + add;
}

__device__ __forceinline__ int addWarpPrefix8(int inc, const int* sred, int w) {
    const int4 A = *(const int4*)(sred);
    const int4 B = *(const int4*)(sred + 4);
    int add = 0;
    if (w > 0) add += A.x;
    if (w > 1) add += A.y;
    if (w > 2) add += A.z;
    if (w > 3) add += A.w;
    if (w > 4) add += B.x;
    if (w > 5) add += B.y;
    if (w > 6) add += B.z;
    return inc + add;
}

__device__ __forceinline__ float sampledScore(unsigned bits) {
    union Bits {
        unsigned u;
        __half2 h;
    } x;
    x.u = bits;
    const float2 f = __half22float2(x.h);
    return f.x + f.y;
}

__device__ __forceinline__ float scoreFromKey(int key) {
    union Bits {
        unsigned short u;
        __half h;
    } x;
    x.u = (unsigned short)key;
    x.u ^= ((x.u >> 15) & 1u) * 0x7FFFu;
    return __half2float(x.h);
}

__device__ __forceinline__ int keyFromScore(float score) {
    union Bits {
        unsigned short u;
        __half h;
    } x;
    x.h = __float2half_rn(score);
    x.u ^= ((x.u >> 15) & 1u) * 0x7FFFu;
    return (int)(short)x.u;
}

__global__ void k_offsets(const int* __restrict__ tile_sizes,
                          int* __restrict__ off, int CQ) {
    const int h = blockIdx.x;
    __shared__ __align__(16) int sred[8];
    __shared__ int carry;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int c0 = 0; c0 < CQ; c0 += 256) {
        int c = c0 + threadIdx.x;
        int v = (c < CQ) ? tile_sizes[(size_t)h * CQ + c] : 0;
        int inc = warpInclScanP(v);
        if (lane == 31) sred[w] = inc;
        __syncthreads();
        inc = addWarpPrefix8(inc, sred, w);
        int excl = inc - v + carry;
        if (c < CQ) off[(size_t)h * CQ + c] = excl;
        __syncthreads();
        if (threadIdx.x == 255) carry += inc;
        __syncthreads();
    }
}

// A parent tile of n rows becomes ceil(n/cap) consecutive virtual tiles in
// the already cluster-grouped rowlist.  Derive the complete virtual layout
// directly from the parent counts: no per-row virtual labels, rank tensor,
// scatter, or bincount are necessary.  The two scans run in the same fixed
// head-local order as k_offsets and only manipulate exact integers.
__global__ void k_virtual_layout(const int* __restrict__ tile_sizes,
                                 int* __restrict__ virtual_sizes,
                                 int* __restrict__ off,
                                 int* __restrict__ vbase,
                                 int* __restrict__ nsplit,
                                 int CQ, int CQV, int cap) {
    const int h = blockIdx.x;
    __shared__ __align__(16) int sred_n[8];
    __shared__ __align__(16) int sred_v[8];
    __shared__ int carry_n, carry_v;
    if (threadIdx.x == 0) {
        carry_n = 0;
        carry_v = 0;
    }
    __syncthreads();
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    for (int c0 = 0; c0 < CQ; c0 += 256) {
        const int c = c0 + threadIdx.x;
        const int n = (c < CQ) ? tile_sizes[(size_t)h * CQ + c] : 0;
        const int nv = (n + cap - 1) / cap;
        int inc_n = warpInclScanP(n);
        int inc_v = warpInclScanP(nv);
        if (lane == 31) {
            sred_n[w] = inc_n;
            sred_v[w] = inc_v;
        }
        __syncthreads();
        inc_n = addWarpPrefix8(inc_n, sred_n, w);
        inc_v = addWarpPrefix8(inc_v, sred_v, w);
        const int q0 = inc_n - n + carry_n;
        const int v0 = inc_v - nv + carry_v;
        if (c < CQ) {
            vbase[(size_t)h * CQ + c] = v0;
            nsplit[(size_t)h * CQ + c] = nv;
            for (int s = 0; s < nv; ++s) {
                const int rem = n - s * cap;
                const int vs = rem < cap ? rem : cap;
                virtual_sizes[(size_t)h * CQV + v0 + s] = vs;
                off[(size_t)h * CQV + v0 + s] = q0 + s * cap;
            }
        }
        __syncthreads();
        if (threadIdx.x == 255) {
            carry_n += inc_n;
            carry_v += inc_v;
        }
        __syncthreads();
    }
    for (int v = carry_v + threadIdx.x; v < CQV; v += blockDim.x) {
        virtual_sizes[(size_t)h * CQV + v] = 0;
        off[(size_t)h * CQV + v] = 0;
    }
}

#define KLO(w) ((int)(short)(w))
#define KHI(w) ((int)(short)((w) >> 16))
#define XFORM(w) ((w) ^= (((w) >> 15) & 0x00010001u) * 0x7FFFu)

#define HBIN1(kk, j) { int _d = vp - (kk); _d = _d < 0 ? 0 : _d; \
                       _d = _d > tmax ? tmax : _d; atomicAdd(&hist[_d], szp[j]); }
#define HBIN(w, j)  { HBIN1(KLO(w), j) HBIN1(KHI(w), (j) + 1) }

#define HHI1(kk, j) atomicAdd(&hist[127 - ((kk) >> 8)], szp[j]);
#define HHI(w, j)   { HHI1(KLO(w), j) HHI1(KHI(w), (j) + 1) }

#define HLO1(kk, j) { int _i = (((kk) >> 8) == bh) ? (511 - ((kk) & 255)) : (512 + lane); \
                      atomicAdd(&hist[_i], szp[j]); }
#define HLO(w, j)   { HLO1(KLO(w), j) HLO1(KHI(w), (j) + 1) }

#define GE1(kk, j)  if ((kk) >= vstar) gmask |= (1u << (j));
#define GE(w, j)    { GE1(KLO(w), j) GE1(KHI(w), (j) + 1) }

#define GT1(kk, j)  { if ((kk) > vstar) gmask |= (1u << (j)); \
                      if ((kk) == vstar) tmask |= (1u << (j)); }
#define TIEFIX(j0, j1, p_init) { int _p = (p_init); \
    _Pragma("unroll") \
    for (int j = (j0); j < (j1); ++j) if (tmask & (1u << j)) { \
        const int sj = szp[j] >> SHF; \
        if (_p < R) { gmask |= (1u << j); \
                      if (_p + sj >= R) rcb[idx_c] = sgt + _p + sj; } \
        _p += sj; } }
#define GT(w, j)    { GT1(KLO(w), j) GT1(KHI(w), (j) + 1) }

#define FOR8(M) { M(rA.x, 0) M(rA.y, 2) M(rA.z, 4) M(rA.w, 6) \
                  M(rB.x, 8) M(rB.y, 10) M(rB.z, 12) M(rB.w, 14) }

__global__ void __launch_bounds__(TPB, 8) k_main(
    const __half* __restrict__ gl,
    const int* __restrict__ sizes,
    const int* __restrict__ rowlist,
    const int* __restrict__ off,
    const int* __restrict__ tile_sizes,
    const int* __restrict__ nsp,
    unsigned char* __restrict__ uni,
    int* __restrict__ rowcnt,
    int N, int CQ, int partial_stride) {
    const int linear = blockIdx.x;
    const int t = linear % CQ;
    const int part = linear / CQ;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;

    __shared__ __align__(16) int hist[544];
    __shared__ __align__(16) int sredA[8];
    __shared__ __align__(16) float smean[4];
    __shared__ int s_b, s_pre, s_hv;

    const int nrows = tile_sizes[(size_t)h * CQ + t];
    int nparts = (nrows + ROWS_PER_PART - 1) / ROWS_PER_PART;
    nparts = nparts > PARTS ? PARTS : nparts;
    unsigned short* __restrict__ partial =
        (unsigned short*)uni + ((size_t)h * CQ + t) * partial_stride
        + part * TPB;
    if (part >= nparts) {
        partial[tid] = 0;
        return;
    }

    const int ns = *nsp;
    const int nsP = ns << SHF;

    int szp[EPT];
    {
        const int4* sp = (const int4*)(sizes + (size_t)h * NC + 8 * tid);
        int4 a = sp[0], b = sp[1];
        szp[0] = (a.x << SHF) + 1; szp[1] = (a.y << SHF) + 1;
        szp[2] = (a.z << SHF) + 1; szp[3] = (a.w << SHF) + 1;
        szp[4] = (b.x << SHF) + 1; szp[5] = (b.y << SHF) + 1;
        szp[6] = (b.z << SHF) + 1; szp[7] = (b.w << SHF) + 1;
        const int4* sq = (const int4*)(sizes + (size_t)h * NC + 1024 + 8 * tid);
        a = sq[0]; b = sq[1];
        szp[8] = (a.x << SHF) + 1; szp[9] = (a.y << SHF) + 1;
        szp[10] = (a.z << SHF) + 1; szp[11] = (a.w << SHF) + 1;
        szp[12] = (b.x << SHF) + 1; szp[13] = (b.y << SHF) + 1;
        szp[14] = (b.z << SHF) + 1; szp[15] = (b.w << SHF) + 1;
    }

    const int rbase = off[(size_t)h * CQ + t];
    const int* __restrict__ rl = rowlist + (size_t)h * N + rbase;
    const __half* __restrict__ glh = gl + (size_t)h * N * NC + 8 * tid;
    int* __restrict__ rcb = rowcnt + (size_t)h * N;

    const int lane = tid & 31, warp = tid >> 5;
    const int tmax = 256 + lane;
    unsigned myacc0 = 0u, myacc1 = 0u;

    int idx_c = 0;
    uint4 rA = make_uint4(0, 0, 0, 0), rB = make_uint4(0, 0, 0, 0);
    {
        idx_c = rl[part];
        const __half* g = glh + (size_t)idx_c * NC;
        rA = *(const uint4*)(g);
        rB = *(const uint4*)(g + 1024);
    }

    float sample = sampledScore(rA.x);
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
        sample += __shfl_down_sync(0xffffffffu, sample, d);
    if (lane == 0) smean[warp] = sample;
    __syncthreads();
    float sampleMean = (smean[0] + smean[1] + smean[2] + smean[3]) *
                       (1.0f / 256.0f);
    float residual = 0.0f;
    const int budget100 = ns * 100;
    const float initialResidual =
        (budget100 <= N) ? 2.461f :
        (budget100 <= 2 * N) ? 2.186f :
        (budget100 <= 5 * N) ? 1.743f : 1.358f;

    int iter = 0;
    for (int rp = part; rp < nrows; rp += nparts, ++iter) {
        int idx_n = 0;
        uint4 nA = make_uint4(0, 0, 0, 0), nB = make_uint4(0, 0, 0, 0);
        if (rp + nparts < nrows) {
            idx_n = rl[rp + nparts];
            const __half* g = glh + (size_t)idx_n * NC;
            nA = *(const uint4*)(g);
            nB = *(const uint4*)(g + 1024);
        }

        XFORM(rA.x); XFORM(rA.y); XFORM(rA.z); XFORM(rA.w);
        XFORM(rB.x); XFORM(rB.y); XFORM(rB.z); XFORM(rB.w);

        const int predicted = keyFromScore(sampleMean +
                                           ((iter == 0) ? initialResidual : residual));
        const int vp = predicted + (MARGIN - 1);
        hist[tid] = 0;
        hist[128 + tid] = 0;
        if (tid == 0) s_b = 256;
        __syncthreads();
        FOR8(HBIN)
        __syncthreads();
        {
            const int2 hh = *(const int2*)(hist + 2 * tid);
            int inc = warpInclScanP(hh.x + hh.y);
            if (lane == 31) sredA[warp] = inc;
            __syncthreads();
            inc = addWarpPrefix4(inc, sredA, warp);
            const int base = inc - hh.x - hh.y;
            const int p0 = base + hh.x;
            if (p0 >= nsP && base < nsP) { s_b = 2 * tid;     s_pre = base; s_hv = hh.x; }
            if (inc >= nsP && p0 < nsP) { s_b = 2 * tid + 1; s_pre = p0;   s_hv = hh.y; }
        }
        __syncthreads();

        int vstar = vp - s_b;
        int packedPre = s_pre;
        int hvb = s_hv;

        if ((unsigned)(s_b - 1) > 253u) {
            hist[tid] = 0; hist[128 + tid] = 0;
            hist[256 + tid] = 0; hist[384 + tid] = 0;
            __syncthreads();
            FOR8(HHI)
            __syncthreads();
            {
                const int2 hh = *(const int2*)(hist + 2 * tid);
                int inc = warpInclScanP(hh.x + hh.y);
                if (lane == 31) sredA[warp] = inc;
                __syncthreads();
                inc = addWarpPrefix4(inc, sredA, warp);
                const int base = inc - hh.x - hh.y;
                const int p0 = base + hh.x;
                if (p0 >= nsP && base < nsP) { s_b = 2 * tid;     s_pre = base; }
                if (inc >= nsP && p0 < nsP) { s_b = 2 * tid + 1; s_pre = p0; }
            }
            __syncthreads();
            const int bh = 127 - s_b;
            const int pre = s_pre;
            FOR8(HLO)
            __syncthreads();
            {
                const int2 hh = *(const int2*)(hist + 256 + 2 * tid);
                int inc = warpInclScanP(hh.x + hh.y);
                if (lane == 31) sredA[warp] = inc;
                __syncthreads();
                inc = addWarpPrefix4(inc, sredA, warp) + pre;
                const int base = inc - hh.x - hh.y;
                const int p0 = base + hh.x;
                if (p0 >= nsP && base < nsP) { s_b = 2 * tid;     s_pre = base; s_hv = hh.x; }
                if (inc >= nsP && p0 < nsP) { s_b = 2 * tid + 1; s_pre = p0;   s_hv = hh.y; }
            }
            __syncthreads();
            vstar = (bh << 8) | (255 - s_b);
            packedPre = s_pre;
            hvb = s_hv;
        }

        const float observedResidual = scoreFromKey(vstar) - sampleMean;
        residual = (iter == 0) ? observedResidual
                               : fmaf(0.125f, observedResidual - residual, residual);

        const int sgt = packedPre >> SHF;
        unsigned gmask = 0u;

        if ((hvb & 0xFFF) == 1) {
            FOR8(GE)
            if (tid == 0) rcb[idx_c] = sgt + (hvb >> SHF);
        } else {
            const int R = ns - sgt;
            unsigned tmask = 0u;
            FOR8(GT)
            int loc0 = 0, loc1 = 0;
            if (tmask) {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    if (tmask & (1u << j)) loc0 += (szp[j] >> SHF);
#pragma unroll
                for (int j = 8; j < EPT; ++j)
                    if (tmask & (1u << j)) loc1 += (szp[j] >> SHF);
            }
            int i0 = warpInclScanP(loc0);
            int i1 = warpInclScanP(loc1);
            if (lane == 31) { sredA[warp] = i0; sredA[4 + warp] = i1; }
            __syncthreads();
            const int4 WA = *(const int4*)(sredA);
            const int4 WB = *(const int4*)(sredA + 4);
            int a0 = 0, a1 = 0;
            if (warp > 0) { a0 += WA.x; a1 += WB.x; }
            if (warp > 1) { a0 += WA.y; a1 += WB.y; }
            if (warp > 2) { a0 += WA.z; a1 += WB.z; }
            const int totLow = WA.x + WA.y + WA.z + WA.w;
            const int p0 = i0 + a0 - loc0;
            const int p1 = totLow + i1 + a1 - loc1;
            if (tmask) {
                TIEFIX(0, 8, p0)
                TIEFIX(8, EPT, p1)
            }
        }

        myacc0 |= (gmask & 0xFFu);
        myacc1 |= (gmask >> 8);

        float nextSample = sampledScore(nA.x);
#pragma unroll
        for (int d = 16; d > 0; d >>= 1)
            nextSample += __shfl_down_sync(0xffffffffu, nextSample, d);
        if (lane == 0) smean[warp] = nextSample;

        rA = nA; rB = nB;
        idx_c = idx_n;
        __syncthreads();
        sampleMean = (smean[0] + smean[1] + smean[2] + smean[3]) *
                     (1.0f / 256.0f);
    }

    partial[tid] = (unsigned short)((myacc0 & 0xFFu) | ((myacc1 & 0xFFu) << 8));
}

__global__ void __launch_bounds__(TPB, 8) k_expand(unsigned char* __restrict__ uni) {
    const int tile = blockIdx.x;
    const int tid = threadIdx.x;
    unsigned short* __restrict__ partial =
        (unsigned short*)(uni + (size_t)tile * NC);
    unsigned bits = 0;
#pragma unroll
    for (int p = 0; p < PARTS; ++p)
        bits |= partial[p * TPB + tid];
    __syncthreads();

    unsigned char* out = uni + (size_t)tile * NC + 8 * tid;
    {
        const unsigned bm = bits & 0xFFu;
        uint2 o;
        o.x = (bm & 1u) | ((bm & 2u) << 7) | ((bm & 4u) << 14) | ((bm & 8u) << 21);
        o.y = ((bm >> 4) & 1u) | ((bm & 32u) << 3) | ((bm & 64u) << 10) | ((bm & 128u) << 17);
        *(uint2*)(out) = o;
    }
    {
        const unsigned bm = bits >> 8;
        uint2 o;
        o.x = (bm & 1u) | ((bm & 2u) << 7) | ((bm & 4u) << 14) | ((bm & 8u) << 21);
        o.y = ((bm >> 4) & 1u) | ((bm & 32u) << 3) | ((bm & 64u) << 10) | ((bm & 128u) << 17);
        *(uint2*)(out + 1024) = o;
    }
}

// k_main already accumulates the selected-cluster state as two 8-bit masks
// per thread.  Publish those masks as the final LSB-first int32 bitmap rather
// than expanding them to 2048 byte flags and packing those again into the
// same 64 words.  In split mode the virtual-tile masks are OR-folded here,
// directly from the partials, so no expanded virtual or parent union exists.
template <bool SPLIT>
__global__ void __launch_bounds__(TPB) k_pack_partials(
    const unsigned short* __restrict__ partial,
    const int* __restrict__ vbase,
    const int* __restrict__ nsplit,
    int* __restrict__ packed,
    int CQ, int CQV) {
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int vb = SPLIT ? vbase[(size_t)h * CQ + t] : t;
    const int nv = SPLIT ? nsplit[(size_t)h * CQ + t] : 1;

    unsigned bits = 0;
    for (int s = 0; s < nv; ++s) {
        const unsigned short* tile = partial
            + ((size_t)h * CQV + vb + s) * (PARTS * TPB);
#pragma unroll
        for (int p = 0; p < PARTS; ++p)
            bits |= tile[p * TPB + tid];
    }

    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int src = lane & ~3;
    const unsigned mask = 0xffffffffu;
    const unsigned b0 = __shfl_sync(mask, bits, src);
    const unsigned b1 = __shfl_sync(mask, bits, src + 1);
    const unsigned b2 = __shfl_sync(mask, bits, src + 2);
    const unsigned b3 = __shfl_sync(mask, bits, src + 3);
    if ((lane & 3) == 0) {
        unsigned lo = (b0 & 0xffu) | ((b1 & 0xffu) << 8)
                    | ((b2 & 0xffu) << 16) | ((b3 & 0xffu) << 24);
        unsigned hi = ((b0 >> 8) & 0xffu) | (((b1 >> 8) & 0xffu) << 8)
                    | (((b2 >> 8) & 0xffu) << 16)
                    | (((b3 >> 8) & 0xffu) << 24);
        const int w = warp * 8 + (lane >> 2);
        int* out = packed + ((size_t)h * CQ + t) * (NC / 32);
        out[w] = (int)lo;
        out[32 + w] = (int)hi;
    }
}

void fscu_launch(const void* gl, const void* sizes, const void* rowlist,
                 const void* tile_sizes, const void* ns,
                 void* uni, void* rowcnt,
                 void* off,
                 int H, int N, int CQ, cudaStream_t stream) {
    k_offsets<<<H, 256, 0, stream>>>((const int*)tile_sizes, (int*)off, CQ);
    dim3 g3(CQ * PARTS, H);
    k_main<<<g3, TPB, 0, stream>>>((const __half*)gl, (const int*)sizes,
                                   (const int*)rowlist, (const int*)off,
                                   (const int*)tile_sizes, (const int*)ns,
                                   (unsigned char*)uni, (int*)rowcnt, N, CQ,
                                   NC / 2);
    k_expand<<<H * CQ, TPB, 0, stream>>>((unsigned char*)uni);
}

void fscu_split_launch(const void* gl, const void* sizes, const void* rowlist,
                       const void* tile_sizes, int cap, const void* ns,
                       void* uni, void* rowcnt,
                       void* virtual_sizes, void* off,
                       void* vbase, void* nsplit,
                       int H, int N, int CQ, int CQV,
                       cudaStream_t stream) {
    k_virtual_layout<<<H, 256, 0, stream>>>(
        (const int*)tile_sizes, (int*)virtual_sizes, (int*)off,
        (int*)vbase, (int*)nsplit, CQ, CQV, cap);
    dim3 g3(CQV * PARTS, H);
    k_main<<<g3, TPB, 0, stream>>>(
        (const __half*)gl, (const int*)sizes, (const int*)rowlist,
        (const int*)off, (const int*)virtual_sizes, (const int*)ns,
        (unsigned char*)uni, (int*)rowcnt, N, CQV, NC / 2);
    k_expand<<<H * CQV, TPB, 0, stream>>>((unsigned char*)uni);
}

void fscu_packed_launch(const void* gl, const void* sizes,
                        const void* rowlist, const void* tile_sizes,
                        const void* ns, void* partial, void* packed,
                        void* rowcnt, void* off,
                        int H, int N, int CQ, cudaStream_t stream) {
    k_offsets<<<H, 256, 0, stream>>>((const int*)tile_sizes, (int*)off, CQ);
    dim3 g3(CQ * PARTS, H);
    k_main<<<g3, TPB, 0, stream>>>(
        (const __half*)gl, (const int*)sizes, (const int*)rowlist,
        (const int*)off, (const int*)tile_sizes, (const int*)ns,
        (unsigned char*)partial, (int*)rowcnt, N, CQ, PARTS * TPB);
    k_pack_partials<false><<<dim3(CQ, H), TPB, 0, stream>>>(
        (const unsigned short*)partial, nullptr, nullptr, (int*)packed,
        CQ, CQ);
}

void fscu_split_packed_launch(
    const void* gl, const void* sizes, const void* rowlist,
    const void* tile_sizes, int cap, const void* ns,
    void* partial, void* packed, void* rowcnt,
    void* virtual_sizes, void* off, void* vbase, void* nsplit,
    int H, int N, int CQ, int CQV, cudaStream_t stream) {
    k_virtual_layout<<<H, 256, 0, stream>>>(
        (const int*)tile_sizes, (int*)virtual_sizes, (int*)off,
        (int*)vbase, (int*)nsplit, CQ, CQV, cap);
    dim3 g3(CQV * PARTS, H);
    k_main<<<g3, TPB, 0, stream>>>(
        (const __half*)gl, (const int*)sizes, (const int*)rowlist,
        (const int*)off, (const int*)virtual_sizes, (const int*)ns,
        (unsigned char*)partial, (int*)rowcnt, N, CQV, PARTS * TPB);
    k_pack_partials<true><<<dim3(CQ, H), TPB, 0, stream>>>(
        (const unsigned short*)partial, (const int*)vbase,
        (const int*)nsplit, (int*)packed, CQ, CQV);
}
