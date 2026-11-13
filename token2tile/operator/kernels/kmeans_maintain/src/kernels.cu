#include "kernels.h"
#include <cstdint>
#include <cstdio>
#include <cmath>

#define R32 32
#define D128 128

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------
#define AW_NW   8            // warps per assign CTA
#define AW_RPW  64           // rows per warp
#define AW_RPC  (AW_NW*AW_RPW)
#define AW_NGR  (AW_RPW/16)
#ifndef AW_NTILE
#define AW_NTILE 64          // clusters staged per iteration
#endif
#ifndef AUNROLL
#define AUNROLL 1
#endif
#ifndef ADIAG
#define ADIAG 0
#endif
#ifndef APREF
#define APREF 0
#endif
#ifndef QUNR
#define QUNR 8
#endif
#ifndef GUNR
#define GUNR 8
#endif
#ifndef CHTARGET
#define CHTARGET 1800
#endif
#define PRAGMA_(x) _Pragma(#x)
#define PRAGMA(x) PRAGMA_(x)
#define UNROLLN(n) PRAGMA(unroll n)
#ifndef ANGR
#define ANGR 2
#endif
#ifndef AMINB
#define AMINB 4
#endif
#define ASTRIDE 40           // smem row stride (halves) for A tile

#define CS_NW   8
#define CBW_Q   16
#define CBW_K   8
#define LCH     512

#define VEPB    8192         // elements per block in vred

static inline int rup(int a, int b) { return ((a + b - 1) / b) * b; }
static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------------------
// centroid preparation:  cenB in mma-B fragment order (holding -2*c), cnorm fp32
// ---------------------------------------------------------------------------
__global__ void prep_cen_kernel(const __half* __restrict__ cen,
                                __half* __restrict__ cenB,
                                float* __restrict__ cnorm, int C, int Cp)
{
    int h = blockIdx.y;
    int g = blockIdx.x;
    int t = threadIdx.x;
    const __half* cenh = cen + (size_t)h * C * 32;
    __half* cbh = cenB + (size_t)h * Cp * 32;
    float* cnh = cnorm + (size_t)h * Cp;

    int q = t & 3, n = t >> 2;
    int c = g * 8 + n;
    const int kb[8] = {0, 1, 8, 9, 16, 17, 24, 25};
    __half vals[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        int k = kb[j] + q * 2;
        vals[j] = (c < C) ? __hmul(__float2half(-2.0f), cenh[(size_t)c * 32 + k])
                          : __ushort_as_half((unsigned short)0);
    }
    *reinterpret_cast<uint4*>(cbh + (size_t)g * 256 + t * 8) =
        *reinterpret_cast<uint4*>(vals);

    if (t < 8) {
        int cc = g * 8 + t;
        if (cc < C) {
            float s = 0.f;
#pragma unroll
            for (int k = 0; k < 32; ++k) {
                float v = __half2float(cenh[(size_t)cc * 32 + k]);
                s += v * v;
            }
            cnh[cc] = s;
        } else {
            cnh[cc] = INFINITY;
        }
    }
}

// ---------------------------------------------------------------------------
// assign: label(i) = argmin_c ( ||c||^2 - 2 x_i . c ), ties -> lowest index
// ---------------------------------------------------------------------------
template <int NGR, int MINB, bool PIN_TAIL = false>
__global__ __launch_bounds__(AW_NW * 32, MINB)
void assign_kernel(const __half* __restrict__ X, const __half* __restrict__ cenB,
                   const float* __restrict__ cnorm, int* __restrict__ labels,
                   int N, int Cp, int pin_begin = 0, int pin_label = 0)
{
    __shared__ __half As[AW_NW * 16 * ASTRIDE];
    __shared__ __half Bs[AW_NTILE * 32];
    __shared__ float  Cs[AW_NTILE];

    int h = blockIdx.y;
    const __half* Xh  = X + (size_t)h * N * 32;
    const __half* cbh = cenB + (size_t)h * Cp * 32;
    const float*  cnh = cnorm + (size_t)h * Cp;
    int* labh = labels + (size_t)h * N;

    int tid = threadIdx.x;
    int w = tid >> 5, lane = tid & 31;
    int q = lane & 3;
    int rowBase  = blockIdx.x * (AW_NW * NGR * 16);
    int wRowBase = rowBase + w * (NGR * 16);

    __half* Aw = As + (size_t)w * 16 * ASTRIDE;

    uint32_t Af[NGR][2][4];
    {
        int m = lane >> 3, r8 = lane & 7;
        int rowOff = (m & 1) ? r8 + 8 : r8;
        int colOff = (m >= 2) ? 8 : 0;
        int ldr = lane >> 2;
        int ldc = (lane & 3) * 8;
#pragma unroll
        for (int rg = 0; rg < NGR; ++rg) {
            // stage 16 rows of A for this row-group into the small per-warp buffer
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int r = i * 8 + ldr;
                uint4 v = make_uint4(0u, 0u, 0u, 0u);
                int gr = wRowBase + rg * 16 + r;
                if (gr < N) v = *reinterpret_cast<const uint4*>(Xh + (size_t)gr * 32 + ldc);
                *reinterpret_cast<uint4*>(Aw + r * ASTRIDE + ldc) = v;
            }
            __syncwarp();
#pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                const __half* p = Aw + rowOff * ASTRIDE + ks * 16 + colOff;
                uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(p));
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(Af[rg][ks][0]), "=r"(Af[rg][ks][1]),
                               "=r"(Af[rg][ks][2]), "=r"(Af[rg][ks][3])
                             : "r"(addr));
            }
            __syncwarp();
        }
    }

    float best[NGR][2];
    int   bidx[NGR][2];
#pragma unroll
    for (int rg = 0; rg < NGR; ++rg) {
        best[rg][0] = INFINITY; best[rg][1] = INFINITY;
        bidx[rg][0] = 0;        bidx[rg][1] = 0;
    }

    for (int cbase = 0; cbase < Cp; cbase += AW_NTILE) {
        __syncthreads();
#pragma unroll
        for (int u = 0; u < AW_NTILE / 64; ++u)
            reinterpret_cast<uint4*>(Bs)[u * 256 + tid] =
                reinterpret_cast<const uint4*>(cbh + (size_t)cbase * 32)[u * 256 + tid];
        for (int u = tid; u < AW_NTILE; u += AW_NW * 32) Cs[u] = cnh[cbase + u];
        __syncthreads();

UNROLLN(AUNROLL)
        for (int g = 0; g < AW_NTILE / 8; ++g) {
            uint4 b = *reinterpret_cast<const uint4*>(Bs + g * 256 + lane * 8);
            float2 cn = *reinterpret_cast<const float2*>(Cs + g * 8 + q * 2);
            // build the mma C-operand quad once per centroid group; without this
            // ptxas re-materialises the consecutive-register quad inside every
            // row-group iteration (4 extra MOVs per mma).
            float cc0, cc1, cc2, cc3;
            asm volatile("mov.f32 %0, %4;\n\tmov.f32 %1, %5;\n\t"
                         "mov.f32 %2, %4;\n\tmov.f32 %3, %5;\n"
                         : "=f"(cc0), "=f"(cc1), "=f"(cc2), "=f"(cc3)
                         : "f"(cn.x), "f"(cn.y));
            int col0 = cbase + g * 8 + q * 2;
            int col1 = col0 + 1;
#pragma unroll
            for (int rg = 0; rg < NGR; ++rg) {
                float d0, d1, d2, d3;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};\n"
                             : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                             : "r"(Af[rg][0][0]), "r"(Af[rg][0][1]),
                               "r"(Af[rg][0][2]), "r"(Af[rg][0][3]),
                               "r"(b.x), "r"(b.y),
                               "f"(cc0), "f"(cc1), "f"(cc2), "f"(cc3));
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                             "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
                             : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
                             : "r"(Af[rg][1][0]), "r"(Af[rg][1][1]),
                               "r"(Af[rg][1][2]), "r"(Af[rg][1][3]),
                               "r"(b.z), "r"(b.w));

#if ADIAG
                best[rg][0] = fminf(best[rg][0], fminf(d0, d1));
                best[rg][1] = fminf(best[rg][1], fminf(d2, d3));
                bidx[rg][0] = col0; bidx[rg][1] = col1;
#else
                float m0 = fminf(d0, d1);
                int   i0 = (d0 <= d1) ? col0 : col1;
                bool  p0 = m0 < best[rg][0];
                bidx[rg][0] = p0 ? i0 : bidx[rg][0];
                best[rg][0] = fminf(best[rg][0], m0);

                float m1 = fminf(d2, d3);
                int   i1 = (d2 <= d3) ? col0 : col1;
                bool  p1 = m1 < best[rg][1];
                bidx[rg][1] = p1 ? i1 : bidx[rg][1];
                best[rg][1] = fminf(best[rg][1], m1);
#endif
            }
        }
    }

#pragma unroll
    for (int rg = 0; rg < NGR; ++rg) {
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            float bv = best[rg][s];
            int   bi = bidx[rg][s];
#pragma unroll
            for (int mask = 1; mask < 4; mask <<= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, mask);
                int   oi = __shfl_xor_sync(0xffffffffu, bi, mask);
                if (ov < bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
            }
            if (q == 0) {
                int row = wRowBase + rg * 16 + (lane >> 2) + s * 8;
                if (row < N) {
                    // Pinned trailing keys (context tokens) are forced into
                    // the reserved last key cluster before the histogram and
                    // grouping consume klab.  Launches without pinned keys
                    // instantiate PIN_TAIL=false and do not carry this
                    // branch.
                    //
                    if constexpr (PIN_TAIL)
                        bi = (row >= pin_begin) ? pin_label : bi;
                    labh[row] = bi;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// deterministic per-cluster sums (owner warps, row order)
// ---------------------------------------------------------------------------

#define SW     4      // warps per CTA in hist/scatter kernels

// per-warp-chunk histogram (deterministic: integer counts)
__global__ __launch_bounds__(SW * 32, 1)
void hist_kernel(const int* __restrict__ lab, int* __restrict__ chist,
                 int N, int C, int nChunk, int chunkR)
{
    extern __shared__ int sh[];
    int h = blockIdx.y;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int chunk = blockIdx.x * SW + w;
    int* hh = sh + w * C;
    for (int i = lane; i < C; i += 32) hh[i] = 0;
    __syncwarp();
    if (chunk < nChunk) {
        const int* labh = lab + (size_t)h * N;
        int r0 = chunk * chunkR;
        int r1 = r0 + chunkR; if (r1 > N) r1 = N;
        for (int r = r0; r < r1; r += 32) {
            int row = r + lane;
            int c = (row < r1) ? labh[row] : -1;
            unsigned peers = __match_any_sync(0xffffffffu, c);
            int leader = __ffs(peers) - 1;
            if (lane == leader && c >= 0) hh[c] += __popc(peers);
        }
        __syncwarp();
        int* out = chist + ((size_t)h * nChunk + chunk) * C;
        for (int i = lane; i < C; i += 32) out[i] = hh[i];
    }
}

__global__ __launch_bounds__(256, 1)
void col_total_kernel(const int* __restrict__ chist, int* __restrict__ counts,
                      int C, int nChunk)
{
    int h = blockIdx.y;
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    const int* ch = chist + (size_t)h * nChunk * C + c;
    int s = 0;
    for (int k = 0; k < nChunk; ++k) s += ch[(size_t)k * C];
    counts[(size_t)h * C + c] = s;
}

__global__ __launch_bounds__(256, 1)
void prefix_kernel(const int* __restrict__ counts, int* __restrict__ cbase, int C)
{
    extern __shared__ int sm[];
    int h = blockIdx.x, tid = threadIdx.x;
    for (int c = tid; c < C; c += 256) sm[c] = counts[(size_t)h * C + c];
    __syncthreads();
    int* a = sm;
    int* b = sm + C;
    for (int off = 1; off < C; off <<= 1) {
        for (int c = tid; c < C; c += 256) b[c] = (c >= off) ? (a[c] + a[c - off]) : a[c];
        __syncthreads();
        int* t = a; a = b; b = t;
        __syncthreads();
    }
    for (int c = tid; c < C; c += 256)
        cbase[(size_t)h * C + c] = a[c] - counts[(size_t)h * C + c];
}

__global__ __launch_bounds__(256, 1)
void col_offset_kernel(int* __restrict__ chist, const int* __restrict__ cbase,
                       int C, int nChunk)
{
    int h = blockIdx.y;
    int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= C) return;
    int* ch = chist + (size_t)h * nChunk * C + c;
    int r = cbase[(size_t)h * C + c];
    for (int k = 0; k < nChunk; ++k) {
        int t = ch[(size_t)k * C];
        ch[(size_t)k * C] = r;
        r += t;
    }
}

// deterministic stable scatter (row order within each cluster)
__global__ __launch_bounds__(SW * 32, 1)
void scatter_kernel(const int* __restrict__ lab, const int* __restrict__ choff,
                    int* __restrict__ sorted, int N, int C, int nChunk, int chunkR)
{
    extern __shared__ int sh2[];
    int h = blockIdx.y;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int chunk = blockIdx.x * SW + w;
    if (chunk >= nChunk) return;
    int* cc = sh2 + w * C;
    const int* off = choff + ((size_t)h * nChunk + chunk) * C;
    for (int i = lane; i < C; i += 32) cc[i] = off[i];
    __syncwarp();
    const int* labh = lab + (size_t)h * N;
    int* sh_out = sorted + (size_t)h * N;
    int r0 = chunk * chunkR;
    int r1 = r0 + chunkR; if (r1 > N) r1 = N;
    for (int r = r0; r < r1; r += 32) {
        int row = r + lane;
        int c = (row < r1) ? labh[row] : -1;
        unsigned peers = __match_any_sync(0xffffffffu, c);
        int leader = __ffs(peers) - 1;
        int rank = __popc(peers & ((1u << lane) - 1u));
        int base = 0;
        if (lane == leader && c >= 0) { base = cc[c]; cc[c] = base + __popc(peers); }
        base = __shfl_sync(0xffffffffu, base, leader);
        if (c >= 0) sh_out[base + rank] = row;
    }
}

// per-cluster sums over the sorted row list (registers, fixed order)
template <bool DOV, int WPB>
__global__ __launch_bounds__(WPB * 32, 1)
void gather_kernel(const __half* __restrict__ X, const __half* __restrict__ V,
                   const int* __restrict__ sorted, const int* __restrict__ counts,
                   const int* __restrict__ cbase,
                   __half* __restrict__ V_sorted,
                   float* __restrict__ xsum, float* __restrict__ vsum,
                   int N, int C)
{
    int h = blockIdx.y;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int c = blockIdx.x * WPB + w;
    if (c >= C) return;
    int cnt = counts[(size_t)h * C + c];
    int bs = cbase[(size_t)h * C + c];
    const int* srt = sorted + (size_t)h * N + bs;
    const __half* Xh = X + (size_t)h * N * 32;
    const __half* Vh = DOV ? (V + (size_t)h * N * 128) : (const __half*)nullptr;
    __half* Vsh = DOV ? (V_sorted + (size_t)h * N * 128) : (__half*)nullptr;

    float ax = 0.f;
    float av0 = 0.f, av1 = 0.f, av2 = 0.f, av3 = 0.f;
#pragma unroll 1
    for (int i = 0; i < cnt; i += GUNR) {
        int rw[GUNR]; float xv[GUNR]; uint2 vr[GUNR];
#pragma unroll
        for (int u = 0; u < GUNR; ++u) rw[u] = (i + u < cnt) ? srt[i + u] : -1;
#pragma unroll
        for (int u = 0; u < GUNR; ++u) {
            int r = (rw[u] >= 0) ? rw[u] : 0;
            xv[u] = (rw[u] >= 0) ? __half2float(Xh[(size_t)r * 32 + lane]) : 0.f;
            if (DOV) {
                uint2 z; z.x = 0u; z.y = 0u;
                vr[u] = (rw[u] >= 0)
                    ? __ldcs(reinterpret_cast<const uint2*>(Vh + (size_t)r * 128 + lane * 4)) : z;
            }
        }
#pragma unroll
        for (int u = 0; u < GUNR; ++u) {
            ax += xv[u];
            if (DOV) {
                if (rw[u] >= 0)
                    *reinterpret_cast<uint2*>(
                        Vsh + (size_t)(bs + i + u) * 128 + lane * 4) = vr[u];
                float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].x));
                float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&vr[u].y));
                av0 += f0.x; av1 += f0.y; av2 += f1.x; av3 += f1.y;
            }
        }
    }
    xsum[((size_t)h * C + c) * 32 + lane] = ax;
    if (DOV) {
        float4 o; o.x = av0; o.y = av1; o.z = av2; o.w = av3;
        *reinterpret_cast<float4*>(vsum + ((size_t)h * C + c) * 128 + lane * 4) = o;
    }
}

// Reuse calls already build cluster-sorted V for exact attention and retain
// the matching counts.  Sum that contiguous layout directly instead of
// rereading original-order V through the permutation, rebuilding a sorted
// row list, or invoking the generic centroid-update path.  One warp owns each
// cluster, so every component is accumulated in a fixed row order.
template <int WPB>
__global__ __launch_bounds__(WPB * 32, 1)
void gather_v_kernel(const __half* __restrict__ V_sorted,
                     const int* __restrict__ counts,
                     const int* __restrict__ cbase,
                     float* __restrict__ vsum,
                     int N, int C)
{
    int h = blockIdx.y;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int c = blockIdx.x * WPB + w;
    if (c >= C) return;
    int cnt = counts[(size_t)h * C + c];
    int bs = cbase[(size_t)h * C + c];
    const __half* Vh = V_sorted + (size_t)h * N * 128;

    float av0 = 0.f, av1 = 0.f, av2 = 0.f, av3 = 0.f;
#pragma unroll 1
    for (int i = 0; i < cnt; i += GUNR) {
        uint2 vr[GUNR];
#pragma unroll
        for (int u = 0; u < GUNR; ++u) {
            uint2 z; z.x = 0u; z.y = 0u;
            vr[u] = (i + u < cnt)
                ? __ldcs(reinterpret_cast<const uint2*>(
                    Vh + (size_t)(bs + i + u) * 128 + lane * 4)) : z;
        }
#pragma unroll
        for (int u = 0; u < GUNR; ++u) {
            float2 f0 = __half22float2(
                *reinterpret_cast<const __half2*>(&vr[u].x));
            float2 f1 = __half22float2(
                *reinterpret_cast<const __half2*>(&vr[u].y));
            av0 += f0.x; av1 += f0.y; av2 += f1.x; av3 += f1.y;
        }
    }
    float4 o; o.x = av0; o.y = av1; o.z = av2; o.w = av3;
    *reinterpret_cast<float4*>(
        vsum + ((size_t)h * C + c) * 128 + lane * 4) = o;
}

// Destination-passing reuse variant.  Each owner warp follows the retained
// stable kperm, writes the exact cluster-major V layout needed by attention,
// and accumulates the same fp32 values in the same sorted-row order as
// gather_v_kernel.  This removes the immediate second read of materialized V.
template <int WPB>
__global__ __launch_bounds__(WPB * 32, 1)
void gather_v_permute_kernel(const __half* __restrict__ V,
                             const int* __restrict__ sorted,
                             const int* __restrict__ counts,
                             const int* __restrict__ cbase,
                             __half* __restrict__ V_sorted,
                             float* __restrict__ vsum,
                             int N, int C)
{
    int h = blockIdx.y;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    int c = blockIdx.x * WPB + w;
    if (c >= C) return;
    int cnt = counts[(size_t)h * C + c];
    int bs = cbase[(size_t)h * C + c];
    const int* srt = sorted + (size_t)h * N + bs;
    const __half* Vh = V + (size_t)h * N * 128;
    __half* Vsh = V_sorted + (size_t)h * N * 128;

    float av0 = 0.f, av1 = 0.f, av2 = 0.f, av3 = 0.f;
#pragma unroll 1
    for (int i = 0; i < cnt; i += GUNR) {
        int rw[GUNR];
        uint2 vr[GUNR];
#pragma unroll
        for (int u = 0; u < GUNR; ++u)
            rw[u] = (i + u < cnt) ? srt[i + u] : -1;
#pragma unroll
        for (int u = 0; u < GUNR; ++u) {
            uint2 z; z.x = 0u; z.y = 0u;
            vr[u] = (rw[u] >= 0)
                ? __ldcs(reinterpret_cast<const uint2*>(
                    Vh + (size_t)rw[u] * 128 + lane * 4)) : z;
        }
#pragma unroll
        for (int u = 0; u < GUNR; ++u) {
            if (rw[u] >= 0)
                *reinterpret_cast<uint2*>(
                    Vsh + (size_t)(bs + i + u) * 128 + lane * 4) = vr[u];
            float2 f0 = __half22float2(
                *reinterpret_cast<const __half2*>(&vr[u].x));
            float2 f1 = __half22float2(
                *reinterpret_cast<const __half2*>(&vr[u].y));
            av0 += f0.x; av1 += f0.y; av2 += f1.x; av3 += f1.y;
        }
    }
    float4 o; o.x = av0; o.y = av1; o.z = av2; o.w = av3;
    *reinterpret_cast<float4*>(
        vsum + ((size_t)h * C + c) * 128 + lane * 4) = o;
}


// ---------------------------------------------------------------------------
// Q-side cluster sums: deterministic shared-memory integer fixed-point
// (integer atomics are order independent -> bitwise reproducible)
// ---------------------------------------------------------------------------
#define QSCALE 32768.0f

__global__ __launch_bounds__(256, 1)
void qsum_kernel(const __half* __restrict__ X, const int* __restrict__ lab,
                 int* __restrict__ part, int* __restrict__ cpart,
                 int N, int C, int rpt)
{
    extern __shared__ int acc[];          // C * 33 ints (stride 33 to spread banks)
    int* cnt = acc + (size_t)C * 33;
    int h = blockIdx.y, t = blockIdx.x;
    int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    for (int i = tid; i < C * 33; i += 256) acc[i] = 0;
    for (int i = tid; i < C; i += 256) cnt[i] = 0;
    __syncthreads();

    const __half* Xh = X + (size_t)h * N * 32;
    const int* labh = lab + (size_t)h * N;
    int r0 = t * rpt;
    int r1 = r0 + rpt; if (r1 > N) r1 = N;

    int sub = lane >> 2;          // row within the 8-row group
    int dc  = (lane & 3) * 8;     // first of 8 dims

    // each warp consumes 32 rows per outer step (4 vector loads of 8 rows)
    for (int base = r0 + w * 32; base < r1; base += 8 * 32) {
        int lb = (base + lane < r1) ? labh[base + lane] : -1;
        uint4 xv[4];
        int cc[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            int row = base + u * 8 + sub;
            cc[u] = __shfl_sync(0xffffffffu, lb, u * 8 + sub);
            uint4 z; z.x = z.y = z.z = z.w = 0u;
            xv[u] = (row < r1) ? *reinterpret_cast<const uint4*>(Xh + (size_t)row * 32 + dc) : z;
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            int c = cc[u];
            if (c >= 0) {
                const __half* hp = reinterpret_cast<const __half*>(&xv[u]);
                int* ap = acc + c * 33 + dc;
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    atomicAdd(ap + j, __float2int_rn(__half2float(hp[j]) * QSCALE));
                if ((lane & 3) == 0) atomicAdd(&cnt[c], 1);
            }
        }
    }
    __syncthreads();
    // Head-major layout matches the generic deterministic grouping kernels
    // below, so the final Q-pass histogram can become qcounts/qperm without
    // another label scan.
    size_t bo = ((size_t)h * gridDim.x + t) * (size_t)C;
    for (int i = tid; i < C * 32; i += 256)
        part[bo * 32 + i] = acc[(i >> 5) * 33 + (i & 31)];
    for (int i = tid; i < C; i += 256) cpart[bo + i] = cnt[i];
}

__global__ __launch_bounds__(256, 1)
void qreduce_kernel(const int* __restrict__ part, const int* __restrict__ cpart,
                    const __half* __restrict__ oldc, __half* __restrict__ out,
                    int* __restrict__ counts, int C, int nTile, int H)
{
    int h = blockIdx.y;
    int idx = blockIdx.x * 256 + threadIdx.x;
    if (idx >= C * 32) return;
    int c = idx >> 5;
    long long s = 0;
    int n = 0;
    for (int t = 0; t < nTile; ++t) {
        size_t bo = ((size_t)h * nTile + t) * (size_t)C;
        s += (long long)part[bo * 32 + idx];
        n += cpart[bo + c];
    }
    if (counts && (idx & 31) == 0)
        counts[(size_t)h * C + c] = n;
    size_t o = (size_t)h * C * 32 + idx;
    out[o] = n ? __float2half((float)s / (QSCALE * (float)n)) : oldc[o];
}

// ---------------------------------------------------------------------------
// finalize means
// ---------------------------------------------------------------------------
__global__ void fin_mean_h_kernel(const float* __restrict__ part,
                                  const int* __restrict__ cntp,
                                  const __half* __restrict__ oldc,
                                  __half* __restrict__ out, int C)
{
    int h = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * 32) return;
    int c = idx >> 5;
    size_t o = (size_t)h * C * 32 + idx;
    int n = cntp[(size_t)h * C + c];
    out[o] = n ? __float2half(part[o] / (float)n) : oldc[o];
}

__global__ void fin_mean_f_kernel(const float* __restrict__ part,
                                  const int* __restrict__ cntp,
                                  const __half* __restrict__ oldc,
                                  float* __restrict__ out, int C)
{
    int h = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * 32) return;
    int c = idx >> 5;
    size_t o = (size_t)h * C * 32 + idx;
    int n = cntp[(size_t)h * C + c];
    out[o] = n ? (part[o] / (float)n) : __half2float(oldc[o]);
}

__global__ void vred_kernel(const float* __restrict__ vsum,
                            float* __restrict__ bmax, int C)
{
    __shared__ float red[256];
    int h = blockIdx.y;
    size_t CE = (size_t)C * 128;
    size_t start = (size_t)blockIdx.x * VEPB;
    size_t stop = start + VEPB; if (stop > CE) stop = CE;
    float mx = 0.f;
    for (size_t i = start + threadIdx.x; i < stop; i += 256)
        mx = fmaxf(mx, fabsf(vsum[(size_t)h * CE + i]));
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0) bmax[(size_t)h * gridDim.x + blockIdx.x] = red[0];
}

__global__ void vscale_kernel(const float* __restrict__ vsum,
                              const float* __restrict__ bmax,
                              __half* __restrict__ sumv,
                              float* __restrict__ lam_s,
                              int C, int nb)
{
    __shared__ float red[256];
    int h = blockIdx.y;
    float mx = 0.f;
    for (int i = threadIdx.x; i < nb; i += 256) mx = fmaxf(mx, bmax[(size_t)h * nb + i]);
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int o = 128; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    float lam = fmaxf(1.0f, red[0] / 32768.0f);
    if (blockIdx.x == 0 && threadIdx.x == 0) lam_s[h] = lam;
    size_t CE = (size_t)C * 128;
    float inv = 1.0f / lam;
    for (size_t i = (size_t)blockIdx.x * 256 + threadIdx.x; i < CE;
         i += (size_t)gridDim.x * 256)
        sumv[(size_t)h * CE + i] = __float2half(vsum[(size_t)h * CE + i] * inv);
}

// ---------------------------------------------------------------------------
// workspace layout
// ---------------------------------------------------------------------------
struct WS {
    __half* cenBq; float* cnormq;
    __half* cenBk; float* cnormk;
    int* qlab1;  __half* cq1;
    int* qpart; int* qcpart;
    float* ksum; int* kcbase; int* kchist;
    float* vsumb; float* bmax;
    size_t total;
    int CQp, CKp, nbv, nChunk, chunkR, rpt, nTile;
};

static WS ws_layout(int H, int N, int CQ, int CK, char* base) {
    WS w;
    w.CQp = rup(CQ, AW_NTILE);
    w.CKp = rup(CK, AW_NTILE);
    w.nbv = cdiv(CK * 128, VEPB);
    {
        long long target = CHTARGET / (H > 0 ? H : 1); if (target < 1) target = 1;
        int cr = (int)((N + target - 1) / target);
        cr = (cr + 31) & ~31;
        if (cr < 128) cr = 128;
        if (cr > 2048) cr = 2048;
        w.chunkR = cr;
    }
    w.nChunk = cdiv(N, w.chunkR);
    {
        int per = 400 / (H > 0 ? H : 1); if (per < 1) per = 1;
        int r = cdiv(N, per);
        if (r > 4096) r = 4096;
        if (r < 256) r = 256;
        w.rpt = r;
        w.nTile = cdiv(N, r);
    }
    size_t off = 0;
    auto alloc = [&](size_t bytes) -> char* {
        char* p = base ? (base + off) : nullptr;
        off += (bytes + 255) & ~(size_t)255;
        return p;
    };
    w.cenBq  = (__half*)alloc((size_t)H * w.CQp * 32 * 2);
    w.cnormq = (float*) alloc((size_t)H * w.CQp * 4);
    w.cenBk  = (__half*)alloc((size_t)H * w.CKp * 32 * 2);
    w.cnormk = (float*) alloc((size_t)H * w.CKp * 4);
    w.qlab1  = (int*)   alloc((size_t)H * N * 4);
    w.cq1    = (__half*)alloc((size_t)H * CQ * 32 * 2);
    w.qpart  = (int*)   alloc((size_t)w.nTile * H * CQ * 32 * 4);
    w.qcpart = (int*)   alloc((size_t)w.nTile * H * CQ * 4);
    w.ksum   = (float*) alloc((size_t)H * CK * 32 * 4);
    w.kcbase = (int*)   alloc((size_t)H * CK * 4);
    w.kchist = (int*)   alloc((size_t)H * w.nChunk * CK * 4);
    w.vsumb  = (float*) alloc((size_t)H * CK * 128 * 4);
    w.bmax   = (float*) alloc((size_t)H * w.nbv * 4);
    w.total = off;
    return w;
}

size_t kmeans_maintain_ws_bytes(int H, int N, int CQ, int CK) {
    return ws_layout(H, N, CQ, CK, nullptr).total;
}

struct VWS {
    int* cbase;
    float* vsumb;
    float* bmax;
    size_t total;
    int nbv;
};

static VWS vws_layout(int H, int CK, char* base) {
    VWS w;
    w.nbv = cdiv(CK * 128, VEPB);
    size_t off = 0;
    auto alloc = [&](size_t bytes) -> char* {
        char* p = base ? (base + off) : nullptr;
        off += (bytes + 255) & ~(size_t)255;
        return p;
    };
    w.cbase = (int*)alloc((size_t)H * CK * 4);
    w.vsumb = (float*)alloc((size_t)H * CK * 128 * 4);
    w.bmax = (float*)alloc((size_t)H * w.nbv * 4);
    w.total = off;
    return w;
}

size_t reuse_vstats_ws_bytes(int H, int CK) {
    return vws_layout(H, CK, nullptr).total;
}

#define WPB 4

void kmeans_maintain_launch(
    const void* QU_, const void* KU_, const void* V_,
    const void* cq0_, const void* ck0_,
    void* qlab_, void* cq2_, void* klab_, void* ck2_,
    void* qcounts_, void* qperm_, void* kcounts_, void* kperm_,
    void* V_sorted_, void* sumv_, void* lam_s_,
    int H, int N, int CQ, int CK, int pin_tail,
    void* wsp, cudaStream_t st)
{
    const __half* QU = (const __half*)QU_;
    const __half* KU = (const __half*)KU_;
    const __half* V  = (const __half*)V_;
    const __half* cq0 = (const __half*)cq0_;
    const __half* ck0 = (const __half*)ck0_;
    int* qlab = (int*)qlab_;
    __half* cq2 = (__half*)cq2_;
    int* klab = (int*)klab_;
    float* ck2 = (float*)ck2_;
    int* qcounts = (int*)qcounts_;
    int* qperm = (int*)qperm_;
    int* kcounts = (int*)kcounts_;
    int* kperm = (int*)kperm_;
    __half* V_sorted = (__half*)V_sorted_;
    __half* sumv = (__half*)sumv_;
    float* lam_s = (float*)lam_s_;

    WS w = ws_layout(H, N, CQ, CK, (char*)wsp);
    int rowTiles = cdiv(N, AW_NW * ANGR * 16);
    int nCh = w.nChunk;
    int hgK = cdiv(nCh, SW);
    int hgQ = cdiv(w.nTile, SW);
    size_t shK = (size_t)SW * CK * 4;
    size_t shQacc = (size_t)CQ * 34 * 4;
    static bool s_attr = false;
    if (!s_attr) {
        cudaFuncSetAttribute(qsum_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 128 * 1024);
        s_attr = true;
    }

    prep_cen_kernel<<<dim3(w.CQp / 8, H), 32, 0, st>>>(cq0, w.cenBq, w.cnormq, CQ, w.CQp);
    prep_cen_kernel<<<dim3(w.CKp / 8, H), 32, 0, st>>>(ck0, w.cenBk, w.cnormk, CK, w.CKp);

    // ---- Q pass 1 ----
    assign_kernel<ANGR, AMINB><<<dim3(rowTiles, H), AW_NW * 32, 0, st>>>(QU, w.cenBq, w.cnormq, w.qlab1, N, w.CQp);
    qsum_kernel<<<dim3(w.nTile, H), 256, shQacc, st>>>(QU, w.qlab1, w.qpart, w.qcpart, N, CQ, w.rpt);
    qreduce_kernel<<<dim3(cdiv(CQ * 32, 256), H), 256, 0, st>>>(
        w.qpart, w.qcpart, cq0, w.cq1, nullptr, CQ, w.nTile, H);

    // ---- Q pass 2 ----
    prep_cen_kernel<<<dim3(w.CQp / 8, H), 32, 0, st>>>(w.cq1, w.cenBq, w.cnormq, CQ, w.CQp);
    assign_kernel<ANGR, AMINB><<<dim3(rowTiles, H), AW_NW * 32, 0, st>>>(QU, w.cenBq, w.cnormq, qlab, N, w.CQp);
    qsum_kernel<<<dim3(w.nTile, H), 256, shQacc, st>>>(QU, qlab, w.qpart, w.qcpart, N, CQ, w.rpt);
    qreduce_kernel<<<dim3(cdiv(CQ * 32, 256), H), 256, 0, st>>>(
        w.qpart, w.qcpart, w.cq1, cq2, qcounts, CQ, w.nTile, H);

    // The final Q-pass partial histograms are still live.  Turn them into a
    // deterministic, stable cluster grouping and publish both facts to the
    // selector instead of rebuilding them with torch.bincount/argsort.
    // qpart is dead after qreduce; its first H*CQ entries are ample cbase
    // scratch and avoid extending the shared workspace.
    int* qcbase = w.qpart;
    prefix_kernel<<<H, 256, (size_t)2 * CQ * 4, st>>>(qcounts, qcbase, CQ);
    col_offset_kernel<<<dim3(cdiv(CQ, 256), H), 256, 0, st>>>(
        w.qcpart, qcbase, CQ, w.nTile);
    scatter_kernel<<<dim3(hgQ, H), SW * 32, (size_t)SW * CQ * 4, st>>>(
        qlab, w.qcpart, qperm, N, CQ, w.nTile, w.rpt);

    // ---- K side ----
    if (pin_tail > 0) {
        assign_kernel<ANGR, AMINB, true>
            <<<dim3(rowTiles, H), AW_NW * 32, 0, st>>>(
                KU, w.cenBk, w.cnormk, klab, N, w.CKp,
                N - pin_tail, CK - 1);
    } else {
        assign_kernel<ANGR, AMINB>
            <<<dim3(rowTiles, H), AW_NW * 32, 0, st>>>(
                KU, w.cenBk, w.cnormk, klab, N, w.CKp);
    }
    hist_kernel<<<dim3(hgK, H), SW * 32, shK, st>>>(klab, w.kchist, N, CK, nCh, w.chunkR);
    col_total_kernel<<<dim3(cdiv(CK,256), H), 256, 0, st>>>(w.kchist, kcounts, CK, nCh);
    prefix_kernel<<<H, 256, (size_t)2 * CK * 4, st>>>(kcounts, w.kcbase, CK);
    col_offset_kernel<<<dim3(cdiv(CK,256), H), 256, 0, st>>>(w.kchist, w.kcbase, CK, nCh);
    scatter_kernel<<<dim3(hgK, H), SW * 32, shK, st>>>(klab, w.kchist, kperm, N, CK, nCh, w.chunkR);
    gather_kernel<true, WPB><<<dim3(cdiv(CK, WPB), H), WPB * 32, 0, st>>>(
        KU, V, kperm, kcounts, w.kcbase, V_sorted,
        w.ksum, w.vsumb, N, CK);
    fin_mean_f_kernel<<<dim3(cdiv(CK * 32, 256), H), 256, 0, st>>>(w.ksum, kcounts, ck0, ck2, CK);
    vred_kernel<<<dim3(w.nbv, H), 256, 0, st>>>(w.vsumb, w.bmax, CK);
    vscale_kernel<<<dim3(w.nbv, H), 256, 0, st>>>(w.vsumb, w.bmax, sumv, lam_s, CK, w.nbv);
}

void reuse_vstats_launch(
    const void* V_sorted_, const void* kcounts_,
    void* sumv_, void* lam_s_,
    int H, int N, int CK,
    void* wsp, cudaStream_t st)
{
    const __half* V_sorted = (const __half*)V_sorted_;
    const int* kcounts = (const int*)kcounts_;
    __half* sumv = (__half*)sumv_;
    float* lam_s = (float*)lam_s_;
    VWS w = vws_layout(H, CK, (char*)wsp);

    prefix_kernel<<<H, 256, (size_t)2 * CK * 4, st>>>(
        kcounts, w.cbase, CK);
    gather_v_kernel<WPB><<<dim3(cdiv(CK, WPB), H), WPB * 32, 0, st>>>(
        V_sorted, kcounts, w.cbase, w.vsumb, N, CK);
    vred_kernel<<<dim3(w.nbv, H), 256, 0, st>>>(
        w.vsumb, w.bmax, CK);
    vscale_kernel<<<dim3(w.nbv, H), 256, 0, st>>>(
        w.vsumb, w.bmax, sumv, lam_s, CK, w.nbv);
}

void reuse_vstats_gather_launch(
    const void* V_, const void* kperm_, const void* kcounts_,
    void* V_sorted_, void* sumv_, void* lam_s_,
    int H, int N, int CK,
    void* wsp, cudaStream_t st)
{
    const __half* V = (const __half*)V_;
    const int* kperm = (const int*)kperm_;
    const int* kcounts = (const int*)kcounts_;
    __half* V_sorted = (__half*)V_sorted_;
    __half* sumv = (__half*)sumv_;
    float* lam_s = (float*)lam_s_;
    VWS w = vws_layout(H, CK, (char*)wsp);

    prefix_kernel<<<H, 256, (size_t)2 * CK * 4, st>>>(
        kcounts, w.cbase, CK);
    gather_v_permute_kernel<WPB>
        <<<dim3(cdiv(CK, WPB), H), WPB * 32, 0, st>>>(
            V, kperm, kcounts, w.cbase, V_sorted, w.vsumb, N, CK);
    vred_kernel<<<dim3(w.nbv, H), 256, 0, st>>>(
        w.vsumb, w.bmax, CK);
    vscale_kernel<<<dim3(w.nbv, H), 256, 0, st>>>(
        w.vsumb, w.bmax, sumv, lam_s, CK, w.nbv);
}
