// Token2Tile tile attention, SM90, D=128, fp16.
// Cluster-task schedule with fp16 QK accumulation
// (scores accumulate in packed halves in the exp2 domain, lazy 3-log2 max, f16x2 exp2).
// Parallel selected-KV planning, exact pairwise run advance, and split K/V
// barriers. Two-block tasks overlap QK(t) with PV(t-1) while preserving packed-half QK/softmax.
// The tail merge reuses dead centroid shared memory to coalesce exact loads and final output stores.
// Exact attention stages original-order Q directly through qperm; K/V remain
// cluster-major.  The indexed copy is bitwise-equivalent to materializing qp.
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <stdexcept>
#include <map>
#include <tuple>
#include <cuda_fp16.h>
#include <stdint.h>

#define DIM 128
#define CKC 2048
#define NCONS 256
#define NPROD 128
#define NTHREADS (NCONS + NPROD)
#define KTILE 112
#define NSTAGE 2
#define QTILE 256
#define MMARGIN 3.0f  // fp16 lazy-max margin with reduced subtraction quantization
#define KSTG (KTILE * 128)      // halfs per K/V stage
#define KBLK (KTILE * 64)       // halfs per 64-wide K/V dim block
#define KSTG_D (KSTG * 2 / 16)  // stage stride, descriptor units
#define KBLK_D (KBLK * 2 / 16)  // dim-block stride, descriptor units

// ---- shared memory descriptor -------------------------------------------
__device__ __forceinline__ uint64_t desc_enc(uint64_t x) { return ((x & 0x3FFFFULL) >> 4); }

__device__ __forceinline__ uint64_t make_desc(const void *ptr, uint64_t lbo, uint64_t sbo) {
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(ptr);
  uint64_t d = desc_enc((uint64_t)addr);
  d |= desc_enc(lbo) << 16;
  d |= desc_enc(sbo) << 32;
  d |= 1ULL << 62;  // 128B swizzle
  return d;
}

__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// Compensate the half-rounded fixed scale with a high/low decomposition.
// Keep packed-half arithmetic while removing its leading scale bias.
__device__ __forceinline__ uint32_t scale_q_pair(uint32_t x, float scale) {
  // High/low fixed-scale decomposition: correct the half-rounded
  // high product using a packed-half FMA. No data-dependent approximation
  // or term dropping; all query components receive the same fixed scale.
  const float hi = 0.1275634765625f;
  const uint32_t h = pack2(hi, hi);
  const uint32_t l = pack2(scale-hi, scale-hi);
  uint32_t correction, result;
  asm("mul.f16x2 %0, %1, %2;" : "=r"(correction) : "r"(x), "r"(l));
  asm("fma.rn.f16x2 %0, %1, %2, %3;"
      : "=r"(result) : "r"(x), "r"(h), "r"(correction));
  return result;
}

// Packed half2 primitives for the fp16-accumulator softmax path.
__device__ __forceinline__ uint32_t hmax2(uint32_t a, uint32_t b) {
  uint32_t r; asm("max.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
__device__ __forceinline__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t r; asm("add.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t r; asm("mul.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b)); return r;
}
__device__ __forceinline__ uint32_t ex2_h2(uint32_t x) {
  uint32_t r; asm("ex2.approx.f16x2 %0, %1;" : "=r"(r) : "r"(x)); return r;
}
__device__ __forceinline__ float h2max_f(uint32_t x) {
  float r;
  asm("{ .reg .f16 a, b, c; mov.b32 {a, b}, %1; max.f16 c, a, b; cvt.f32.f16 %0, c; }" : "=f"(r) : "r"(x));
  return r;
}
__device__ __forceinline__ float h2sum_f(uint32_t x) {
  float r;
  asm("{ .reg .f16 a, b, c; mov.b32 {a, b}, %1; add.f16 c, a, b; cvt.f32.f16 %0, c; }" : "=f"(r) : "r"(x));
  return r;
}
// Keep the fp32 online-max bookkeeping exactly consistent with the fp16 shift applied to the scores.
__device__ __forceinline__ float snap_h(float x) {
  float r; asm("{ .reg .f16 a; cvt.rn.f16.f32 a, %1; cvt.f32.f16 %0, a; }" : "=f"(r) : "f"(x)); return r;
}

__device__ __forceinline__ void wg_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wg_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N> __device__ __forceinline__ void wg_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
#define WGMMA64_SS(d, descA, descB, sd) \
  asm volatile( \
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {" \
      "%0, %1, %2, %3, %4, %5, %6, %7," \
      "%8, %9, %10, %11, %12, %13, %14, %15," \
      "%16, %17, %18, %19, %20, %21, %22, %23," \
      "%24, %25, %26, %27, %28, %29, %30, %31}, " \
      "%32, %33, %34, 1, 1, 0, 0;" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), \
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]) \
      : "l"(descA), "l"(descB), "n"(sd))

#define WGMMA96_SS(d, descA, descB, sd) \
  asm volatile( \
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.f16.f16 {" \
      "%0, %1, %2, %3, %4, %5, %6, %7," \
      "%8, %9, %10, %11, %12, %13, %14, %15," \
      "%16, %17, %18, %19, %20, %21, %22, %23," \
      "%24, %25, %26, %27, %28, %29, %30, %31," \
      "%32, %33, %34, %35, %36, %37, %38, %39," \
      "%40, %41, %42, %43, %44, %45, %46, %47}, " \
      "%48, %49, %50, 1, 1, 0, 0;" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), \
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), \
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), \
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), \
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), \
        "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]) \
      : "l"(descA), "l"(descB), "n"(sd))

#define WGMMA128_RS(d, a0, a1, a2, a3, descB) \
  asm volatile( \
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {" \
      "%0, %1, %2, %3, %4, %5, %6, %7," \
      "%8, %9, %10, %11, %12, %13, %14, %15," \
      "%16, %17, %18, %19, %20, %21, %22, %23," \
      "%24, %25, %26, %27, %28, %29, %30, %31," \
      "%32, %33, %34, %35, %36, %37, %38, %39," \
      "%40, %41, %42, %43, %44, %45, %46, %47," \
      "%48, %49, %50, %51, %52, %53, %54, %55," \
      "%56, %57, %58, %59, %60, %61, %62, %63}, " \
      "{%64, %65, %66, %67}, %68, 1, 1, 1, 1;" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), \
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), \
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), \
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), \
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), \
        "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), \
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), \
        "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), \
        "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), \
        "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63]) \
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(descB))

#define WGMMA112_SS(d, descA, descB, sd) \
  asm volatile( \
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.f16.f16 {" \
      "%0, %1, %2, %3, %4, %5, %6, %7," \
      "%8, %9, %10, %11, %12, %13, %14, %15," \
      "%16, %17, %18, %19, %20, %21, %22, %23," \
      "%24, %25, %26, %27, %28, %29, %30, %31," \
      "%32, %33, %34, %35, %36, %37, %38, %39," \
      "%40, %41, %42, %43, %44, %45, %46, %47," \
      "%48, %49, %50, %51, %52, %53, %54, %55}, " \
      "%56, %57, %58, 1, 1, 0, 0;" \
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), \
        "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), \
        "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), \
        "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), \
        "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), \
        "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), \
        "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), \
        "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), \
        "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), \
        "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), \
        "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), \
        "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), \
        "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), \
        "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]) \
      : "l"(descA), "l"(descB), "n"(sd))

// ---- cp.async helpers ---------------------------------------------------
__device__ __forceinline__ void cp16(uint32_t dst, const void *src, int bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
// fp16-accumulator QK: the score tile lands in the packed half pairs consumed by softmax and PV.
#define WGMMA112_SS_F16(d, descA, descB, sd) \
  asm volatile( \
      "wgmma.mma_async.sync.aligned.m64n112k16.f16.f16.f16 {" \
      "%0, %1, %2, %3, %4, %5, %6, %7," \
      "%8, %9, %10, %11, %12, %13, %14, %15," \
      "%16, %17, %18, %19, %20, %21, %22, %23," \
      "%24, %25, %26, %27}, " \
      "%28, %29, %30, 1, 1, 0, 0;" \
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), \
        "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), \
        "+r"(d[8]), "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), \
        "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), \
        "+r"(d[16]), "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), \
        "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23]), \
        "+r"(d[24]), "+r"(d[25]), "+r"(d[26]), "+r"(d[27]) \
      : "l"(descA), "l"(descB), "n"(sd))

template <int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// swizzled half offset inside a 64-wide block: row r, 16B-chunk index cc (0..7)
__device__ __forceinline__ int swz(int r, int cc) { return r * 64 + ((cc ^ (r & 7)) << 3); }

// ---------------------------------------------------------------- plan ----
__device__ __forceinline__ int blk_scan(int v, int *ws, int &total) {
  int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int y = __shfl_up_sync(0xffffffff, x, d);
    if (lane >= d) x += y;
  }
  __syncthreads();
  if (lane == 31) ws[warp] = x;
  __syncthreads();
  if (threadIdx.x == 0) {
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { int t = ws[i]; ws[i] = acc; acc += t; }
    ws[8] = acc;
  }
  __syncthreads();
  total = ws[8];
  return x - v + ws[warp];
}

// Selected-KV lengths, one warp per query cluster.  This used to be a serial
// per-thread popcount-gather inside plan_kernel, which ran on only H blocks and
// also forced an 8 KB cluster-size scratch array into that block.  Spreading it
// over H*CQ warps removes both.
__global__ void selected_kv_lengths_kernel(const int *__restrict__ qsz, const int *__restrict__ ksz,
                                           const uint32_t *__restrict__ ub, int *__restrict__ lengths,
                                           int CQ) {
  int h = blockIdx.y, c = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (c >= CQ) return;
  int total = 0;
  if (qsz[(size_t)h * CQ + c] > 0) {
    const uint32_t *words = ub + ((size_t)h * CQ + c) * (CKC / 32);
    for (int w = lane; w < CKC / 32; w += 32) {
      uint32_t bits = words[w];
      while (bits) { int bit = __ffs(bits) - 1; total += ksz[(size_t)h * CKC + w * 32 + bit]; bits &= bits - 1; }
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) total += __shfl_down_sync(0xffffffffu, total, off);
  if (lane == 0) lengths[(size_t)h * CQ + c] = total;
}

#define PLAN_MAXT 2048   // tasks per head handled by the in-block sort (H3 15 s: CQ=852 -> <=~900 tasks)

__global__ __launch_bounds__(256) void plan_kernel(const int *__restrict__ qsz,
                                                   const int *__restrict__ ksz, int H, int N,
                                                   int CQ, int *__restrict__ k_off,
                                                   int4 *__restrict__ tasks,
                                                   int *__restrict__ counters, int* kv_lens) {
  __shared__ int ws[16];
  __shared__ int sbase;
  __shared__ unsigned long long skey[PLAN_MAXT];   // (work << 32) | task index, sorted descending
  int h = blockIdx.x;
  int tid = threadIdx.x;

  int running = 0;
  for (int base = 0; base < CKC; base += 256) {
    int c = base + tid;
    int v = ksz[(size_t)h * CKC + c];
    int tot;
    int ex = blk_scan(v, ws, tot);
    k_off[(size_t)h * (CKC + 1) + c] = running + ex;
    running += tot;
  }
  if (tid == 0) k_off[(size_t)h * (CKC + 1) + CKC] = running;

  const int QT = QTILE;  // one task per q-cluster up to 256 rows
  // pass A: count tasks
  int ntask = 0;
  for (int base = 0; base < CQ; base += 256) {
    int c = base + tid;
    int v = (c < CQ) ? qsz[(size_t)h * CQ + c] : 0;
    int nt = (v + QT - 1) / QT;
    int tot;
    blk_scan(nt, ws, tot);
    ntask += tot;
  }
  if (tid == 0) sbase = atomicAdd(&counters[0], ntask);
  __syncthreads();
  int gbase = sbase;

  int qrun = 0, trun = 0;
  for (int base = 0; base < CQ; base += 256) {
    int c = base + tid;
    int v = (c < CQ) ? qsz[(size_t)h * CQ + c] : 0;
    int totq;
    int exq = blk_scan(v, ws, totq);
    int nt = (v + QT - 1) / QT;
    int tott;
    int ext = blk_scan(nt, ws, tott);
    int qstart = qrun + exq;
    int toff = gbase + trun + ext;
    for (int i = 0; i < nt; ++i) {
      int s = qstart + i * QT;
      int rows = v - i * QT;
      if (rows > QT) rows = QT;
      tasks[toff + i] = make_int4(h, c, s, rows);
    }
    qrun += totq;
    trun += tott;
  }
  __syncthreads();
  if (ntask > PLAN_MAXT) return;   // keep cluster order when the sort buffer is too small
  // LPT: sort this head's tasks by descending estimated work (rows x selected keys); ties by index.
  // Deterministic: pure function of the snapshot's cluster sizes and union bits; task order does not
  // change any task's arithmetic.
  const int P2 = 1 << (32 - __clz(ntask - 1 > 0 ? ntask - 1 : 1));   // next pow2 >= ntask (>=2)
  for (int i = tid; i < P2; i += 256) {
    unsigned long long key = 0ull;
    if (i < ntask) {
      int4 tk = tasks[gbase + i];
      const int kv=kv_lens[(size_t)h*CQ+tk.y];
      unsigned long long work = (unsigned long long)tk.w * (unsigned long long)kv;
      if (work > 0xFFFFFFFFull) work = 0xFFFFFFFFull;
      key = (work << 32) | (unsigned long long)(0x7FFFFFFF - i);   // larger work first; lower index first on ties
    }
    skey[i] = key;
  }
  __syncthreads();
  for (int k = 2; k <= P2; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      for (int i = tid; i < P2; i += 256) {
        int ixj = i ^ j;
        if (ixj > i) {
          unsigned long long a = skey[i], b = skey[ixj];
          bool up = ((i & k) == 0);           // descending overall: keep larger first in "up" blocks
          if (up ? (a < b) : (a > b)) { skey[i] = b; skey[ixj] = a; }
        }
      }
      __syncthreads();
    }
  }
  // gather the reordered tasks (read all originals first, then write) via a second pass through smem-free staging:
  // each thread reads its destination's source task before anyone writes.
  int4 mine[8];
  int n_mine = 0;
  for (int i = tid; i < ntask; i += 256) {
    int src = 0x7FFFFFFF - (int)(skey[i] & 0xFFFFFFFFull);
    mine[n_mine++] = tasks[gbase + src];
  }
  __syncthreads();
  n_mine = 0;
  for (int i = tid; i < ntask; i += 256) tasks[gbase + i] = mine[n_mine++];
}

#define FULLM 0xffffffffu
#define QBLK (QTILE * 64)  // halfs per 64-wide Q dim-block

// Each task owns a fresh barrier generation. CTA synchronization at the
// next task boundary proves that all users of the previous task are done.
__device__ __forceinline__ void mbar_inval(uint64_t *b) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(a) : "memory");
}
__device__ __forceinline__ void mbar_init(uint64_t *b, int cnt) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(cnt));
}
__device__ __forceinline__ void mbar_wait(uint64_t *b, int ph) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_W%=:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra.uni DONE%=;\n"
      "bra.uni LAB_W%=;\n"
      "DONE%=:\n"
      "}\n" ::"r"(a),
      "r"(ph));
}
__device__ __forceinline__ void mbar_arrive(uint64_t *b) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void cp_mbar_arrive(uint64_t *b) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void pbar() { asm volatile("bar.sync 1, 128;" ::: "memory"); }
// consumer ping-pong tokens: 128 threads arrive, the other 128 wait
template <int ID> __device__ __forceinline__ void nb_sync() {
  asm volatile("bar.sync %0, 256;" ::"n"(ID) : "memory");
}
template <int ID> __device__ __forceinline__ void nb_arrive() {
  asm volatile("bar.arrive %0, 256;" ::"n"(ID) : "memory");
}

__device__ __forceinline__ int p_scan(int v, int *ws, int pt, int &total) {
  int lane = pt & 31, warp = pt >> 5;
  int x = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int y = __shfl_up_sync(FULLM, x, d);
    if (lane >= d) x += y;
  }
  pbar();
  if (lane == 31) ws[warp] = x;
  pbar();
  if (pt == 0) {
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) { int t = ws[i]; ws[i] = acc; acc += t; }
    ws[4] = acc;
  }
  pbar();
  total = ws[4];
  return x - v + ws[warp];
}

__device__ __forceinline__ void spbar() { asm volatile("bar.sync 1, 64;" ::: "memory"); }

__device__ __forceinline__ int sp_scan(int v, int *ws, int pt, int &total) {
  int lane = pt & 31, warp = pt >> 5;
  int x = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    int y = __shfl_up_sync(FULLM, x, d);
    if (lane >= d) x += y;
  }
  spbar();
  if (lane == 31) ws[warp] = x;
  spbar();
  if (pt == 0) { ws[2] = ws[0] + ws[1]; }
  spbar();
  total = ws[2];
  return x - v + (warp ? ws[0] : 0);
}


__device__ __forceinline__ void reg_alloc232() { asm volatile("setmaxnreg.inc.sync.aligned.u32 216;\n" ::: "memory"); }
__device__ __forceinline__ void reg_dealloc32() { asm volatile("setmaxnreg.dec.sync.aligned.u32 64;\n" ::: "memory"); }

// QK^T for one 64-row block against the 96-key stage tile: S[48] fp32 accumulators.
__device__ __forceinline__ void qk_block(float (&S)[56], uint64_t dQ, uint64_t dk, int QB) {
  wg_fence();
  WGMMA112_SS(S, dQ, dk, 0);
  WGMMA112_SS(S, dQ + 2, dk + 2, 1);
  WGMMA112_SS(S, dQ + 4, dk + 4, 1);
  WGMMA112_SS(S, dQ + 6, dk + 6, 1);
  WGMMA112_SS(S, dQ + QB, dk + KBLK_D, 1);
  WGMMA112_SS(S, dQ + QB + 2, dk + KBLK_D + 2, 1);
  WGMMA112_SS(S, dQ + QB + 4, dk + KBLK_D + 4, 1);
  WGMMA112_SS(S, dQ + QB + 6, dk + KBLK_D + 6, 1);
  wg_commit();
  wg_wait<0>();
}

// Online softmax over the 96-key tile for one block; leaves exp values in S (fp32) and packed P.
__device__ __forceinline__ void softmax_block(float (&S)[56], uint32_t (&P)[28], float &mA, float &mB,
                                              float &lA, float &lB, float &cA, float &cB, int valid,
                                              int q4, float qk_scale) {
  if (valid < KTILE) {
#pragma unroll
    for (int i = 0; i < 56; ++i) {
      int col = ((i >> 2) << 3) + 2 * q4 + (i & 1);
      if (col >= valid) S[i] = -INFINITY;
    }
  }
  float ra = -INFINITY, ra2 = -INFINITY, rb = -INFINITY, rb2 = -INFINITY;
#pragma unroll
  for (int g = 0; g < 14; g += 2) {
    ra = fmaxf(ra, fmaxf(S[4 * g], S[4 * g + 1]));
    rb = fmaxf(rb, fmaxf(S[4 * g + 2], S[4 * g + 3]));
    ra2 = fmaxf(ra2, fmaxf(S[4 * g + 4], S[4 * g + 5]));
    rb2 = fmaxf(rb2, fmaxf(S[4 * g + 6], S[4 * g + 7]));
  }
  ra = fmaxf(ra, ra2);
  rb = fmaxf(rb, rb2);
  ra = fmaxf(ra, __shfl_xor_sync(FULLM, ra, 1));
  rb = fmaxf(rb, __shfl_xor_sync(FULLM, rb, 1));
  ra = fmaxf(ra, __shfl_xor_sync(FULLM, ra, 2));
  rb = fmaxf(rb, __shfl_xor_sync(FULLM, rb, 2));
  const float tmA = ra * qk_scale, tmB = rb * qk_scale;
  const float nmA = (tmA > mA) ? (tmA + MMARGIN) : mA;
  const float nmB = (tmB > mB) ? (tmB + MMARGIN) : mB;
  cA = exp2f(mA - nmA);
  cB = exp2f(mB - nmB);
  mA = nmA;
  mB = nmB;
  float sa = 0.f, sb = 0.f;
#pragma unroll
  for (int g = 0; g < 14; ++g) {
    float e0 = exp2f(fmaf(S[4 * g + 0], qk_scale, -nmA));
    float e1 = exp2f(fmaf(S[4 * g + 1], qk_scale, -nmA));
    float e2 = exp2f(fmaf(S[4 * g + 2], qk_scale, -nmB));
    float e3 = exp2f(fmaf(S[4 * g + 3], qk_scale, -nmB));
    sa += e0 + e1;
    sb += e2 + e3;
    P[2 * g] = pack2(e0, e1);
    P[2 * g + 1] = pack2(e2, e3);
  }
  sa += __shfl_xor_sync(FULLM, sa, 1);
  sb += __shfl_xor_sync(FULLM, sb, 1);
  sa += __shfl_xor_sync(FULLM, sa, 2);
  sb += __shfl_xor_sync(FULLM, sb, 2);
  lA = lA * cA + sa;
  lB = lB * cB + sb;
}

// fp16 QK^T for one 64-row block against the K stage tile: scores land packed in P (exp2 domain, Q pre-scaled).
// _issue leaves the wgmma group in flight so the caller can overlap it with other work.
__device__ __forceinline__ void qk_issue_f16(uint32_t (&P)[28], uint64_t dQ, uint64_t dk, int QB) {
  wg_fence();
  WGMMA112_SS_F16(P, dQ, dk, 0);
  WGMMA112_SS_F16(P, dQ + 2, dk + 2, 1);
  WGMMA112_SS_F16(P, dQ + 4, dk + 4, 1);
  WGMMA112_SS_F16(P, dQ + 6, dk + 6, 1);
  WGMMA112_SS_F16(P, dQ + QB, dk + KBLK_D, 1);
  WGMMA112_SS_F16(P, dQ + QB + 2, dk + KBLK_D + 2, 1);
  WGMMA112_SS_F16(P, dQ + QB + 4, dk + KBLK_D + 4, 1);
  WGMMA112_SS_F16(P, dQ + QB + 6, dk + KBLK_D + 6, 1);
  wg_commit();
}
__device__ __forceinline__ void qk_block_f16(uint32_t (&P)[28], uint64_t dQ, uint64_t dk, int QB) {
  qk_issue_f16(P, dQ, dk, QB);
  wg_wait<0>();
}

// fp16 online softmax over the K tile for one block: lazy 3-log2 max with the rare
// fp32 O/L rescale inside the update branch, f16x2 exp2, packed-half row sums; P is rewritten in place.
__device__ __forceinline__ void softmax_block_f16(uint32_t (&P)[28], float (&O)[64], float &mA, float &mB,
                                                  float &lA, float &lB, uint32_t &nA2, uint32_t &nB2,
                                                  int valid, int q4) {
  if (valid < KTILE) {
#pragma unroll
    for (int g = 0; g < 14; ++g) {
      int col = (g << 3) + 2 * q4;
      uint32_t m = (col + 1 < valid) ? 0xffffffffu : ((col < valid) ? 0x0000ffffu : 0u);
      P[2 * g] = (P[2 * g] & m) | (~m & 0xfc00fc00u);
      P[2 * g + 1] = (P[2 * g + 1] & m) | (~m & 0xfc00fc00u);
    }
  }
  uint32_t xa = P[0], xb = P[1], xa2 = P[2], xb2 = P[3];
#pragma unroll
  for (int g = 2; g < 14; g += 2) {
    xa = hmax2(xa, P[2 * g]);
    xb = hmax2(xb, P[2 * g + 1]);
    xa2 = hmax2(xa2, P[2 * g + 2]);
    xb2 = hmax2(xb2, P[2 * g + 3]);
  }
  const float la = h2max_f(hmax2(xa, xa2));
  const float lb = h2max_f(hmax2(xb, xb2));
  if (__any_sync(FULLM, (la > mA) | (lb > mB))) {
    float ra = la, rb = lb;
    ra = fmaxf(ra, __shfl_xor_sync(FULLM, ra, 1));
    rb = fmaxf(rb, __shfl_xor_sync(FULLM, rb, 1));
    ra = fmaxf(ra, __shfl_xor_sync(FULLM, ra, 2));
    rb = fmaxf(rb, __shfl_xor_sync(FULLM, rb, 2));
    const float nmA = (ra > mA) ? snap_h(ra + MMARGIN) : mA;
    const float nmB = (rb > mB) ? snap_h(rb + MMARGIN) : mB;
    const float cA = exp2f(mA - nmA);
    const float cB = exp2f(mB - nmB);
#pragma unroll
    for (int i = 0; i < 64; ++i) O[i] *= ((i & 2) ? cB : cA);
    lA *= cA;
    lB *= cB;
    mA = nmA;
    mB = nmB;
    nA2 = pack2(-nmA, -nmA);
    nB2 = pack2(-nmB, -nmB);
  }
#pragma unroll
  for (int g = 0; g < 14; ++g) {
    P[2 * g] = ex2_h2(hadd2(P[2 * g], nA2));
    P[2 * g + 1] = ex2_h2(hadd2(P[2 * g + 1], nB2));
  }
  uint32_t ua[7], ub2[7];
#pragma unroll
  for (int j = 0; j < 7; ++j) {
    ua[j] = hadd2(P[4 * j], P[4 * j + 2]);
    ub2[j] = hadd2(P[4 * j + 1], P[4 * j + 3]);
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    ua[j] = hadd2(ua[2 * j], ua[2 * j + 1]);
    ub2[j] = hadd2(ub2[2 * j], ub2[2 * j + 1]);
  }
  ua[0] = hadd2(hadd2(ua[0], ua[1]), hadd2(ua[2], ua[6]));
  ub2[0] = hadd2(hadd2(ub2[0], ub2[1]), hadd2(ub2[2], ub2[6]));
  float sa = h2sum_f(ua[0]);
  float sb = h2sum_f(ub2[0]);
  sa += __shfl_xor_sync(FULLM, sa, 1);
  sb += __shfl_xor_sync(FULLM, sb, 1);
  sa += __shfl_xor_sync(FULLM, sa, 2);
  sb += __shfl_xor_sync(FULLM, sb, 2);
  lA += sa;
  lB += sb;
}

// Same arithmetic and order as softmax_block_f16, but the rare O rescale factors are returned rather
// than applied, so a caller running an asynchronous PV of the previous tile can apply them once that
// PV has retired (which is the same point in the accumulation order).
__device__ __forceinline__ void softmax_block_f16_c(uint32_t (&P)[28], float &mA, float &mB, float &lA,
                                                    float &lB, uint32_t &nA2, uint32_t &nB2, float &cA,
                                                    float &cB, int valid, int q4) {
  if (valid < KTILE) {
#pragma unroll
    for (int g = 0; g < 14; ++g) {
      int col = (g << 3) + 2 * q4;
      uint32_t m = (col + 1 < valid) ? 0xffffffffu : ((col < valid) ? 0x0000ffffu : 0u);
      P[2 * g] = (P[2 * g] & m) | (~m & 0xfc00fc00u);
      P[2 * g + 1] = (P[2 * g + 1] & m) | (~m & 0xfc00fc00u);
    }
  }
  uint32_t xa = P[0], xb = P[1], xa2 = P[2], xb2 = P[3];
#pragma unroll
  for (int g = 2; g < 14; g += 2) {
    xa = hmax2(xa, P[2 * g]);
    xb = hmax2(xb, P[2 * g + 1]);
    xa2 = hmax2(xa2, P[2 * g + 2]);
    xb2 = hmax2(xb2, P[2 * g + 3]);
  }
  const float la = h2max_f(hmax2(xa, xa2));
  const float lb = h2max_f(hmax2(xb, xb2));
  cA = 1.0f;
  cB = 1.0f;
  if (__any_sync(FULLM, (la > mA) | (lb > mB))) {
    float ra = la, rb = lb;
    ra = fmaxf(ra, __shfl_xor_sync(FULLM, ra, 1));
    rb = fmaxf(rb, __shfl_xor_sync(FULLM, rb, 1));
    ra = fmaxf(ra, __shfl_xor_sync(FULLM, ra, 2));
    rb = fmaxf(rb, __shfl_xor_sync(FULLM, rb, 2));
    const float nmA = (ra > mA) ? snap_h(ra + MMARGIN) : mA;
    const float nmB = (rb > mB) ? snap_h(rb + MMARGIN) : mB;
    cA = exp2f(mA - nmA);
    cB = exp2f(mB - nmB);
    lA *= cA;
    lB *= cB;
    mA = nmA;
    mB = nmB;
    nA2 = pack2(-nmA, -nmA);
    nB2 = pack2(-nmB, -nmB);
  }
#pragma unroll
  for (int g = 0; g < 14; ++g) {
    P[2 * g] = ex2_h2(hadd2(P[2 * g], nA2));
    P[2 * g + 1] = ex2_h2(hadd2(P[2 * g + 1], nB2));
  }
  uint32_t ua[7], ub2[7];
#pragma unroll
  for (int j = 0; j < 7; ++j) {
    ua[j] = hadd2(P[4 * j], P[4 * j + 2]);
    ub2[j] = hadd2(P[4 * j + 1], P[4 * j + 3]);
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    ua[j] = hadd2(ua[2 * j], ua[2 * j + 1]);
    ub2[j] = hadd2(ub2[2 * j], ub2[2 * j + 1]);
  }
  ua[0] = hadd2(hadd2(ua[0], ua[1]), hadd2(ua[2], ua[6]));
  ub2[0] = hadd2(hadd2(ub2[0], ub2[1]), hadd2(ub2[2], ub2[6]));
  float sa = h2sum_f(ua[0]);
  float sb = h2sum_f(ub2[0]);
  sa += __shfl_xor_sync(FULLM, sa, 1);
  sb += __shfl_xor_sync(FULLM, sb, 1);
  sa += __shfl_xor_sync(FULLM, sa, 2);
  sb += __shfl_xor_sync(FULLM, sb, 2);
  lA += sa;
  lB += sb;
}

__device__ __forceinline__ void pv_issue(float (&O)[64], const uint32_t (&P)[28], uint64_t dv) {
  wg_fence();
  WGMMA128_RS(O, P[0], P[1], P[2], P[3], dv);
  WGMMA128_RS(O, P[4], P[5], P[6], P[7], dv + 128);
  WGMMA128_RS(O, P[8], P[9], P[10], P[11], dv + 256);
  WGMMA128_RS(O, P[12], P[13], P[14], P[15], dv + 384);
  WGMMA128_RS(O, P[16], P[17], P[18], P[19], dv + 512);
  WGMMA128_RS(O, P[20], P[21], P[22], P[23], dv + 640);
  WGMMA128_RS(O, P[24], P[25], P[26], P[27], dv + 768);
  wg_commit();
}
__device__ __forceinline__ void pv_block(float (&O)[64], const uint32_t (&P)[28], uint64_t dv) {
  pv_issue(O, P, dv);
  wg_wait<0>();
}
// Empty asm with the accumulators as read-modify-write operands: forces the compiler to treat the
// wgmma accumulator registers as defined here, i.e. after the matching wgmma.wait_group, when the
// wait is no longer adjacent to the mma group.
__device__ __forceinline__ void fence_prob(uint32_t (&x)[28]) { asm volatile("" : "+r"(x[0]), "+r"(x[1]), "+r"(x[2]), "+r"(x[3]), "+r"(x[4]), "+r"(x[5]), "+r"(x[6]), "+r"(x[7]), "+r"(x[8]), "+r"(x[9]), "+r"(x[10]), "+r"(x[11]), "+r"(x[12]), "+r"(x[13]), "+r"(x[14]), "+r"(x[15]), "+r"(x[16]), "+r"(x[17]), "+r"(x[18]), "+r"(x[19]), "+r"(x[20]), "+r"(x[21]), "+r"(x[22]), "+r"(x[23]), "+r"(x[24]), "+r"(x[25]), "+r"(x[26]), "+r"(x[27]) :: "memory"); }
__device__ __forceinline__ void fence_out(float (&x)[64]) { asm volatile("" : "+f"(x[0]), "+f"(x[1]), "+f"(x[2]), "+f"(x[3]), "+f"(x[4]), "+f"(x[5]), "+f"(x[6]), "+f"(x[7]), "+f"(x[8]), "+f"(x[9]), "+f"(x[10]), "+f"(x[11]), "+f"(x[12]), "+f"(x[13]), "+f"(x[14]), "+f"(x[15]), "+f"(x[16]), "+f"(x[17]), "+f"(x[18]), "+f"(x[19]), "+f"(x[20]), "+f"(x[21]), "+f"(x[22]), "+f"(x[23]), "+f"(x[24]), "+f"(x[25]), "+f"(x[26]), "+f"(x[27]), "+f"(x[28]), "+f"(x[29]), "+f"(x[30]), "+f"(x[31]), "+f"(x[32]), "+f"(x[33]), "+f"(x[34]), "+f"(x[35]), "+f"(x[36]), "+f"(x[37]), "+f"(x[38]), "+f"(x[39]), "+f"(x[40]), "+f"(x[41]), "+f"(x[42]), "+f"(x[43]), "+f"(x[44]), "+f"(x[45]), "+f"(x[46]), "+f"(x[47]), "+f"(x[48]), "+f"(x[49]), "+f"(x[50]), "+f"(x[51]), "+f"(x[52]), "+f"(x[53]), "+f"(x[54]), "+f"(x[55]), "+f"(x[56]), "+f"(x[57]), "+f"(x[58]), "+f"(x[59]), "+f"(x[60]), "+f"(x[61]), "+f"(x[62]), "+f"(x[63]) :: "memory"); }

__device__ __forceinline__ void rescale_block(float (&O)[64], float cA, float cB) {
  if (__any_sync(FULLM, (cA != 1.0f) | (cB != 1.0f))) {
#pragma unroll
    for (int i = 0; i < 64; ++i) O[i] *= ((i & 2) ? cB : cA);
  }
}

// fp16 quantisation residual of P for tiny unions (kvlen <= 128): second accumulate pass.
__device__ __forceinline__ void hiprec_block(float (&O)[64], float (&S)[56], uint32_t (&P)[28], uint64_t dv) {
#pragma unroll
  for (int i = 0; i < 56; ++i) S[i] -= __half2float(__float2half_rn(S[i]));
#pragma unroll
  for (int g = 0; g < 14; ++g) {
    P[2 * g] = pack2(S[4 * g + 0], S[4 * g + 1]);
    P[2 * g + 1] = pack2(S[4 * g + 2], S[4 * g + 3]);
  }
  pv_block(O, P, dv);
}

__device__ __forceinline__ void store_block(const float (&O)[64], float mA, float mB, float lA, float lB,
                                            half *__restrict__ out, float *__restrict__ lse, size_t rowbase,
                                            int r0, int r1, int qrows, int q4) {
  const float rA = 1.f / lA, rB = 1.f / lB;
  const float lseA = mA + __log2f(lA), lseB = mB + __log2f(lB);
  half *o0 = out + (rowbase + r0) * DIM;
  half *o1 = out + (rowbase + r1) * DIM;
#pragma unroll
  for (int g = 0; g < 16; ++g) {
    int col = (g << 3) + 2 * q4;
    if (r0 < qrows) *(uint32_t *)(o0 + col) = pack2(O[4 * g + 0] * rA, O[4 * g + 1] * rA);
    if (r1 < qrows) *(uint32_t *)(o1 + col) = pack2(O[4 * g + 2] * rB, O[4 * g + 3] * rB);
  }
  if (q4 == 0) {
    if (r0 < qrows) lse[rowbase + r0] = lseA;
    if (r1 < qrows) lse[rowbase + r1] = lseB;
  }
}

__device__ __forceinline__ void cta_sync() { asm volatile("bar.cta.sync 6, 384;" ::: "memory"); }
__device__ __forceinline__ void cons_sync() { asm volatile("bar.cta.sync 7, 256;" ::: "memory"); }

// Owner-only pipelined task body: exactly one private 64-row block per warpgroup (65..128-row tasks,
// about half of all tasks for a ~128-row ragged query-cluster distribution).  The QK of tile t is issued
// before the PV of tile t-1 and only wgmma.wait_group 1 separates them, so the packed-half softmax of
// tile t runs while the PV wgmma group of tile t-1 is still in flight.  O is rescaled by the tile-t max
// update after that PV retires, which is the same position in the accumulation order as the reference rescale
// (tiles 0..t-1 scaled, then tile t added), so the fp16 score path is unchanged.
__device__ __forceinline__ void fa3_task_f16(half *sQ, half *sK, half *sV, uint64_t *bar, half *out,
                                             float *lse, size_t rowbase, int qrows, int kvlen, int wg,
                                             int tid, int QB) {
  const int lane = tid & 31, q4 = lane & 3, rowA = 16 * ((tid >> 5) & 3) + (lane >> 2);
  const uint64_t dq = make_desc(sQ + wg * 4096, 1, 1024), dk0 = make_desc(sK, 1, 1024),
                 dv0 = make_desc(sV, KBLK * 2, 1024);
  float O[64];
  uint32_t P[28], Pn[28];
#pragma unroll
  for (int i = 0; i < 64; ++i) O[i] = 0.f;
#pragma unroll
  for (int i = 0; i < 28; ++i) { P[i] = 0u; Pn[i] = 0u; }
  float ma = -INFINITY, mb = -INFINITY, la = 0.f, lb = 0.f, ca = 1.f, cb = 1.f;
  uint32_t na2 = 0u, nb2 = 0u;
  const int nt = (kvlen + KTILE - 1) / KTILE;
  int stage = 0, phase = 0;
  mbar_wait(bar, 0);
  qk_issue_f16(P, dq, dk0, QB);
  wg_wait<0>();
  fence_prob(P);
  mbar_arrive(bar + 2 * NSTAGE);                       // K stage 0 consumed
  softmax_block_f16_c(P, ma, mb, la, lb, na2, nb2, ca, cb, kvlen, q4);
#pragma unroll 1
  for (int t = 1; t < nt; ++t) {
    int ns = stage + 1, np = phase;
    if (ns == NSTAGE) { ns = 0; np ^= 1; }
    mbar_wait(bar + ns, np);                           // K of tile t
    const int next = __shfl_sync(FULLM, ns, 0), prev = __shfl_sync(FULLM, stage, 0);
    qk_issue_f16(Pn, dq, dk0 + (uint64_t)next * KSTG_D, QB);
    mbar_wait(bar + NSTAGE + stage, phase);            // V of tile t-1
    pv_issue(O, P, dv0 + (uint64_t)prev * KSTG_D);
    wg_wait<1>();
    fence_prob(Pn);
    mbar_arrive(bar + 2 * NSTAGE + ns);                // K of tile t consumed
    softmax_block_f16_c(Pn, ma, mb, la, lb, na2, nb2, ca, cb, kvlen - t * KTILE, q4);
    wg_wait<0>();
    fence_out(O);
    fence_prob(P);
    mbar_arrive(bar + 3 * NSTAGE + stage);             // V of tile t-1 consumed
    rescale_block(O, ca, cb);
#pragma unroll
    for (int i = 0; i < 28; ++i) asm volatile("mov.b32 %0, %1;" : "=r"(P[i]) : "r"(Pn[i]));
    stage = ns;
    phase = np;
  }
  const int us = __shfl_sync(FULLM, stage, 0);
  mbar_wait(bar + NSTAGE + stage, phase);
  pv_issue(O, P, dv0 + (uint64_t)us * KSTG_D);
  wg_wait<0>();
  fence_out(O);
  fence_prob(P);
  mbar_arrive(bar + 3 * NSTAGE + stage);
  store_block(O, ma, mb, la, lb, out, lse, rowbase, wg * 64 + rowA, wg * 64 + rowA + 8, qrows, q4);
}

__global__ __launch_bounds__(NTHREADS, 1) void attn_kernel(
    const half *__restrict__ q, const half *__restrict__ kp, const half *__restrict__ vp,
    const uint8_t *__restrict__ ub, const int *__restrict__ qperm,
    const int *__restrict__ k_off, half *__restrict__ out, float *__restrict__ lse,
    const int4 *__restrict__ tasks, int *__restrict__ counters, int H, int N, int CQ,const int* kv_lens) {
  extern __shared__ __align__(1024) char smem_raw[];
  half *sQ = (half *)smem_raw;               // 2 dim-blocks x 256 rows x 64 halfs
  half *sK = sQ + 2 * QBLK;
  half *sV = sK + NSTAGE * KSTG;
  uint64_t *bar = (uint64_t *)(sV + NSTAGE * KSTG);
  int *sInt = (int *)(bar + 4 * NSTAGE);
  int *sGidx = sInt;                 // 2 x 128 ints (96 used per half), 16B aligned
  int *sWs = sInt + 256;
  int *sMisc = sInt + 256 + 16;
  int *sSelStart = sInt + 256 + 24;   // <= 1024 merged runs
  int *sSelCum = sInt + 256 + 1048;   // 1025

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wg = __shfl_sync(0xffffffffu, tid >> 7, 0);
  const int numTasks = counters[0];

  // Producer/consumer branches meet the exact same CTA barrier PCs.
  reg_dealloc32();
  bool barriers_initialized=false;
  while(true) {
    __syncthreads();
    if(tid==0)sMisc[0]=atomicAdd(&counters[1],1);
    __syncthreads();
    const int ti=sMisc[0];if(ti>=numTasks)break;
    const int4 tk=tasks[ti];const int h=tk.x,qc=tk.y,qs=tk.z,qrows=tk.w;
    if(tid>=NCONS) {
      const int pt=tid-NCONS;
      if (pt < 4 * NSTAGE) {
        if (barriers_initialized) mbar_inval(bar + pt);
        mbar_init(bar + pt, pt < 2 * NSTAGE ? NPROD : NCONS);
      }
      barriers_initialized = true;
      const int *koh = k_off + (size_t)h * (CKC + 1);
      const uint8_t *ubr = ub + ((size_t)h * CQ + qc) * (CKC / 8);
      const int c0 = pt * 16;
      const uint16_t bw = *(const uint16_t *)(ubr + pt * 2);
      const int base0 = koh[c0];
      // was cluster c0-1 selected *and* non-empty?  If so its tokens are
      // immediately followed by c0's, so the two belong to the same run.
      bool pne0 = false;
      if (c0 > 0) {
        int pm1 = koh[c0 - 1];
        pne0 = (((ubr[(c0 - 1) >> 3] >> ((c0 - 1) & 7)) & 1u) != 0u) && (base0 > pm1);
      }
      int cnt = 0, sz = 0;
      {
        int prev = base0;
        bool pne = pne0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          int nx = koh[c0 + j + 1];
          bool cne = (((bw >> j) & 1u) != 0u) && (nx > prev);
          if (cne) {
            if (!pne) cnt++;
            sz += nx - prev;
          }
          pne = cne;
          prev = nx;
        }
      }
      int totc, tots;
      int k = p_scan(cnt, sWs, pt, totc);
      int p = p_scan(sz, sWs, pt, tots);
      {
        int prev = base0;
        bool pne = pne0;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
          int nx = koh[c0 + j + 1];
          bool cne = (((bw >> j) & 1u) != 0u) && (nx > prev);
          if (cne) {
            if (!pne) {
              sSelStart[k] = prev;
              sSelCum[k] = p;
              k++;
            }
            p += nx - prev;
          }
          pne = cne;
          prev = nx;
        }
      }
      if (pt == 0) {
        sSelCum[totc] = tots;

        sMisc[2] = totc;
      }

    } else {
    const float qk_scale = 0.08838834764831845f * 1.4426950408889634f;
#pragma unroll
    for (int i = 0; i < QTILE * 16 / NCONS; ++i) {
      int slot = tid + i * NCONS;
      int r = slot >> 4, cc = slot & 15;
      int rr = (r < qrows) ? r : 0;
      int orig = qperm[(size_t)h * N + qs + rr];
      const half *src = q + ((size_t)h * N + orig) * DIM + cc * 8;
      cp16((uint32_t)__cvta_generic_to_shared(sQ + (cc >> 3) * QBLK + swz(r, cc & 7)),
           src, (r < qrows) ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    {
      // Fold sm_scale*log2(e) into the staged Q (same thread re-touches the chunks it copied): the fp16 QK
      // accumulator then emits scores directly in the exp2 domain.

#pragma unroll
      for (int i = 0; i < QTILE * 16 / NCONS; ++i) {
        int slot = tid + i * NCONS;
        int r = slot >> 4, cc = slot & 15;
        uint4 *dst = (uint4 *)(sQ + (cc >> 3) * QBLK + swz(r, cc & 7));
        uint4 vq = *dst;
        vq.x = scale_q_pair(vq.x, qk_scale); vq.y = scale_q_pair(vq.y, qk_scale); vq.z = scale_q_pair(vq.z, qk_scale); vq.w = scale_q_pair(vq.w, qk_scale);
        *dst = vq;
      }
    }

    }
    __syncthreads();
    const int kvlen=kv_lens[(size_t)h*CQ+qc];
    if(kvlen==0) {
      if(tid<NCONS) {
      for (int i = tid; i < qrows * 64; i += NCONS) {
        int r = i >> 6, cidx = i & 63;
        ((uint32_t *)(out + ((size_t)h * N + qs + r) * DIM))[cidx] = 0u;
      }
      for (int r = tid; r < qrows; r += NCONS) lse[(size_t)h * N + qs + r] = -INFINITY;

      }
      continue;
    }
    if(tid>=NCONS) {
    const int pt = tid - NCONS;
    const int p_cc = pt & 15;
    const int p_r0 = pt >> 4;               // rows p_r0, p_r0+8, ... p_r0+88
    const uint32_t p_dstK =
        (uint32_t)__cvta_generic_to_shared(sK + (p_cc >> 3) * KBLK + swz(p_r0, p_cc & 7));
    const uint32_t p_dstV =
        (uint32_t)__cvta_generic_to_shared(sV + (p_cc >> 3) * KBLK + swz(p_r0, p_cc & 7));

      const int nsel = sMisc[2];
      const int ntiles = (kvlen + KTILE - 1) / KTILE;
      int cur = 0, phe = 0, s = 0, wc = 0;
      const half *kbase = kp + (size_t)h * N * DIM + p_cc * 8;
      const half *vbase = vp + (size_t)h * N * DIM + p_cc * 8;
      if (pt < KTILE) {
        while (cur + 1 < nsel && sSelCum[cur + 1] <= pt) cur++;
        sGidx[(pt & 7) * 16 + (pt >> 3)] =
            (pt < kvlen) ? (sSelStart[cur] + pt - sSelCum[cur]) * DIM : 0;
      }
      pbar();
      for (int t = 0; t < ntiles; ++t) {
        if (t >= NSTAGE) {
          mbar_wait(bar + 2 * NSTAGE + s, phe);
          if (++wc == NSTAGE) { wc = 0; phe ^= 1; }
        }
        const uint32_t dk = p_dstK + s * (KSTG * 2);
        const uint32_t dv = p_dstV + s * (KSTG * 2);
        const int4 *g4 = (const int4 *)(sGidx + (t & 1) * 128 + p_r0 * 16);
        const int4 ga = g4[0], gb = g4[1], gc = g4[2], gd = g4[3];
        const int tkr[14] = {ga.x, ga.y, ga.z, ga.w, gb.x, gb.y, gb.z, gb.w,
                             gc.x, gc.y, gc.z, gc.w, gd.x, gd.y};
#pragma unroll
        for (int u = 0; u < 14; ++u) cp16(dk + u * 1024, kbase + (size_t)tkr[u], 16);
        cp_mbar_arrive(bar + s);                            // K of this tile is on its way
        if (t >= NSTAGE) mbar_wait(bar + 3 * NSTAGE + s, (t / NSTAGE - 1) & 1);
#pragma unroll
        for (int u = 0; u < 14; ++u) cp16(dv + u * 1024, vbase + (size_t)tkr[u], 16);
        cp_mbar_arrive(bar + NSTAGE + s);
        if (++s == NSTAGE) s = 0;
        if (pt < KTILE && t + 1 < ntiles) {
          int pn = (t + 1) * KTILE + pt;
          // two consecutive selected runs advance in one step, with an exact single-step fixup
          while (cur + 2 < nsel && sSelCum[cur + 2] <= pn) cur += 2;
          if (cur + 1 < nsel && sSelCum[cur + 1] <= pn) cur++;
          sGidx[((t + 1) & 1) * 128 + (pt & 7) * 16 + (pt >> 3)] =
              (pn < kvlen) ? (sSelStart[cur] + pn - sSelCum[cur]) * DIM : 0;
        }
        pbar();
      }
    } else {
  reg_alloc232();
  const int warpid = (tid >> 5) & 3;
  const int q4 = lane & 3;
  const int rowA = 16 * warpid + (lane >> 2);
  const float qk_scale = 0.08838834764831845f * 1.4426950408889634f;
  const uint64_t dK0 = make_desc(sK, 1, 1024);
  const uint64_t dV0 = make_desc(sV, KBLK * 2, 1024);
  const int QB = (QBLK * 2) >> 4;  // Q dim-block stride in descriptor units

    const int ntiles = (kvlen + KTILE - 1) / KTILE;
    // 64-row blocks of this task (nblk in 1..4).  Private blocks: nblk>=2 -> block wg (slot a); nblk==4 -> block wg+2 (slot b).
    // Odd block counts leave one block (0 for nblk==1, 2 for nblk==3) shared: warpgroup wg takes its K tiles with (t&1)==wg
    // into slot b and the two partial softmax states are merged deterministically after the loop.
    const int nblk = (qrows + 63) >> 6;
    if (nblk == 2) {
      fa3_task_f16(sQ, sK, sV, bar, out, lse, (size_t)h * N + qs, qrows, kvlen, wg, tid, QB);
      reg_dealloc32();
      continue;
    }
    const int priv_a = (nblk >= 2) ? wg : -1;
    const int priv_b = (nblk == 4) ? (wg + 2) : -1;
    const int shared = (nblk == 1) ? 0 : ((nblk == 3) ? 2 : -1);
    const uint64_t dQa = make_desc(sQ + (priv_a < 0 ? 0 : priv_a) * 4096, 1, 1024);
    const uint64_t dQb = make_desc(sQ + (priv_b >= 0 ? priv_b : (shared >= 0 ? shared : 0)) * 4096, 1, 1024);
    float Oa[64], Ob[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) { Oa[i] = 0.f; Ob[i] = 0.f; }
    float maA = -INFINITY, maB = -INFINITY, laA = 0.f, laB = 0.f;
    float mbA = -INFINITY, mbB = -INFINITY, lbA = 0.f, lbB = 0.f;
    int ph = 0, s = 0;
    uint32_t P[28];
    uint32_t naA2 = 0u, naB2 = 0u, nbA2 = 0u, nbB2 = 0u;
#pragma unroll 1
    for (int t = 0; t < ntiles; ++t) {
      mbar_wait(bar + s, ph);
      mbar_wait(bar + NSTAGE + s, ph);
      const int uniform_s = __shfl_sync(FULLM, s, 0);
      const uint64_t dk = dK0 + (uint64_t)(uniform_s * KSTG_D);
      const uint64_t dv = dV0 + (uint64_t)(uniform_s * KSTG_D);
      const int valid = kvlen - t * KTILE;
      if (priv_a >= 0) {
        qk_block_f16(P, dQa, dk, QB);
        softmax_block_f16(P, Oa, maA, maB, laA, laB, naA2, naB2, valid, q4);
        pv_block(Oa, P, dv);
      }
      if (priv_b >= 0 || (shared >= 0 && (t & 1) == wg)) {
        qk_block_f16(P, dQb, dk, QB);
        softmax_block_f16(P, Ob, mbA, mbB, lbA, lbB, nbA2, nbB2, valid, q4);
        pv_block(Ob, P, dv);
      }
      mbar_arrive(bar + 2 * NSTAGE + s);
      mbar_arrive(bar + 3 * NSTAGE + s);
      if (++s == NSTAGE) { s = 0; ph ^= 1; }
    }

    const size_t rowbase = (size_t)h * N + qs;
    if (priv_a >= 0) store_block(Oa, maA, maB, laA, laB, out, lse, rowbase, 64 * priv_a + rowA, 64 * priv_a + rowA + 8, qrows, q4);
    if (priv_b >= 0) store_block(Ob, mbA, mbB, lbA, lbB, out, lse, rowbase, 64 * priv_b + rowA, 64 * priv_b + rowA + 8, qrows, q4);
    if (shared >= 0) {
      // deterministic two-way merge of the shared block: warpgroup 1 publishes, warpgroup 0 combines (fixed order).
      float *xO = (float *)sQ;                      // 128 x 64 floats (32 KB) + 128 x 4 floats
      float *xM = xO + 128 * 64;
      const int ct = tid & 127;
      cons_sync();                                   // all QK reads of sQ finished
      if (wg == 1) {
#pragma unroll
        for (int i = 0; i < 64; ++i) xO[i * 128 + ct] = Ob[i];
        xM[ct * 4 + 0] = mbA; xM[ct * 4 + 1] = mbB; xM[ct * 4 + 2] = lbA; xM[ct * 4 + 3] = lbB;
      }
      cons_sync();
      if (wg == 0) {
        const float m1A = xM[ct * 4 + 0], m1B = xM[ct * 4 + 1], l1A = xM[ct * 4 + 2], l1B = xM[ct * 4 + 3];
        const float MA = fmaxf(mbA, m1A), MB = fmaxf(mbB, m1B);
        const float w0A = exp2f(mbA - MA), w1A = exp2f(m1A - MA);
        const float w0B = exp2f(mbB - MB), w1B = exp2f(m1B - MB);
#pragma unroll
        for (int i = 0; i < 64; ++i) {
          const float o1 = xO[i * 128 + ct];
          Ob[i] = (i & 2) ? fmaf(Ob[i], w0B, o1 * w1B) : fmaf(Ob[i], w0A, o1 * w1A);
        }
        lbA = fmaf(lbA, w0A, l1A * w1A);
        lbB = fmaf(lbB, w0B, l1B * w1B);
        mbA = MA; mbB = MB;
        store_block(Ob, mbA, mbB, lbA, lbB, out, lse, rowbase, 64 * shared + rowA, 64 * shared + rowA + 8, qrows, q4);
      }
    }
    reg_dealloc32();
    }
  }
}

static int g_nsm = 0;
static bool g_attr = false;

void launch_cluster_attn(const void *q, const void *kp, const void *vp, const void *ub,
                         const void *qperm, void *out, void *lse, const void *qsz,
                         const void *ksz, int H, int N, int CQ, int *ws,
                         bool rebuild_plan, cudaStream_t stream) {
  if (g_nsm == 0) {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, dev);
  }
  const int smem = (2 * QTILE * 64 + 2 * NSTAGE * KSTG) * 2 + 4 * NSTAGE * 8 + (256 + 2073) * 4;
  if (!g_attr) {
    cudaFuncSetAttribute(attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    g_attr = true;
  }
  int *k_off = ws;
  int *counters = ws + (size_t)H * (CKC + 1);
  long toff = ((long)H * (CKC + 1) + 4 + 3) & ~3L;
  int4 *tasks = (int4 *)(ws + toff);

  const int qt=N>=8192?128:64;
  const long max_tasks=(long)H*(CQ+(N+qt-1)/qt+1);
  int* kv_lens=ws+toff+4*max_tasks;
  if (rebuild_plan) {
    cudaMemsetAsync(counters, 0, 3 * sizeof(int), stream);
    selected_kv_lengths_kernel<<<dim3((CQ + 7) / 8, H), 256, 0, stream>>>(
        (const int *)qsz, (const int *)ksz, (const uint32_t *)ub, kv_lens, CQ);
    plan_kernel<<<H, 256, 0, stream>>>(
        (const int *)qsz, (const int *)ksz, H, N, CQ, k_off, tasks, counters, kv_lens);
  } else {
    // k_off, tasks, and counters[0] depend only on the immutable snapshot
    // cluster sizes.  The persistent-CTA task cursor is the only per-dispatch
    // state in a prepared plan.
    cudaMemsetAsync(counters + 1, 0, 2 * sizeof(int), stream);
  }
  attn_kernel<<<g_nsm, NTHREADS, smem, stream>>>(
      (const half *)q, (const half *)kp, (const half *)vp, (const uint8_t *)ub,
      (const int *)qperm, k_off, (half *)out, (float *)lse, tasks, counters, H, N, CQ,kv_lens);
}

// Batched cluster-tail correction and production merge. One warpgroup handles
// 64 sorted queries and streams the 2048 centroids in 64-column tiles.
#define TAIL_Q 128
#define TAIL_C 64
#define TAIL_PAD 64
#define TAIL_VBLK (TAIL_C * 64)
#define TAIL_CENTROID_SMEM \
  (2 * TAIL_C * TAIL_PAD * (int)sizeof(half) + 2 * TAIL_C * 128 * (int)sizeof(half) + \
   2 * TAIL_Q * (int)sizeof(unsigned long long) + 2 * TAIL_C * (int)sizeof(int))

#define TAIL_MARGIN 2.0f
// 136 halfs puts consecutive rows four banks apart for the accumulator fragment write pattern.
#define TAIL_PITCH 136
#define TAIL_MERGE_SMEM (TAIL_Q * TAIL_PITCH * (int)sizeof(half))
// Centroid staging is dead before merge, so both phases share the same dynamic-smem allocation.
#define TAIL_STAGE_SMEM \
  (TAIL_CENTROID_SMEM > TAIL_MERGE_SMEM ? TAIL_CENTROID_SMEM : TAIL_MERGE_SMEM)


__device__ __forceinline__ void tail_tma_load(half* dst,const CUtensorMap* map,uint64_t* bar,int x,int y,int h) {
  const uint32_t d=(uint32_t)__cvta_generic_to_shared(dst),b=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2,%3,%4}], [%5];"
               :: "r"(d),"l"(map),"r"(x),"r"(y),"r"(h),"r"(b):"memory");
}
__device__ __forceinline__ void tail_tma_stage(half* c,half* v,const CUtensorMap* cm,const CUtensorMap* vm,uint64_t* bar,int tile,int h) {
  const uint32_t b=(uint32_t)__cvta_generic_to_shared(bar);
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 24576;" ::"r"(b):"memory");
  tail_tma_load(c,cm,bar,0,tile*64,h);
  tail_tma_load(v,vm,bar,0,tile*64,h);
  tail_tma_load(v+4096,vm,bar,64,tile*64,h);
}

__global__ __launch_bounds__(256, 2) void tail_wgmma_kernel(
    const half *__restrict__ attn_out, const float *__restrict__ attn_lse,
    const half *__restrict__ QU, const half *__restrict__ cks,
    const half *__restrict__ sumv, const int *__restrict__ ksz,
    const uint8_t *__restrict__ ub, const int *__restrict__ qlab,
    const int *__restrict__ qperm, const float *__restrict__ lam_s,
    const float *__restrict__ tailw_ptr, half *__restrict__ out,
    int N, int CQ, const __grid_constant__ CUtensorMap cm, const __grid_constant__ CUtensorMap vm) {
  __shared__ __align__(1024) half sQ[TAIL_Q * TAIL_PAD];
  __shared__ int sOrig[TAIL_Q];
  __shared__ int sLabel[TAIL_Q];
  extern __shared__ __align__(1024) char tail_stage_raw[];
  half (*sC)[TAIL_C * TAIL_PAD] = (half (*)[TAIL_C * TAIL_PAD])tail_stage_raw;
  half (*sV)[TAIL_C * 128] = (half (*)[TAIL_C * 128])(sC + 2);
  unsigned long long (*sMask)[TAIL_Q] = (unsigned long long (*)[TAIL_Q])(sV + 2);
  float (*sCnt)[TAIL_C] = (float (*)[TAIL_C])(sMask + 2);

  __shared__ __align__(8) uint64_t tbar[2];
  const int tid = threadIdx.x;
  if(tid<2)mbar_init(tbar+tid,1);
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  const int warpgroup = __shfl_sync(0xffffffffu,tid >> 7,0);
  const int wtid = tid & 127;
  const int lane = wtid & 31;
  const int warpid = wtid >> 5;
  const int q4 = lane & 3;
  const int row_base = warpgroup * 64;
  const int rowA = row_base + 16 * warpid + (lane >> 2);
  const int rowB = rowA + 8;
  const int h = blockIdx.y;
  const int qs = blockIdx.x * TAIL_Q;

  if (tid < TAIL_Q) {
    const int j = qs + tid;
    const int orig = (j < N) ? qperm[(size_t)h * N + j] : 0;
    sOrig[tid] = orig;
    sLabel[tid] = qlab[(size_t)h * N + orig];
  }
  __syncthreads();

  // Gather the randomly permuted rank-32 query rows into the canonical SW128
  // physical layout expected by the SS WGMMA descriptor.
  for (int slot = tid; slot < TAIL_Q * 4; slot += 256) {
    const int r = slot >> 2;
    const int cc = slot & 3;
    const int j = qs + r;
    const half *src = QU + ((size_t)h * N + sOrig[r]) * 32 + cc * 8;
    const uint32_t dst = (uint32_t)__cvta_generic_to_shared(sQ + swz(r, cc));
    cp16(dst, src, j < N ? 16 : 0);
  }
  cp_commit();
  cp_wait<0>();
  __syncthreads();

  const uint64_t dQ = make_desc(sQ + warpgroup * 64 * TAIL_PAD, 1, 1024);
  const uint64_t dC0 = make_desc(sC[0], 1, 1024);
  const uint64_t dC1 = make_desc(sC[1], 1, 1024);
  const uint64_t dV0 = make_desc(sV[0], TAIL_VBLK * 2, 1024);
  const uint64_t dV1 = make_desc(sV[1], TAIL_VBLK * 2, 1024);

  if (tid < TAIL_C) sCnt[0][tid] = (float)ksz[(size_t)h * CKC + tid];
  if (tid < TAIL_Q) {
    const uint8_t *bits = ub + ((size_t)h * CQ + sLabel[tid]) * (CKC / 8);
    sMask[0][tid] = *(const unsigned long long *)bits;
  }
  if(tid==0)tail_tma_stage(sC[0],sV[0],&cm,&vm,tbar,0,h);
  mbar_wait(tbar,0);
  __syncthreads();

  float O[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) O[i] = 0.0f;
  float mA = -INFINITY, mB = -INFINITY;
  float phiA = 0.0f, phiB = 0.0f;

#pragma unroll 1
  for (int tile = 0; tile < CKC / TAIL_C; ++tile) {
    const int stage = tile & 1;
    const uint64_t dC = stage ? dC1 : dC0;
    const uint64_t dV = stage ? dV1 : dV0;

    float S[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) S[i] = 0.0f;
    wg_fence();
    WGMMA64_SS(S, dQ, dC, 0);
    WGMMA64_SS(S, dQ + 2, dC + 2, 1);
    wg_commit();

    if (tile + 1 < CKC / TAIL_C) {
      const int next_stage = stage ^ 1;
      const int next_tile = tile + 1;
      const int c0 = next_tile * TAIL_C;
      if (tid < TAIL_C) sCnt[next_stage][tid] = (float)ksz[(size_t)h * CKC + c0 + tid];
      if (tid < TAIL_Q) {
        const uint8_t *bits =
            ub + ((size_t)h * CQ + sLabel[tid]) * (CKC / 8) + next_tile * 8;
        sMask[next_stage][tid] = *(const unsigned long long *)bits;
      }
      if(tid==0)tail_tma_stage(sC[next_stage],sV[next_stage],&cm,&vm,tbar+next_stage,next_tile,h);
    }

    const unsigned long long mwA = sMask[stage][rowA] >> (2 * q4);
    const unsigned long long mwB = sMask[stage][rowB] >> (2 * q4);
    wg_wait<0>();

    // The reference explicitly rounds gl to fp16 before rowmax and exp.  Round
    // two accumulators at a time (cvt.rn.f16x2) instead of one at a time.
#pragma unroll
    for (int k = 0; k < 16; ++k) {
      const uint32_t hp = pack2(S[2 * k], S[2 * k + 1]);
      S[2 * k] = __low2float(*(const half2 *)&hp);
      S[2 * k + 1] = __high2float(*(const half2 *)&hp);
    }

    float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      mxA = fmaxf(mxA, fmaxf(S[4 * g], S[4 * g + 1]));
      mxB = fmaxf(mxB, fmaxf(S[4 * g + 2], S[4 * g + 3]));
    }
    mxA = fmaxf(mxA, __shfl_xor_sync(FULLM, mxA, 1));
    mxB = fmaxf(mxB, __shfl_xor_sync(FULLM, mxB, 1));
    mxA = fmaxf(mxA, __shfl_xor_sync(FULLM, mxA, 2));
    mxB = fmaxf(mxB, __shfl_xor_sync(FULLM, mxB, 2));
    // Exact max-margin: overshoot the running max by TAIL_MARGIN natural-log
    // units whenever it advances.  exp() is shift-invariant and the shift
    // cancels in st*Phi / st*phi, so Phi/phi/out are unchanged; but later
    // centroid tiles then almost never advance the max, which lets the
    // alpha==1 guard skip the 64-register O rescale on most of the 32 tiles.
    const float nmA = (mxA > mA) ? (mxA + TAIL_MARGIN) : mA;
    const float nmB = (mxB > mB) ? (mxB + TAIL_MARGIN) : mB;
    const float alphaA = expf(mA - nmA);
    const float alphaB = expf(mB - nmB);
    mA = nmA;
    mB = nmB;

    if (__any_sync(FULLM, (alphaA != 1.0f) | (alphaB != 1.0f))) {
#pragma unroll
      for (int i = 0; i < 64; ++i) O[i] *= (i & 2) ? alphaB : alphaA;
    }

    // Hoist the two 64-bit selection words (pre-shifted by this lane's column
    // offset) and turn exp(gl-m) into a single fma + ex2; the counts are already
    // floats in smem, so the inner body is 4 instructions per centroid.
    const float eaA = -nmA * 1.4426950408889634f, eaB = -nmB * 1.4426950408889634f;
    float psA = 0.0f, psB = 0.0f, psA2 = 0.0f, psB2 = 0.0f;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      const int col = (g << 3) + 2 * q4;
      const float c0 = sCnt[stage][col], c1 = sCnt[stage][col + 1];
      const int b = g << 3;
      const float p0 =
          ((mwA >> b) & 1ull) ? 0.0f : exp2f(fmaf(S[4 * g + 0], 1.4426950408889634f, eaA));
      const float p1 =
          ((mwA >> (b + 1)) & 1ull) ? 0.0f : exp2f(fmaf(S[4 * g + 1], 1.4426950408889634f, eaA));
      const float p2 =
          ((mwB >> b) & 1ull) ? 0.0f : exp2f(fmaf(S[4 * g + 2], 1.4426950408889634f, eaB));
      const float p3 =
          ((mwB >> (b + 1)) & 1ull) ? 0.0f : exp2f(fmaf(S[4 * g + 3], 1.4426950408889634f, eaB));
      S[4 * g + 0] = p0;
      S[4 * g + 1] = p1;
      S[4 * g + 2] = p2;
      S[4 * g + 3] = p3;
      psA = fmaf(p0, c0, psA);
      psA2 = fmaf(p1, c1, psA2);
      psB = fmaf(p2, c0, psB);
      psB2 = fmaf(p3, c1, psB2);
    }
    psA += psA2;
    psB += psB2;
    psA += __shfl_xor_sync(FULLM, psA, 1);
    psB += __shfl_xor_sync(FULLM, psB, 1);
    psA += __shfl_xor_sync(FULLM, psA, 2);
    psB += __shfl_xor_sync(FULLM, psB, 2);
    phiA = phiA * alphaA + psA;
    phiB = phiB * alphaB + psB;

    uint32_t a[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; ++kk) {
      a[kk][0] = pack2(S[8 * kk + 0], S[8 * kk + 1]);
      a[kk][1] = pack2(S[8 * kk + 2], S[8 * kk + 3]);
      a[kk][2] = pack2(S[8 * kk + 4], S[8 * kk + 5]);
      a[kk][3] = pack2(S[8 * kk + 6], S[8 * kk + 7]);
    }
    wg_fence();
    WGMMA128_RS(O, a[0][0], a[0][1], a[0][2], a[0][3], dV);
    WGMMA128_RS(O, a[1][0], a[1][1], a[1][2], a[1][3], dV + 128);
    WGMMA128_RS(O, a[2][0], a[2][1], a[2][2], a[2][3], dV + 256);
    WGMMA128_RS(O, a[3][0], a[3][1], a[3][2], a[3][3], dV + 384);
    wg_commit();
    wg_wait<0>();
    if (tile + 1 < CKC / TAIL_C) {
      mbar_wait(tbar+((tile+1)&1),((tile+1)/2)&1);
      __syncthreads();
    }
  }

  const float lam = lam_s[h];
  const float tw = *tailw_ptr;
  const int jA = qs + rowA;
  const int jB = qs + rowB;
  const bool activeA = jA < N;
  const bool activeB = jB < N;
  const int origA = sOrig[rowA];
  const int origB = sOrig[rowB];
  const float leA = activeA ? attn_lse[(size_t)h * N + jA] * 0.6931471805599453f : -INFINITY;
  const float leB = activeB ? attn_lse[(size_t)h * N + jB] * 0.6931471805599453f : -INFINITY;
  float exA, exB, stA, stB;
  if (leA >= mA) {
    exA = 1.0f;
    stA = expf(mA - leA);
  } else {
    exA = expf(leA - mA);
    stA = 1.0f;
  }
  if (leB >= mB) {
    exB = 1.0f;
    stB = expf(mB - leB);
  } else {
    exB = expf(leB - mB);
    stB = 1.0f;
  }
  const float phA = stA * phiA, phB = stB * phiB;
  const bool badA = phA <= 1.0e-6f * exA;
  const bool badB = phB <= 1.0e-6f * exB;
  const float denA = fmaxf(exA + phA * tw, 1.0e-12f);
  const float denB = fmaxf(exB + phB * tw, 1.0e-12f);
  const float invDenA = 1.0f / denA;
  const float invDenB = 1.0f / denB;
  const float exactWeightA = exA * invDenA;
  const float exactWeightB = exB * invDenB;
  const float tailWeightA = stA * tw * lam * invDenA;
  const float tailWeightB = stB * tw * lam * invDenB;

  // Reuse the dead centroid buffer as a transposition scratch. Global traffic is 16 B/lane in
  // row order; FP32 merge arithmetic and the pre-tailw negligible-tail test are unchanged.
  half *sMerge = (half *)tail_stage_raw;
  __syncthreads();
  for (int i = tid; i < TAIL_Q * 16; i += 256) {
    const int r = i >> 4, c = i & 15;
    const int j = qs + r;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (j < N) v = ((const uint4 *)(attn_out + ((size_t)h * N + j) * DIM))[c];
    ((uint4 *)(sMerge + r * TAIL_PITCH))[c] = v;
  }
  __syncthreads();
  if (activeA && !badA) {
    uint32_t *pA = (uint32_t *)(sMerge + rowA * TAIL_PITCH) + q4;
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      const uint32_t sv = pA[4 * g];
      const half2 h2 = *(const half2 *)&sv;
      const float x0 = fmaf(tailWeightA, O[4 * g], exactWeightA * __low2float(h2));
      const float x1 = fmaf(tailWeightA, O[4 * g + 1], exactWeightA * __high2float(h2));
      pA[4 * g] = pack2(x0, x1);
    }
  }
  if (activeB && !badB) {
    uint32_t *pB = (uint32_t *)(sMerge + rowB * TAIL_PITCH) + q4;
#pragma unroll
    for (int g = 0; g < 16; ++g) {
      const uint32_t sv = pB[4 * g];
      const half2 h2 = *(const half2 *)&sv;
      const float x0 = fmaf(tailWeightB, O[4 * g + 2], exactWeightB * __low2float(h2));
      const float x1 = fmaf(tailWeightB, O[4 * g + 3], exactWeightB * __high2float(h2));
      pB[4 * g] = pack2(x0, x1);
    }
  }
  __syncthreads();
  for (int i = tid; i < TAIL_Q * 16; i += 256) {
    const int r = i >> 4, c = i & 15;
    if (qs + r < N) {
      ((uint4 *)(out + ((size_t)h * N + sOrig[r]) * DIM))[c] =
          ((const uint4 *)(sMerge + r * TAIL_PITCH))[c];
    }
  }
}


static CUtensorMap encode_tail_map(const void* ptr,int width,int H) {
  static thread_local std::map<std::tuple<uintptr_t,int,int>,CUtensorMap> cache;
  auto key=std::make_tuple((uintptr_t)ptr,width,H);
  auto it=cache.find(key);
  if(it!=cache.end())return it->second;
  alignas(64) CUtensorMap m;
  const uint64_t shape[3]={(uint64_t)width,2048,(uint64_t)H};
  const uint64_t strides[2]={(uint64_t)width*2,(uint64_t)width*2*2048};
  const uint32_t box[3]={64,64,1}, elem[3]={1,1,1};
  static auto encode=[]() {
    void* driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
    if(!driver) throw std::runtime_error("Cannot load CUDA driver for tensor map encoding");
    auto fn=reinterpret_cast<decltype(&cuTensorMapEncodeTiled)>(dlsym(driver,"cuTensorMapEncodeTiled"));
    if(!fn) throw std::runtime_error("CUDA driver lacks cuTensorMapEncodeTiled");
    return fn;
  }();
  CUresult err=encode(&m,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,3,const_cast<void*>(ptr),shape,strides,box,elem,
    CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if(err!=CUDA_SUCCESS)throw std::runtime_error("tail TMA descriptor encoding failed: "+std::to_string((int)err));
  cache.emplace(key,m);
  return m;
}

static bool g_tail_attr = false;

void launch_tail_merge(const void *attn_out, const void *attn_lse, const void *QU,
                       const void *cks, const void *sumv, const void *ksz,
                       const void *union_bits, const void *qlab, const void *qperm,
                       const void *lam_s, const void *tailw, void *out,
                       int H, int N, int CQ, cudaStream_t stream) {
  if (!g_tail_attr) {
    cudaFuncSetAttribute(tail_wgmma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)TAIL_STAGE_SMEM);
    g_tail_attr = true;
  }
  const CUtensorMap cm=encode_tail_map(cks,32,H),vm=encode_tail_map(sumv,128,H);
  dim3 grid((N + TAIL_Q - 1) / TAIL_Q, H);
  tail_wgmma_kernel<<<grid, 256, TAIL_STAGE_SMEM, stream>>>(
      (const half *)attn_out, (const float *)attn_lse, (const half *)QU,
      (const half *)cks, (const half *)sumv, (const int *)ksz,
      (const uint8_t *)union_bits, (const int *)qlab, (const int *)qperm,
      (const float *)lam_s, (const float *)tailw, (half *)out, N, CQ, cm, vm);
}
