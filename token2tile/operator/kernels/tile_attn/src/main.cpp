#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

void launch_cluster_attn(const void *q, const void *kp, const void *vp, const void *ub,
                         const void *qperm, void *out, void *lse, const void *qsz,
                         const void *ksz, int H, int N, int CQ, int *ws,
                         bool rebuild_plan, cudaStream_t stream);

void launch_tail_merge(const void *attn_out, const void *attn_lse, const void *QU,
                       const void *cks, const void *sumv, const void *ksz,
                       const void *union_bits, const void *qlab, const void *qperm,
                       const void *lam_s, const void *tailw, void *out,
                       int H, int N, int CQ, cudaStream_t stream);

static long plan_workspace_ints(int H, int N, int CQ) {
  const int qtile = N >= 8192 ? 128 : 64;
  const long max_tasks =
      static_cast<long>(H) * (CQ + (N + qtile - 1) / qtile + 1);
  const long task_offset = (static_cast<long>(H) * 2049 + 4 + 3) & ~3L;
  return task_offset + 4 * max_tasks + (long)H*CQ;
}

static void run_impl(torch::Tensor q, torch::Tensor kp, torch::Tensor vp,
                     torch::Tensor q_cluster_size,
                     torch::Tensor k_cluster_size,
                     torch::Tensor union_bits, torch::Tensor QU,
                     torch::Tensor cks, torch::Tensor sumv,
                     torch::Tensor lam_s, torch::Tensor qlab,
                     torch::Tensor qperm, torch::Tensor tailw,
                     torch::Tensor out, torch::Tensor plan_workspace,
                     bool rebuild_plan) {
  const int H = static_cast<int>(q.size(0));
  const int N = static_cast<int>(q.size(1));
  const int CQ = static_cast<int>(q_cluster_size.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();

  auto attn_out = torch::empty_like(out);
  auto attn_lse = torch::empty(
      {H, N}, torch::TensorOptions().dtype(torch::kFloat32).device(q.device()));

  launch_cluster_attn(q.data_ptr(), kp.data_ptr(), vp.data_ptr(), union_bits.data_ptr(),
                      qperm.data_ptr(), attn_out.data_ptr(), attn_lse.data_ptr(),
                      q_cluster_size.data_ptr(), k_cluster_size.data_ptr(), H, N, CQ,
                      plan_workspace.data_ptr<int>(), rebuild_plan, stream.stream());
  launch_tail_merge(attn_out.data_ptr(), attn_lse.data_ptr(), QU.data_ptr(), cks.data_ptr(),
                    sumv.data_ptr(), k_cluster_size.data_ptr(), union_bits.data_ptr(),
                    qlab.data_ptr(), qperm.data_ptr(), lam_s.data_ptr(), tailw.data_ptr(),
                    out.data_ptr(), H, N, CQ, stream.stream());
}

void run(torch::Tensor q, torch::Tensor kp, torch::Tensor vp,
         torch::Tensor q_cluster_size, torch::Tensor k_cluster_size,
         torch::Tensor union_bits, torch::Tensor QU, torch::Tensor cks,
         torch::Tensor sumv, torch::Tensor lam_s, torch::Tensor qlab,
         torch::Tensor qperm, torch::Tensor tailw, torch::Tensor out) {
  const int H = static_cast<int>(q.size(0));
  const int N = static_cast<int>(q.size(1));
  const int CQ = static_cast<int>(q_cluster_size.size(1));
  auto workspace = torch::empty(
      {plan_workspace_ints(H, N, CQ)},
      torch::TensorOptions().dtype(torch::kInt32).device(q.device()));
  run_impl(q, kp, vp, q_cluster_size, k_cluster_size, union_bits, QU, cks,
           sumv, lam_s, qlab, qperm, tailw, out, workspace, true);
}

void run_planned(torch::Tensor q, torch::Tensor kp, torch::Tensor vp,
                 torch::Tensor q_cluster_size,
                 torch::Tensor k_cluster_size,
                 torch::Tensor union_bits, torch::Tensor QU,
                 torch::Tensor cks, torch::Tensor sumv,
                 torch::Tensor lam_s, torch::Tensor qlab,
                 torch::Tensor qperm, torch::Tensor tailw,
                 torch::Tensor out, torch::Tensor plan_workspace,
                 bool rebuild_plan) {
  run_impl(q, kp, vp, q_cluster_size, k_cluster_size, union_bits, QU, cks,
           sumv, lam_s, qlab, qperm, tailw, out, plan_workspace,
           rebuild_plan);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "Fused cluster-union attention tail merge");
  m.def("run_planned", &run_planned,
        "Fused attention using a snapshot-owned task plan");
}
