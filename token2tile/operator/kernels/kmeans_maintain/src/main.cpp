#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include "kernels.h"

namespace py = pybind11;

static void* g_ws = nullptr;
static size_t g_ws_bytes = 0;

static void* ensure_ws(size_t bytes) {
    if (bytes > g_ws_bytes) {
        if (g_ws) cudaFree(g_ws);
        cudaMalloc(&g_ws, bytes);
        g_ws_bytes = bytes;
    }
    return g_ws;
}

void run(at::Tensor QU, at::Tensor KU, at::Tensor V, at::Tensor cq0, at::Tensor ck0,
         at::Tensor qlab, at::Tensor cq2, at::Tensor klab, at::Tensor ck2,
         at::Tensor qcounts, at::Tensor qperm,
         at::Tensor kcounts, at::Tensor kperm,
         at::Tensor V_sorted, at::Tensor sumv, at::Tensor lam_s,
         int pin_tail) {
    int H = (int)QU.size(0);
    int N = (int)QU.size(1);
    int CQ = (int)cq0.size(1);
    int CK = (int)ck0.size(1);
    TORCH_CHECK(pin_tail >= 0 && pin_tail <= N,
                "pin_tail must be in [0, N]");
    TORCH_CHECK(pin_tail == 0 || CK > 0,
                "pinned K maintenance requires a reserved cluster");

    size_t need = kmeans_maintain_ws_bytes(H, N, CQ, CK);
    void* ws = ensure_ws(need);

    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    kmeans_maintain_launch(QU.data_ptr(), KU.data_ptr(), V.data_ptr(),
                           cq0.data_ptr(), ck0.data_ptr(),
                           qlab.data_ptr(), cq2.data_ptr(), klab.data_ptr(),
                           ck2.data_ptr(), qcounts.data_ptr(), qperm.data_ptr(),
                           kcounts.data_ptr(), kperm.data_ptr(),
                           V_sorted.data_ptr(), sumv.data_ptr(), lam_s.data_ptr(),
                           H, N, CQ, CK, pin_tail, ws, st);
}

void run_vstats(at::Tensor V_sorted, at::Tensor kcounts,
                at::Tensor sumv, at::Tensor lam_s) {
    int H = (int)V_sorted.size(0);
    int N = (int)V_sorted.size(1);
    int CK = (int)kcounts.size(1);

    size_t need = reuse_vstats_ws_bytes(H, CK);
    void* ws = ensure_ws(need);

    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    reuse_vstats_launch(V_sorted.data_ptr(), kcounts.data_ptr(),
                        sumv.data_ptr(), lam_s.data_ptr(),
                        H, N, CK, ws, st);
}

void run_vstats_gather(at::Tensor V, at::Tensor kperm,
                       at::Tensor kcounts, at::Tensor V_sorted,
                       at::Tensor sumv, at::Tensor lam_s) {
    int H = (int)V.size(0);
    int N = (int)V.size(1);
    int CK = (int)kcounts.size(1);

    size_t need = reuse_vstats_ws_bytes(H, CK);
    void* ws = ensure_ws(need);

    cudaStream_t st = at::cuda::getCurrentCUDAStream();
    reuse_vstats_gather_launch(
        V.data_ptr(), kperm.data_ptr(), kcounts.data_ptr(),
        V_sorted.data_ptr(), sumv.data_ptr(), lam_s.data_ptr(),
        H, N, CK, ws, st);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "fused kmeans maintenance",
          py::arg("QU"), py::arg("KU"), py::arg("V"),
          py::arg("cq0"), py::arg("ck0"), py::arg("qlab"),
          py::arg("cq2"), py::arg("klab"), py::arg("ck2"),
          py::arg("qcounts"), py::arg("qperm"), py::arg("kcounts"),
          py::arg("kperm"), py::arg("V_sorted"), py::arg("sumv"),
          py::arg("lam_s"), py::arg("pin_tail") = 0);
    m.def("run_vstats", &run_vstats,
          "reuse V sums from cluster-sorted V");
    m.def("run_vstats_gather", &run_vstats_gather,
          "gather cluster-sorted V and compute reuse V sums");
}
