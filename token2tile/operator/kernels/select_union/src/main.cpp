#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "kernel.h"

void run(const torch::Tensor& gl, const torch::Tensor& sizes,
         const torch::Tensor& rowlist, const torch::Tensor& tile_sizes,
         const torch::Tensor& ns, torch::Tensor& uni, torch::Tensor& rowcnt) {
    const int H = (int)gl.size(0);
    const int N = (int)gl.size(1);
    const int CQ = (int)tile_sizes.size(1);
    auto opts = torch::TensorOptions().dtype(torch::kInt32).device(gl.device());
    auto off = torch::empty({H, CQ}, opts);
    fscu_launch(gl.data_ptr(), sizes.data_ptr(), rowlist.data_ptr(),
                tile_sizes.data_ptr(), ns.data_ptr(),
                uni.data_ptr(), rowcnt.data_ptr(),
                off.data_ptr(),
                H, N, CQ, at::cuda::getCurrentCUDAStream().stream());
}

void run_split(const torch::Tensor& gl, const torch::Tensor& sizes,
               const torch::Tensor& rowlist,
               const torch::Tensor& tile_sizes, int64_t cap,
               const torch::Tensor& ns, torch::Tensor& uni,
               torch::Tensor& rowcnt, torch::Tensor& virtual_sizes,
               torch::Tensor& off, torch::Tensor& vbase,
               torch::Tensor& nsplit) {
    const int H = (int)gl.size(0);
    const int N = (int)gl.size(1);
    const int CQ = (int)tile_sizes.size(1);
    const int CQV = (int)virtual_sizes.size(1);
    fscu_split_launch(gl.data_ptr(), sizes.data_ptr(), rowlist.data_ptr(),
                      tile_sizes.data_ptr(), (int)cap, ns.data_ptr(),
                      uni.data_ptr(), rowcnt.data_ptr(),
                      virtual_sizes.data_ptr(), off.data_ptr(),
                      vbase.data_ptr(), nsplit.data_ptr(),
                      H, N, CQ, CQV,
                      at::cuda::getCurrentCUDAStream().stream());
}

void run_packed(const torch::Tensor& gl, const torch::Tensor& sizes,
                const torch::Tensor& rowlist,
                const torch::Tensor& tile_sizes, const torch::Tensor& ns,
                torch::Tensor& partial, torch::Tensor& packed,
                torch::Tensor& rowcnt, torch::Tensor& off) {
    const int H = (int)gl.size(0);
    const int N = (int)gl.size(1);
    const int CQ = (int)tile_sizes.size(1);
    fscu_packed_launch(
        gl.data_ptr(), sizes.data_ptr(), rowlist.data_ptr(),
        tile_sizes.data_ptr(), ns.data_ptr(), partial.data_ptr(),
        packed.data_ptr(), rowcnt.data_ptr(), off.data_ptr(),
        H, N, CQ, at::cuda::getCurrentCUDAStream().stream());
}

void run_split_packed(
        const torch::Tensor& gl, const torch::Tensor& sizes,
        const torch::Tensor& rowlist, const torch::Tensor& tile_sizes,
        int64_t cap, const torch::Tensor& ns, torch::Tensor& partial,
        torch::Tensor& packed, torch::Tensor& rowcnt,
        torch::Tensor& virtual_sizes, torch::Tensor& off,
        torch::Tensor& vbase, torch::Tensor& nsplit) {
    const int H = (int)gl.size(0);
    const int N = (int)gl.size(1);
    const int CQ = (int)tile_sizes.size(1);
    const int CQV = (int)virtual_sizes.size(1);
    fscu_split_packed_launch(
        gl.data_ptr(), sizes.data_ptr(), rowlist.data_ptr(),
        tile_sizes.data_ptr(), (int)cap, ns.data_ptr(), partial.data_ptr(),
        packed.data_ptr(), rowcnt.data_ptr(), virtual_sizes.data_ptr(),
        off.data_ptr(), vbase.data_ptr(), nsplit.data_ptr(),
        H, N, CQ, CQV, at::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, "per-query cluster selection and query-cluster union");
    m.def("run_split", &run_split,
          "the same with large query clusters split into sub-tiles");
    m.def("run_packed", &run_packed,
          "per-query cluster selection and packed query-cluster union");
    m.def("run_split_packed", &run_split_packed,
          "the packed variant with large query clusters split");
}
