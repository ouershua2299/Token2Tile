#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>

void kmeans_maintain_launch(
    const void* QU, const void* KU, const void* V,
    const void* cq0, const void* ck0,
    void* qlab, void* cq2, void* klab, void* ck2,
    void* qcounts, void* qperm, void* kcounts, void* kperm,
    void* V_sorted, void* sumv, void* lam_s,
    int H, int N, int CQ, int CK, int pin_tail,
    void* ws, cudaStream_t stream);

size_t kmeans_maintain_ws_bytes(int H, int N, int CQ, int CK);

void reuse_vstats_launch(
    const void* V_sorted, const void* kcounts,
    void* sumv, void* lam_s,
    int H, int N, int CK,
    void* ws, cudaStream_t stream);

void reuse_vstats_gather_launch(
    const void* V, const void* kperm, const void* kcounts,
    void* V_sorted, void* sumv, void* lam_s,
    int H, int N, int CK,
    void* ws, cudaStream_t stream);

size_t reuse_vstats_ws_bytes(int H, int CK);
