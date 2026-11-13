#pragma once
#include <cuda_runtime.h>

void fscu_launch(const void* gl, const void* sizes, const void* rowlist,
                 const void* tile_sizes, const void* ns,
                 void* uni, void* rowcnt,
                 void* off,
                 int H, int N, int CQ, cudaStream_t stream);

void fscu_split_launch(const void* gl, const void* sizes, const void* rowlist,
                       const void* tile_sizes, int cap, const void* ns,
                       void* uni, void* rowcnt,
                       void* virtual_sizes, void* off,
                       void* vbase, void* nsplit,
                       int H, int N, int CQ, int CQV,
                       cudaStream_t stream);

void fscu_packed_launch(const void* gl, const void* sizes,
                        const void* rowlist, const void* tile_sizes,
                        const void* ns, void* partial, void* packed,
                        void* rowcnt, void* off,
                        int H, int N, int CQ, cudaStream_t stream);

void fscu_split_packed_launch(
    const void* gl, const void* sizes, const void* rowlist,
    const void* tile_sizes, int cap, const void* ns,
    void* partial, void* packed, void* rowcnt,
    void* virtual_sizes, void* off, void* vbase, void* nsplit,
    int H, int N, int CQ, int CQV, cudaStream_t stream);
