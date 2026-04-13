
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>

void batch_h2d_dispatch(
    const std::vector<std::tuple<int64_t, int64_t, int64_t>>& ops,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    py::gil_scoped_release release;
    for (const auto& op : ops) {
        auto src = reinterpret_cast<const void*>(std::get<0>(op));
        auto dst = reinterpret_cast<void*>(std::get<1>(op));
        auto nbytes = static_cast<size_t>(std::get<2>(op));
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch", &batch_h2d_dispatch,
          "Batch H2D async copies on a CUDA stream without holding GIL.");
}
