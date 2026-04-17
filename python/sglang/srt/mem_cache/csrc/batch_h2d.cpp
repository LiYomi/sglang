
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include <tuple>
#include <stdexcept>
#include <string>

void batch_h2d_dispatch(
    const std::vector<std::tuple<int64_t, int64_t, int64_t>>& ops,
    int64_t stream_ptr
) {
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    py::gil_scoped_release release;
    for (size_t i = 0; i < ops.size(); ++i) {
        const auto& op = ops[i];
        auto src = reinterpret_cast<const void*>(std::get<0>(op));
        auto dst = reinterpret_cast<void*>(std::get<1>(op));
        auto nbytes = static_cast<size_t>(std::get<2>(op));
        cudaError_t err = cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, stream);
        if (err != cudaSuccess) {
            // Re-acquire the GIL before raising so the Python runtime is in a
            // valid state when the exception unwinds back through pybind11.
            py::gil_scoped_acquire acquire;
            throw std::runtime_error(
                "batch_h2d cudaMemcpyAsync failed at op " + std::to_string(i) +
                ": " + cudaGetErrorString(err));
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dispatch", &batch_h2d_dispatch,
          "Batch H2D async copies on a CUDA stream without holding GIL.");
}
