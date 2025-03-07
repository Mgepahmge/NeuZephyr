#ifndef NEUZEPHYRCUDAERRORHANDLING_CUH
#define NEUZEPHYRCUDAERRORHANDLING_CUH
#include <stdexcept>
#include <cstdio>
#include "dl_export.cuh"

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace nz {
    /**
     * @class CudaException
     * @brief A final class that represents CUDA exceptions, inheriting from std::runtime_error.
     *
     * This class is designed to handle and provide detailed information about CUDA-related errors. It captures the file, line number, CUDA error code, and the expression that caused the error. By overriding the `what()` method, it can return a formatted error message with all these details.
     *
     * ### Type Definitions:
     * - None
     *
     * @details
     * ### Key Features:
     * - **Error Information Capture**: Records the file, line number, CUDA error code, and the expression where the error occurred.
     * - **Formatted Error Message**: Generates a detailed and human - readable error message that includes all the captured information.
     *
     * ### Usage Example:
     * ```cpp
     * #include <iostream>
     * #include <cuda_runtime.h>
     *
     * // Assume DL_API is a valid macro for visibility
     * class DL_API CudaException final : public std::runtime_error {
     *     // Class definition as provided
     * };
     *
     * void someCudaFunction() {
     *     cudaError_t err = cudaMalloc(nullptr, 1024);
     *     if (err!= cudaSuccess) {
     *         throw CudaException(__FILE__, __LINE__, err, "cudaMalloc(nullptr, 1024)");
     *     }
     * }
     *
     * int main() {
     *     try {
     *         someCudaFunction();
     *     } catch (const CudaException& e) {
     *         std::cerr << e.what() << std::endl;
     *     }
     *     return 0;
     * }
     * ```
     *
     * @note
     * - Ensure that the buffer size of 1024 characters is sufficient for the error message. If the message exceeds this size, it will be truncated.
     * - The `format_message()` method should be called during object construction to ensure the error message is properly formatted.
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     *
     * @date
     * 2024/07/24
     */
    class DL_API CudaException final : public std::runtime_error {
    public:
        CudaException(const char* file, int line,
                      cudaError_t code, const char* expr)
            : std::runtime_error(cudaGetErrorString(code)),
              file(file),
              line(line),
              code(code),
              expr(expr) {
            format_message();
        }

        const char* what() const noexcept override {
            return msg.c_str();
        }

        void format_message() {
            snprintf(buffer, sizeof(buffer),
                     "CUDA Exception:\n"
                     "\tFile:       %s\n"
                     "\tLine:       %d\n"
                     "\tError code: %d\n"
                     "\tError text: %s\n"
                     "\tExpression: %s",
                     file, line, code, std::runtime_error::what(), expr);

            msg = buffer;
        }

    private:
        const char* file;
        int line;
        cudaError_t code;
        const char* expr;
        char buffer[1024];
        std::string msg;
    };
}

#define CHECK(call)                                                             \
do {                                                                            \
cudaError_t __err = call;                                                   \
if (__err != cudaSuccess) {                                                 \
cudaDeviceReset();                                                      \
throw nz::CudaException(__FILE__, __LINE__, __err, #call);                  \
}                                                                           \
} while (0)

#define CHECK_LAST()                                                            \
do {                                                                            \
cudaError_t __err = cudaGetLastError();                                     \
if (__err != cudaSuccess) {                                                 \
cudaDeviceReset();                                                      \
throw nz::CudaException(__FILE__, __LINE__, __err, "cudaGetLastError()");   \
}                                                                           \
} while (0)

#endif
#endif //NEUZEPHYRCUDAERRORHANDLING_CUH
