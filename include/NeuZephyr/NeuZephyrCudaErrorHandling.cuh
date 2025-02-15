#ifndef NEUZEPHYRCUDAERRORHANDLING_CUH
#define NEUZEPHYRCUDAERRORHANDLING_CUH

#endif //NEUZEPHYRCUDAERRORHANDLING_CUH
#include <stdexcept>
#include <cstdio>
#include "dl_export.cuh"

#ifdef __CUDACC__
#include <cuda_runtime.h>

namespace nz {
    class DL_API CudaException final : public std::runtime_error {
    public:
        CudaException(const char* file, int line,
                     cudaError_t code, const char* expr)
            : std::runtime_error(cudaGetErrorString(code)),
              file(file),
              line(line),
              code(code),
              expr(expr)
        {
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
throw CudaException(__FILE__, __LINE__, __err, #call);                  \
}                                                                           \
} while (0)

#define CHECK_LAST()                                                            \
do {                                                                            \
cudaError_t __err = cudaGetLastError();                                     \
if (__err != cudaSuccess) {                                                 \
cudaDeviceReset();                                                      \
throw CudaException(__FILE__, __LINE__, __err, "cudaGetLastError()");   \
}                                                                           \
} while (0)

#endif



