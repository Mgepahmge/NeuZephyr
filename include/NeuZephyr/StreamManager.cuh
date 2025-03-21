#ifndef STREAMMANAGER_CUH
#define STREAMMANAGER_CUH
#include <curand.h>

#ifdef __CUDACC__

#include <cuda_fp16.h>
#include "EventPool.cuh"

namespace nz::cuStrm {
    template <typename T>
    class StreamManager {
    public:
        StreamManager(const StreamManager&) = delete;

        StreamManager& operator=(const StreamManager&) = delete;

        static StreamManager& Instance() {
            static StreamManager instance(16, 128);
            return instance;
        }

        ~StreamManager() {
            sync();
            for (auto& s : streamPool) {
                cudaStreamDestroy(s);
            }
            eventPool.reset();
        }

        void malloc(T** data, const size_t size) {
            cudaStream_t stream = getStream();
            cudaMallocAsync(data, sizeof(T) * size, stream);
            eventPool->recordData(stream, data);
        }

        void malloc(half** data, const size_t size) {
            cudaStream_t stream = getStream();
            cudaMallocAsync(data, sizeof(half) * size, stream);
            eventPool->recordData(stream, data);
        }

        void free(T* data) {
            syncData(data);
            cudaFree(data);
        }

        void freeHost(T* data) {
            syncData(data);
            cudaFreeHost(data);
        }

        void freeAsync(T* data) {
            cudaStream_t stream = getStream();
            streamWait(data, stream);
            cudaFreeAsync(data, stream);
        }

        void freeAsync(half* data) {
            cudaStream_t stream = getStream();
            streamWait(data, stream);
            cudaFreeAsync(data, stream);
        }

        void memset(T* data, const int value, const size_t count) {
            cudaStream_t stream = getStream();
            streamWait(data, stream);
            cudaMemsetAsync(data, value, count, stream);
            eventPool->recordData(stream, data);
        }

        void memcpy(T* dst, T* src, const size_t size, const cudaMemcpyKind kind) {
            cudaStream_t stream = getStream();
            streamWait(src, stream);
            streamWait(dst, stream);
            cudaMemcpyAsync(dst, src, size, kind, stream);
            eventPool->recordData(stream, dst);
        }

        template <typename F, typename... Args>
        void submit(F func, dim3 grid, dim3 block, size_t shared, T* odata, T* idata, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata, stream);
            streamWait(odata, stream);
            func<<<grid, block, shared, stream>>>(odata, idata, args...);
            eventPool->recordData(stream, odata);
        }

        template <typename F, typename... Args>
        void submit(F func, dim3 grid, dim3 block, size_t shared, T* odata, T* idata1, T* idata2, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(idata2, stream);
            streamWait(odata, stream);
            func<<<grid, block, shared, stream>>>(odata, idata1, idata2, args...);
            eventPool->recordData(stream, odata);
        }

        template <typename F, typename... Args>
        void submit(F func, dim3 grid, dim3 block, size_t shared, T* odata, T* idata1, T* idata2, T* idata3, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(idata2, stream);
            streamWait(idata3, stream);
            streamWait(odata, stream);
            func<<<grid, block,shared, stream>>>(odata, idata1, idata2, idata3, args...);
            eventPool->recordData(stream, odata);
        }

        template <typename F, typename... Args>
        void submit(F func, dim3 grid, dim3 block, size_t shared, half* odata, T* idata1, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(odata, stream);
            func<<<grid, block, shared, stream>>>(odata, idata1, args...);
            eventPool->recordData(stream, odata);
        }

        template <typename F, typename... Args>
        void submit(F func, dim3 grid, dim3 block, size_t shared, T* odata, half* idata1, half* idata2, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(idata2, stream);
            streamWait(odata, stream);
            func<<<grid, block, shared, stream>>>(odata, idata1, idata2, args...);
            eventPool->recordData(stream, odata);
        }

        template <typename F, typename... Args>
        void submitDualOut(F func, dim3 grid, dim3 block, size_t shared, T* odata1, T* odata2, T* idata1, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(odata1, stream);
            streamWait(odata2, stream);
            func<<<grid, block, shared, stream>>>(odata1, odata2, idata1, args...);
            eventPool->recordData(stream, odata1);
            eventPool->recordData(stream, odata2);
        }

        template <typename F, typename... Args>
        void submit(F func, dim3 grid, dim3 block, size_t shared, T* odata, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(odata, stream);
            func<<<grid, block, shared, stream>>>(odata, args...);
            eventPool->recordData(stream, odata);
        }

        template <typename F, typename... Args>
        void submitTripleOut(F func, dim3 grid, dim3 block, size_t shared, T* odata1, T* odata2, T* odata3, T* idata1, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(odata1, stream);
            streamWait(odata2, stream);
            streamWait(odata3, stream);
            func<<<grid, block, shared, stream>>>(odata1, odata2, odata3, idata1, args...);
            eventPool->recordData(stream, odata1);
            eventPool->recordData(stream, odata2);
            eventPool->recordData(stream, odata3);
        }

        template <typename F, typename... Args>
        void submitQuadOut(F func, dim3 grid, dim3 block, size_t shared, T* odata1, T* odata2, T* odata3, T* odata4, T* idata1, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(odata1, stream);
            streamWait(odata2, stream);
            streamWait(odata3,stream);
            streamWait(odata4, stream);
            func<<<grid, block, shared, stream>>>(odata1, odata2, odata3, odata4, idata1, args...);
            eventPool->recordData(stream, odata1);
            eventPool->recordData(stream, odata2);
            eventPool->recordData(stream, odata3);
            eventPool->recordData(stream, odata4);
        }

        void sync() const {
            for (const auto s : streamPool) {
                cudaStreamSynchronize(s);
            }
        }

        void syncData(T* data) {
            eventPool->syncData(data);
        }

        void randomize(T* data, size_t size, size_t seed, curandRngType_t rngType) {
            cudaStream_t stream = getStream();
            curandGenerator_t gen;
            curandCreateGenerator(&gen, rngType);
            curandSetStream(gen, stream);
            curandSetPseudoRandomGeneratorSeed(gen, seed);
            streamWait(data, stream);
            curandGenerateUniform(gen, data, size);
            eventPool->recordData(stream, data);
            curandDestroyGenerator(gen);
        }

    private:
        std::vector<cudaStream_t> streamPool;
        std::queue<int> streamQueue;
        std::shared_ptr<EventPool> eventPool;
        uint32_t maxStream;
        std::mutex mtx;

        explicit StreamManager(const uint32_t maxStream = 16, const uint32_t maxEvent = 128) :
            eventPool(std::make_shared<EventPool>(maxEvent)), maxStream(maxStream) {
            cudaFree(0);
            for (uint32_t i = 0; i < maxStream; ++i) {
                cudaStream_t stream;
                cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
                streamPool.push_back(stream);
                streamQueue.push(i);
            }
        }

        cudaStream_t getStream() {
            std::lock_guard<std::mutex> lock(mtx);
            const auto id = streamQueue.front();
            streamQueue.pop();
            streamQueue.push(id);
            return streamPool[id];
        }

        void streamWait(T* data, cudaStream_t stream) {
            for (const auto e : eventPool->getEvents(data)) {
                cudaStreamWaitEvent(stream, e, 0);
            }
        }

        void streamWait(half* data, cudaStream_t stream) {
            for (const auto e : eventPool->getEvents(data)) {
                cudaStreamWaitEvent(stream, e, 0);
            }
        }
    };
}

#endif
#endif //STREAMMANAGER_CUH
