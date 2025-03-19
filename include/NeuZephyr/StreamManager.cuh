#ifndef STREAMMANAGER_CUH
#define STREAMMANAGER_CUH

#ifdef __CUDACC__

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

        void free(T* data) {
            syncData(data);
            cudaFree(data);
        }

        void freeHost(T* data) {
            syncData(data);
            cudaFreeHost(data);
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

        void sync() const {
            for (const auto s : streamPool) {
                cudaStreamSynchronize(s);
            }
        }

        void syncData(T* data) {
            eventPool->syncData(data);
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
    };
}

#endif
#endif //STREAMMANAGER_CUH
