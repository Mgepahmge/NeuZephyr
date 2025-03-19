#ifndef EVENTPOOL_CUH
#define EVENTPOOL_CUH

#ifdef __CUDACC__

#include <map>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace nz::cuStrm {
    class EventPool {
    public:
        explicit EventPool(const size_t maxEvent) : maxEvent(maxEvent) {
            for (size_t i = 0; i < maxEvent; ++i) {
                cudaEvent_t e;
                cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
                free.insert(e);
            }
        }

        ~EventPool() {
            for (const auto e : free) {
                cudaEventDestroy(e);
            }
            for (const auto e : busy) {
                cudaEventDestroy(e);
            }
            for (const auto e : released) {
                cudaEventDestroy(e);
            }
        }

        cudaEvent_t recordData(cudaStream_t stream, void* data) {
            std::lock_guard lock(mtx);
            auto event = acquire();
            eventMap[data].insert(event);
            eventMapR[event] = data;
            record(stream, event);
            return event;
        }

        std::unordered_set<cudaEvent_t> getEvents(void* data) {
            std::lock_guard lock(mtx);
            if (const auto it = eventMap.find(data); it != eventMap.end()) {
                return it->second;
            }
            return {};
        }

        void syncData(void* data) {
            std::unique_lock lock(mtx);
            cv.wait(lock, [this, data]() -> bool {
                return eventMap.find(data) == eventMap.end();
            });
        }

    private:
        std::unordered_set<cudaEvent_t> free;
        std::unordered_set<cudaEvent_t> busy;
        std::unordered_set<cudaEvent_t> released;
        std::unordered_map<void*, std::unordered_set<cudaEvent_t>> eventMap;
        std::unordered_map<cudaEvent_t, void*> eventMapR;
        size_t maxEvent;
        std::mutex mtx;
        std::condition_variable cv;

        void transfer() {
            free.insert(released.begin(), released.end());
            released.clear();
        }

        cudaEvent_t acquire() {
            if (!free.empty()) {
                const cudaEvent_t e = *free.begin();
                free.erase(e);
                busy.insert(e);
                transfer();
                return e;
            }

            cudaEvent_t e;
            cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
            busy.insert(e);
            maxEvent++;
            transfer();
            return e;
        }

        void release(cudaEvent_t e) {
            std::lock_guard lock(mtx);
            if (busy.find(e) != busy.end()) {
                busy.erase(e);
                released.insert(e);
            }
            if (const auto it = eventMapR.find(e); it != eventMapR.end()) {
                const auto data = eventMapR.find(e)->second;
                eventMapR.erase(e);
                auto& events = eventMap.find(data)->second;
                events.erase(e);
                if (events.empty()) {
                    eventMap.erase(data);
                    cv.notify_all();
                }
            }
        }

        void record(cudaStream_t stream, cudaEvent_t event) {
            cudaEventRecord(event, stream);
            cudaStreamAddCallback(stream, [](cudaStream_t, cudaError_t, void* params) {
                const auto* param = static_cast<std::pair<EventPool*, cudaEvent_t>*>(params);
                param->first->release(param->second);
                delete param;
            }, new std::pair(this, event), 0);
        }
    };
}

#endif
#endif //EVENTPOOL_CUH
