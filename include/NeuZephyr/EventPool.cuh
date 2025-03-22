/**
 * @file EventPool.cuh
 * @brief Definition of the EventPool class for managing CUDA events in multi-stream environments.
 *
 * This file provides the declaration of the `EventPool` class, which is designed to efficiently manage
 * CUDA event allocation, recycling, and synchronization across multiple streams. It supports tracking
 * events associated with specific data pointers for advanced synchronization use cases in GPU computing.
 *
 * @details
 * The `EventPool` class encapsulates the following key features:
 * - **Event Pooling**: Pre-allocates a configurable number of CUDA events (`cudaEvent_t`) and dynamically
 *   expands the pool when demand exceeds supply, minimizing runtime overhead.
 * - **Event-Data Association**: Tracks relationships between CUDA events and user data pointers, enabling
 *   synchronization based on data completion status.
 * - **Thread Safety**: Utilizes mutexes and condition variables to ensure safe concurrent access across threads.
 * - **Automatic Recycling**: Automatically reclaims completed events via CUDA stream callbacks, reducing manual
 *   resource management.
 * - **Efficient Queries**: Provides fast lookups for events associated with specific data through hash-based mappings.
 *
 * This class is part of the `nz::cuStrm` namespace and is optimized for scenarios requiring fine-grained
 * synchronization between GPU operations and host-side data management.
 *
 * @note
 * 1. All events are created with `cudaEventDisableTiming` for minimal overhead.
 * 2. Users must ensure proper destruction of the pool to avoid CUDA resource leaks.
 * 3. The `syncData` method blocks until all events associated with a data pointer complete.
 *
 * @warning
 * This class is an internal component of the nz::cuStrm::StreamManager system. It should not be instantiated or
 * accessed directly in most use cases. Event management should be handled exclusively through StreamManager's
 * interfaces to maintain proper synchronization and resource lifecycle management.
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/11/29
 */
#ifndef EVENTPOOL_CUH
#define EVENTPOOL_CUH

#ifdef __CUDACC__

#include <map>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/**
 * @namespace nz::cuStrm
 * @brief Provides core components for CUDA stream and event lifecycle management in GPU computing environments.
 *
 * The `nz::cuStrm` namespace implements advanced stream resource management and operation scheduling
 * mechanisms for CUDA-enabled devices. It serves as the central coordination layer for GPU operation
 * execution and synchronization in the nz project.
 *
 * @details
 * Key components within this namespace include:
 * - **StreamManager**: Central controller class that:
 *   - Hosts all CUDA kernel executions through managed streams
 *   - Implements intelligent stream allocation policies (Round-Robin, Priority-based, etc.)
 *   - Maintains operation dependency graphs through event tracking
 *   - Provides automatic synchronization primitives
 * - **EventPool**: Internal event management system (see EventPool.cuh documentation)
 *
 * The architecture is designed to:
 * - Guarantee CUDA stream usage safety in multi-threaded environments
 * - Prevent resource contention through stream/event ownership management
 * - Enable fine-grained operation sequencing via dependency tracking
 *
 * @note
 * 1. All CUDA operations should be dispatched through StreamManager interfaces
 * 2. Manual stream creation/destruction is strictly prohibited in this paradigm
 * 3. Event synchronization must use the provided abstraction layers
 *
 * @warning
 * Direct CUDA stream/event API calls outside this namespace may cause:
 * - Undefined synchronization behavior
 * - Resource ownership conflicts
 * - Memory consistency issues
 *
 * @author
 * Mgepahmge(https://github.com/Mgepahmge)
 *
 * @date
 * 2024/11/29
 */
namespace nz::cuStrm {
    /**
     * @class EventPool
     * @brief Internal event management system for CUDA stream synchronization (Part of StreamManager)
     *
     * This class implements a thread-safe CUDA event pool with automatic recycling and data-aware
     * synchronization capabilities. It serves as the foundational event infrastructure for
     * nz::cuStrm::StreamManager's operation scheduling system.
     *
     * @warning
     * - **This class must not be directly instantiated or invoked**
     * - Event lifecycle management should **exclusively** be handled through StreamManager interfaces
     * - Direct usage may lead to:
     *   - Undefined synchronization behavior
     *   - Event pool corruption
     *   - CUDA resource leaks
     *
     * @details
     * ### Core Functionality:
     * - **Pool Management**:
     *   - Pre-allocates `cudaEventDisableTiming` events during initialization
     *   - Dynamically expands when concurrent demands exceed initial capacity
     *   - Implements triple-state tracking (free/busy/released) with atomic transfers
     * - **Data-Event Binding**:
     *   - Maintains bidirectional mappings between CUDA events and user data pointers
     *   - Enables data-centric synchronization through `syncData()`
     * - **Automatic Recycling**:
     *   - Utilizes CUDA stream callbacks for event release detection
     *   - Implements lock-protected resource transitions between states
     *
     * ### Critical Methods (Internal Use Only):
     * - `recordData()`:
     *   - Binds event recording to specific data pointer
     *   - Triggers automatic recycling via stream callback
     * - `getEvents()`:
     *   - Retrieves all events associated with data pointer
     *   - Used for dependency graph construction
     * - `syncData()`:
     *   - Blocks until all events linked to data complete
     *   - Implements condition variable-based waiting
     *
     * The pool maintains three distinct event states to ensure safe CUDA event reuse:
     *
     * - **Free Pool**:
     *   Contains immediately available `cudaEvent_t` instances ready for allocation.
     *   Events are drawn from this pool when servicing new recording requests through `acquire()`.
     *
     * - **Busy Pool**:
     *   Tracks actively used events currently associated with in-flight CUDA operations.
     *   Events remain here until their host stream completes execution, at which point
     *   the stream callback moves them to the released state.
     *
     * - **Released Pool**:
     *   Holds events that have completed execution but haven't been recycled. These events
     *   are transferred back to the free pool during subsequent `acquire()` calls via the
     *   internal `transfer()` method, ensuring safe temporal separation between event usage
     *   cycles and preventing premature reuse.
     *
     * @note
     * 1. All public methods employ `std::lock_guard` for thread safety
     * 2. Event destruction occurs only during pool destruction
     * 3. Exceeding maxEvent capacity triggers silent pool expansion
     * 4. Callback parameters use heap-allocated memory for cross-stream safety
     *
     * @internal
     * Architecture Diagram:
     * [StreamManager] -- manages --> [EventPool]
     *                -- uses     --> (CUDA Events)
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    class EventPool {
    public:
        /**
         * @brief Construct an EventPool object with a specified maximum number of events.
         *
         * This constructor initializes an EventPool object with a given maximum number of CUDA events. It creates `maxEvent` number of CUDA events with the `cudaEventDisableTiming` flag and inserts them into the `free` set, indicating that these events are initially available for use.
         *
         * @param maxEvent The maximum number of CUDA events that the EventPool can manage. Memory location: host.
         *
         * @return None
         *
         * Memory management: The constructor allocates memory for CUDA events using `cudaEventCreateWithFlags`. The responsibility of deallocating these events lies with the destructor of the `EventPool` class.
         * Exception handling: This constructor does not have an explicit exception - handling mechanism. It relies on the CUDA runtime to report any errors that occur during event creation. If an error occurs during `cudaEventCreateWithFlags`, the program's behavior may be undefined.
         * Relationship with other components: This constructor is part of the `EventPool` class, which is likely used in a larger CUDA - related application to manage the lifecycle of CUDA events.
         *
         * @note
         * - The time complexity of this constructor is O(n), where n is the value of `maxEvent`, as it iterates `maxEvent` times to create and insert events.
         * - Ensure that the CUDA environment is properly initialized before creating an `EventPool` object.
         *
         * @code
         * ```cpp
         * size_t maxEvents = 10;
         * EventPool pool(maxEvents);
         * ```
         * @endcode
         */
        explicit EventPool(const size_t maxEvent) : maxEvent(maxEvent) {
            for (size_t i = 0; i < maxEvent; ++i) {
                cudaEvent_t e;
                cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
                free.insert(e);
            }
        }

        /**
         * @brief Destruct the EventPool object, releasing all managed CUDA events.
         *
         * This destructor iterates through the sets of free, busy, and released CUDA events and destroys each event using `cudaEventDestroy`. This ensures that all resources allocated for these events are properly released.
         *
         * @param None
         *
         * @return None
         *
         * Memory management: The destructor is responsible for deallocating the memory associated with the CUDA events that were created during the lifetime of the `EventPool` object. It destroys all events in the `free`, `busy`, and `released` sets.
         * Exception handling: This destructor does not have an explicit exception - handling mechanism. It relies on the CUDA runtime to report any errors that occur during event destruction. If an error occurs during `cudaEventDestroy`, the program's behavior may be undefined.
         * Relationship with other components: This destructor is part of the `EventPool` class and is crucial for proper resource management in a CUDA - related application that uses the `EventPool` to manage CUDA events.
         *
         * @note
         * - The time complexity of this destructor is O(n), where n is the total number of events in the `free`, `busy`, and `released` sets combined.
         * - Ensure that all CUDA operations associated with the events have completed before the `EventPool` object is destroyed.
         *
         * @code
         * ```cpp
         * // Assume EventPool is defined and an instance is created
         * EventPool pool(10);
         * // Some operations with the pool
         * // ...
         * // The pool will be destroyed automatically when it goes out of scope
         * ```
         * @endcode
         */
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

        /**
         * @brief Record an event in a CUDA stream associated with a given data pointer.
         *
         * This function records a CUDA event in the specified CUDA stream and associates it with the provided data pointer. It first acquires an available event from the event pool, then updates the mapping between data pointers and events and vice - versa. Finally, it records the event in the stream and returns the event handle.
         *
         * @param stream The CUDA stream in which the event will be recorded. Memory location: host.
         * @param data A pointer to the data associated with the event. Memory location: host or device, depending on the context.
         *
         * @return A handle to the recorded CUDA event.
         *
         * Memory management: The function does not allocate or deallocate any memory directly. It uses an existing event pool and updates mapping data structures. The responsibility of event memory management lies with the event pool's constructor and destructor.
         * Exception handling: This function does not have an explicit exception - handling mechanism. It relies on the underlying CUDA functions (such as `acquire` and `record`) to report any errors. If an error occurs during event acquisition or recording, the program's behavior may be undefined.
         * Relationship with other components: This function is part of the `EventPool` class. It interacts with the event pool's internal state (event sets and mapping data structures) and the CUDA runtime to record events.
         *
         * @note
         * - The time complexity of this function is O(log n) due to the operations on the mapping data structures (assuming they are implemented as balanced trees), where n is the number of elements in the mapping.
         * - Ensure that the CUDA stream is properly initialized before calling this function.
         */
        cudaEvent_t recordData(cudaStream_t stream, void* data) {
            std::lock_guard lock(mtx);
            auto event = acquire();
            eventMap[data].insert(event);
            eventMapR[event] = data;
            record(stream, event);
            return event;
        }

        /**
         * @brief Retrieve all CUDA events associated with a given data pointer.
         *
         * This function searches for the provided data pointer in the internal mapping and returns a set of all CUDA events associated with it. If no events are found for the given data pointer, an empty set is returned.
         *
         * @param data A pointer to the data for which the associated events are to be retrieved. Memory location: host or device, depending on the context.
         *
         * @return An unordered set of CUDA event handles associated with the given data pointer. If no events are associated, an empty set is returned.
         *
         * Memory management: The function does not allocate or deallocate any memory directly. It only accesses the internal mapping data structure of the `EventPool` class.
         * Exception handling: This function does not have an explicit exception - handling mechanism. It relies on the underlying standard library functions for `std::unordered_map` operations. If an error occurs during the map lookup, the program's behavior may be undefined.
         * Relationship with other components: This function is part of the `EventPool` class. It interacts with the internal `eventMap` data structure to retrieve the associated events.
         *
         * @note
         * - The average time complexity of this function is O(1) because it uses an `std::unordered_map` for lookup. In the worst - case scenario, the time complexity is O(n), where n is the number of elements in the `eventMap`.
         * - Ensure that the data pointer is valid and has been previously used in the `recordData` function to associate events with it.
         */
        std::unordered_set<cudaEvent_t> getEvents(void* data) {
            std::lock_guard lock(mtx);
            if (const auto it = eventMap.find(data); it != eventMap.end()) {
                return it->second;
            }
            return {};
        }

        /**
         * @brief Synchronize the program execution with the completion of all events associated with a given data pointer.
         *
         * This function waits until all CUDA events associated with the provided data pointer have completed. It uses a condition variable (`cv`) to block the current thread until the `eventMap` no longer contains any events for the given data pointer, indicating that all associated events have finished.
         *
         * @param data A pointer to the data for which the associated events need to be synchronized. Memory location: host or device, depending on the context.
         *
         * @return None.
         *
         * Memory management: The function does not allocate or deallocate any memory directly. It only accesses the internal `eventMap` data structure of the `EventPool` class.
         * Exception handling: This function does not have an explicit exception - handling mechanism. It relies on the underlying standard library functions for mutex and condition variable operations. If an error occurs during locking, unlocking, or waiting, the program's behavior may be undefined.
         * Relationship with other components: This function is part of the `EventPool` class. It interacts with the internal `eventMap` data structure and the condition variable (`cv`) to wait for event completion.
         *
         * @note
         * - The time complexity of this function is not fixed as it depends on when the associated events complete. It will block until all relevant events are finished.
         * - Ensure that the data pointer is valid and has been previously used in the `recordData` function to associate events with it.
         * - The function assumes that the internal state of the `eventMap` is updated correctly when events are completed to signal the condition variable.
         *
         */
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

        /**
         * @brief Transfer elements from the `released` container to the `free` container and clear the `released` container.
         *
         * This private method is used to move all elements from the `released` container into the `free` container, which is likely part of a resource management mechanism within the class. After the transfer, the `released` container is cleared to prepare for new elements.
         *
         * @param None.
         *
         * @return None.
         *
         * Memory management: The function does not allocate or deallocate memory directly. It only modifies the internal containers `free` and `released`. The elements themselves are just moved from one container to another.
         * Exception handling: This function does not have an explicit exception - handling mechanism. It relies on the underlying standard library functions for container operations. If an error occurs during the insertion or clearing operations, the program's behavior may be undefined.
         * Relationship with other components: This is a private method of the class, and it is likely called by other public or private methods within the class to manage resource availability.
         *
         * @note
         * - The time complexity of this function is O(n), where n is the number of elements in the `released` container, because it needs to insert each element from `released` into `free`.
         * - This method should only be called when it is appropriate to transfer the resources from the `released` state to the `free` state according to the class's resource management logic.
         */
        void transfer() {
            free.insert(released.begin(), released.end());
            released.clear();
        }

        /**
         * @brief Acquire a CUDA event from the available pool or create a new one if none are available.
         *
         * This private method is used to obtain a CUDA event for use within the class. It first checks if there are any free events in the `free` set. If so, it takes one from the `free` set, moves it to the `busy` set, and calls the `transfer` method to manage resource availability. If no free events are available, it creates a new CUDA event with the `cudaEventDisableTiming` flag, adds it to the `busy` set, increments the `maxEvent` counter, and also calls the `transfer` method.
         *
         * @param None.
         *
         * @return A CUDA event (`cudaEvent_t`) that is now marked as busy and ready for use.
         *
         * Memory management: The function creates a new CUDA event using `cudaEventCreateWithFlags` if there are no free events. The responsibility of destroying these events is likely handled elsewhere in the class. The function also manages the movement of events between the `free` and `busy` sets.
         * Exception handling: This function does not have an explicit exception - handling mechanism. It relies on the underlying CUDA API functions and standard library functions for container operations. If an error occurs during CUDA event creation or container operations, the program's behavior may be undefined.
         * Relationship with other components: This is a private method of the class and is likely called by other public or private methods within the class to obtain CUDA events for recording and synchronization. It also calls the `transfer` method to manage resource availability.
         *
         * @throws std::runtime_error If the CUDA event creation fails (although not explicitly handled in this code).
         *
         * @note
         * - The time complexity of this function is O(1) on average for the case where there are free events, as set insertion and deletion operations in `std::unordered_set` have an average constant time complexity. In the case where a new event needs to be created, the time complexity is dominated by the CUDA event creation operation, which is not strictly O(1).
         * - Ensure that the CUDA environment is properly initialized before calling this method.
         * - This method should only be called when it is appropriate to acquire a CUDA event according to the class's resource management logic.
         *
         * @warning
         * - Failure to properly manage the lifecycle of the acquired CUDA events can lead to resource leaks.
         */
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

        /**
         * @brief Release a CUDA event and perform associated resource management.
         *
         * This method is used to mark a CUDA event as released. It first locks the mutex to ensure thread - safety. If the event is in the `busy` set, it is removed from the `busy` set and added to the `released` set. Additionally, if the event is found in the `eventMapR`, it is removed from both `eventMapR` and the corresponding event list in the `eventMap`. If the event list in the `eventMap` becomes empty, the corresponding entry in the `eventMap` is removed, and a notification is sent to all waiting threads.
         *
         * @param e A CUDA event (`cudaEvent_t`) to be released. The event is passed from the calling context (host - to - method).
         *
         * @return None.
         *
         * Memory management: The function does not allocate or deallocate memory directly related to the CUDA events. It only modifies the internal data structures (`busy`, `released`, `eventMapR`, `eventMap`). The responsibility of destroying the CUDA events is likely handled elsewhere in the class.
         * Exception handling: This function does not have an explicit exception - handling mechanism. It relies on the underlying standard library functions for container operations and mutex locking. If an error occurs during these operations, the program's behavior may be undefined.
         * Relationship with other components: This method is likely called by other parts of the class or the program when a CUDA event is no longer needed. It interacts with the `busy`, `released`, `eventMapR`, and `eventMap` data structures and can notify other threads through the `cv` condition variable.
         *
         * @note
         * - The time complexity of this function is O(log n) in the average case for the set and map operations, where n is the number of elements in the respective containers.
         * - This method should only be called when it is appropriate to release a CUDA event according to the class's resource management logic.
         *
         */
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

        /**
         * @brief Record a CUDA event in a specified CUDA stream and schedule its release upon stream completion.
         *
         * This method records a given CUDA event in a specified CUDA stream. It then adds a callback to the stream. When the stream reaches the point where the event is recorded and completes the related operations, the callback is executed. The callback releases the event by calling the `release` method of the `EventPool` class and deletes the dynamically allocated parameter passed to it.
         *
         * @param stream A CUDA stream (`cudaStream_t`) in which the event will be recorded. The stream is passed from the calling context (host - to - method).
         * @param event A CUDA event (`cudaEvent_t`) to be recorded in the stream. The event is passed from the calling context (host - to - method).
         *
         * @return None.
         *
         * Memory management: The function dynamically allocates memory for a `std::pair` containing a pointer to the `EventPool` object and the CUDA event. This memory is deleted within the callback function. The responsibility of destroying the CUDA stream and event themselves is likely handled elsewhere in the program.
         * Exception handling: This function does not have an explicit exception - handling mechanism. It relies on the underlying CUDA API functions (`cudaEventRecord` and `cudaStreamAddCallback`). If an error occurs during these operations, the program's behavior may be undefined.
         * Relationship with other components: This method is part of the `EventPool` class. It calls the `cudaEventRecord` and `cudaStreamAddCallback` functions from the CUDA API and the `release` method of the `EventPool` class.
         *
         * @throws std::bad_alloc If the dynamic memory allocation for the `std::pair` fails.
         *
         * @note
         * - The time complexity of this function is O(1) as it mainly involves calling CUDA API functions and a simple memory allocation and callback setup.
         * - Ensure that the CUDA stream and event are properly initialized before calling this method.
         * - This method should only be called when it is appropriate to record an event in a stream according to the program's CUDA execution logic.
         *
         * @warning
         * - Improper handling of the dynamically allocated `std::pair` can lead to memory leaks if the callback is not executed as expected.
         *
         */
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
