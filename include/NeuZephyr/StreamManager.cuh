#ifndef STREAMMANAGER_CUH
#define STREAMMANAGER_CUH
#include <curand.h>

#ifdef __CUDACC__

#include <cuda_fp16.h>
#include "EventPool.cuh"

namespace nz::cuStrm {
    template <typename T>
    /**
     * @class StreamManager
     * @brief Centralized CUDA stream and resource management system with automatic dependency tracking
     *
     * This singleton class implements a high-level abstraction layer for CUDA concurrency management,
     * combining stream scheduling, event-based dependency tracking, and resource lifecycle management
     * into a unified interface. As the core of NVIDIA GPU task scheduling infrastructure, it enforces
     * strict execution ordering constraints while maximizing concurrent throughput.
     *
     * @warning
     * - **Singleton Pattern**: Must be accessed exclusively through Instance() method
     * - **Type-Specific Instantiation**: Template parameter T must match allocation/free types
     * - **Resource Ownership**: All CUDA resources managed through this class must not be externally modified
     *
     * @design
     * ### 1. Stream Scheduling Strategy: Queue-based Least Recently Used (LRU)
     * Implements lightweight load balancing through cyclic stream allocation:
     * - **Pool Initialization**: Creates `maxStream` non-blocking CUDA streams at construction
     * - **Rotation Mechanism**: Maintains a queue of stream indices, cycling through available streams
     *   ```cpp
     *   // Acquisition pseudocode:
     *   lock();
     *   stream_id = queue.front();  // Get least recently used
     *   queue.pop();
     *   queue.push(stream_id);      // Cycle to end as most recently used
     *   unlock();
     *   ```
     * - **Contention Handling**:
     *   - Fixed pool size prevents CUDA context bloat
     *   - Queue rotation naturally balances workload across streams
     *   - Mutex protection ensures thread-safe access
     *
     * ### 2. CUDA Operation Orchestration: Managed Execution Pipeline
     * Standardizes all GPU operations through a four-stage protocol:
     * 1. **Stream Acquisition**: Obtain execution channel via LRU scheduler
     * 2. **Dependency Resolution**:
     *    - Query all events associated with input/output buffers
     *    - Insert stream wait commands for pending operations
     *    ```cpp
     *    // streamWait implementation:
     *    for (event in EventPool::getEvents(data)) {
     *        cudaStreamWaitEvent(stream, event, 0);
     *    }
     *    ```
     * 3. **Operation Execution**:
     *    - Dispatch kernels/memops with CUDA API wrappers
     *    - Template methods handle variable input/output configurations
     * 4. **Event Recording & Cleanup**:
     *    - Attach completion event to output buffers
     *    - Register CUDA callback for automatic event recycling
     *
     * ### 3. Memory-Centric Synchronization
     * Extends EventPool's data-event binding with type-aware management:
     * - **Allocation Tracking**: Records events for all allocated memory regions
     * - **Smart Free Operations**:
     *   - Synchronous free: Full sync before deallocation
     *   - Async free: Stream-ordered deallocation with dependency enforcement
     * - **Type Specialization**: Explicit handling of half-precision types
     *
     * @interface
     * ### Core Functionality
     * - **Resource Management**:
     *   - `malloc/mallocAsync`: Stream-ordered memory allocation
     *   - `free/freeAsync`: Type-specific deallocation with safety checks
     * - **Kernel Submission**:
     *   - Family of `submit*` methods supporting 1-4 outputs and mixed data types
     *   - Automatic dependency injection for input/output buffers
     * - **System Control**:
     *   - `sync/syncData`: Full pipeline or data-specific synchronization
     *   - `randomize`: Managed CURAND initialization and execution
     *
     * @internal
     * ### Architecture Integration
     * - **EventPool Collaboration**:
     *   - Uses EventPool for cross-stream dependency tracking
     *   - Delegates event lifecycle management to specialized component
     * - **CUDA Resource Isolation**:
     *   - Encapsulates all CUDA API calls
     *   - Prevents direct stream/event access from external code
     *
     * ### Usage Example:
     * ```cpp
     * // Get singleton instance
     * StreamManager<float>& manager = StreamManager<float>::Instance();
     *
     * // Allocate device memory
     * float* d_data;
     * manager.malloc(&d_data, 1024);
     *
     * // Initialize memory
     * manager.memset(d_data, 0, 1024);
     *
     * // Launch kernel (axpy example)
     * manager.submit(
     *     axpy_kernel,            // Kernel function
     *     dim3(1024/256),         // Grid size
     *     dim3(256),              // Block size
     *     0,                      // Shared memory
     *     d_data,                 // Output
     *     d_data, d_data,         // Inputs
     *     2.0f                    // alpha parameter
     * );
     *
     * // Asynchronous memory release
     * manager.freeAsync(d_data);
     * ```
     *
     * @note
     * 1. All public methods are thread-safe via mutex protection
     * 2. Destruction triggers full pipeline synchronization
     * 3. CURAND generators are created per-operation for thread safety
     * 4. Template specialization handles half-precision math requirements
     *
     * @author
     * Mgepahmge(https://github.com/Mgepahmge)
     *
     * @date
     * 2024/11/29
     */
    class StreamManager {
    public:
        StreamManager(const StreamManager&) = delete;

        StreamManager& operator=(const StreamManager&) = delete;

        /**
         * @brief Returns a reference to the singleton instance of the StreamManager.
         *
         * @return A reference to the singleton instance of the StreamManager.
         *
         * This function implements the singleton pattern for the StreamManager class. It ensures that only one instance of the StreamManager is created throughout the program's lifetime. The instance is created with initial parameters 16 and 128. Memory management of the instance is handled automatically by the static keyword, which means the instance is created on the first call to this function and destroyed when the program terminates. There is no specific exception handling mechanism for this function as it is a simple singleton accessor. It serves as a central point of access for other components to interact with the StreamManager instance.
         *
         * @note
         * - This function is thread-safe in C++11 and later due to the static variable initialization being guaranteed to be thread-safe.
         * - Do not attempt to create additional instances of StreamManager manually; always use this function to access the singleton instance.
         *
         * @code
         * ```cpp
         * StreamManager<float>& manager = StreamManager<float>::Instance();
         * ```
         * @endcode
         */
        static StreamManager& Instance() {
            static StreamManager instance(16, 128);
            return instance;
        }

        /**
         * @brief Destructor for the StreamManager class.
         *
         * This destructor is responsible for cleaning up the resources used by the StreamManager. It first synchronizes all the streams in the stream pool, then destroys each CUDA stream in the pool, and finally resets the event pool.
         *
         * @return None.
         *
         * **Memory Management Strategy**:
         * - The CUDA streams in the `streamPool` are explicitly destroyed using `cudaStreamDestroy`, which releases the resources associated with these streams.
         * - The `eventPool` is reset, which should release any resources held by the event pool.
         *
         * **Exception Handling Mechanism**:
         * - This destructor does not throw exceptions. However, `cudaStreamDestroy` can return an error code indicating a failure to destroy the stream. These errors are not explicitly handled in this destructor, but it is assumed that the calling code or the CUDA runtime will handle such errors appropriately.
         *
         * **Relationship with Other Components**:
         * - This destructor depends on the `sync` function to synchronize the streams before destroying them. It also interacts with the `cudaStreamDestroy` function from the CUDA library to release the stream resources and the `reset` method of the `eventPool` object.
         *
         * @note
         * - Ensure that all CUDA operations in the streams have completed before the destructor is called. Otherwise, destroying the streams prematurely may lead to undefined behavior.
         *
         */
        ~StreamManager() {
            sync();
            for (auto& s : streamPool) {
                cudaStreamDestroy(s);
            }
            eventPool.reset();
        }

        /**
         * @brief Asynchronously allocates device memory for type-specific data with stream-ordered dependency tracking
         *
         * @param data Double pointer to device memory (host-to-device parameter). Receives the allocated memory address.
         *            - Must be a valid pointer to device memory pointer (T**)
         *            - The allocated memory is accessible only after stream synchronization
         * @param size Number of elements to allocate (host-to-device parameter)
         *            - Determines total allocation size as sizeof(T) * size
         *            - Must be > 0
         *
         * This method implements a stream-ordered memory allocation workflow:
         * 1. Acquires CUDA stream using LRU scheduling policy
         * 2. Executes cudaMallocAsync on the acquired stream
         * 3. Records allocation event in EventPool for dependency tracking
         *
         * The allocation operation becomes visible to subsequent operations through:
         * - Implicit stream ordering within the same CUDA stream
         * - Explicit event dependencies managed by EventPool
         *
         * @throws No explicit exceptions, but CUDA errors can be checked using cudaGetLastError()
         *
         * @note
         * - Thread-safe through internal mutex protection
         * - Allocation lifetime is managed by CUDA's async memory system
         * - Subsequent operations using this memory must call streamWait() for dependency resolution
         * - Requires CUDA 11.2+ for async memory APIs
         * - For half-precision allocations, use the explicit half** overload
         *
         * @code
         * StreamManager<float>& manager = StreamManager<float>::Instance();
         * float* device_buffer = nullptr;
         *
         * // Allocate 1MB buffer
         * manager.malloc(&device_buffer, 1024*1024/sizeof(float));
         *
         * // Check CUDA errors
         * cudaError_t err = cudaGetLastError();
         * if (err != cudaSuccess) {
         *     // Handle allocation error
         * }
         * @endcode
         */
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

        /**
         * @brief Frees the CUDA device memory pointed to by the given pointer.
         *
         * @param data A pointer to the CUDA device memory to be freed (device-to-host).
         *
         * @return None.
         *
         * This function is responsible for releasing the CUDA device memory pointed to by `data`. Before freeing the memory, it calls the `syncData` function to ensure that any pending data synchronization operations are completed. The `cudaFree` function from the CUDA library is then used to release the device memory.
         *
         * **Memory Management Strategy**:
         * - The function ensures that the memory is synchronized before freeing it to avoid data inconsistency. After calling `cudaFree`, the memory pointed to by `data` is released and should not be accessed further.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw exceptions. However, `cudaFree` can return an error code indicating a failure to free the memory. These errors are not explicitly handled in this function, but it is assumed that the calling code or the CUDA runtime will handle such errors appropriately.
         *
         * **Relationship with Other Components**:
         * - This function depends on the `syncData` function to synchronize the data before freeing the memory. It also interacts with the `cudaFree` function from the CUDA library.
         *
         * @note
         * - Ensure that the pointer `data` points to valid CUDA device memory. Passing a null pointer or a pointer to non - CUDA device memory will lead to undefined behavior.
         */
        void free(T* data) {
            syncData(data);
            cudaFree(data);
        }

        /**
         * @brief Frees the pinned host memory pointed to by the given pointer.
         *
         * @param data A pointer to the pinned host memory to be freed (host-to-host).
         *
         * @return None.
         *
         * This function is designed to release the pinned host memory allocated by CUDA. Before freeing the memory, it invokes the `syncData` function to make sure that all data synchronization operations related to this memory are finished. Subsequently, it uses the `cudaFreeHost` function from the CUDA library to free the pinned host memory.
         *
         * **Memory Management Strategy**:
         * - The function guarantees that the data in the memory is synchronized before deallocation to prevent data loss or inconsistency. Once `cudaFreeHost` is called, the memory pointed to by `data` is released and should no longer be accessed.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw exceptions. However, `cudaFreeHost` may return an error code if it fails to free the memory. These errors are not explicitly handled within this function, and it is assumed that the calling code or the CUDA runtime will deal with such issues appropriately.
         *
         * **Relationship with Other Components**:
         * - It relies on the `syncData` function for data synchronization. Additionally, it interacts with the `cudaFreeHost` function from the CUDA library to perform the actual memory deallocation.
         *
         * @note
         * - Ensure that the pointer `data` points to valid pinned host memory allocated by CUDA. Passing a null pointer or a pointer to non - pinned host memory will result in undefined behavior.
         *
         * @code
         * ```cpp
         * T* pinnedHostData;
         * cudaMallocHost((void**)&pinnedHostData, sizeof(T));
         * // Use pinnedHostData
         * StreamManager manager;
         * manager.freeHost(pinnedHostData);
         * ```
         * @endcode
         */
        void freeHost(T* data) {
            syncData(data);
            cudaFreeHost(data);
        }

        /**
         * @brief Asynchronously frees the CUDA device memory pointed to by the given pointer.
         *
         * @param data A pointer to the CUDA device memory to be freed (device-to-host).
         *
         * @return None.
         *
         * This function is responsible for asynchronously releasing the CUDA device memory pointed to by `data`. First, it retrieves a CUDA stream using the `getStream` function. Then, it calls `streamWait` to ensure that all operations related to the `data` in the retrieved stream are completed. Finally, it uses `cudaFreeAsync` to asynchronously free the device memory in the specified stream.
         *
         * **Memory Management Strategy**:
         * - The function ensures that all stream - related operations on the memory are finished before scheduling the memory to be freed asynchronously. After `cudaFreeAsync` is called, the memory will be released once all previous operations in the stream have completed. The memory should not be accessed after this call.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw exceptions. However, `getStream`, `streamWait`, and `cudaFreeAsync` can return error codes indicating failures. These errors are not explicitly handled in this function, and it is assumed that the calling code or the CUDA runtime will handle them appropriately.
         *
         * **Relationship with Other Components**:
         * - It depends on the `getStream` function to obtain a CUDA stream, the `streamWait` function to synchronize operations in the stream, and the `cudaFreeAsync` function from the CUDA library to perform the asynchronous memory deallocation.
         *
         * @note
         * - Ensure that the pointer `data` points to valid CUDA device memory. Passing a null pointer or a pointer to non - CUDA device memory will lead to undefined behavior.
         */
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

        /**
         * @brief Asynchronously sets a block of CUDA device memory to a specified value.
         *
         * @param data A pointer to the CUDA device memory to be set (device-to-host).
         * @param value The value to set each byte of the memory block to.
         * @param count The number of bytes to set.
         *
         * @return None.
         *
         * This function is designed to asynchronously initialize a block of CUDA device memory to a given value. It first retrieves a CUDA stream using the `getStream` function. Then, it calls `streamWait` to ensure that all previous operations related to the `data` in the retrieved stream are completed. After that, it uses `cudaMemsetAsync` to asynchronously set the specified number of bytes in the device memory to the given value. Finally, it records the data operation in the `eventPool` for future reference.
         *
         * **Memory Management Strategy**:
         * - The function ensures that all previous operations on the memory block are finished before scheduling the `cudaMemsetAsync` operation. The memory should not be accessed until the `cudaMemsetAsync` operation has completed. The `eventPool` can be used to check the completion status.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw exceptions. However, `getStream`, `streamWait`, `cudaMemsetAsync`, and the operations related to `eventPool` can return error codes indicating failures. These errors are not explicitly handled in this function, and it is assumed that the calling code or the CUDA runtime will handle them appropriately.
         *
         * **Relationship with Other Components**:
         * - It depends on the `getStream` function to obtain a CUDA stream, the `streamWait` function to synchronize operations in the stream, the `cudaMemsetAsync` function from the CUDA library to perform the asynchronous memory setting, and the `eventPool` object to record the data operation.
         *
         * @note
         * - Ensure that the pointer `data` points to valid CUDA device memory. Passing a null pointer or a pointer to non - CUDA device memory will lead to undefined behavior.
         */
        void memset(T* data, const int value, const size_t count) {
            cudaStream_t stream = getStream();
            streamWait(data, stream);
            cudaMemsetAsync(data, value, count, stream);
            eventPool->recordData(stream, data);
        }

        /**
         * @brief Asynchronously copies data between CUDA device and host memory based on the specified memory copy kind.
         *
         * @param dst A pointer to the destination memory (memory flow depends on `kind`).
         * @param src A pointer to the source memory (memory flow depends on `kind`).
         * @param size The number of bytes to copy.
         * @param kind The type of memory copy operation (`cudaMemcpyKind`). This determines the direction of the memory transfer (e.g., host - to - device, device - to - host, etc.).
         *
         * @return None.
         *
         * This function is responsible for performing an asynchronous memory copy operation. It first retrieves a CUDA stream using the `getStream` function. Then, it waits for all previous operations related to both the source (`src`) and destination (`dst`) memory in the retrieved stream to complete by calling `streamWait` twice. After that, it uses `cudaMemcpyAsync` to asynchronously copy the specified number of bytes from the source to the destination memory according to the given `kind` in the retrieved stream. Finally, it records the data operation in the `eventPool` for future reference.
         *
         * **Memory Management Strategy**:
         * - The function ensures that all previous operations on both the source and destination memory are finished before scheduling the `cudaMemcpyAsync` operation. The memory should not be accessed until the `cudaMemcpyAsync` operation has completed. The `eventPool` can be used to check the completion status.
         *
         * **Exception Handling Mechanism**:
         * - This function does not throw exceptions. However, `getStream`, `streamWait`, `cudaMemcpyAsync`, and the operations related to `eventPool` can return error codes indicating failures. These errors are not explicitly handled in this function, and it is assumed that the calling code or the CUDA runtime will handle them appropriately.
         *
         * **Relationship with Other Components**:
         * - It depends on the `getStream` function to obtain a CUDA stream, the `streamWait` function to synchronize operations in the stream, the `cudaMemcpyAsync` function from the CUDA library to perform the asynchronous memory copy, and the `eventPool` object to record the data operation.
         *
         * @note
         * - Ensure that both `src` and `dst` pointers point to valid memory locations appropriate for the specified `cudaMemcpyKind`. Passing null pointers or pointers to incorrect memory types will lead to undefined behavior.
         */
        void memcpy(T* dst, T* src, const size_t size, const cudaMemcpyKind kind) {
            cudaStream_t stream = getStream();
            streamWait(src, stream);
            streamWait(dst, stream);
            cudaMemcpyAsync(dst, src, size, kind, stream);
            eventPool->recordData(stream, dst);
        }

        /**
         * @brief Asynchronously submits a CUDA kernel with stream-ordered dependency management
         *
         * @tparam F CUDA kernel function type (automatically deduced)
         * @tparam Args Variadic template parameter types (automatically deduced)
         * @param func CUDA kernel function pointer (host-to-device)
         *            - Must comply with CUDA kernel calling conventions
         * @param grid Grid dimension configuration (host-to-device)
         *            - Use dim3 to define computation grid structure
         * @param block Thread block dimension configuration (host-to-device)
         *            - Use dim3 to define thread block structure
         * @param shared Dynamic shared memory size in bytes (host-to-device)
         *            - Set to 0 if not using dynamic shared memory
         * @param odata Output data device pointer (device-to-device)
         *            - First output parameter of the kernel
         *            - Automatically records completion event
         * @param idata Input data device pointer (device-to-device)
         *            - First input parameter of the kernel
         *            - Automatically inserts stream wait dependencies
         * @param args Additional kernel arguments (host-to-device)
         *            - Supports scalar type parameter passing
         *            - Must satisfy CUDA kernel parameter passing rules
         *
         * This method implements full lifecycle management for stream-ordered kernel execution:
         * 1. Acquires CUDA stream through LRU policy
         * 2. Inserts stream wait operations for input/output data dependencies
         * 3. Submis configured CUDA kernel to target stream
         * 4. Records output data completion event
         *
         * Dependency management mechanisms:
         * - Input data dependencies: Ensures producer-consumer order through streamWait
         * - Output data visibility: Guarantees subsequent operation visibility via eventPool
         *
         * @note
         * - This is the base implementation for single input/output, multiple overloads exist for different parameter counts
         * - Kernel function signature must match parameter order (output first, input second)
         * - Default stream-ordered launch, no manual synchronization required
         * - Thread-safe: Internal stream acquisition protected by mutex
         * - CUDA errors should be checked via cudaPeekAtLastError
         *
         * @code
         * // Example: Vector addition kernel
         * __global__ void vecAdd(float* out, const float* a, const float* b, int n) {
         *     int i = blockIdx.x * blockDim.x + threadIdx.x;
         *     if (i < n) out[i] = a[i] + b[i];
         * }
         *
         * // Usage example
         * float *d_out, *d_a, *d_b;
         * manager.malloc(&d_out, N);
         * manager.malloc(&d_a, N);
         * manager.malloc(&d_b, N);
         *
         * manager.submit(vecAdd,
         *               dim3((N+255)/256),  // grid
         *               dim3(256),          // block
         *               0,                  // shared memory
         *               d_out,              // output pointer
         *               d_a,                // input pointer
         *               d_b, N);            // additional args
         *
         * cudaError_t err = cudaPeekAtLastError();
         * if (err != cudaSuccess) {
         *     // Handle kernel configuration errors
         * }
         * @endcode
         *
         * @warning
         * - Additional arguments must persist until kernel execution completes
         * - Avoid passing host pointers in kernel arguments
         */
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
        void submit(F func, dim3 grid, dim3 block, size_t shared, T* odata, T* idata1, T* idata2, T* idata3,
                    Args... args) {
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
        void submitDualOut(F func, dim3 grid, dim3 block, size_t shared, T* odata1, T* odata2, T* idata1,
                           Args... args) {
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
        void submitTripleOut(F func, dim3 grid, dim3 block, size_t shared, T* odata1, T* odata2, T* odata3, T* idata1,
                             Args... args) {
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
        void submitQuadOut(F func, dim3 grid, dim3 block, size_t shared, T* odata1, T* odata2, T* odata3, T* odata4,
                           T* idata1, Args... args) {
            cudaStream_t stream = getStream();
            streamWait(idata1, stream);
            streamWait(odata1, stream);
            streamWait(odata2, stream);
            streamWait(odata3, stream);
            streamWait(odata4, stream);
            func<<<grid, block, shared, stream>>>(odata1, odata2, odata3, odata4, idata1, args...);
            eventPool->recordData(stream, odata1);
            eventPool->recordData(stream, odata2);
            eventPool->recordData(stream, odata3);
            eventPool->recordData(stream, odata4);
        }

        template <typename F, typename... Args>
        void submitParallel(F func, dim3 grid, dim3 block, size_t shared, T* odata, T* idata1, T* idata2,
                            const std::vector<size_t>& offset_o,
                            const std::vector<size_t>& offset_i1, const std::vector<size_t>& offset_i2, Args... args) {
            if (offset_o.size() != offset_i1.size() || offset_o.size() != offset_i2.size()) {
                throw std::invalid_argument("Size of offset is not uniform");
            }
            std::vector<cudaStream_t> streams;
            for (size_t i = 0; i < offset_o.size(); ++i) {
                cudaStream_t stream = getStream();
                streams.push_back(stream);
                streamWait(idata1, stream);
                streamWait(idata2, stream);
                streamWait(odata, stream);
                func<<<grid, block, shared, stream>>>(odata, idata1, idata2, args..., offset_o[i], offset_i1[i],
                                                      offset_i2[i]);
            }
            for (auto stream : streams) {
                eventPool->recordData(stream, odata);
            }
        }

        template <typename F, typename... Args>
        void submitParallel(F func, dim3 grid, dim3 block, size_t shared, T* odata, T* idata1,
                            const std::vector<size_t>& offset_o,
                            Args... args) {
            std::vector<cudaStream_t> streams;
            for (unsigned long long i : offset_o) {
                cudaStream_t stream = getStream();
                streams.push_back(stream);
                streamWait(idata1, stream);
                streamWait(odata, stream);
                func<<<grid, block, shared, stream>>>(odata, idata1, args..., i);
            }
            for (auto stream : streams) {
                eventPool->recordData(stream, odata);
            }
        }

        template <typename F, typename... Args>
        void submitParallel(F func, dim3 grid, dim3 block, size_t shared, T* odata, T* idata1,
                            const std::vector<size_t>& offset_o,
                            const std::vector<size_t>& offset_i,
                            Args... args) {
            std::vector<cudaStream_t> streams;
            for (auto i = 0; i < offset_o.size(); ++i) {
                cudaStream_t stream = getStream();
                streams.push_back(stream);
                streamWait(idata1, stream);
                streamWait(odata, stream);
                func<<<grid, block, shared, stream>>>(odata, idata1, args..., offset_o[i], offset_i[i]);
            }
            for (auto stream : streams) {
                eventPool->recordData(stream, odata);
            }
        }

        /**
         * @brief Synchronizes all CUDA streams in the stream pool by blocking the host thread
         *
         * This function performs a full barrier synchronization across all managed CUDA streams
         * in the stream pool. It sequentially waits for the completion of all operations enqueued
         * in every stream, ensuring no pending GPU work remains after return.
         *
         * @note
         * - Heavyweight Operation: Introduces host-side wait for entire stream pool completion
         * - Execution Order: Synchronizes streams regardless of their dependency relationships
         * - Alternative: Use event-based synchronization for partial stream dependencies
         * - Thread Safety: Safe to call concurrently if streamPool remains unmodified
         *
         * @warning
         * - Host Blocking: Freezes calling thread until all streams complete (millisecond~second scale)
         * - Error Propagation: Does not handle CUDA errors internally; check errors post-call
         * - Stream Validity: Assumes all streams in streamPool are valid CUDA streams
         *
         * @throws No explicit exceptions thrown. CUDA runtime errors may surface through:
         *         - Subsequent CUDA API calls returning error codes
         *         - External CUDA error handlers (if configured)
         *
         * @code
         * // Benchmark timing with full synchronization
         * auto start = std::chrono::high_resolution_clock::now();
         *
         * // Submit multiple GPU workloads
         * manager.submit(kernel1, ...);
         * manager.submit(kernel2, ...);
         *
         * // Full system synchronization
         * manager.sync();
         *
         * auto end = std::chrono::high_resolution_clock::now();
         *
         * // Check for asynchronous errors
         * cudaError_t err = cudaDeviceSynchronize();
         * if (err != cudaSuccess) {
         *     std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
         * }
         * @endcode
         *
         * @see cudaStreamSynchronize, submit()
         */
        void sync() const {
            for (const auto s : streamPool) {
                cudaStreamSynchronize(s);
            }
        }

        /**
         * @brief Synchronizes host thread with completion events for a specific data object
         *
         * @param data Device data pointer to synchronize (device-to-host)
         *            - Must be a valid device pointer previously used in submit() calls
         *            - Synchronizes based on last recorded event for this data
         *
         * This function provides targeted synchronization for a specific data object by:
         * 1. Querying the event pool for the most recent completion event associated with the data
         * 2. Blocking host execution until all operations preceding the event complete
         * 3. Maintaining other stream operations' asynchronous execution
         *
         * @note
         * - Precision Synchronization: Only waits for operations affecting this specific data
         * - Event Reuse: Completion event is preserved for future dependency tracking
         * - Thread Safety: Safe for concurrent access if data isn't being modified
         * - Lightweight Alternative: Prefer over full sync() for partial workflow completion
         *
         * @warning
         * - Host Blocking: Freezes thread until target data's operations complete
         * - Stale Pointers: Undefined behavior if data has been deallocated
         * - Temporal Scope: Only synchronizes with events recorded since last submit()
         *
         * @code
         * // Producer-consumer workflow
         * manager.submit(producerKernel, ..., output_data, ...);
         * manager.syncData(output_data);  // Wait for producer completion
         * manager.submit(consumerKernel, ..., input_data=output_data, ...);
         *
         * // Host data access
         * manager.submit(computeKernel, ..., result_data, ...);
         * manager.syncData(result_data);
         * cudaMemcpy(host_result, result_data, ..., cudaMemcpyDeviceToHost);
         * @endcode
         *
         * @see submit(), eventPool
         */
        void syncData(T* data) {
            eventPool->syncData(data);
        }

        /**
        * @brief Generates uniformly distributed random numbers on GPU using CURAND
        *
        * @param data Device pointer to allocated memory for random numbers
        * @param size Number of elements to generate
        * @param seed Seed value for pseudo-random generator
        * @param rngType CURAND RNG algorithm type (e.g., CURAND_RNG_PSEUDO_XORWOW)
        *
        * This function:
        * 1. Acquires a CUDA stream from the pool
        * 2. Initializes CURAND generator with specified configuration
        * 3. Ensures prior operations on data complete via stream synchronization
        * 4. Generates random numbers in device memory
        * 5. Records completion event for subsequent synchronization
        *
        * @note
        * - Data Requirements: Memory must be pre-allocated with correct type/size
        * - RNG Performance: XORWOW vs MRG32K3A have different speed/quality tradeoffs
        * - Stream Isolation: Uses dedicated stream to avoid RNG sequence corruption
        * - Float Generation: Produces values in [0,1) range for 32-bit float types
        *
        * @warning
        * - Type Safety: Requires T=float for correct operation with curandGenerateUniform
        * - Seed Size: Implicit cast to unsigned long long may truncate 64-bit values
        * - Generator Overhead: Repeated create/destroy calls impact performance
        * - Concurrent Access: Not thread-safe for same data pointer
        *
        * @code
        * // Initialize 1M element array with random values
        * float* d_data;
        * cudaMalloc(&d_data, 1<<20 * sizeof(float));
        *
        * manager.randomize(d_data, 1<<20, 12345, CURAND_RNG_PSEUDO_XORWOW);
        * manager.syncData(d_data);  // Wait for completion
        * @endcode
        *
        * @see curandCreateGenerator, curandGenerateUniform, eventPool
        *
        */
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

        /**
        * @brief Acquires CUDA stream from pool using round-robin scheduling
        *
        * This function:
        * 1. Locks access to stream queue with mutex for thread safety
        * 2. Retrieves next available stream ID from front of queue
        * 3. Rotates queue by moving front element to end (FIFO rotation)
        * 4. Returns corresponding CUDA stream from preallocated pool
        *
        * @note
        * - Thread Safety: Protected by mutex for concurrent access
        * - Stream Pool: Requires pre-initialized stream pool and queue
        * - Fair Scheduling: Round-robin prevents stream starvation
        * - Stream Lifetime: Streams remain owned by pool manager
        *
        * @warning
        * - Pool Capacity: Fixed stream count may lead to saturation
        * - No Stream Creation: Assumes streams already initialized
        * - Queue State: Undefined behavior if called before pool initialization
        * - Thread Blocking: Mutex contention may impact performance at scale
        *
        * @code
        * // Typical usage pattern:
        * cudaStream_t stream = manager.getStream();
        * kernel<<<blocks, threads, 0, stream>>>(...);
        * cudaStreamSynchronize(stream);
        * @endcode
        *
        * @see cudaStreamCreate, cudaStreamSynchronize, std::lock_guard
        */
        cudaStream_t getStream() {
            std::lock_guard<std::mutex> lock(mtx);
            const auto id = streamQueue.front();
            streamQueue.pop();
            streamQueue.push(id);
            return streamPool[id];
        }

        /**
        * @brief Synchronizes CUDA stream execution until data writes complete
        *
        * @tparam T Data type (inferred from pointer)
        * @param data Device memory pointer with pending write operations
        * @param stream CUDA stream to apply synchronization constraints
        *
        * This function:
        * 1. Retrieves all CUDA events associated with write operations on target data
        * 2. Applies stream wait operations for each pending write event
        * 3. Ensures subsequent stream operations execute only after data writes complete
        *
        * @note
        * - Event Lifetime: Events remain owned by event pool until explicitly released
        * - Write Synchronization: Only affects write-type CUDA events for this data
        * - Thread Safety: Requires external synchronization for concurrent data access
        * - Pointer Association: Data pointer must match previous async operation records
        *
        * @warning
        * - Event Pool Validity: Undefined behavior if event pool not initialized
        * - Data Ownership: Incorrect pointers may wait on unrelated operations
        * - Stream State: Target stream must be valid and not destroyed
        * - Partial Completion: Does NOT guarantee host-side data readiness
        *
        * @code
        * // Typical producer-consumer pattern:
        * asyncWrite(d_data, host_buffer, writeStream);  // Records write event
        * streamWait(d_data, computeStream);  // Make compute wait
        * processData<<<..., computeStream>>>(d_data);  // Safe access
        * @endcode
        *
        * @see cudaStreamWaitEvent, cudaEventRecord, EventPool
        */
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

        /**
        * @brief Records write completion event for asynchronous data operations
        *
        * @tparam T Data type (inferred from pointer)
        * @param data Device memory pointer tracking write completion
        * @param stream CUDA stream where write operation occurred
        *
        * This method:
        * 1. Creates CUDA event through event pool
        * 2. Records event on specified stream at current execution point
        * 3. Registers automatic cleanup callback upon event completion
        * 4. Associates event with target data for dependency tracking
        *
        * @note
        * - Event Lifetime: Managed by event pool with automatic recycling
        * - Write Tracking: Enables cross-stream synchronization via streamWait()
        * - Thread Safety: Requires external synchronization for concurrent data access
        * - Stream Ordering: Captures all preceding operations in the stream
        *
        * @warning
        * - Event Pool Initialization: Must be properly initialized before use
        * - Data Ownership: Pointer must match subsequent synchronization calls
        * - Stream Validity: Target stream must be active during recording
        * - Host Visibility: Does NOT guarantee host memory consistency
        *
        * @code
        * // Typical asynchronous write pattern:
        * cudaMemcpyAsync(devPtr, hostPtr, size, cudaMemcpyHostToDevice, stream);
        * recordData(devPtr, stream);  // Bookmark write completion
        * streamWait(devPtr, computeStream);  // Enforce dependency
        * @endcode
        *
        * @see cudaEventRecord, cudaStreamAddCallback, EventPool::recordData
        */
        void recordData(T* data, cudaStream_t stream) {
            eventPool->recordData(stream, data);
        }

        void recordData(half* data, cudaStream_t stream) {
            eventPool->recordData(stream, data);
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
    };
}

#endif
#endif //STREAMMANAGER_CUH
