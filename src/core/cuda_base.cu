
#include "cuda_base.h"

namespace tw
{

/*
CUDA calls:
<<<dimBlockGrid,dimThreadGrid,dynShared,stream>>>


cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
kernel1<<<grid, block, 0, stream1>>>(data_1);
kernel2<<<grid, block, 0, stream2>>>(data_2);

cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaMemcpyAsync(a_d, a_h, size, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream2>>>(otherData_d);

size=N*sizeof(float)/nStreams;
for(i=0; i<nStreams; i++)
{
  offset = i*N/nStreams;
  cudaMemcpyAsync(a_d+offset, a_h+offset, size, dir, stream[i]);
  kernel<<<N/(nThreads*nStreams), nThreads, 0, stream[i]>>>(a_d+offset);
}

cudaStream_t streams[5];
for(int i = 0; i < 5; i++)
{
  cudaStreamCreate(&streams[i]);
  busy<<<1, 1, 0, streams[i]>>>();
}
cudaDeviceSynchronize();
for(int i = 0; i < 5; i++)
  cudaStreamDestroy(streams[i]);

template <typename T>
__global__ void pipeline_kernel_async(T *global, uint64_t *clock, size_t copy_count)
{
  extern __shared__ char s[];
  T *shared = reinterpret_cast<T *>(s);

  uint64_t clock_start = clock64();

  //pipeline pipe;
  for(size_t i = 0; i < copy_count; ++i)
  {
    __pipeline_memcpy_async(&shared[blockDim.x * i + threadIdx.x],
                            &global[blockDim.x * i + threadIdx.x], sizeof(T));
  }
  __pipeline_commit();
  __pipeline_wait_prior(0);

  uint64_t clock_end = clock64();

  atomicAdd(reinterpret_cast<unsigned long long *>(clock), clock_end - clock_start);
}

__global__ void test_pipe_intr(float* input) {
    __shared__ float smem[32*4];
    int idx4 = threadIdx.x * 4;
    // for tread 0 get it imput[0 - 4], for tread 1 get it imput[4 - 8]
    __pipeline_memcpy_async(smem + idx4, input + idx4, 16);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    printf("% i = %f \n", threadIdx.x, smem[idx4]);
}*/

void cuda_info::print()
{
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if(error_id != cudaSuccess)
  {
    printf("cudaGetDeviceCount returned %d\n-> %s\n", static_cast<int>(error_id), cudaGetErrorString(error_id));
    printf("Result = FAIL\n");
  }

  if(deviceCount == 0)
  {
    printf("There are no available device(s) that support CUDA\n");
  }
  else
  {
    printf("Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  for(dev = 0; dev < deviceCount; ++dev)
  {
    cudaSetDevice(dev);
    cudaGetDeviceProperties(&deviceProp, dev);

    printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

    // Console log
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
           deviceProp.major, deviceProp.minor);

    char msg[256];
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    sprintf_s(msg, sizeof(msg),
              "  Total amount of global memory:                 %.0f MBytes "
              "(%llu bytes)\n",
              static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
              (unsigned long long)deviceProp.totalGlobalMem);
#else
    snprintf(msg, sizeof(msg),
             "  Total amount of global memory:                 %.0f MBytes "
             "(%llu bytes)\n",
             static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
             (unsigned long long)deviceProp.totalGlobalMem);
#endif
    printf("%s", msg);

    printf("  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
           deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);
    printf(
        "  GPU Max Clock rate:                            %.0f MHz (%0.2f "
        "GHz)\n",
        deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

#if CUDART_VERSION >= 5000
    // This is supported in CUDA 5.0 (runtime API device properties)
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           deviceProp.memoryClockRate * 1e-3f);
    printf("  Memory Bus Width:                              %d-bit\n",
           deviceProp.memoryBusWidth);

    if (deviceProp.l2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             deviceProp.l2CacheSize);
    }

#else
    // This only available in CUDA 4.0-4.2 (but these were only exposed in the
    // CUDA Driver API)
    int memoryClock;
    getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
                          dev);
    printf("  Memory Clock rate:                             %.0f Mhz\n",
           memoryClock * 1e-3f);
    int memBusWidth;
    getCudaAttribute<int>(&memBusWidth,
                          CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
    printf("  Memory Bus Width:                              %d-bit\n",
           memBusWidth);
    int L2CacheSize;
    getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

    if (L2CacheSize) {
      printf("  L2 Cache Size:                                 %d bytes\n",
             L2CacheSize);
    }

#endif

    printf(
        "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
        "%d), 3D=(%d, %d, %d)\n",
        deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
        deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
        deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf(
        "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
        deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
    printf(
        "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
        "layers\n",
        deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
        deviceProp.maxTexture2DLayered[2]);

    printf("  Total amount of constant memory:               %zu bytes\n",
           deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %zu bytes\n",
           deviceProp.sharedMemPerBlock);
    printf("  Total shared memory per multiprocessor:        %zu bytes\n",
           deviceProp.sharedMemPerMultiprocessor);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
           deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
           deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %zu bytes\n",
           deviceProp.memPitch);
    printf("  Texture alignment:                             %zu bytes\n",
           deviceProp.textureAlignment);
    printf(
        "  Concurrent copy and kernel execution:          %s with %d copy "
        "engine(s)\n",
        (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
    printf("  Run time limit on kernels:                     %s\n",
           deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           deviceProp.integrated ? "Yes" : "No");
    printf("  Support host page-locked memory mapping:       %s\n",
           deviceProp.canMapHostMemory ? "Yes" : "No");
    printf("  Alignment requirement for Surfaces:            %s\n",
           deviceProp.surfaceAlignment ? "Yes" : "No");
    printf("  Device has ECC support:                        %s\n",
           deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n",
           deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)"
                                : "WDDM (Windows Display Driver Model)");
#endif
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           deviceProp.unifiedAddressing ? "Yes" : "No");
    printf("  Device supports Managed Memory:                %s\n",
           deviceProp.managedMemory ? "Yes" : "No");
    printf("  Device supports Compute Preemption:            %s\n",
           deviceProp.computePreemptionSupported ? "Yes" : "No");
    printf("  Supports Cooperative Kernel Launch:            %s\n",
           deviceProp.cooperativeLaunch ? "Yes" : "No");
    printf("  Supports MultiDevice Co-op Kernel Launch:      %s\n",
           deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
    printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
           deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

    const char *sComputeMode[] = {
        "Default (multiple host threads can use ::cudaSetDevice() with device "
        "simultaneously)",
        "Exclusive (only one host thread in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Prohibited (no host thread can use ::cudaSetDevice() with this "
        "device)",
        "Exclusive Process (many threads in one process is able to use "
        "::cudaSetDevice() with this device)",
        "Unknown", NULL};
    printf("  Compute Mode:\n");
    printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
  }

  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (deviceCount >= 2) {
    cudaDeviceProp prop[64];
    int gpuid[64];  // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

      // Only boards based on Fermi or later can support P2P
      if ((prop[i].major >= 2)
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
          // on Windows (64-bit), the Tesla Compute Cluster driver for windows
          // must be enabled to support this
          && prop[i].tccDriver
#endif
          ) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;


    if (gpu_p2p_count >= 2)
    {
      printf("GPU combinations:\n");

      for (int i = 0; i < gpu_p2p_count; i++)
      {
        //printf("GPU%d: ", gpuid[i]); 
        checkCudaErrors(cudaSetDevice(gpuid[i]));

        for (int j = 0; j < gpu_p2p_count; j++)
        {
          if (gpuid[i] == gpuid[j])
            continue;

          //printf("GPU%d, ", gpuid[j]);
          checkCudaErrors(cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));

          printf("> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n", prop[gpuid[i]].name, 
          gpuid[i], prop[gpuid[j]].name, gpuid[j], can_access_peer ? "Yes" : "No");
              
          if (can_access_peer)
            cudaDeviceEnablePeerAccess(gpuid[j], 0);
        }
      }

      checkCudaErrors(cudaSetDevice(gpuid[0]));
    }
  }

  gpuErrchk(cudaPeekAtLastError());
}

// host <-> device + staged (pinned)
// global <-> shared + staged
// streams/pipelines/graphs
// vectorization?
// multi load per threads
// avoid kernel launches

// reduction
// cuda graphs
//__device__ void solve();
// lazy loading
// CUDA_MODULE_LOADING = LAZY

// pinned memory and zero copy: https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
// cudaError_t status = cudaMallocHost((void**)&h_aPinned, bytes);
// if (status != cudaSuccess)
//  printf("Error allocating pinned host memory\n");
/*
template <typename T>
__global__ void pipeline_test(T *global1, T *global2, size_t subset_count)
{
  extern __shared__ T s[];
  constexpr unsigned stages_count = 2;
  auto group = cooperative_groups::this_thread_block();
  T *shared[stages_count] = {s, s + 2*group.size() };
  // Create a synchronization object (cuda::pipeline)
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, stages_count> shared_state;
  auto pipeline = cuda::make_pipeline(group, &shared_state);
  size_t fetch;
  size_t subset;
  for (subset = fetch = 0; subset < subset_count; ++subset)
  {
    // Fetch ahead up to stages_count subsets
    for (; fetch < subset_count && fetch < (subset + stages_count); ++fetch )
    {
        pipeline.producer_acquire();
        cuda::memcpy_async(group, shared[fetch%2], &global1[fetch*group.size()], sizeof(T)*group.size(), pipeline);
        cuda::memcpy_async(group, shared[fetch%2] + group.size(), &global2[fetch*group.size()], sizeof(T)*group.size(), pipeline);
        pipeline.producer_commit(); // Commit the fetch-ahead stage
    }
    pipeline.consumer_wait(); // Wait for ‘subset’ stage to be available
    compute(shared[subset%2]);
    pipeline.consumer_release();
  }
}*/
/*
void stream_test() // and pinned memory test
{
  const size_t N = 128;
  double *data;
  cudaMallocHost(&hostPtr, 2*size); // -> host vector data?
  const size_t N_streams = 10;
  cudaStream_t streams[N_streams];

  for(size_t it = 0 ; it < N_streams ; ++it)
  {
    cudaStreamCreate(&streams[it]);
  }

  for(size_t it = 0 ; it < N_streams ; ++it)
  {
    cudaMemcpyAsync(inputDevPtr + i*size, hostPtr + i*size, size, cudaMemcpyHostToDevice, streams[i]);
    MyKernel<<<100, 512, 0, stream[i]>>>(outputDevPtr + i*size, inputDevPtr + i*size, size);
    cudaMemcpyAsync(hostPtr + i*size, outputDevPtr + i*size, size, cudaMemcpyDeviceToHost, streams[i]);
  }
  
  for(size_t it = 0 ; it < N_streams ; ++it)
  {
    cudaStreamDestroy(streams[it]);
  }
}
*/

// split data for each warp (shared) and apply something using threads
// 1 warp -> 32 threads
// thread block gets split into warps
// grid of blocks, thread block cluster
__global__ void memory_test(double data[], const size_t N)
{
  extern __shared__ double smem[];//blockDim.x*32*sizeof(double)
  const int tIdx = blockIdx.x*blockDim.x + threadIdx.x;

  //auto grid = cooperative_groups::this_grid();
  //auto block = cooperative_groups::this_thread_block();
  //assert(size == batch_sz * grid.size()); // Exposition: input size fits batch_sz * grid_size

  //cuda::memcpy_async(smem + tIdx, data + tIdx, 16);
  //__pipeline_memcpy_async(smem + tIdx, data + tIdx, 16);
  //__pipeline_commit();
  //__pipeline_wait_prior(0);
  
  smem[tIdx] = data[tIdx];
  smem[tIdx] = 1./smem[tIdx];
  data[tIdx] = smem[tIdx];

  //__pipeline_memcpy_async(data + tIdx, smem + tIdx, 16);
  //__pipeline_commit();
  //__pipeline_wait_prior(0);

  __syncthreads();
}

// UNIT TEST ==================================================================
void cuda_base_test()
{
  log_info("=== cuda_base_test ===");
  /*
  // allocate and initialize test data
  log_info("allocate and init");
  const size_t N = 128;
  std::vector<double> host_data(N);
  for(size_t it = 0 ; it < N ; ++it)
  {
    host_data[it] = static_cast<double>(it+1);
  }

  // test copy from host to device memory
  log_info("copy to device");
  double* device_data;
  checkCudaErrors(cudaMalloc(&device_data, sizeof(double)*N));
  checkCudaErrors(cudaMemcpy(device_data, host_data.data(), sizeof(double)*N, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaDeviceSynchronize());

  // test copy from global memory to shared memory / call kernel which loads into shared
  // next: test shared memory allocation, warps, blocks, threads, ...
  log_info("call kernel");
  const int numBlocks = 2;
  dim3 threadsPerBlock(64); // 32 threads per warp, 2 warps per block, 2 blocks
  memory_test<<<numBlocks, threadsPerBlock>>>(device_data, N);

  // copy back and check
  log_info("copy back");
  checkCudaErrors(cudaMemcpy(host_data.data(), device_data, sizeof(double)*N, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaDeviceSynchronize());

  log_info("check");
  double dif = 0.;
  for(size_t it = 0 ; it < N ; ++it)
  {
    dif += 1./host_data[it] - static_cast<double>(it+1);
  }
  std::cout<<"dif: "<<dif<<std::endl;
  for(const auto& val : host_data) std::cout<<val<<" ";
  std::cout<<std::endl;

  log_info("free");
  checkCudaErrors(cudaFree(device_data));
  */

  /*
  // SC Ass1 Task 3 (cublas test)
  const int n = 10, m = 10, k = 10;
  const float alpha = 0.5, beta = 0.3;
  
  std::vector<float> x_h(n, 1.0), y_h(n, 2.0), z_h(n, 0.0);
  float *x_d, *y_d, *z_d, *A, *M, *r, res = 0.0;

  cublasHandle_t handle;
  //cudaEvent_t start, stop;

  checkCudaErrors(cublasCreate(&handle));
  //checkCudaErrors(cudaEventCreate(&start));
  //checkCudaErrors(cudaEventCreate(&stop));

  checkCudaErrors(cudaMalloc(&x_d, sizeof(float)*n));
  checkCudaErrors(cudaMalloc(&y_d, sizeof(float)*n));
  //checkCudaErrors(cudaMalloc(&z_d, sizeof(float)*n));

  checkCudaErrors(cudaMemcpy(x_d, x_h.data(), sizeof(float)*n, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(y_d, y_h.data(), sizeof(float)*n, cudaMemcpyHostToDevice));
  //checkCudaErrors(cudaDeviceSynchronize());

  //checkCudaErrors(cudaEventRecord(start, NULL));

  checkCudaErrors(cublasSaxpy(handle, n, &alpha, x_d, 1, y_d, 1)); // i)
  //cublasSscal(handle, n, (1.0 - alpha), x_d, 1);                   // ii) (-> saxpy does not change x, avoids copy!)
  //cublasSscal(handle, n, &beta, x_d, 1);                           // iii) (-> double saxpy, avoids copy!)
  //checkCudaErrors(cublasSaxpy(handle, n, &alpha, x_d, 1, y_d, 1));

  //cublasSdot(handle, n, x_d, 1, y_d, 1, &res);
  //cublasSnrm2(handle, n, x_d, 1, &res);
  //cublasSgemv(handle, trans, m, n, &alpha, A, lda, x_d, 1, &beta, y_d, 1);
  //cublasStrmv(handle, uplo, trans, diag, n, A, lda, x_d, 1);
  //cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &alpha, M, lda, M, ldb, &beta, A, ldc); // CUBLAS_OP_N
  //cublasScopy(handle, N, x, incx, y, incy);

  //checkCudaErrors(cudaEventRecord(stop, NULL));
  //checkCudaErrors(cudaEventSynchronize(stop));

  checkCudaErrors(cudaMemcpy(x_h.data(), x_d, sizeof(float)*n, cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(y_h.data(), y_d, sizeof(float)*n, cudaMemcpyDeviceToHost));
  //checkCudaErrors(cudaDeviceSynchronize());

  checkCudaErrors(cudaFree(x_d));
  checkCudaErrors(cudaFree(y_d));
  //checkCudaErrors(cudaFree(z_d));

  checkCudaErrors(cublasDestroy(handle));
  cudaDeviceReset();*/
}
// ============================================================================
}// namespace tw

