/**
 * Copyright (c) 2022 Thomas Schrotter. All rights reserved.
 * @file cuda_fim.h
 * @brief Implements the Fast Iterative Method for solving the eikonal equation
 * @author Thomas Schrotter
 * @version 0.0.0
 * @date 2022-12
 */

#ifndef TW_CUDA_FIM_H_
#define TW_CUDA_FIM_H_

#define FULL_MASK 0xffffffff

// tw includes
#include "../core/types.h"
#include "../core/cuda_base.h"
#include "../io/vtk.h"
#include "../io/io_json.h"

// libs
//#include "../../lib/indicators.hpp"

// stdlib includes
#include <vector>
#include <map>
#include <set>

#define is_aligned(POINTER, BYTE_COUNT) \
  (((uintptr_t)(const void *)(POINTER)) % (BYTE_COUNT) == 0) // very useful!

#define aligned_size(BYTES, BYTE_COUNT) \
  (BYTES + ((BYTES % BYTE_COUNT)?(BYTE_COUNT - (BYTES % BYTE_COUNT)):0))

//size_t aligned_size(size_t Nbytes, size_t align_size=8UL)
//{
//  return Nbytes + (align_size - (Nbytes % align_size));
//}
namespace tw { // =============================================================

struct gray_symbols
{
  const idx_t adf[12] = {4, 5, 3, 1, 3, 0, 4, 0, 2, 1, 2, 5};
  const idx_t bce[12] = {2, 0, 1, 5, 2, 4, 3, 5, 1, 0, 3, 4};
  const idx_t phi[12] = {1, 2, 3, 2, 3, 0, 3, 0, 1, 0, 1, 2}; // not checked!
};

__constant__ gray_symbols d_gray_mappings;

enum memtype_e {host_e, device_e, pinned_e, const_e, shallow_e, NA_e};
/*
ek_struct data_global;
...
solve_submesh<<<1, 32, Nshared>>>(data_global); // pass by value (copy constructor) // shallow copy?
{
  extern __shared__ char sdata[]; // Nbytes
  const int tidx = threadIdx.x + blockIdx.x*blockDim.x;

  ek_struct data_shared(sdata); 
  data_global.copy_to(data_shared);

  ek_fim_cuda solver(sdata, data_global);
}
*/
// ----------------------------------------------------------------------------
// -> use pointer?
/*
template <typename T, typename = void> struct has_size : std::false_type {};
template <typename T> struct has_size<T, decltype(void(std::declval<T &>().size()))> : std::true_type {};

// ----------------------------------------------------------------------------
//template<typename T>
//struct ActualType { typedef T type; };
//template<typename T>
//struct ActualType<T*> { typedef typename ActualType<T>::type type; };
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// https://stackoverflow.com/questions/1198260/how-can-you-iterate-over-the-elements-of-an-stdtuple
// https://stackoverflow.com/questions/16387354/template-tuple-calling-a-function-on-each-element
// https://stackoverflow.com/questions/14471224/c11-how-to-get-the-type-a-pointer-or-iterator-points-to
// currently no container templates allowed!
template<class... T>
struct mem_t // auto-align!
{
  char* dptr;
  const int N;
  const memtype_e memtype;
  std::tuple<T*...> ptrs;
  
  mem_t(const int N, memtype_e memtype) : N(N), memtype(memtype)
  {
    const size_t Nbytes = count_bytes();

    switch(memtype)
    {
      case host_e: { dptr = new char[Nbytes]; break; }
      case device_e: { checkCudaErrors(cudaMalloc(&dptr, Nbytes)); break; }
      case pinned_e: { checkCudaErrors(cudaMallocHost(&dptr, Nbytes)); break; }
      case const_e: { break; }
      case shallow_e: break;
      default: throw_tw_error("mem_t constructor: invalid memtype!");
    }
  }
  mem_t(char *ptr, const int Nbytes) : memtype(shallow_e)
  {
    if(Nbytes != count_bytes())
      throw_tw_error("mem_t constructor: bytes mismatch!");
    // warning?
  }
  mem_t(const mem_t &mem) : dptr(mem.dptr), N(mem.N), memtype(shallow_e), ptrs(mem.ptrs)
  {

  }
  ~mem_t()
  {
    switch(memtype)
    {
      case host_e: { delete dptr; break; }
      case device_e: { checkCudaErrors(cudaFree(dptr)); break; }
      case pinned_e: { checkCudaErrors(cudaFreeHost(dptr)); break; }
      case const_e: { break; }
      case shallow_e: break;
      default: break; //throw_tw_error("mem_t destructor: invalid memtype!");
    }
  }
  int count_bytes() const
  {
    int Nbytes = 0;
    auto count_bytes_func = [&Nbytes](auto&& ptr)
    {
      Nbytes += sizeof(*std::declval<decltype(ptr)>());
    };

    std::apply([&](auto& ...ptr)
    {
      (..., count_bytes_func(ptr));
    }, ptrs);

    return N*Nbytes;
  }
};
*/

/*
// ----------------------------------------------------------------------------
template<typename T>
struct mat_t
{
  char      *dptr;  // 1x64bit (holds data)
  const memtype_e memtype;
  const int  Ndims; // 1x32bit (start of data) [merge with dptr?]
  const int *dims;  // Nx32bit
  T         *data;  // MxXYbit
  // --------------------------------------------------------------------------
  mat_t(const int N, memtype_e memtype) : N(N), memtype(memtype)
  {
    int Nbytes = sizeof(int)*Ndims;
    int dim_product = 1;

    for(int it = 0 ; it < Ndims ; ++it)
      dim_product *= dims[it];

    if(has_size<T>::value)
      Nbytes += T.count_bytes()*Ndims;
    else
      Nbytes += sizeof(int)*Ndims + 

    switch(memtype)
    {
      //case host_e: { dptr = new char[Nbytes]; break; }
      //case device_e: { checkCudaErrors(cudaMalloc((&dptr, Nbytes))); break; }
      //case pinned_e: { checkCudaErrors(cudaMallocHost(&dptr, Nbytes)); break; }
      //case const_e: { break; }
      case shallow_e: break;
      default: break; // throw
    }
  }
  // --------------------------------------------------------------------------
  ~mat_t()
  {
    switch(memtype)
    {
      //case host_e: { delete dptr; break; }
      //case device_e: { checkCudaErrors(cudaFree(dptr)) break; }
      //case pinned_e: { checkCudaErrors(cudaFreeHost(dptr)); break; }
      //case const_e: { break; }
      case shallow_e: break;
      default: break; // throw
    }
  }



  mat_t(char *ptr, const int Nbytes, const int alignment=8) : 
        dptr(ptr), N(*reinterpret_cast<int>(ptr))
  {
    if(N < 1)
      throw_tw_error("mat_t: invalid size: " + std::to_string(N));

    ptr += aligned_size(sizeof(mat_t<T>) - sizeof(char*), alignment);
    cnt = reinterpret_cast<cnt_t*>(ptr);
    ptr += aligned_size(N*sizeof(cnt_t), alignment);
    dsp = reinterpret_cast<idx_t*>(ptr);
    ptr += aligned_size(N*sizeof(idx_t), alignment);
    con = reinterpret_cast<T*>(ptr);

    const int bytes = count_bytes();

    if(bytes > Nbytes)
      throw_tw_error("mat_t: init array too small: " + std::to_string(N));
    else if(bytes < Nbytes)
      log_warning("mat_t: init array larger than data.");
  }
  // --------------------------------------------------------------------------
  mat_t(const mat_t &arr) : dptr(arr.dptr), N(arr.N), 
                                    cnt(arr.cnt), dsp(arr.dsp), con(arr.con)
  {
    if(N < 1)
      throw_tw_error("mat_t: invalid size: " + std::to_string(N));
    // rebuild?
  }
  // --------------------------------------------------------------------------
  int count_bytes(const int alignment=8)
  {
    int Ncon = 0;

    for(int it = 0 ; it < N ; ++it)
      Ncon += cnt[it];

    return aligned_size(sizeof(mat_t<T>), alignment) + // base
           aligned_size(     N*sizeof(cnt_t), alignment) + // cnt
           aligned_size(     N*sizeof(idx_t), alignment) + // dsp
           aligned_size(      Ncon*sizeof(T), alignment);  // con
  }
};
// ----------------------------------------------------------------------------
// ek_memory_data &test = *reinterpret_cast<ek_memory_data*>(ptr);
// test.construct();
// struct must be coupled to entire memory section + data must be aligned (to what?)
// this data structure is nice for ambigous mappings!
template<typename T>
struct aligned_t // better name (aligned_data)
{
  char *dptr; // 1x64bit (holds data)
  int     &N; // 1x32bit (start of data) [merge with dptr?]
  cnt_t *cnt; // Nx16bit
  idx_t *dsp; // Nx64bit ptr_t adr_t
  T     *con; // MxXYbit
  // maybe save M? or multiple data? -> K
  
  // --------------------------------------------------------------------------
  aligned_t(char *ptr, const int Nbytes, const int alignment=8) : 
        dptr(ptr), N(*reinterpret_cast<int>(ptr))
  {
    if(N < 1)
      throw_tw_error("aligned_t: invalid size: " + std::to_string(N));

    ptr += aligned_size(sizeof(aligned_t<T>) - sizeof(char*), alignment);
    cnt = reinterpret_cast<cnt_t*>(ptr);
    ptr += aligned_size(N*sizeof(cnt_t), alignment);
    dsp = reinterpret_cast<idx_t*>(ptr);
    ptr += aligned_size(N*sizeof(idx_t), alignment);
    con = reinterpret_cast<T*>(ptr);

    const int bytes = count_bytes();

    if(bytes > Nbytes)
      throw_tw_error("aligned_t: init array too small: " + std::to_string(N));
    else if(bytes < Nbytes)
      log_warning("aligned_t: init array larger than data.");
  }
  // --------------------------------------------------------------------------
  aligned_t(const aligned_t &arr) : dptr(arr.dptr), N(arr.N), 
                                    cnt(arr.cnt), dsp(arr.dsp), con(arr.con)
  {
    if(N < 1)
      throw_tw_error("araligned_tr_t: invalid size: " + std::to_string(N));
    // rebuild?
  }
  // --------------------------------------------------------------------------
  int count_bytes(const int alignment=8)
  {
    int Ncon = 0;

    for(int it = 0 ; it < N ; ++it)
      Ncon += cnt[it];

    return aligned_size(sizeof(aligned_t<T>), alignment) + // base
           aligned_size(     N*sizeof(cnt_t), alignment) + // cnt
           aligned_size(     N*sizeof(idx_t), alignment) + // dsp
           aligned_size(      Ncon*sizeof(T), alignment);  // con
  }
};
// ----------------------------------------------------------------------------
struct ek_node_data // possibly all internal
{
  int N = -1;
  dbl_t *phi_ms; // activation times
  int *states;   // node states [int2]
  dbl_t *di_ms;  // diastolic interval
  dbl_t* curv;   // wavefront curvature
  // dbl_t *act, apd;
};
// ----------------------------------------------------------------------------
// if visualized through vulkan -> no cpu memcpy is required!
struct ek_subsolver : public ek_node_data
{
  // mesh properties
  aligned_t<idx_t> e2n, n2e, n2n;
  dbl_t *xyz, *lon;
  // tag_t *tag;

  // solver properties [global, constant]
  //ek_options *opts;
  //temp_item *stims, *blocks;
  //ek_region_properties *rp;
  dbl_t wavefront_width, phi_start;
};
*/

// ----------------------------------------------------------------------------
struct ek_params_cuda
{
  char* meshdata, *nodedata, *solverdata;
  const int aidx, eidx;
};
// ----------------------------------------------------------------------------
// ptr, aidx, Nagg -> gives pointers to each section
// use int2, int4 ...
struct ek_data_cuda
{
public:
  // mesh properties [global, read only] const?
  int aidx, eidx, Nn, Ne, Nenc, Nnnc;
  cnt_t *e2n_cnt, *n2e_cnt, *n2n_cnt;
  idx_t *e2n_dsp, *n2e_dsp, *n2n_dsp;
  idx_t *e2n_con, *n2e_con, *n2n_con;
  dbl_t *xyz, *lon;

  // node properties [global, pinned]
  dbl_t *phi_ms; // activation times
  int *states;   // node states [int2]
  dbl_t *di_ms;  // diastolic interval
  dbl_t* curv;   // wavefront curvature
  // dbl_t *act, apd;

  // solver properties [global, constant]
  //ek_options *opts;
  //temp_item *stims, *blocks;
  //ek_region_properties *rp;
  dbl_t wavefront_width, phi_start;

public:
  ek_data_cuda(ek_params_cuda *params);
  void bytes(int &inBytes, int &outBytes, int &totalBytes, const int byte_count);
};
// ----------------------------------------------------------------------------
// use intX here?
class ek_solver_cuda : public ek_data_cuda
{
protected:
  dbl_t cur_time = 0.;

public: 
  ek_solver_cuda(ek_params_cuda *params);
  ~ek_solver_cuda();
};
// ----------------------------------------------------------------------------
class fim_cuda final : public ek_solver_cuda
{
protected:
  size_t active_size = 0;
  idx_t *active_list;

public: 
  fim_cuda(ek_params_cuda *params);
  ~fim_cuda();
};
// ----------------------------------------------------------------------------



















































































#define CMEM_T_MAGIC 0x1337

template<typename T>
__host__ __device__ void assign_ptr(T *&ptr_T, char *&ptr, const size_t N, const size_t byte_count=8UL)
{
  ptr_T = reinterpret_cast<T*>(ptr);
  ptr += N*sizeof(T);
  ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
}

// ----------------------------------------------------------------------------
// handles auto-aligned, continuous memory vector of class T
// |--static--|---dynamic---|
// maybe add method to switch memory type (eg from host to device -> allocates and moves data)
template<class T>
struct cmem_t // auto-align! -> cvec_t
{
  char* dptr = nullptr;
  size_t N;
  memtype_e memtype;
  size_t *data_bytes = nullptr; // always first entry in cmem -> unify with dptr?
  T** data = nullptr; // char**, unify with dptr?

  /*
  cmem_t(memtype_e memtype, const int N, const int totalBytes) : N(N), memtype(memtype)
  {
    allocate(totalBytes);
  }
  cmem_t(memtype_e memtype, std::vector<int> nBytes) : N(nBytes.size()), memtype(memtype)
  {
    //allocate(totalBytes);
  }
  */

  cmem_t() : dptr(nullptr), N(0), memtype(NA_e), data_bytes(nullptr), data(nullptr)
  {
    //log_info("cmem_t");
  }

  //cmem_t(cmem_t<T> & other, memtype); ...

  void init(memtype_e memory_type, const std::vector<size_t> &bytes_per_element) // recursive member copy? what if cmem_t?
  {
    //log_info("init");
    if (this->memtype < NA_e)
      throw_tw_error("memory already allocated!"); // free + realloc + warning?

    this->N = bytes_per_element.size(); this->memtype = memory_type;
    size_t total_bytes = sizeof(size_t)*N + sizeof(T*)*N; // aligned

    for (size_t it = 0 ; it < N ; ++it)
      total_bytes += bytes_per_element[it];

    //printf("tp1: %lu\n", total_bytes);

    switch(memtype)
    {
      case host_e: { dptr = new char[total_bytes]; break; }//std::cout<<"allocated on host\n";
      case device_e: { checkCudaErrors(cudaMalloc(&dptr, total_bytes)); break; }//std::cout<<"allocated on device\n";
      case pinned_e: { checkCudaErrors(cudaMallocHost(&dptr, total_bytes));  break; }//std::cout<<"allocated on device (pinned)\n";
      case const_e: {  break; }//std::cout<<"allocated on device (const)\n";
      case shallow_e: {  break; }//std::cout<<"allocated nothing (shallow)\n";
      case NA_e: { std::cout<<"allocated nothing\n"; break; }
      default: throw_tw_error("cmem_t constructor: invalid memtype!");
    }

    //printf("tp2: %p\n", dptr);

    char *ptr = dptr;
    data_bytes = reinterpret_cast<size_t*>(ptr); ptr += sizeof(size_t)*N; // align!
    if (memtype == device_e)
      checkCudaErrors(cudaMemcpy(data_bytes, bytes_per_element.data(), sizeof(size_t)*N, cudaMemcpyHostToDevice)); 
    else
      memcpy(data_bytes, bytes_per_element.data(), sizeof(size_t)*N); // always host init?
    data = reinterpret_cast<T**>(ptr); ptr += sizeof(T*)*N;

    //printf("tp3: %p %p %p %lu\n", data_bytes, data, ptr, N);

    std::vector<T*> data_ptrs(N);

    for (size_t it = 0 ; it < N ; ++it)
    {
      //printf("it %lu\n", it);
      data_ptrs[it] = reinterpret_cast<T*>(ptr);
      ptr += bytes_per_element[it];
    }

    if (memtype == device_e) // merge with data_bytes cpy
      checkCudaErrors(cudaMemcpy(data, data_ptrs.data(), sizeof(T*)*N, cudaMemcpyHostToDevice)); 
    else
      memcpy(data, data_ptrs.data(), sizeof(T*)*N); // always host init? nope!

    //printf("tp4\n");
  }

  //cudaMemcpyHostToHost          =   0,      00
  //cudaMemcpyHostToDevice        =   1,      01
  //cudaMemcpyDeviceToHost        =   2,      10
  //cudaMemcpyDeviceToDevice      =   3,      11
  void copy_to(cmem_t &dst) const // what about async?
  {
    std::vector<size_t> dst_bytes(N), src_bytes(N);

    if (dst.memtype < NA_e)
    {
      if (dst.N != N)
        throw_tw_error("num elements mismatch");

      if (dst.memtype == device_e)
        checkCudaErrors(cudaMemcpy(dst_bytes.data(), dst.data_bytes, sizeof(size_t)*N, cudaMemcpyDeviceToHost)); 
      else
        memcpy(dst_bytes.data(), dst.data_bytes, sizeof(size_t)*N);

      if (memtype == device_e)
        checkCudaErrors(cudaMemcpy(src_bytes.data(), data_bytes, sizeof(size_t)*N, cudaMemcpyDeviceToHost)); 
      else
        memcpy(src_bytes.data(), data_bytes, sizeof(size_t)*N);

      for (size_t it = 0 ; it < N ; ++it)
      {
        if (dst_bytes[it] != src_bytes[it]) // might by on different devices!
          throw_tw_error("element size mismatch");
      }
    }
    else
    {
      throw_tw_error("not implemented yet");
    }

    //const size_t total_bytes = bytes();
    size_t total_bytes = 0;

    for (size_t it = 0 ; it < N ; ++it)
      total_bytes += dst_bytes[it];
    const cudaMemcpyKind copy_dir = static_cast<cudaMemcpyKind>(((memtype > host_e) << 1) | ((dst.memtype > host_e) << 0));

    const size_t ptr_offset = N*(sizeof(size_t) + sizeof(T*));

    if (!copy_dir)
      memcpy(dst.dptr + ptr_offset, dptr + ptr_offset, total_bytes);
    else
      checkCudaErrors(cudaMemcpy(dst.dptr + ptr_offset, dptr + ptr_offset, total_bytes, copy_dir)); 

    checkCudaErrors(cudaDeviceSynchronize());
    // dst assemble
  }

  __host__ __device__ T& operator[](const size_t idx)
  {
    if (idx >= N)
      printf("cmem: invalid index: %lu\n", idx);
    return *data[idx];
  }

  /*__host__ __device__ size_t bytes(const int byte_count=8UL) const
  {
    size_t total_bytes = sizeof(size_t)*N + sizeof(T*)*N; // aligned

    for (size_t it = 0 ; it < N ; ++it)
      total_bytes += data_bytes[it];

    return total_bytes;
  }*/

  ~cmem_t()
  {
    //log_info("~cmem_t");
  }

  void destroy()
  {
    log_info("destroy");

    switch(memtype)
    {
      case host_e: { delete[] dptr; break; }
      case device_e: { checkCudaErrors(cudaFree(dptr)); break; }
      case pinned_e: { checkCudaErrors(cudaFreeHost(dptr)); break; }
      case const_e: { break; }
      case shallow_e: { break; }
      case NA_e: { break; }
      default: break; //throw_tw_error("mem_t destructor: invalid memtype!");
    }
  }
};


















// ----------------------------------------------------------------------------
struct ek_event2 // active/blocked
{
  int id, active = 0;
  dbl_t tstart = 0., dur = 0.;
  size_t Nagg, Nall;
  idx_t *adx;
  cnt_t *vtx_cnt;
  idx_t *vtx_dsp;
  idx_t *vtx_con;

  __host__ __device__ void assemble(const size_t byte_count=8UL)
  {
    char *ptr = reinterpret_cast<char*>(this) + aligned_size(sizeof(ek_event2), byte_count);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    adx = reinterpret_cast<idx_t*>(ptr); ptr += Nagg*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    vtx_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Nagg*sizeof(cnt_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    
    vtx_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Nagg*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    vtx_con = reinterpret_cast<idx_t*>(ptr); ptr += Nall*sizeof(idx_t);
  }

  __host__ __device__ size_t bytes(const int byte_count=8UL) const
  {
    return sbytes(Nagg, Nall, byte_count);
  }

  static size_t sbytes(const int Nagg, const int Nall, const int byte_count=8UL)
  {
    return aligned_size( sizeof(ek_event2), byte_count) + // base
           aligned_size(Nagg*sizeof(idx_t), byte_count) + // adx
           aligned_size(Nagg*sizeof(cnt_t), byte_count) + // vtx_cnt
           aligned_size(Nagg*sizeof(idx_t), byte_count) + // vtx_dsp
           aligned_size(Nall*sizeof(idx_t), byte_count);  // vtx_con
  }
  
  void construct_from(const ek_event &event, 
                      const std::map<idx_t, std::vector<idx_t>> events_per_agg) // always cpu?
  {
    id = event.id; active = event.active;
    tstart = event.tstart; dur = event.dur;

    Nagg = 0; Nall = 0;

    for (const auto &item : events_per_agg)
      Nall += item.second.size();

    Nagg = events_per_agg.size();
    
    assemble();

    int ait = 0, adsp = 0;

    for (const auto &it : events_per_agg)
    {
      adx[ait] = it.first;
      vtx_cnt[ait] = it.second.size();
      vtx_dsp[ait] = adsp;
      memcpy(&vtx_con[adsp], it.second.data(), sizeof(idx_t)*it.second.size());
      ait++;
      adsp += it.second.size();
    }
  }
};













// ----------------------------------------------------------------------------
// lon might be 6*sizeof(dbl_t)!
struct smsh_t // 
{
  size_t Nn, Ne, Nenc, Nnnc; //rdx
  cnt_t *e2n_cnt, *n2e_cnt, *n2n_cnt;
  idx_t *e2n_dsp, *n2e_dsp, *n2n_dsp;
  idx_t *e2n_con, *n2e_con, *n2n_con;
  dbl_t *geo; // *xyz,  = nullptr

  __host__ __device__ void assemble(const size_t byte_count=8UL)
  {
    //printf("is_aligned: %d\n", is_aligned(this, byte_count));
    char *ptr = reinterpret_cast<char*>(this) + aligned_size(sizeof(smsh_t), byte_count);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    e2n_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Ne*sizeof(cnt_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    n2e_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Nn*sizeof(cnt_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    n2n_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Nn*sizeof(cnt_t);
    //printf("is_aligned: %d\n", is_aligned(ptr, byte_count));
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    //printf("is_aligned: %d\n", is_aligned(ptr, byte_count));

    e2n_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Ne*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    n2e_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Nn*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    n2n_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Nn*sizeof(idx_t);
    //printf("is_aligned: %d\n", is_aligned(ptr, byte_count));
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    //printf("is_aligned: %d\n", is_aligned(ptr, byte_count));

    e2n_con = reinterpret_cast<idx_t*>(ptr); ptr += Nenc*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    n2e_con = reinterpret_cast<idx_t*>(ptr); ptr += Nenc*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    n2n_con = reinterpret_cast<idx_t*>(ptr); ptr += Nnnc*sizeof(idx_t);
    //printf("is_aligned: %d\n", is_aligned(ptr, byte_count));
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    //printf("is_aligned: %d\n", is_aligned(ptr, byte_count));

    //xyz = reinterpret_cast<dbl_t*>(ptr); ptr += 3*Nn*sizeof(dbl_t);
    //ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    geo = reinterpret_cast<dbl_t*>(ptr); ptr += 6*Ne*sizeof(dbl_t);
  }

  void construct_from(const mesh_t &mesh)
  {
    Nn = mesh.n2e_cnt.size(); Ne = mesh.e2n_cnt.size();
    Nenc = mesh.n2e_con.size(); Nnnc = mesh.n2n_con.size();

    assemble();

    // mesh_t is always on cpu?
    memcpy(e2n_cnt, mesh.e2n_cnt.data(), Ne*sizeof(cnt_t));
    memcpy(e2n_dsp, mesh.e2n_dsp.data(), Ne*sizeof(idx_t));
    memcpy(e2n_con, mesh.e2n_con.data(), Nenc*sizeof(idx_t));

    memcpy(n2e_cnt, mesh.n2e_cnt.data(), Nn*sizeof(cnt_t));
    memcpy(n2e_dsp, mesh.n2e_dsp.data(), Nn*sizeof(idx_t));
    memcpy(n2e_con, mesh.n2e_con.data(), Nenc*sizeof(idx_t));

    memcpy(n2n_cnt, mesh.n2n_cnt.data(), Nn*sizeof(cnt_t));
    memcpy(n2n_dsp, mesh.n2n_dsp.data(), Nn*sizeof(idx_t));
    memcpy(n2n_con, mesh.n2n_con.data(), Nnnc*sizeof(idx_t));

    //memcpy(xyz, mesh.xyz.data(), 3*Nn*sizeof(dbl_t));
    memcpy(geo, mesh.geo.data(), 6*Ne*sizeof(dbl_t)); // Nedges
  }

  __host__ __device__ size_t bytes(const int byte_count=8UL) const
  {
    return static_bytes(Nn, Ne, Nenc, Nnnc, byte_count);
  }
  static size_t static_bytes(const int Nn, const int Ne, const int Nenc, const int Nnnc, const int byte_count=8UL)
  {
    return aligned_size(    sizeof(smsh_t), byte_count) + // struct
           aligned_size(  Ne*sizeof(cnt_t), byte_count) + // e2n_cnt
           aligned_size(  Nn*sizeof(cnt_t), byte_count) + // n2e_cnt
           aligned_size(  Nn*sizeof(cnt_t), byte_count) + // n2n_cnt
           aligned_size(  Ne*sizeof(idx_t), byte_count) + // e2n_dsp
           aligned_size(  Nn*sizeof(idx_t), byte_count) + // n2e_dsp
           aligned_size(  Nn*sizeof(idx_t), byte_count) + // n2n_dsp
           aligned_size(Nenc*sizeof(idx_t), byte_count) + // e2n_con
           aligned_size(Nenc*sizeof(idx_t), byte_count) + // n2e_con
           aligned_size(Nnnc*sizeof(idx_t), byte_count) + // n2n_con
           //aligned_size(3*Nn*sizeof(dbl_t), byte_count) + // xyz
           aligned_size(6*Ne*sizeof(dbl_t), byte_count);  // geo
  }
};
// ----------------------------------------------------------------------------
struct eksv_t
{
  int nNodes, Nenc;
  //ek_multistate agg_state;
  ek_state agg_state;
  // size_t active_size = 0;

  dbl_t *phi_ms; // activation times
  int *states;   // node states [int2]
  //dbl_t *di90_ms;  // diastolic interval
  //dbl_t* curv;   // wavefront curvature
  int *tmp_state = nullptr; // move to eksv?
  dbl_t *tmp_phi_ms = nullptr;

  __host__ __device__ void assemble(const int byte_count=8UL)
  {
    char *ptr = reinterpret_cast<char*>(this) + aligned_size(sizeof(eksv_t), byte_count);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    phi_ms = reinterpret_cast<dbl_t*>(ptr); ptr += nNodes*sizeof(dbl_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    states = reinterpret_cast<int*>(ptr);   ptr += nNodes*sizeof(int);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    //di90_ms  = reinterpret_cast<dbl_t*>(ptr); ptr += nNodes*sizeof(dbl_t);
    //ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    //curv   = reinterpret_cast<dbl_t*>(ptr); ptr += nNodes*sizeof(dbl_t);
    //ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    tmp_state = reinterpret_cast<int*>(ptr); ptr += nNodes*sizeof(int);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
    tmp_phi_ms = reinterpret_cast<dbl_t*>(ptr); ptr += Nenc*sizeof(dbl_t);
    //ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
  }

  __host__ __device__ int bytes(const int byte_count=8UL) const
  {
    return this->static_bytes(nNodes, byte_count);
  }

  static int static_bytes(const int Nn, const int Nenc, const int byte_count=8UL)
  {
    return aligned_size(  sizeof(eksv_t), byte_count) + // struct
           aligned_size(Nn*sizeof(dbl_t), byte_count) + // phi_ms
           aligned_size(  Nn*sizeof(int), byte_count) + // states
           //aligned_size(Nn*sizeof(dbl_t), byte_count) + // di90_ms
           //aligned_size(Nn*sizeof(dbl_t), byte_count);  // curv
           aligned_size(    Nn*sizeof(int), byte_count) + // tmp_state
           aligned_size(Nenc*sizeof(dbl_t), byte_count);  // phi_start_ms
  }

  void construct_from(const mesh_t &mesh) // always on cpu?
  {
    nNodes = mesh.n2e_cnt.size();
    Nenc = mesh.n2e_con.size();

    assemble();

    std::fill(phi_ms, phi_ms + nNodes, DBL_INF);
    std::fill(states, states + nNodes, ek_far);
    //std::fill(di90_ms, di90_ms + nNodes, DBL_INF);
    //std::fill(curv, curv + nNodes, 0.0);

    std::fill(tmp_state, tmp_state + nNodes, ek_far); // not required ..
    std::fill(tmp_phi_ms, tmp_phi_ms + Nenc, DBL_INF);
  }
};
// ----------------------------------------------------------------------------









/*
// ----------------------------------------------------------------------------
struct slvd_t
{
  size_t Nn = 0, Nenc = 0, active_size = 0;
  dbl_t front_width_ms = DBL_INF;
  dbl_t cur_time_ms = 0.0;
  dbl_t phi_start_ms = 0.0;

  //smsh_t *mesh = nullptr; // just reference for convenience (computation)
  //eksv_t *states = nullptr;

  int *tmp_state = nullptr; // move to eksv?
  dbl_t *tmp_phi_ms = nullptr;

  void construct_from(const mesh_t &submesh, const dbl_t wfw_ms) // always on cpu?
  {
    Nn = submesh.n2e_cnt.size();
    Nenc = submesh.n2e_con.size();
    active_size = 0;
    front_width_ms = wfw_ms;
    cur_time_ms = 0.0;
    phi_start_ms = 0.0;
    //mesh = nullptr;
    //states = nullptr;

    assemble();

    std::fill(tmp_state, tmp_state + Nn, ek_far); // not required ..
    std::fill(tmp_phi_ms, tmp_phi_ms + Nenc, DBL_INF);
  }

  __host__ __device__ void assemble(const int byte_count=8UL)
  {
    char *ptr = reinterpret_cast<char*>(this) + aligned_size(sizeof(slvd_t), byte_count);

    tmp_state = reinterpret_cast<int*>(ptr); ptr += Nn*sizeof(int);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    tmp_phi_ms = reinterpret_cast<dbl_t*>(ptr); ptr += Nenc*sizeof(dbl_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
  }

  int bytes(const int byte_count=8UL) const//const int byte_count=8UL
  {
    return this->static_bytes(Nn, Nenc, byte_count);
  }

  static int static_bytes(const size_t Nn, const size_t Nenc, const int byte_count=8UL)
  {
    return aligned_size(    sizeof(slvd_t), byte_count) + // base
           aligned_size(    Nn*sizeof(int), byte_count) + // tmp_state
           aligned_size(Nenc*sizeof(dbl_t), byte_count);  // phi_start_ms
  }
  // --------------------------------------------------------------------------
};
*/









struct reg_t
{
  dbl_t vx = 0.0, vy = 0.0, vz = 0.0;
  dbl_t apd = 0.0;

  dbl_t *tapdr, *fapdr;
  dbl_t *tcvr,  *fcvr;
};













/*
// ----------------------------------------------------------------------------
struct inter_t
{
  size_t Na = 0;
  size_t Nc = 0;
  idx_t *adx_con = nullptr;
  cnt_t *gcd_cnt = nullptr; // bad name!
  idx_t *gcd_dsp = nullptr;
  idx_t *gcd_con = nullptr;

  void construct_from(const std::map<idx_t,std::vector<idx_t>> &vtxcon) // cpu only?
  {
    int it = 0;
    Na = vtxcon.size();
    Nc = 0;
    
    for (const auto &item : vtxcon)
      Nc += item.second.size();

    assemble();

    Nc = 0;
    
    //std::cout<<"inter_t construct_from: ";
    for (const auto &item : vtxcon)
    {
      const size_t cnt = item.second.size();
      gcd_cnt[it] = cnt;
      gcd_dsp[it] = Nc;

      //std::cout<<is_aligned(gcd_con, 8UL)<<" "<<is_aligned(gcd_dsp, 8UL)<<" "<<is_aligned(gcd_cnt, 8UL)<<"\n";
      //std::cout<<"cnt: "<<cnt<<" ptr: "<<gcd_con<<" "<<gcd_dsp<<" "<<gcd_cnt<<" "<<adx_con<<" "<<this<<" ";

      for (size_t jt = 0 ; jt < cnt ; ++jt)
        gcd_con[Nc + jt] = item.second[jt];

      //memcpy(gcd_con + Nc, item.second.data(), sizeof(idx_t)*cnt);
      adx_con[it] = item.first;

      //std::cout<<"A"<<item.first<<": ";
      //for (const auto &jt : item.second)
      //  std::cout<<jt<<" ";
      //std::cout<<" - ";
      //for (int jt = 0 ; jt < item.second.size() ; ++jt)
      //  std::cout<<gcd_con[Nc+jt]<<" ";

      Nc += cnt;
      it++;
    }
    //std::cout<<" - Na: "<<Na<<" Nc: "<<Nc<<"\n";

    //for (size_t it = 0 ; it < Na ; ++it)
    //{
    //  const idx_t tdsp = gcd_dsp[it];

    //  for (int16_t tcnt = 0 ; tcnt < gcd_cnt[it] ; ++tcnt)
    //    std::cout<<gcd_con[tdsp+tcnt]<<" ";
    //  std::cout<<"\n";
    //}

    //memcpy(adx_con, a2a_con.data(), sizeof(idx_t)*Na);
    //for (size_t it = 0 ; it < vtxcon.size() ; ++it)
    //{
    //  const size_t cnt = vtxcon[it].size();
    //  gcd_cnt[it] = cnt;
    //  gcd_dsp[it] = Nc;
    //  memcpy(&gcd_con[Nc], vtxcon[it].data(), sizeof(idx_t)*cnt);
    //  Nc += cnt;
    //}
  }

  __host__ __device__ void assemble(const int byte_count=8UL)
  {
    char *ptr = reinterpret_cast<char*>(this) + aligned_size(sizeof(inter_t), byte_count);

    //assign_ptr(gcd_cnt, ptr, Na, byte_count);
    //assign_ptr(gcd_dsp, ptr, Na, byte_count);
    //assign_ptr(gcd_con, ptr, Nc, byte_count);
    //assign_ptr(adx_con, ptr, Na, byte_count);
    adx_con = reinterpret_cast<idx_t*>(ptr); ptr += Na*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    gcd_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Na*sizeof(cnt_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    gcd_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Na*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    gcd_con = reinterpret_cast<idx_t*>(ptr); ptr += Nc*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
  }

  int bytes(const int byte_count=8UL) const
  {
    return this->static_bytes(Na, Nc, byte_count);
  }

  static int static_bytes(const int Na1, const int Nc1, const int byte_count=8UL)
  {
    return aligned_size(  sizeof(inter_t), byte_count) + // base
           aligned_size(Na1*sizeof(idx_t), byte_count) + // adx_con
           aligned_size(Na1*sizeof(cnt_t), byte_count) + // gcd_cnt
           aligned_size(Na1*sizeof(idx_t), byte_count) + // gcd_dsp
           aligned_size(Nc1*sizeof(idx_t), byte_count);  // gcd_con
  }
};
// ----------------------------------------------------------------------------
struct inter2_t
{
  size_t Nn = 0;
  size_t Nc = 0;
  cnt_t *int_cnt = nullptr;
  idx_t *int_dsp = nullptr;
  //ipair_t *int_con = nullptr;
  idx_t *vtx_con = nullptr;
  idx_t *adx_con = nullptr;

  void construct_from(const std::vector<std::vector<ipair_t>> &n2i) // cpu only?
  {
    Nn = n2i.size();
    Nc = 0;
    
    for (const auto &item : n2i)
      Nc += item.size();

    assemble();

    int it = 0;
    Nc = 0;
    
    for (const auto &item : n2i)
    {
      const size_t cnt = item.size();
      int_cnt[it] = cnt;
      int_dsp[it] = Nc;

      for (size_t jt = 0 ; jt < cnt ; ++jt)
      {
        //int_con[Nc + jt] = item[jt];
        vtx_con[Nc + jt] = item[jt].vtx;
        adx_con[Nc + jt] = item[jt].adx;
      }

      Nc += cnt;
      it++;
    }
  }

  __host__ __device__ void assemble(const int byte_count=8UL)
  {
    char *ptr = reinterpret_cast<char*>(this) + aligned_size(sizeof(inter2_t), byte_count);

    int_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Nn*sizeof(cnt_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    int_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Nn*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    //int_con = reinterpret_cast<ipair_t*>(ptr); ptr += Nc*sizeof(ipair_t);
    //ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    vtx_con = reinterpret_cast<idx_t*>(ptr); ptr += Nc*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    adx_con = reinterpret_cast<idx_t*>(ptr); ptr += Nc*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
  }

  int bytes(const int byte_count=8UL) const
  {
    return this->static_bytes(Nn, Nc, byte_count);
  }

  static int static_bytes(const int Nn1, const int Nc1, const int byte_count=8UL)
  {
    return aligned_size( sizeof(inter2_t), byte_count) + // base
           aligned_size(Nn1*sizeof(cnt_t), byte_count) + // adx_con
           aligned_size(Nn1*sizeof(idx_t), byte_count) + // gcd_cnt
           //aligned_size(Nc1*sizeof(ipair_t), byte_count);// gcd_dsp
           aligned_size(Nc1*sizeof(idx_t), byte_count) + // gcd_dsp
           aligned_size(Nc1*sizeof(idx_t), byte_count);// gcd_dsp
  }
};*/





// ----------------------------------------------------------------------------
struct inter3_t
{
  size_t Nint = 0;
  idx_t *vtx = nullptr;
  idx_t *adx = nullptr;

  void construct_from(const std::vector<ipair_t> &n2i) // cpu only?
  {
    int it = 0;
    Nint = n2i.size();

    assemble();

    for (const auto &item : n2i)
    {
      vtx[it] = item.vtx;
      adx[it] = item.adx;
      it++;
    }
  }

  __host__ __device__ void assemble(const int byte_count=8UL)
  {
    char *ptr = reinterpret_cast<char*>(this) + aligned_size(sizeof(inter3_t), byte_count);

    vtx = reinterpret_cast<idx_t*>(ptr); ptr += Nint*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));

    adx = reinterpret_cast<idx_t*>(ptr); ptr += Nint*sizeof(idx_t);
    ptr += is_aligned(ptr, byte_count)?0:(byte_count - (reinterpret_cast<std::uintptr_t>(ptr) % byte_count));
  }

  int bytes(const int byte_count=8UL) const
  {
    return this->static_bytes(Nint, byte_count);
  }

  static int static_bytes(const int Nint1, const int byte_count=8UL)
  {
    return aligned_size(   sizeof(inter3_t), byte_count) + // base
           aligned_size(Nint1*sizeof(idx_t), byte_count) +
           aligned_size(Nint1*sizeof(idx_t), byte_count);
  }
};















// ----------------------------------------------------------------------------
__forceinline__ __device__
dbl_t solve_tri_cu(const dbl_t &e12Me12, const dbl_t &e13Me13, const dbl_t &e23Me23,
                   const dbl_t &phi1,    const dbl_t &phi2)
{
  const dbl_t &a =       e23Me23;
  const dbl_t  b = -0.5*(e12Me12 + e23Me23 - e13Me13);
  const dbl_t &c =       e12Me12;
  const dbl_t phi12 = phi2 - phi1;

  //const dbl_t sqt = (phi12*phi12*(a*c - b*b))/(c*c*(c - phi12*phi12)); // what about precision and stability?
  //const double discriminant = fma(b, b, -4.0*a*c);
  //if (discriminant < 0.0)  // No real solutions
  //  return 1e50;
  //const double denominator = 2.0*a;
  //const double x1 = (-b - sqrt(discriminant)) / denominator;
  //const double x2 = c / (a * x1);
  //return fmin(x1, x2);
  // clamp solutions?

  float divident = phi12*phi12 - c;
  float sqrt_val = divident*(b*b - a*c);
  //float sqrt_val = (f12*f12-c)*(b*b-a*c);

	if (divident != 0.0 && sqrt_val >= 0.0)
  {
		divident = c*divident;
    const float sqrtTmp = sqrt(sqrt_val);
		float l1 = -(b*phi12*phi12 - b*c - sqrtTmp*phi12)/divident;
		float l2 = -(b*phi12*phi12 - b*c + sqrtTmp*phi12)/divident;

		float ret1 = phi2 - l1*phi12 + sqrt(a + 2.0*l1*b + l1*l1*c);

		if ( !((l1 >= 0.0) && (l1 <= 1.0)))
			ret1 = DBL_INF;

    float ret2 = phi2 - l2*phi12 + sqrt(a + 2.0*l2*b + l2*l2*c);

		if ( !((l2 >= 0.0) && (l2 <= 1.0)) )
			ret2 = DBL_INF;

		return fmin(ret1, ret2);
	}
  else
    return DBL_INF;
}
// ----------------------------------------------------------------------------
__forceinline__ __device__
dbl_t solve_tet_cu(const dbl_t &e13Me13, const dbl_t &e23Me23, const dbl_t &e34Me34,
                   const dbl_t &e12Me12, const dbl_t &e14Me14, const dbl_t &e24Me24,
                   const dbl_t &phi1,    const dbl_t &phi2,    const dbl_t &phi3)
{
  const dbl_t &a =       e13Me13;
  const dbl_t  b =  0.5*(e23Me23 + e13Me13 - e12Me12);
  const dbl_t  c = -0.5*(e34Me34 + e13Me13 - e14Me14);
  const dbl_t &d =       e23Me23;
  const dbl_t  e = -0.5*(e34Me34 + e23Me23 - e24Me24);
  const dbl_t &f =       e34Me34;

  // tetrahedron solutions:
  const dbl_t phi13 = phi3 - phi1;
  const dbl_t phi23 = phi3 - phi2;
  
  const dbl_t sqt = 
    -e*e*c*d*d + a*d*d*d*d - (b*b*c - a*c*c)*f*f + ((b*b - a*c)*f*f -
    (e*e*c - a*d*d - 2.*(e*c - b*d)*e)*f)*phi13*phi13 - 2.*(e*e*c*d -
    2.*e*b*d*d + a*d*d*d + (b*b - a*c)*d*f)*phi13*phi23 + (e*e*c*c -
    2.*e*b*c*d + a*c*d*d + (b*b*c - a*c*c)*f)*phi23*phi23 + 2.*(e*c*d*d -
    b*d*d*d)*e + (e*e*c*c + (b*b - 2.*a*c)*d*d - 2.*(e*c*c - b*c*d)*e)*f;

  if(sqt < 0.)
    return DBL_INF;

  /*l1 = fmin(fmax(-p + sqrt(sqt), 0.0), 1.0);
  l2 = fmin(fmax( l1*g + h,      0.0), 1.0);
  l3 = fmin(fmax( 1.0 - l1 - l2, 0.0), 1.0); // not necessary?
  phi4 = fmin(phi4, l1*phi1 + l2*phi2 + l3*phi3 + sqrt());

  l1 = fmin(fmax(-p - sqrt(sqt), 0.0), 1.0);
  l2 = fmin(fmax( l1*g + h,      0.0), 1.0);
  l3 = fmin(fmax( 1.0 - l1 - l2, 0.0), 1.0);
  phi4 = fmin(phi4, l1*phi1 + l2*phi2 + l3*phi3 + sqrt());*/

  const dbl_t p1 = e*c*d*d - b*d*d*d + (e*c - b*d)*f*phi13*phi13 - 2.*(e*c*d - b*d*d)*phi13*phi23 +
    (e*c*c - b*c*d)*phi23*phi23 - (e*c*c - b*c*d)*f;
  const dbl_t p2 = (d*d*d*d - 2.*c*d*d*f + c*c*f*f + (d*d*f - c*f*f)*phi13*phi13 - 2.*(d*d*d - c*d*f)*phi13*phi23 +
    (c*d*d - c*c*f)*phi23*phi23);

  const dbl_t l2_1 = (p1 + sqt*(d*phi13 - c*phi23))/p2;
  const dbl_t l2_2 = (p1 - sqt*(d*phi13 - c*phi23))/p2;

  const dbl_t l1_1 = -((f*l2_1 + e)*phi13 - (d*l2_1 + b)*phi23)/(d*phi13 - c*phi23);
  const dbl_t l1_2 = -((f*l2_2 + e)*phi13 - (d*l2_2 + b)*phi23)/(d*phi13 - c*phi23);

  dbl_t l3 = 1 - l1_1 - l2_1, phi4_1 = DBL_INF, phi4_2 = DBL_INF;

  if((l1_1 > 0. && l1_1 < 1.) && (l2_1 > 0. && l2_1 < 1.) && l3 > 0.)
    phi4_1 = l1_1*phi1 + l2_1*phi2 + l3*phi3 + sqrt(a + 2.*l1_1*b + l1_1*l1_1*c + 2.*l1_1*l2_1*d + 2.*l2_1*e + l2_1*l2_1*f);
  
  l3 = 1 - l1_2 - l2_2;

  if((l1_2 > 0. && l1_2 < 1.) && (l2_2 > 0. && l2_2 < 1.) && l3 > 0.)
    phi4_2 = l1_2*phi1 + l2_2*phi2 + l3*phi3 + sqrt(a + 2.*l1_2*b + l1_2*l1_2*c + 2.*l1_2*l2_2*d + 2.*l2_2*e + l2_2*l2_2*f);

  return fmin(phi4_1, phi4_2);
}
// ----------------------------------------------------------------------------
__forceinline__ __device__
  dbl_t update_per_element(const smsh_t *mesh, const eksv_t *states, const idx_t vtx, const idx_t edx, 
                           const ek_state threshold, const dbl_t phi_start_ms, const dbl_t front_width_ms)
{
  cnt_t idx = 0; // array -> local
  dbl_t phi_tmp_ms[4];
  const idx_t &dsp = mesh->e2n_dsp[edx]; // move this to loop (or make reference) and check if it makes a difference in register count
  const dbl_t *geo = mesh->geo + 6*edx; // reference?
  const gray_symbols &gidx = d_gray_mappings;

  for (cnt_t cnt = 0 ; cnt < mesh->e2n_cnt[edx] ; ++cnt)
  {
    const idx_t &nvtx = mesh->e2n_con[dsp+cnt]; // move to if?

    if (nvtx == vtx)
      idx = cnt;
    else if ((states->states[nvtx] >= threshold) && (states->phi_ms[nvtx] > (phi_start_ms - front_width_ms)))
      phi_tmp_ms[cnt] = states->phi_ms[nvtx];
    else
      phi_tmp_ms[cnt] = DBL_INF;
  }
  
  const idx_t &i0 = gidx.adf[3*idx + 0], &i1 = gidx.adf[3*idx + 1], &i2 = gidx.adf[3*idx + 2], // and here
              &i3 = gidx.bce[3*idx + 0], &i4 = gidx.bce[3*idx + 1], &i5 = gidx.bce[3*idx + 2];

  const dbl_t &e13Me13 = geo[i0], &e23Me23 = geo[i1], &e34Me34 = geo[i2], // also references here
              &e12Me12 = geo[i3], &e14Me14 = geo[i4], &e24Me24 = geo[i5];

  const dbl_t &phi1 = phi_tmp_ms[gidx.phi[3*idx + 0]]; // use reference here, check if register count changes
  const dbl_t &phi2 = phi_tmp_ms[gidx.phi[3*idx + 1]];
  const dbl_t &phi3 = phi_tmp_ms[gidx.phi[3*idx + 2]];

  //dbl_t phi4 = DBL_INF;
  dbl_t phi4 = solve_tet_cu(e13Me13, e23Me23, e34Me34, e12Me12, e14Me14, e24Me24, phi1, phi2, phi3);

  // triangle solutions:
  phi4 = fmin(phi4, solve_tri_cu(e12Me12, e14Me14, e24Me24, phi1, phi2));
  phi4 = fmin(phi4, solve_tri_cu(e23Me23, e24Me24, e34Me34, phi2, phi3));
  phi4 = fmin(phi4, solve_tri_cu(e13Me13, e14Me14, e34Me34, phi1, phi3));

  // edge solutions
  phi4 = fmin(phi4, phi1 + sqrt(e14Me14));
  phi4 = fmin(phi4, phi2 + sqrt(e24Me24));
  phi4 = fmin(phi4, phi3 + sqrt(e34Me34));

  //if (edx == 5 && vtx == 18)
  //  printf("edx: %3ld (idx: %d) [%ld %ld %ld %ld] -> (%d %d %d) (%lf %lf %lf) phi4: %lf\n", edx, idx, mesh->e2n_con[dsp+0], mesh->e2n_con[dsp+1], mesh->e2n_con[dsp+2], mesh->e2n_con[dsp+3], phi1 < DBL_INF, phi2 < DBL_INF, phi3 < DBL_INF, sqrt(e14Me14), sqrt(e24Me24), sqrt(e34Me34), phi4);//phi1, phi2, phi3,  phi1: %lf, phi2: %lf, phi3: %lf, 

  return phi4;
}





























//void assemble_pointers(cmem_t<smsh_t> &cdata);

class CudaSolverFIM : public ek_data
{
protected:
  const int threads_per_block = 32;
  const int num_blocks = 4;
  size_t max_shared_bytes = 0;

  // cuda solver
  dbl_t cur_time = 0., min_edge = 0., max_edge = 0.;
  size_t e2n_max = 0, n2e_max = 0, n2n_max = 0;

  std::vector<idx_t> g2l;
  std::vector<std::vector<idx_t>> l2g, a2a;

  std::vector<size_t> aggs_sizes;
  std::vector<ek_state> agg_states;

  cmem_t<ek_event2> h_events, d_events;
  cmem_t<smsh_t> h_smsh, d_smsh; // host mesh not needed
  cmem_t<eksv_t> h_eksv, d_eksv;
  //cmem_t<slvd_t> h_slvd, d_slvd; // host solver data not needed

  // agg solve
  //cmem_t<inter_t> h_inter, d_inter;
  //cmem_t<inter2_t> h_inter, d_inter;
  cmem_t<inter3_t> h_inter, d_inter;

  void update_events();
  void update_delayed();

  void transpose_solution(std::vector<std::vector<dbl_t>> &act_ms, std::vector<std::vector<dbl_t>> &apd_ms);

    // --------------------------------------------------------------------------
  void read_xyz2(const idx_t idx, vec3_t& pt) // to mesh class?
  {
    const dbl_t* xp = mesh.xyz.data() + idx*3;
    pt.x = *xp++, pt.y = *xp++, pt.z = *xp++;
  }
  dbl_t aMb_sym2(vec3_t& a, const dbl_t* M, vec3_t& b) // const
  {
    dbl_t M0 = M[0], M1 = M[1], M2 = M[2],
                     M4 = M[3], M5 = M[4],
                                M8 = M[5];

    return (M0*a.x + M1*a.y + M2*a.z)*b.x + 
           (M1*a.x + M4*a.y + M5*a.z)*b.y + 
           (M2*a.x + M5*a.y + M8*a.z)*b.z;
  }
  // --------------------------------------------------------------------------
  void compute_velocity_tensor2(dbl_t *M)//const dbl_t eidx, const vec3_t &v, 
  {
    const dbl_t val = 1./(.6*.6);
    M[0] = val;//M[0] = fa*vf + sa*vs - (fa + sa - 1.)*vn;
    M[1] = 0.0;//M[1] = fb*vf + sb*vs - (fb + sb)*vn;
    M[2] = 0.0;//M[2] = fc*vf + sc*vs - (fc + sc)*vn;
    M[3] = val;//M[3] = fd*vf + sd*vs - (fd + sd - 1.)*vn; // M4
    M[4] = 0.0;//M[4] = fe*vf + se*vs - (fe + se)*vn;      // M5
    M[5] = val;//M[5] = ff*vf + sf*vs - (ff + sf - 1.)*vn; // M8
  }

public:
  CudaSolverFIM(const ek_data &ekdata);
  ~CudaSolverFIM();

  void solve(std::vector<std::vector<dbl_t>> &act_ms,
             std::vector<std::vector<dbl_t>> &apd_ms,
             const std::string file);

  void operator()(std::vector<std::vector<dbl_t>> &act_ms,
                  std::vector<std::vector<dbl_t>> &apd_ms,
                  const std::string file)
  {
    solve(act_ms, apd_ms, file);
  }

  void step(const dbl_t dt);
  void activate(const idx_t idx);
  void cleanup();

  void clear_igb_data();
  void extract_node_data(const size_t output_id, std::vector<dbl_t> &data_out);
  void write_state_slice(const bool OE_flag, const size_t t, 
                         const std::string outfile_path,
                         const bool VTK_flag);
  // ----------------------------------------------------------------------------
  void read_xyz(const idx_t idx, const dbl_t* xyz, vec3_t& pt)
  {
    const dbl_t* xp = xyz + idx*3;
    pt.x = *xp++, pt.y = *xp++, pt.z = *xp++;
  }
  // ----------------------------------------------------------------------------
  dbl_t compute_dwf(const mesh_t &mesh)
  {
    vec3_t A, B;
    dbl_t max_edge_len_um = 0.0, max_velocity_m_s = 0.6;

    //#ifdef EK_OMP_EN
    //#pragma omp parallel for reduction(max:max_edge_len_um) num_threads(8)
    //#endif
    for (size_t vtx = 0 ; vtx < mesh.xyz.size()/3 ; ++vtx) // parallel does not work (?)
    {
      const idx_t ndsp = mesh.n2n_dsp[vtx];
      read_xyz(vtx, mesh.xyz.data(), A);
      dbl_t edge_len_um = 0.0;

      for (cnt_t ncnt = 0 ; ncnt < mesh.n2n_cnt[vtx] ; ++ncnt) // collapse?
      {
        const idx_t nvtx = mesh.n2n_con[ndsp+ncnt];
        read_xyz(nvtx, mesh.xyz.data(), B);
        const vec3_t BA = (B - A);
        const dbl_t tmp = BA.x*BA.x + BA.y*BA.y + BA.z*BA.z;

        if (tmp > edge_len_um)
          edge_len_um = tmp;
      }

      if (edge_len_um > max_edge_len_um)
        max_edge_len_um = edge_len_um;
    }

    max_edge_len_um = sqrt(max_edge_len_um);

    //for (const auto &region : regions)
    //{
    //  max_velocity_m_s = std::max(max_velocity_m_s, region.default_cv_m_s.vf);
    //  max_velocity_m_s = std::max(max_velocity_m_s, region.default_cv_m_s.vs);
    //  max_velocity_m_s = std::max(max_velocity_m_s, region.default_cv_m_s.vn);
    //}

    std::cout<<"DWF: "<<1.01*max_edge_len_um/max_velocity_m_s<<"\n";
    return 1.01*max_edge_len_um/max_velocity_m_s;
  }
};

// ----------------------------------------------------------------------------
__host__ void test_gray_code(const mesh_t &mesh);
__host__ void cuda_fim_test();
__host__ void cuda_fim_test2();
__host__ void cuda_fim_test3();
} // namespace tw -------------------------------------------------------
#endif // TW_CUDA_FIM_H_