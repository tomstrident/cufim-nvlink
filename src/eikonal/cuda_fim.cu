
#include "cuda_fim.h"

namespace tw { // =============================================================
// ----------------------------------------------------------------------------
ek_data_cuda::ek_data_cuda(ek_params_cuda *params)
{
  char *ptr = params->meshdata;

  e2n_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Ne*sizeof(cnt_t);
  n2e_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Nn*sizeof(cnt_t);
  n2n_cnt = reinterpret_cast<cnt_t*>(ptr); ptr += Nn*sizeof(cnt_t);

  e2n_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Ne*sizeof(idx_t);
  n2e_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Nn*sizeof(idx_t);
  n2n_dsp = reinterpret_cast<idx_t*>(ptr); ptr += Nn*sizeof(idx_t);

  e2n_con = reinterpret_cast<idx_t*>(ptr); ptr += Nenc*sizeof(idx_t);
  n2e_con = reinterpret_cast<idx_t*>(ptr); ptr += Nenc*sizeof(idx_t);
  n2n_con = reinterpret_cast<idx_t*>(ptr); ptr += Nnnc*sizeof(idx_t);

  xyz = reinterpret_cast<dbl_t*>(ptr); ptr += 3*Nn*sizeof(dbl_t);

  ptr = params->nodedata;

  phi_ms = reinterpret_cast<dbl_t*>(ptr); ptr += Nn*sizeof(dbl_t);
  states = reinterpret_cast<int*>(ptr);   ptr += Nn*sizeof(int);

  ptr = params->solverdata;
}
// ----------------------------------------------------------------------------
void ek_data_cuda::bytes(int &inBytes, int &outBytes, int &totalBytes, const int byte_count=8UL)
{
  inBytes  = aligned_size(    (2*Nn + Ne)*sizeof(cnt_t), byte_count) + // cnt
             aligned_size(    (2*Nn + Ne)*sizeof(idx_t), byte_count) + // dsp
             aligned_size((2*Nenc + Nnnc)*sizeof(idx_t), byte_count) + // con
             aligned_size(           3*Nn*sizeof(dbl_t), byte_count);  // xyz
  outBytes = aligned_size(             Nn*sizeof(dbl_t), byte_count) + // phi_ms
             aligned_size(             Nn*sizeof(int)  , byte_count);  // states
  totalBytes = sizeof(ek_data_cuda) + inBytes + outBytes;// + 
  //           aligned_size(             Nn*sizeof(idx_t), byte_count);  // heap/active/wavefront list
}
// ----------------------------------------------------------------------------
ek_solver_cuda::ek_solver_cuda(ek_params_cuda *params) : ek_data_cuda(params)
{

}
// ----------------------------------------------------------------------------
ek_solver_cuda::~ek_solver_cuda()
{
  
}
// ----------------------------------------------------------------------------
fim_cuda::fim_cuda(ek_params_cuda *params) : ek_solver_cuda(params)
{
  
}
// ----------------------------------------------------------------------------
fim_cuda::~fim_cuda()
{
  
}
// ----------------------------------------------------------------------------
void compute_bytes(const mesh_t &mesh, size_t &inBytes, size_t &outBytes, 
                   size_t &totalBytes, const size_t byte_count=8UL)
{
  const size_t Nn = mesh.n2e_cnt.size();
  const size_t Ne = mesh.e2n_cnt.size();
  const size_t Nenc = mesh.e2n_con.size();
  const size_t Nnnc = mesh.n2n_con.size();

  inBytes  = aligned_size(             Ne*sizeof(tag_t), byte_count) + // etags
             aligned_size(    (2*Nn + Ne)*sizeof(cnt_t), byte_count) + // cnt
             aligned_size(    (2*Nn + Ne)*sizeof(idx_t), byte_count) + // dsp
             aligned_size((2*Nenc + Nnnc)*sizeof(idx_t), byte_count) + // con
             aligned_size(           3*Nn*sizeof(dbl_t), byte_count);  // xyz
  outBytes = aligned_size(             Nn*sizeof(dbl_t), byte_count) + // phi
             aligned_size(             Nn*sizeof(int)  , byte_count);  // states
  totalBytes = sizeof(ek_data_cuda) + inBytes + outBytes + 
             aligned_size(             Nn*sizeof(idx_t), byte_count);  // heap/active/wavefront list
}











// ----------------------------------------------------------------------------
__device__ int batch_idx(const int tidx, const int N, const int threadsPerBlock)
{
  const int tmp = tidx - (N % threadsPerBlock);
  return tidx*(N/threadsPerBlock + 1) - tmp*(tmp >= 0);
}
























































/*
// ----------------------------------------------------------------------------
// what do i want?
// |----mesh----| |----states----|  host
// |-|-|-|-|-|-|  |-|-|-|-|-|-|     host
// |-|-|-|-|-|-|  |-|-|-|-|-|-|     global
//     |-|-|   |-|-|   |-|-|        shared
template<typename T>
__device__ void batch_copy(T *dst, const T *src, const int nBytes, const int tidx)
{
  const int idx_start = compute_idx_start(tidx, nBytes, blockDim.x);
  const int idx_end = compute_idx_start(tidx + 1, nBytes, blockDim.x);

  const char *bsrc = reinterpret_cast<const char*>(src);
        char *bdst = reinterpret_cast<char*>(dst);

  for(int it = idx_start ; it < idx_end ; ++it)
    bdst[it] = bsrc[it];

  // alignment check?
  __syncthreads(); // __syncwarp();
}
__global__ void solve_on_submeshes(cmem_t<smsh_t> meshdata, cmem_t<eksv_t> eksvdata, 
                                   cmem_t<ekfd_t> fimdata,  cmem_t<idx_t> aidxs, const int cur_time) // <- constant memory (continuous, aligned)
{
  extern __shared__ char sptr[];

  const idx_t aidx = aidxs[blockIdx.x]; // size check!
  smsh_t *mesh = meshdata.get(aidx); // big problem .. size needs to be saved within cmem_t !
  eksv_t *eksv = eksvdata.get(aidx);

  const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
  const int mesh_bytes = mesh->bytes(), eksv_bytes = eksv->bytes();

  smsh_t *smesh = reinterpret_cast<smsh_t*>(sptr);
  eksv_t *seksv = reinterpret_cast<eksv_t*>(sptr + mesh_bytes);
  ekfd_t *sekfd = reinterpret_cast<ekfd_t*>(sptr + mesh_bytes + eksv_bytes);

  batch_copy(smesh, mesh, mesh_bytes, tidx);
  batch_copy(seksv, eksv, eksv_bytes, tidx);

  // rebuild pointers on shared
  if(tidx == 0)
  {
    smesh->assemble();
    seksv->assemble();
    sekfd->assemble();
  }

  __syncthreads(); // __syncwarp();

  // update delayed idx
  // update blocked
  // update activations

  // perform eikonal solve on submesh data
  sekfd->update_active_list(tidx, smesh, seksv);

  while(sekfd->active_size > 0)
  {
    const int idx_start = compute_idx_start(tidx, sekfd->active_size, blockDim.x);
    const int idx_end = compute_idx_start(tidx + 1, sekfd->active_size, blockDim.x);

    for(int it = idx_start ; it < idx_end ; ++it)
    {
      const idx_t idx = sekfd->active_list[it];

      if(seksv->states[idx] != ek_block)
      {
        const dbl_t old_phi = seksv->phi_ms[idx];
        dbl_t new_phi = sekfd->update_phi(idx, ek_narrow);
        new_phi = (new_phi < old_phi)?new_phi:old_phi;
        seksv->phi_ms[idx] = new_phi;

        if(fabs(old_phi - new_phi) < (1e-6*(1. + (new_phi + old_phi)*.5)))
        {
          seksv->states[idx] = ek_frozen;

          // add neighbours
          const idx_t ndsp = smesh->n2n_dsp[idx];

          for(cnt_t jt = 0 ; jt < smesh->n2n_cnt[idx] ; ++jt)
          {
            const idx_t nidx = smesh->n2n_con[ndsp+jt];
            const dbl_t old_phi = seksv->phi_ms[nidx];

            if(!(seksv->states[nidx] % 2) && (old_phi > seksv->phi_ms[idx]))
            {
              const dbl_t new_phi = sekfd->update_phi(nidx, ek_narrow);
              //printf("AN: nidx: %d old: %e new: %e\n", nidx, old_phi, new_phi);
              //dbl_t *phi_ptr = &(agg->phi[nidx]);
              //atomicMin(reinterpret_cast<int*>(phi_ptr), agg->update_phi(nidx, ek_narrow));
              //atomicExch(agg->states, ek_narrow);

              if(new_phi < old_phi)
              {
                seksv->states[nidx] = ek_narrow;
                seksv->phi_ms[nidx] = new_phi;
                //atomicMin(&(agg->phi[nidx]), new_phi);
                //printf("    added nidx %d -> %f - (%e,%e,%e) to (%e,%e,%e)\n", nidx, new_phi, agg->xyz[3*idx], agg->xyz[3*idx+1], agg->xyz[3*idx+2], agg->xyz[3*nidx], agg->xyz[3*nidx+1], agg->xyz[3*nidx+2]);
              }
            }
          }
        }
      }
    }

    sekfd->update_active_list(tidx, smesh, seksv);
  }

  // update delayed (batch style) / all in one update?
  const int idx_start = compute_idx_start(tidx, nBytes, blockDim.x);
  const int idx_end = compute_idx_start(tidx + 1, nBytes, blockDim.x);
  
  for (int idx = idx_start ; idx < idx_end ; ++idx)
  {
    if (states[idx] == ek_narrow)
      act_list[act_list_size++] = idx;

    if (states[idx] != ek_frozen) continue;
    if (phi[idx] + apd[idx].back() > cur_time) continue;

    phi[idx] = MT_REAL_INF;
    states[idx] = ek_far;
    curv[idx] = 0.;
    num_updates[idx] = 0;

    // integrate_ghost_cell_data (only if idx is overlap)
    const int gidx = l2g[idx];

    for (all neighbours in a2a[aidx])
    {
      lidx = g2l[gidx];

      if (phi[lidx] > phi[idx])
      {
        states[lidx] = ek_narrow;
        phi[lidx] = phi[idx];
      }
    }
  }

  // copy states back to global
  batch_copy(eksv, seksv, eksv_bytes, tidx);
}

__global__ void update_neighbouring_submeshes(cmem_t<smsh_t> meshdata, cmem_t<eksv_t> eksvdata, 
                                              cmem_t<ekfd_t> fimdata, idx_t *aidxs, const int N)
{
  extern __shared__ char sptr[];

  const idx_t aidx = aidxs[blockIdx.x]; // size check!
  smsh_t *mesh = meshdata.get(aidx);
  eksv_t *eksv = eksvdata.get(aidx);

  const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
  const int mesh_bytes = mesh->bytes(), eksv_bytes = eksv->bytes();
  
  smsh_t *smesh = reinterpret_cast<smsh_t*>(sptr);
  eksv_t *seksv = reinterpret_cast<eksv_t*>(sptr + mesh_bytes);

  batch_copy(seksv, eksv, eksv_bytes, tidx);

  // synchronize states / update neighbours?
  __syncthreads();

  const int idx_start = compute_idx_start(tidx, nBytes, blockDim.x);
  const int idx_end = compute_idx_start(tidx + 1, nBytes, blockDim.x);
  
  for (int idx = idx_start ; idx < idx_end ; ++idx)
  {
    const int gidx = l2g[idx];
    const idx_t dsp = a2a_dsp[aidx];

    for (int it = 0 ; it < a2a_cnt[aidx] ; ++it)
    {
      const int naidx = a2a_con[dsp+it];
      const int lidx = g2l[gidx];

      if (eksv->phi[] < seksv->phi_ms[])
      {
        states[lidx] = ek_narrow;
        phi[lidx] = phi[idx];
      }
    }
  }

  batch_copy(eksv, seksv, eksv_bytes, tidx);
}
*/
















template<typename T>
__global__ void assemble_gpu_kernel(cmem_t<T> cdata)
{
  //const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
  const idx_t adx = static_cast<idx_t>(blockIdx.x);

  if (threadIdx.x == 0)//tidx < cdata.N
    cdata[adx].assemble();

  __syncthreads(); // uselesss
}

template<typename T>
void assemble_pointers(cmem_t<T> &cdata)
{
  switch (cdata.memtype)
  {
    case host_e:
    {
      for (size_t it = 0 ; it < cdata.N ; ++it)
        cdata[it].assemble();

      break;
    }
    case device_e:
    {
      assemble_gpu_kernel<<<cdata.N, 32>>>(cdata); // pretty wasteful ..
      checkCudaErrors(cudaDeviceSynchronize());
      break;
    }
    default: { throw_tw_error("invalid memtype"); };
  }

  checkCudaErrors(cudaDeviceSynchronize());
}












template<typename T>
__global__ void cmem_check_gpu(cmem_t<T> cdata)
{
  const int tidx = threadIdx.x + blockIdx.x*blockDim.x;

  if (tidx == 0)
  {
    printf("%p %lu %d\n", cdata.dptr, cdata.N, cdata.memtype);
    int asdf = 0;

    for (size_t it = 0 ; it < cdata.N ; ++it)
    {
      //printf("%lu -> %p\n", cdata.data_bytes[it], cdata.data[it]);
      //break;
      if (cdata.data[it])
        asdf++;
    }
    
    printf("asdf: %d\n", asdf);
  }
}

template<typename T>
void cmem_check(cmem_t<T> &cdata)
{
  if (cdata.memtype == host_e)
  {
    printf("cmem_check cpu: %p %lu %d\n", cdata.dptr, cdata.N, cdata.memtype);
    int asdf = 0;

    for (size_t it = 0 ; it < cdata.N ; ++it)
    {
      //printf("%lu -> %p\n", cdata.data_bytes[it], cdata.data[it]);
      //break;
      if (cdata.data[it])
        asdf++;
    }

    printf("asdf: %d\n", asdf);
  }
  else
  {
    printf("cmem_check gpu: %p %lu %d\n", cdata.dptr, cdata.N, cdata.memtype);
    cmem_check_gpu<<<1,32>>>(cdata);//(cdata.N)?cdata.N:1
    checkCudaErrors(cudaDeviceSynchronize());
  }
}
//cmem_t<smsh_t> d_smsh, cmem_t<eksv_t> d_slvd, cmem_t<ekfd_t> d_sfdd, cmem_t<ekfd_main_t> d_agg_solver, cmem_t<inter_t> d_inter, cmem_t<ek_event2> d_events
//__global__ void check_smsh(cmem_t<smsh_t> d_smsh, cmem_t<eksv_t> d_slvd, cmem_t<ekfd_t> d_sfdd, cmem_t<ekfd_main_t> d_agg_solver, cmem_t<inter_t> d_inter, cmem_t<ek_event2> d_events)
//{
//  const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
//}
















// ----------------------------------------------------------------------------
// resets states of all aggs and clears active lists (global, local)
// requires parallel lists, agg states and apd/act!
__global__ void reset_states_gpu(cmem_t<eksv_t> d_slvd) //cmem_t<ekfd_t> d_sfdd, cmem_t<ekfd_main_t> d_agfd
{
  //const int tidx = threadIdx.x + blockIdx.x*blockDim.x;

  //const int adx_begin = batch_idx(blockIdx.x, d_slvd.N, gridDim.x);
  //const int adx_end = batch_idx(blockIdx.x + 1, d_slvd.N, gridDim.x);

  //if (threadIdx.x == 0 && blockIdx.x == 1)
  //  printf("astart: %d aend: %d\n", adx_begin, adx_end);

  //if (tidx == 0)
  //  printf("adx_begin: %d adx_end: %d\n", adx_begin, adx_end);

  const idx_t adx = static_cast<idx_t>(blockIdx.x);

  //for (int adx = adx_begin ; adx < adx_end ; ++adx)
  //{
    eksv_t &sv = d_slvd[adx];
    //ekfd_t &fd = d_sfdd[adx];

    if (threadIdx.x == 0)
    {
      sv.assemble();
      //fd.assemble();
    }

    __syncthreads();

    const int vtx_begin = batch_idx(threadIdx.x, sv.nNodes, blockDim.x);
    const int vtx_end = batch_idx(threadIdx.x + 1, sv.nNodes, blockDim.x);

    //if (tidx == 0)
    //{
    //  printf("vtx_begin: %d vtx_end: %d - %d ptr: %p\n", vtx_begin, vtx_end, sv.nNodes, sv.phi_ms);
    //}

    for (int vtx = vtx_begin ; vtx < vtx_end ; ++vtx)
    {
      //printf("vtx: %d\n", vtx);
      sv.phi_ms[vtx] = DBL_INF;
      sv.states[vtx] = ek_far;
      //sv.di90_ms[vtx] = DBL_INF;
      //sv.curv[vtx] = 0.0;

      //act[it].clear(); // reassign!
      //apd[it].clear();
      //deact[it].clear();
    }
    //}

    if (threadIdx.x == 0) // clear active list of agg
    {
      //fd.active_size = 0;
      sv.agg_state = ek_far; //sv.agg_state = EKMS_FAR;
    }

    //__syncthreads(); //__syncwarp();
  //}

  //if (tidx == 0) // first thread clears global active list
  //  d_agfd[0].active_size = 0;

  //__syncthreads();
}

/*
// ----------------------------------------------------------------------------
// 
__global__ void update_delayed_gpu(cmem_t<eksv_t> d_slvd, cmem_t<ekfd_t> d_sfdd, ekfd_t d_agfd)
{
  const int tidx = threadIdx.x + blockIdx.x*blockDim.x;

  const int adx_begin = batch_idx(blockIdx.x, d_slvd.N, blockDim.x/d_slvd.N + 1);
  const int adx_end = batch_idx(blockIdx.x + 1, d_slvd.N, blockDim.x/d_slvd.N + 1);

  for (int adx = adx_begin ; adx < adx_end ; ++adx)
  {
    eksv_t &sv = d_slvd[adx];
    ekfd_t &fd = d_sfdd[adx];

    const int vtx_begin = batch_idx(tidx, sv.nNodes, blockDim.x);
    const int vtx_end = batch_idx(tidx+1, sv.nNodes, blockDim.x);

    for (int vtx = vtx_begin ; vtx < vtx_end ; ++vtx)
    {
      //if (sv.states[vtx] == ek_frozen)
      //{
      //  const dbl_t act_time = act[vtx].back() + apd[vtx].back();
      //
      //  if (act_time > cur_time)
      //    continue;

      //  sv.states[vtx] = ek_refractory;
      //  sv.deact[vtx].push_back(act_time);
      //}
      //else 
      if (sv.states[vtx] == ek_frozen) //ek_refractory
      {
        const dbl_t act_time = act[vtx].back() + apd[vtx].back()/0.91; // apd90

        if (act_time > cur_time) 
          continue;

        sv.phi_ms[vtx] = DBL_INF;
        sv.states[vtx] = ek_far;
        sv.curv[vtx] = 0.;
      }
    }

    if (tidx == 0 && added_far) // clear active list of agg
      sv.agg_state |= EKMS_FAR;
  }
}
*/































// ----------------------------------------------------------------------------
// sets vtx in aggs to narrow and adds (agg, vtx) to global/local active lists
// each block processes one event from the list and sets inactive?
// each block processes one agg from an event?
// events sorted? keep count of active events to prevent kernel calls?
// requires parallel list!
// provide/precompute aggs and vtx per agg? yes!
//cmem_t<ekfd_t> d_sfdd, cmem_t<ekfd_main_t> d_agg_solver, 
__global__ void update_events_gpu(cmem_t<eksv_t> d_slvd, cmem_t<ek_event2> d_events, const dbl_t cur_time)
{
  const idx_t adx = static_cast<idx_t>(blockIdx.x);

  for (int edx = 0 ; edx < d_events.N ; ++edx)
  {
    ek_event2 &event = d_events[edx]; // must be assembled before call!

    if (!(event.tstart >= cur_time && event.active))
      break; // continue?

    for (int ait = 0 ; ait < event.Nagg ; ++ait)
    {
      if (event.adx[ait] == adx)
      {
        eksv_t &sv = d_slvd[adx];
        //ekfd_t &fd = d_sfdd[adx];

        if (threadIdx.x == 0)
        {
          sv.assemble();
          //fd.assemble();
        }
        __syncthreads();

        const int vit_begin = batch_idx(threadIdx.x, event.vtx_cnt[ait], blockDim.x);
        const int vit_end = batch_idx(threadIdx.x + 1, event.vtx_cnt[ait], blockDim.x);

        for (int vit = vit_begin ; vit < vit_end ; ++vit)
        {
          const idx_t vtx = event.vtx_con[event.vtx_dsp[ait] + vit]; // must be local idx

          // only narrow if not active! add logic later ...
          sv.phi_ms[vtx] = cur_time; //event.tstart
          sv.states[vtx] = ek_narrow;
          //sv.di_ms[vtx] = compute_diastolic_interval(vtx);
          //sv.curv[vtx] = 0.0;

          //fd.active_list[fd.active_size + vit] = vtx;
        }

        if (threadIdx.x == 0)
        {
          //fd.active_size += event.vtx_cnt[ait];
          //agg_solver.active_list[agg_solver.active_size + ait] = adx;
          sv.agg_state = ek_narrow; //sv.agg_state |= EKMS_NARROW;
        }

        __syncthreads();//__syncwarp();
      }
    }
  }












  /*
  const unsigned int tidx = threadIdx.x + blockIdx.x*blockDim.x;
  //ekfd_main_t &agg_solver = d_agg_solver[0];

  //if (threadIdx.x == 0)
  //  agg_solver.assemble();

  //__syncthreads();

  //if (tidx == 32)
  //  printf("tg: %d tl: %d bidx: %d bdim: %d gdim: %d\n", tidx, threadIdx.x, blockIdx.x, blockDim.x, gridDim.x);

  for (int edx = 0 ; edx < d_events.N ; ++edx)
  {
    ek_event2 &event = d_events[edx];

    if (threadIdx.x == 0)
      event.assemble();

    __syncthreads();

    if (!(event.tstart >= cur_time && event.active))
      break; // continue?

    const int ait_begin = batch_idx(blockIdx.x, event.Nagg, gridDim.x);
    const int ait_end = batch_idx(blockIdx.x + 1, event.Nagg, gridDim.x);

    if (tidx == 0)
      printf("ait_begin: %d ait_end: %d\n", ait_begin, ait_end);

    for (int ait = ait_begin ; ait < ait_end ; ++ait)
    {
      const idx_t adx = event.adx[ait]; // contains no duplicates
      eksv_t &sv = d_slvd[adx];
      //ekfd_t &fd = d_sfdd[adx];

      if (threadIdx.x == 0)
      {
        sv.assemble();
        //fd.assemble();
      }
      __syncthreads();

      const int vit_begin = batch_idx(threadIdx.x, event.vtx_cnt[ait], blockDim.x);
      const int vit_end = batch_idx(threadIdx.x + 1, event.vtx_cnt[ait], blockDim.x);

      for (int vit = vit_begin ; vit < vit_end ; ++vit)
      {
        const idx_t vtx = event.vtx_con[event.vtx_dsp[ait] + vit]; // must be local idx

        // only narrow if not active! add logic later ...
        sv.phi_ms[vtx] = cur_time; //event.tstart
        sv.states[vtx] = ek_narrow;
        //sv.di_ms[vtx] = compute_diastolic_interval(vtx);
        sv.curv[vtx] = 0.0;

        //fd.active_list[fd.active_size + vit] = vtx;
      }

      if (threadIdx.x == 0)
      {
        //fd.active_size += event.vtx_cnt[ait];
        //agg_solver.active_list[agg_solver.active_size + ait] = adx;
        sv.agg_state = ek_narrow; //sv.agg_state |= EKMS_NARROW;
      }

      __syncthreads();//__syncwarp();
    }

    //if (tidx == 0)
    //  agg_solver.active_size += event.Nagg;

    __syncthreads();
  }*/
}













/*
__device__ void merge_active_lists(ekfd_t &fd, size_t &t_size, idx_t *t_list)
{
  __syncthreads();//__syncwarp();

  int pos_start = 0;//fd.active_size; // 0
  fd.cnt[threadIdx.x] = t_size;

  __syncthreads();//__syncwarp();

  for (int it = 1 ; it <= threadIdx.x ; ++it)
    pos_start += fd.cnt[it-1];

  for (int it = 0 ; it < t_size ; ++it)
    fd.active_list[pos_start + it] = t_list[it];//fd.active_size + 
  
  __syncthreads();//__syncwarp();

  if (threadIdx.x == 0) // reduction .. (maybe duplicate check)
  {
    int num_added = 0;

    for (int it = 0 ; it < blockDim.x ; ++it)
      num_added += fd.cnt[it];
    
    fd.active_size = num_added; // fd.active_size += num_added;
  }

  t_size = 0;
  
  __syncthreads();//__syncwarp();
}

__device__ void merge_active_lists_global(ekfd_main_t &fd, size_t &t_size, idx_t *t_list)
{
  __syncthreads();

  if ((threadIdx.x == 0) && (blockIdx.x == 0))
  {
    for (int it = 0 ; it < gridDim.x ; ++it)
      fd.cnt[it] = 0;
  }

  __syncthreads();

  if (threadIdx.x == 0)
    fd.cnt[blockIdx.x] = t_size;

  __syncthreads();

  if (threadIdx.x == 0)
  {
    int pos_start = fd.active_size;//fd.active_size; // fd.active_size? or 0?

    for (int it = 1 ; it <= blockIdx.x ; ++it)
      pos_start += fd.cnt[it-1];

    for (int it = 0 ; it < t_size ; ++it)
      fd.active_list[pos_start + it] = t_list[it];
  }

  __syncthreads();

  if ((threadIdx.x == 0) && (blockIdx.x == 0)) // reduction .. (maybe duplicate check)
  {
    int num_added = 0;

    for (int it = 0 ; it < gridDim.x ; ++it)
      num_added += fd.cnt[it];
    
    fd.active_size += num_added;

    printf("merge_active_lists_global - num_added: %d\n", num_added);
  }

  t_size = 0;

  __syncthreads();
}

*/





































/*
// ----------------------------------------------------------------------------
// solves eikonal on all active aggs, adds new aggs [block(0,0,0) thread(0,0,0)]
// syncthreads does not sync all blocks!
// collect active list in parallel?
__global__ void fim_step_gpu(cmem_t<smsh_t> d_smsh, cmem_t<eksv_t> d_slvd, cmem_t<ekfd_t> d_sfdd, cmem_t<ekfd_main_t> d_agg_solver, cmem_t<inter_t> d_inter, cmem_t<ek_event2> d_events, const dbl_t dt, const size_t n_steps)
{
  const int tidx = threadIdx.x + blockIdx.x*blockDim.x;
  ekfd_main_t &agg_solver = d_agg_solver[0];

  size_t t_active_size = 0, t_next_size = 0;
  idx_t t_active_list[64], t_next_list[64]; // used for threads and blocks? <- Nnodes/Nthreads (or n2n_max?)

  size_t b_next_size = 0, b_active_size = 0;
  idx_t b_next_list[64], b_active_list[64];

  if (tidx == 0)
    printf("solver start size: %d\n", agg_solver.active_size);
  __syncthreads(); 

  if (tidx == 0) // for some reason, cur_time is not always 0 ...
    agg_solver.cur_time = 0.0;

  //if (tidx == 0)
  //{
  //  const auto &ninter = d_inter[0];
  //  for (size_t nait = 0 ; nait < ninter.Na ; ++nait)
  //  {
  //    const idx_t nadx = ninter.adx_con[nait];
  //    const idx_t dsp1 = ninter.gcd_dsp[nait];
  //    printf("A%ld: ", nadx);
  //    for (cnt_t cit = 0 ; cit < ninter.gcd_cnt[nait] ; ++cit)
  //      printf("%ld ", ninter.gcd_con[dsp1+cit]);
  //    printf(" - Na: %lu Nc: %lu\n", ninter.Na, ninter.Nc);
  //  }
  //}
  
  __syncthreads(); 

  // main loop
  for (size_t t = 0 ; t < n_steps ; t++)
  {
    if (tidx == 0)
    {
      agg_solver.phi_start = agg_solver.cur_time;
      agg_solver.cur_time += dt;
      printf("----- fd.phi_start: %f dt: %f step: %lu -------------------------------------------\n", agg_solver.cur_time, dt, t);
    }

    __syncthreads();

    // step
    while (agg_solver.active_size > 0)
    {
      const int ait_begin = batch_idx(blockIdx.x, agg_solver.active_size, gridDim.x);
      const int ait_end = batch_idx(blockIdx.x + 1, agg_solver.active_size, gridDim.x);

      if (tidx == 0) 
      {
        agg_solver.active_size = 0; // remove? gets reset anyway
        printf("ait_begin: %d ait_end: %d\n", ait_begin, ait_end);
      }

      __syncthreads();

      for (int ait = ait_begin ; ait < ait_end ; ++ait)
      {
        const idx_t adx = agg_solver.active_list[ait];
        smsh_t &msh = d_smsh[adx];
        eksv_t &sv = d_slvd[adx];
        ekfd_t &fd = d_sfdd[adx];
        inter_t &inter = d_inter[adx];

        if (threadIdx.x == 0)
        {
          fd.msh = &msh;
          fd.svd = &sv;
          fd.phi_start = agg_solver.phi_start;
          fd.cur_time = agg_solver.cur_time;
        }

        if (tidx == 0)
          printf("ADX: %ld, fd.phi_start: %f dt: %f\n", adx, fd.phi_start, dt);

        __syncthreads();//__syncwarp();

        if (sv.agg_state != ek_narrow) // if (!(sv.agg_state & EKMS_NARROW))
          continue;

        // copy to shared
        __syncthreads();//__syncwarp();

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
          printf("=== Active: block %d: size: %d -> ", blockIdx.x, fd.active_size);
          for (size_t tit = 0 ; tit < fd.active_size ; ++tit)
            printf("%ld,", fd.active_list[tit]);
          printf("\n");
        }

        while (fd.active_size > 0)
        {
          __syncthreads();//__syncwarp();

          const int it_begin = batch_idx(threadIdx.x, fd.active_size, blockDim.x);
          const int it_end = batch_idx(threadIdx.x + 1, fd.active_size, blockDim.x);

          __syncthreads();//__syncwarp();

          if (threadIdx.x == 0) // remove? gets reset anyway
            fd.active_size = 0;

          __syncthreads();//__syncwarp();

          for (int it = it_begin ; it < it_end ; ++it)
          {
            const idx_t vtx = fd.active_list[it];

            if (sv.states[vtx] != ek_narrow)
              continue;
            
            const dbl_t old_phi = sv.phi_ms[vtx];
            dbl_t new_phi = fd.update_phi(vtx, ek_narrow);
            new_phi = (new_phi < old_phi)?new_phi:old_phi;
            sv.phi_ms[vtx] = new_phi;

            if (new_phi - fd.phi_start > dt)
              t_next_list[t_next_size++] = vtx;
            else if(fabs(old_phi - new_phi) < (1e-6*(1. + (new_phi + old_phi)*.5)))
            {
              sv.states[vtx] = ek_frozen;
              // compute diastolic interval, set act, compute apd

              // add neighbours (conflicts possible)
              const idx_t ndsp = msh.n2n_dsp[vtx];

              for(cnt_t jt = 0 ; jt < msh.n2n_cnt[vtx] ; ++jt)
              {
                const idx_t nvtx = msh.n2n_con[ndsp+jt];

                if (sv.states[nvtx] == ek_far)
                {
                  sv.phi_ms[nvtx] = fd.update_phi(nvtx, ek_narrow);
                  sv.states[nvtx] = ek_narrow;
                  t_active_list[t_active_size++] = nvtx;
                }
              }
            }
            else
              t_active_list[t_active_size++] = vtx;
          }

          __syncthreads();//__syncwarp();

          merge_active_lists(fd, t_active_size, t_active_list);

          __syncthreads();//__syncwarp();

          if (threadIdx.x == 0 && blockIdx.x == 0)
          {
            printf("  === Active: block %d: size: %d -> ", blockIdx.x, fd.active_size);
            for (size_t tit = 0 ; tit < fd.active_size ; ++tit)
              printf("%ld,", fd.active_list[tit]);
            printf("\n");
          }

          __syncthreads();//__syncwarp();
        }

        __syncthreads(); //__syncwarp();?

        // merge next lists
        merge_active_lists(fd, t_next_size, t_next_list);

        // copy shared back?
        __syncthreads(); //__syncwarp();?

        if (threadIdx.x == 0 && blockIdx.x == 0)
        {
          printf("=== Active: block %d: size: %d -> ", blockIdx.x, fd.active_size);
          for (size_t tit = 0 ; tit < fd.active_size ; ++tit)
            printf("%ld,", fd.active_list[tit]);
          printf("\n");
        }

        // ----------------------------------------------------------------------
        // add aggs to main active list + update active lists and states

        __syncthreads(); //__syncwarp();?

        // since one warp can handle multiple aggs -> t_active_list not possible!
        // TODO: better logic -> converged to next, ensure no duplicates
        if (threadIdx.x == 0) // check and add neighbours
        {
          printf("add begin - block %d: active_size: %d\n", blockIdx.x, agg_solver.active_size);

          if (fd.active_size > 0) // add to next
            b_next_list[b_next_size++] = adx;
          else
          {
            sv.agg_state = ek_frozen;//&= !EKMS_NARROW;
          }

            for (size_t nait = 0 ; nait < inter.Na ; ++nait)
            {
              const idx_t nadx = inter.adx_con[nait];
              const idx_t dsp1 = inter.gcd_dsp[nait];

              eksv_t &nsv = d_slvd[nadx];
              ekfd_t &nfd = d_sfdd[nadx];
              inter_t &ninter = d_inter[nadx];
              //printf("block %d: nait: %lu Na: %lu Nc: %lu nadx: %d cnt: %d aptr: %p con: %p dsp: %p cnt: %p\n", blockIdx.x, nait, inter.Na, inter.Nc, static_cast<int>(nadx), static_cast<int>(inter.gcd_cnt[nait]), inter.adx_con, inter.gcd_con, inter.gcd_dsp, inter.gcd_cnt);
              
              // update intersections
              for (size_t nait2 = 0 ; nait2 < ninter.Na ; ++nait2) // linear search -> bad -> replace with fast search
              {
                if (ninter.adx_con[nait2] == adx)
                {
                  // must be same size -> check? inter.gcd_cnt[nait] == ninter.gcd_cnt[nait2]
                  if (inter.gcd_cnt[nait] != ninter.gcd_cnt[nait2])
                    printf("inter size mismatch!");
                  
                  const idx_t dsp2 = ninter.gcd_dsp[nait2];

                  for (cnt_t cit = 0 ; cit < inter.gcd_cnt[nait] ; ++cit)
                  {
                    const idx_t vtx1 = inter.gcd_con[dsp1+cit];
                    const idx_t vtx2 = ninter.gcd_con[dsp2+cit];

                    // change neighbouring vertices -> bad
                    if (sv.phi_ms[vtx1] < nsv.phi_ms[vtx2])
                    {
                      nsv.phi_ms[vtx2] = sv.phi_ms[vtx1];
                      nsv.states[vtx2] = ek_narrow;
                      // t_next_list[t_next_size++] = vtx;
                      //fd.active_list[fd.active_size++] = vtx1;
                      nfd.active_list[nfd.active_size++] = vtx2;

                      if (nsv.agg_state == ek_far)//(!(nsv.agg_state & EKMS_NARROW & EKMS_NARROW)) && 
                      {
                        printf("ADX%ld updated vdx of ADX%ld\n", adx, nadx);
                        nsv.agg_state = ek_narrow; //|= EKMS_NARROW;
                        //t_active_list[t_active_size++] = nadx;
                        //agg_solver.active_list[agg_solver.active_size++] = nadx;
                        b_active_list[b_active_size++] = nadx;
                      }
                    }

                    // change own vertices -> better (BUT DOES THIS WORK???) -> only if we move this to the beginning ...
                    //if (nsv.phi_ms[vtx2] < sv.phi_ms[vtx1])
                    //{
                    //  sv.phi_ms[vtx1] = nsv.phi_ms[vtx2];
                    //  sv.states[vtx1] = ek_narrow;
                    //  //t_next_list[t_next_size++] = vtx1;
                    //  fd.active_list[fd.active_size++] = vtx1; // only one thread so ok
                    //}
                  }

                  break;
                }
              }
              // error check if nothing was found!
            }
          //}

          printf("add end - block %d: active_size: %d\n", blockIdx.x, agg_solver.active_size);
        }

        __syncthreads();//__syncwarp();
      }

      // add neighbours to aggs
      merge_active_lists_global(agg_solver, b_active_size, b_active_list);
      //__syncthreads();

      __syncthreads();//__syncwarp();
    }

    // add next aggs
    merge_active_lists_global(agg_solver, b_next_size, b_next_list);

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
      printf("||| Active: block %d: size: %d -> ", blockIdx.x, agg_solver.active_size);
      for (size_t tit = 0 ; tit < agg_solver.active_size ; ++tit)
        printf("%ld,", agg_solver.active_list[tit]);
      printf("\n");
    }

     __syncthreads();

    //update_delayed();
    //update_blocked();
    //update_activations();
  }
}
*/



















CudaSolverFIM::CudaSolverFIM(const ek_data &ekdata) : ek_data(ekdata), 
                             h_smsh(), d_smsh(), h_eksv(), d_eksv(), 
                             h_events(), d_events(), 
                             h_inter(), d_inter()
{
  log_info("CudaSolverFIM");

  reset_states();

  // compute n2n_max
  n2e_max = 0, n2n_max = 0;
  for (size_t it = 0 ; it < Npts ; it++)
  {
    n2e_max = std::max(n2e_max, static_cast<size_t>(mesh.n2e_cnt[it]));
    n2n_max = std::max(n2n_max, static_cast<size_t>(mesh.n2n_cnt[it]));
  }

  // compute a2a_max
  //const int a2a_max = 3;//!!
  std::cout<<"n2e_max: "<<n2e_max<<", n2n_max: "<<n2n_max<<"\n";
  std::vector<std::vector<idx_t>> n2a;
  std::vector<std::map<idx_t, idx_t>> a2n;

  std::vector<std::map<idx_t,std::vector<idx_t>>> a2i;
  //std::vector<std::vector<std::vector<ipair_t>>> n2i;
  std::vector<std::vector<ipair_t>> n2i;

  // atags or e2a (replace with smsh_t)
  std::vector<mesh_t> submeshes;
  const size_t nAgg = extract_submeshes2(mesh, mesh.atags, submeshes, l2g, g2l, n2a, a2n, a2a, a2i, n2i); 
  aggs_sizes.assign(nAgg, 0);
  agg_states.assign(nAgg, ek_far);

  /*
  for(size_t adx = 0 ; adx < nAgg ; ++adx)
  {
    std::cout<<"adx"<<adx<<":\n";
    for (size_t vtx = 0 ; vtx < n2i[adx].size() ; ++vtx)
    {
      std::cout<<"  vtx"<<vtx<<": ";
      for (const auto &jtem : n2i[adx][vtx])
      {
        std::cout<<"("<<jtem.adx<<","<<jtem.vtx<<") ";
        //if (adx == 0 && jtem.adx == 1)
        //{
          //std::cout<<vtx<<",";
        //  std::cout<<jtem.vtx<<",";
        //}
      }
        
      std::cout<<"\n";

      //if (n2i[adx][vtx].size() > 0)
      //  std::cout<<vtx<<",";
    }
    std::cout<<"\n";
  }*/

  std::cout<<"num_submeshes: "<<nAgg<<"\n";
  wavefront_width = compute_dwf(mesh);

  /*int Na2a_con = 0;
  for (const auto &item : a2a)
    Na2a_con += item.size();

  std::cout<<"Na2a_con: "<<Na2a_con<<"\n";*/

  std::vector<size_t> inter3_bytes;

  for (const auto &item : n2i)
  {
    if (!item.empty())
      inter3_bytes.push_back(inter3_t::static_bytes(item.size()));
  }

  std::cout<<"num inter: "<<inter3_bytes.size()<<"\n";

  // mesh states and solver data ----------------------------------------------
  std::vector<size_t> mesh_bytes(nAgg), eksv_bytes(nAgg);//, slvd_bytes(nAgg); //inter2_bytes(nAgg); // inter_bytes(nAgg), agfd_bytes = {static_cast<size_t>(ekfd_main_t::static_bytes(nAgg, num_blocks))};
  max_shared_bytes = 0;
  size_t max_bytes_adx = 0, total_bytes = 0;

  for(size_t adx = 0 ; adx < nAgg ; ++adx)
  {
    const auto &submesh = submeshes[adx];

    mesh_bytes[adx] = smsh_t::static_bytes(submesh.n2e_cnt.size(), submesh.e2n_cnt.size(),
                                           submesh.e2n_con.size(), submesh.n2n_con.size());

    eksv_bytes[adx] = eksv_t::static_bytes(submesh.n2e_cnt.size(), submesh.n2e_con.size());
    //slvd_bytes[adx] = slvd_t::static_bytes(submesh.n2e_cnt.size(), submesh.n2e_con.size()); // constant!? -> threads_per_block
    
    //size_t a2i_con_size = 0;

    //for (const auto &item : a2i[adx])
    //  a2i_con_size += item.second.size();

    //size_t n2i_con_size = 0;

    //for (const auto &item : n2i[adx])
    //  n2i_con_size += item.size();

    //std::cout<<"a2i[adx].size(): "<<a2i[adx].size()<<" a2i_con_size: "<<a2i_con_size<<"\n";
    //inter_bytes[adx] = inter_t::static_bytes(a2i[adx].size(), a2i_con_size);
    //inter2_bytes[adx] = inter2_t::static_bytes(submesh.n2e_cnt.size(), n2i_con_size);

    //std::cout<<"  A"<<adx<<": "<<mesh_bytes[adx]<<" + "<<eksv_bytes[adx]<<" + "<<ekfd_bytes[adx]<<" + "<<inter_bytes[adx]<<" (Nn: "<<submesh.n2e_cnt.size()<<", Ne: "<<submesh.e2n_cnt.size()<<")\n";
    const size_t tmp_max_bytes = mesh_bytes[adx] + eksv_bytes[adx];// + slvd_bytes[adx];

    if (tmp_max_bytes > max_shared_bytes)
    {
      max_shared_bytes = tmp_max_bytes;
      max_bytes_adx = adx;
      std::cout<<"  A"<<adx<<": "<<tmp_max_bytes<<" = "<<mesh_bytes[adx]<<" + "<<eksv_bytes[adx]<<" (Nn: "<<submesh.n2e_cnt.size()<<", Ne: "<<submesh.e2n_cnt.size()<<", Nenc: "<<submesh.e2n_con.size()<<", Nnnc: "<<submesh.n2n_con.size()<<")\n";
    }

    //for (const auto &item : a2i[adx])
    //{
    //  std::cout<<"    A"<<item.first<<": ";
    //  for (const auto &jtem : item.second)
    //    std::cout<<jtem<<","; // jtem l2g[adx][jtem]
    //    
    //  std::cout<<"\n";
    //}
    total_bytes += tmp_max_bytes;

    //for (int it = 0 ; it < static_cast<int>(submesh.e2n_cnt.size()) ; ++it)
    //  std::cout<<"edx"<<it<<": "<<submesh.geo[6*it+0]<<" "<<submesh.geo[6*it+1]<<" "<<submesh.geo[6*it+2]<<" "<<submesh.geo[6*it+3]<<" "<<submesh.geo[6*it+4]<<" "<<submesh.geo[6*it+5]<<" "<<"\n";
  }

  std::cout<<"mesh_bytes: "<<mesh_bytes.size()<<"\n";
  std::cout<<"max_shared_bytes: "<<max_shared_bytes<<"(A"<<max_bytes_adx<<")"<<"\n";
  std::cout<<"total_bytes: "<<total_bytes/(1024.0*1024.0)<<"MB\n";

  //std::cout<<"  first\n";
  //cmem_check(h_inter);
  //cmem_check(d_inter);

  //std::cout<<"smsh: -------------------------\n";
  //cmem_check(h_smsh);
  //cmem_check(d_smsh);
  //std::cout<<"eksv: -------------------------\n";
  //cmem_check(h_eksv);
  //cmem_check(d_eksv);
  //std::cout<<"ekfd: -------------------------\n";
  //cmem_check(h_ekfd);
  //cmem_check(d_ekfd);
  //std::cout<<"-------------------------------\n";

  //std::cout<<"  second\n";
  h_smsh.init(host_e, mesh_bytes);   d_smsh.init(device_e, mesh_bytes);
  h_eksv.init(host_e, eksv_bytes);   d_eksv.init(device_e, eksv_bytes);
  //h_slvd.init(host_e, slvd_bytes);   d_slvd.init(device_e, slvd_bytes);
  //h_inter.init(host_e, inter2_bytes); d_inter.init(device_e, inter2_bytes);
  h_inter.init(host_e, inter3_bytes); d_inter.init(device_e, inter3_bytes);
  //h_agfd.init(host_e, agfd_bytes);   d_agfd.init(device_e, agfd_bytes);
  //cmem_check(h_inter);
  //cmem_check(d_inter);
  
  for(size_t adx = 0 ; adx < nAgg ; ++adx)
  {
    const mesh_t &submesh = submeshes[adx];
    h_smsh[adx].construct_from(submesh);
    h_eksv[adx].construct_from(submesh);
    //h_slvd[adx].construct_from(submesh, wavefront_width);
    //h_inter[adx].construct_from(n2i[adx]);
  }

  size_t n2i_it = 0;

  for (const auto &item : n2i)
  {
    if (!item.empty())
    {
      h_inter[n2i_it].construct_from(item);
      n2i_it++;
    }
  }

  //h_agfd[0].construct_from(nAgg, num_blocks, wavefront_width);

  h_smsh.copy_to(d_smsh);
  h_eksv.copy_to(d_eksv);
  //h_slvd.copy_to(d_slvd);
  h_inter.copy_to(d_inter);
  //h_agfd.copy_to(d_agfd);

  //cmem_check(h_inter);
  //cmem_check(d_inter);

  assemble_pointers(d_smsh); // would be nice if in copy_to() [but has to be done every time data moves]
  assemble_pointers(d_eksv);
  //assemble_pointers(d_slvd);
  assemble_pointers(d_inter);
  //assemble_pointers(d_agfd);

  // inter debug
  //cmem_t<inter_t> h_debug;
  //h_debug.init(host_e, inter_bytes);
  //h_inter.copy_to(h_debug);
  //assemble_pointers(h_debug);

  submeshes.clear();

  // apply activations --------------------------------------------
  std::cout<<"num events: "<<events.size()<<"\n";

  std::vector<size_t> event_bytes(events.size());
  std::vector<std::map<idx_t, std::vector<idx_t>>> events_per_agg(events.size());

  //std::cout<<"ebytes: ";
  for (size_t it = 0 ; it < events.size() ; ++it)
  {
    int e_Nagg = 0, e_Nall = 0;

    for (const auto &vtx : events[it].vtx)
    {
      for (const auto &adx : n2a[vtx])
      {
        const auto &item = events_per_agg[it].find(adx);

        if (item == events_per_agg[it].end())
          events_per_agg[it][adx].push_back(a2n[adx].at(vtx));
        else
          item->second.push_back(a2n[adx].at(vtx));

        e_Nall++;
      }
    }

    e_Nagg = events_per_agg[it].size();
    event_bytes[it] = ek_event2::sbytes(e_Nagg, e_Nall);
    std::cout<<event_bytes[it]<<" ";
  }
  
  /*log_info("events_per_agg:");
  for (const auto &item : events_per_agg)
  {
    std::cout<<"event: \n";
    for (const auto &it : item)
    {
      std::cout<<"  A"<<it.first<<": ";
      for (const auto &jt : it.second)
        std::cout<<jt<<" ";
      std::cout<<"\n";
    }
      
    std::cout<<"\n";
  }*/

  std::cout<<"  firstA\n";
  //cmem_check(h_events);
  //cmem_check(d_events);

  std::cout<<"  secondA\n";
  h_events.init(host_e, event_bytes);
  d_events.init(device_e, event_bytes);
  //cmem_check(h_events);
  //cmem_check(d_events);

  for (size_t it = 0 ; it < events.size() ; ++it)
    h_events[it].construct_from(events[it], events_per_agg[it]);

  h_events.copy_to(d_events);

  //std::cout<<"  thirdA\n";
  //cmem_check(h_events);
  //cmem_check(d_events);

  assemble_pointers(d_events);

  //std::cout<<"  fourthA\n";
  //cmem_check(h_events);
  //cmem_check(d_events);

  //log_info("CudaSolverFIM end");

  /*for (idx_t adx = 0 ; adx < static_cast<idx_t>(nAgg) ; ++adx)
  {
    for (size_t it = 0 ; it < h_inter[adx].Na ; ++it)
      std::cout<<h_inter[adx].adx_con[it]<<" ";
    std::cout<<"\n";
  }*/
}

CudaSolverFIM::~CudaSolverFIM()
{
  //log_info("CudaSolverFIM destr");
  h_smsh.destroy();
  d_smsh.destroy();
}
// ----------------------------------------------------------------------------
// check if current time allows for an event to happen (stim, block) (activating, deactivating)
void CudaSolverFIM::update_events()
{
  /*if (cur_time >= eitem->start)
  {

  }*/
}
























// use DMA to poll or exchange data?
void CudaSolverFIM::step(const dbl_t dt)
{
  log_info("CudaFIM::step");

  /*
  const int threads_per_block = 32;
  const int max_shared_bytes = ...;

  while(active_size > 0)
  {
    const size_t AL_size = active_size;
    solve_on_submeshes<<<AL_size, threads_per_block, max_shared_bytes>>>(d_smsh, d_eksv, cur_time);
  
    // add neighbouring submeshes
  }
  */
  //a2a = {{1, 2, 3}, {0, 2, 3}, {0, 1, 3}, {0, 1, 2}};

  //const size_t Nagg = a2a.size();
  //active_list.resize(Nagg);
  //std::vector<idx_t> tmp_list(Nagg);

  // add aidx (using n2a)
  //active_size = 1;
  //active_list[0] = 0;
  //agg_states[n2a[idx]] |= ek_narrow;

  //states[0] = ek_narrow;
  //phi[0] = 0.;

  // other
  //const int threads_per_block = 32;
  //const int max_shared_bytes = ...;

  //while(active_size > 0)
  //{
  //  const size_t AL_size = active_size;

  //  solve_on_submeshes<<<AL_size, threads_per_block, max_shared_bytes>>>(meshdata, eksvdata, cur_time);

    // construct UL list and push async to gpu?
    // a2a[aidx] -> union
  //  checkCudaErrors(cudaMemcpyAsync(d_neighbour_indices, h_neighbour_indices, 
  //    num_neighbour_indices*sizeof(idx_t), cudaMemcpyHostToDevice, streams[0]));

  //  checkCudaErrors(cudaDeviceSynchronize());

  //  update_neighbouring_submeshes<<<num_neighbour_indices, threads_per_block, max_shared_bytes>>>(meshdata, eksvdata, d_neighbour_indices, num_neighbour_indices);

  //  checkCudaErrors(cudaDeviceSynchronize());

    // memcpy to host and collect agg_states

    //agg_states[aidx] |= ek_frozen;

    // collect? keep unconverged
    //active_size = 0;
    //for (idx_t idx = 0 ; idx < static_cast<idx_t>(Npts) ; idx++) // 2nd array + swap faster (but not possible with mp .. better to reserve space and iterate over subsections with added idx?)
    //{
    //  if (states[idx] == ek_narrow && (phi[idx] - phi_start) <= dt)
    //    act_list[act_list_size++] = idx;
    //}

    // add neighbours (watch out! neighbours get added only if one idx is activated)
    //for(size_t it = 0 ; it < AL_size ; ++it)
    //{
    //  const idx_t aidx = active_list[it];
      
    //  for(const idx_t naidx : a2a[aidx])
    //  {
    //    if(agg_states[naidx] == ek_far)
    //    {
    //      agg_states[naidx] = ek_narrow;
    //      tmp_list[active_size++] = naidx;
    //    }
    //  }
    //}

    //active_list.assign(tmp_list.begin(), tmp_list.end());
  //}
}


















//__inline__ __device__ int warpReduceSum(int val)
//{
//  for (int offset = warpSize/2; offset > 0; offset /= 2) 
//    val += __shfl_down(val, offset);
//    
//  return val;
//}
// ----------------------------------------------------------------------------
// at least 2 errors present:
// - corner never/wrong value (logic fail) (DONE)
// - sporadic 0 value at inter (race condition) (DONE)
// - phi values dont match!
/*
__global__ void sync_step(cmem_t<smsh_t> d_smsh, cmem_t<eksv_t> d_esvd, 
                          cmem_t<slvd_t> d_slvd, cmem_t<inter2_t> d_inter)
{
    extern __shared__ char sptr[];
  const idx_t adx = static_cast<idx_t>(blockIdx.x);

  const size_t states_bytes = d_esvd.data_bytes[adx];
  char *states_ptr = reinterpret_cast<char*>(&d_esvd[adx]);
  eksv_t *states = reinterpret_cast<eksv_t*>(sptr);
  inter2_t *inter = &d_inter[adx];

  // batch copy mesh, states and solver into shared memory --------------------
  for (size_t bit = batch_idx(threadIdx.x, states_bytes, blockDim.x) ; 
       bit < batch_idx(threadIdx.x + 1, states_bytes, blockDim.x) ; ++bit)
    sptr[bit] = states_ptr[bit];

  __syncthreads();

  if (threadIdx.x == 0)
  {
    states->assemble();
    inter->assemble();
    printf("inter%ld: Nn: %lu Nc: %lu\n", adx, inter->Nn, inter->Nc);
  }

  __syncthreads(); // global sync?

  // vtx structure correct? yes
  // sync problem? atomics?
  // assemble or construct from correct? yes
  // nsv? <- pointer difference? nope
  // visualization?
  for (idx_t vtx = batch_idx(threadIdx.x, inter->Nn, blockDim.x) ; 
       vtx < batch_idx(threadIdx.x + 1, inter->Nn, blockDim.x) ; ++vtx)
  {
    //if (states->states[vtx] < 0)
    //  printf("sync: invalid state!\n");
    if (inter->int_cnt[vtx] > 0)
    {
      const idx_t dsp = inter->int_dsp[vtx];

      for (cnt_t cnt = 0 ; cnt < inter->int_cnt[vtx] ; ++cnt)
      {
        //const ipair_t item = inter->int_con[dsp+cnt];
        const idx_t nvtx = inter->vtx_con[dsp+cnt];
        const idx_t nadx = inter->adx_con[dsp+cnt];
        //if (adx == 0 && item.adx == 1)
        //  printf("%ld %ld,", vtx, item.vtx);
        const eksv_t &nsv = d_esvd[nadx];
        //const eksv_t &nsv = d_esvd[item.adx];
        //if (adx == 0)
        //  printf("A %p\n", &d_esvd[adx]);
        //else if(item.adx == 0)
        //  printf("B %p %p\n", &d_esvd[item.adx], &nsv);
        //printf("  item%ld: nadx%ld nvtx%ld\n", vtx, item.adx, item.vtx);

        //if (nsv.phi_ms[item.vtx] < states->phi_ms[vtx])
        if (nsv.phi_ms[nvtx] < states->phi_ms[vtx])
        {
          //printf("adx%ld: %ld -> narrow adx%ld %ld (%lf %d %lf | %lf)\n", adx, vtx, item.adx, item.vtx, nsv.phi_ms[item.vtx], nsv.states[item.vtx], nsv.curv[item.vtx], states->phi_ms[vtx]);
          //printf("adx%ld: %ld -> narrow adx%ld %ld (%lf %d %lf | %lf)\n", adx, vtx, nadx, nvtx, nsv.phi_ms[nvtx], nsv.states[nvtx], nsv.curv[nvtx], states->phi_ms[vtx]);
          //if (adx == 0)
          //  printf("%ld,", vtx);
          //states->phi_ms[vtx] = nsv.phi_ms[item.vtx];
          states->phi_ms[vtx] = nsv.phi_ms[nvtx];
          states->states[vtx] = ek_narrow;
        }
      }
    }
  }

  __syncthreads();

  // batch copy states back to global memory ----------------------------------
  for (size_t bit = batch_idx(threadIdx.x, states_bytes, blockDim.x) ; 
       bit < batch_idx(threadIdx.x + 1, states_bytes, blockDim.x) ; ++bit)
    states_ptr[bit] = sptr[bit];

  __syncthreads();

  if (threadIdx.x == 0)
    d_esvd[adx].assemble();
}
*/












// ----------------------------------------------------------------------------
__global__ void sync_step2(cmem_t<smsh_t> d_smsh, cmem_t<eksv_t> d_esvd, cmem_t<inter3_t> d_inter)
{
  //if (threadIdx.x == 0 && blockIdx.x == 0)
  //  printf("sync_step2 kernel started\n");

  const size_t t_total = blockDim.x*gridDim.x;
  const idx_t tdx = static_cast<idx_t>(threadIdx.x + blockIdx.x*blockDim.x);

  for (size_t bit = batch_idx(tdx, d_inter.N, t_total) ; 
       bit < batch_idx(tdx + 1, d_inter.N, t_total) ; ++bit)
  {
    const auto &item = d_inter[tdx];
    dbl_t new_phi = DBL_INF;

    for (size_t it = 0 ; it < item.Nint ; ++it)
    {
      const idx_t vtx = item.vtx[it], adx = item.adx[it];
      const dbl_t cur_phi = d_esvd[adx].phi_ms[vtx];

      if (cur_phi < new_phi)
        new_phi = cur_phi;
    }

    for (size_t it = 0 ; it < item.Nint ; ++it)
    {
      const idx_t vtx = item.vtx[it], adx = item.adx[it];
      const dbl_t cur_phi = d_esvd[adx].phi_ms[vtx];

      if (new_phi < cur_phi)
      {
        d_esvd[adx].phi_ms[vtx] = new_phi;
        d_esvd[adx].states[vtx] = ek_narrow;
      }
    }
  }
}
// ----------------------------------------------------------------------------
// state lookup function
__device__ int update_states(smsh_t *mesh, eksv_t *states, const dbl_t cur_time_ms)
{
  int active_size = 0;

  for (idx_t vtx = batch_idx(threadIdx.x, mesh->Nn, blockDim.x) ; 
       vtx < batch_idx(threadIdx.x + 1, mesh->Nn, blockDim.x) ; ++vtx)
  {
    const int state = states->states[vtx];
    //update_function[state];

    if (state == ek_far)
    {

    }
    else if (state == ek_narrow)
      active_size++;
    else if (state == ek_next)// && (states->phi_ms[vtx] - phi_start_ms <= dt_ms)
    {
      active_size++;
      states->states[vtx] = ek_narrow; // only if timestep is ok?
    }
    else if (state == ek_frozen)
    {
      //const dbl_t act_time_ms = act_ms[vtx].back() + apd90_ms[vtx].back(); // apd90
      const dbl_t act_time_ms = states->phi_ms[vtx] + 3e5;

      if (!(act_time_ms > cur_time_ms))
      {
        states->states[vtx] = ek_tail;
        //deact_ms[vtx].push_back(act_time_ms);
      }
    }
    else if (state == ek_tail)
    {
      //const dbl_t act_time_ms = act_ms[vtx].back() + apd90_ms[vtx].back()/0.91; // apd100
      const dbl_t act_time_ms = states->phi_ms[vtx] + 3e5/0.91; // apd100

      if (!(act_time_ms > cur_time_ms))
      {
        states->phi_ms[vtx] = DBL_INF;
        states->states[vtx] = ek_far;
        //states->curv[vtx] = 0.;
      }
    }
    else
    {
      // throw error?
      printf("invalid state!\n");
    }

    // missing: update based on neighbours!
    // missing: update based on events!
  }

  return active_size;
}













/*
__inline__ __device__
int warpReduceSum(int val)
{
  for (int offset = warpSize/2; offset > 0; offset /= 2)
    val += __shfl_down_sync(FULL_MASK, val, offset);
  return val;
}
__inline__ __device__
int blockReduceSum(int val)
{
  static __shared__ int shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpReduceSum(val);     // Each warp performs partial reduction

  if (lane==0) 
    shared[wid]=val; // Write reduced value to shared memory

  __syncthreads();              // Wait for all partial reductions

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

  if (wid==0) 
    val = warpReduceSum(val); //Final reduce within first warp

  return val;
}*/

/*
template <class T>
__global__ void
myreduce(T *g_idata, T *g_odata, unsigned int n)
{
  static __shared__ T shared[512];

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  __syncthreads();

  // do reduction in shared mem
  for (unsigned int s=blockDim.x/2; s>0; s>>=1)
  {
      if (tid < s)
      {
          sdata[tid] += sdata[tid + s];
      }

      __syncthreads();
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
*/

//int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);



__device__
int reduce_active_size(int active_size, int *tmp)
{
  unsigned int tid = threadIdx.x;
  tmp[tid] = active_size;
  __syncthreads();

  if ((blockDim.x >= 512) && (tid < 256))
    tmp[tid] = active_size = active_size + tmp[tid + 256];

  __syncthreads();

  if ((blockDim.x >= 256) &&(tid < 128))
    tmp[tid] = active_size = active_size + tmp[tid + 128];

    __syncthreads();

  if ((blockDim.x >= 128) && (tid <  64))
    tmp[tid] = active_size = active_size + tmp[tid +  64];

  __syncthreads();

  if (tid < 32)
  {
    if (blockDim.x >=  64) // Fetch final intermediate sum from 2nd warp
      active_size += tmp[tid + 32];
    
    for (int offset = warpSize/2 ; offset > 0 ; offset /= 2) // Reduce final warp using shuffle
      active_size += __shfl_xor_sync(FULL_MASK, active_size, offset);

    tmp[0] = active_size;
  }

  __syncthreads();

  return tmp[0];
}



















// ----------------------------------------------------------------------------
// CONSIDERATIONS:
// X each kernel gets called for each agglomerate -> block = agg
// - inter -> global (const too small, shared limited, access limited)
// - regions -> constant (frequent access, small)
// - events -> constant/global (might be large -> entire mesh, access limited) [rect, sphere, agg or limited select vtx?]
// X solver -> shared (high access)
// - one step generates only one act/deact/apd per node max
// X parallel workflow! (temp phi, states ...)
// - next logic might hamper accuracy (what if a new update leads to a narrow node?)
// - int16 sufficient index size for submeshes? -> massive saves!
__global__ void compute_step(cmem_t<smsh_t> d_smsh, cmem_t<eksv_t> d_esvd, cmem_t<inter3_t> d_inter, // cmem_t<ek_event2> d_events, 
                             const dbl_t front_width_ms, const dbl_t cur_time_ms, const dbl_t phi_start_ms, const dbl_t dt_ms)
{
  //if (threadIdx.x == 0 && blockIdx.x == 0)
  //{
  //  printf("kernel started\n");
  //}
    
  extern __shared__ char sptr[];
  static __shared__ int tmp[512];
  //const int wdx = threadIdx.x/warpSize; // warp index
  //const int ldx = threadIdx.x % warpSize; // lane index
  //num_warps = blockDim.x/warpSize;

  //const idx_t tdx = static_cast<idx_t>(threadIdx.x + blockIdx.x*blockDim.x);
  //if (tdx == 0)
  //  printf("A%d,", blockIdx.x);
  const idx_t adx = static_cast<idx_t>(blockIdx.x);

  //smsh_t *mesh = &d_smsh[adx];
  //eksv_t *states = &d_esvd[adx];
  //slvd_t *solver = &d_slvd[adx];

  int active_size = update_states(&d_smsh[adx], &d_esvd[adx], cur_time_ms); // move to separate kernel?
  active_size = reduce_active_size(active_size, tmp);

  if (active_size == 0)
    return;
  
  const size_t mesh_bytes = d_smsh.data_bytes[adx];
  const size_t states_bytes = d_esvd.data_bytes[adx];
  //const size_t solver_bytes = d_slvd.data_bytes[adx];
  const size_t total_bytes = mesh_bytes + states_bytes;// + solver_bytes;

  char *mesh_ptr = reinterpret_cast<char*>(&d_smsh[adx]);
  char *states_ptr = reinterpret_cast<char*>(&d_esvd[adx]);
  //char *solver_ptr = reinterpret_cast<char*>(&d_slvd[adx]);

  smsh_t *mesh = reinterpret_cast<smsh_t*>(sptr);
  eksv_t *states = reinterpret_cast<eksv_t*>(sptr + mesh_bytes);
  //slvd_t *solver = reinterpret_cast<slvd_t*>(sptr + mesh_bytes + states_bytes);

  // batch copy mesh, states and solver into shared memory --------------------
  size_t bit_start = batch_idx(threadIdx.x, total_bytes, blockDim.x);
  size_t bit_end = batch_idx(threadIdx.x + 1, total_bytes, blockDim.x);

  //if (adx == 0 && threadIdx.x > 508)
  //  printf("tdx%d: start: %lu end: %lu total: %lu\n", threadIdx.x, bit_start, bit_end, total_bytes);
  
  // own function (get, set), separate data vs all data
  for (size_t bit = bit_start ; bit < bit_end ; ++bit)
  {
    if (bit < mesh_bytes)
      sptr[bit] = mesh_ptr[bit];
    //else if (bit < mesh_bytes + states_bytes)
    else
      sptr[bit] = states_ptr[bit - mesh_bytes];
    //else
    //  sptr[bit] = solver_ptr[bit - mesh_bytes - states_bytes];
    // copy inter bytes? probably not -> no frequent access
  }

  __syncthreads();

  if (threadIdx.x == 0)
  {
    mesh->assemble();
    states->assemble();
    //solver->assemble();
    //inter->assemble();
  }

  __syncthreads(); // global sync?
  
  // update global and active list --------------------------------------------
  //int active_size = update_states(mesh, states, cur_time_ms);
 // __syncthreads(); // global sync? ||||||||||||||||||||||||||||||||||||||||||||

  //if (threadIdx.x == 0 && adx == 0)
  //  printf("yey\n");
  //  printf("arch: %d\n", __CUDA_ARCH__);

  // reduce all of active list (works only because 32 threads per warp!)
  //for (int offset = warpSize/2; offset > 0; offset >>= 1)
  //  active_size += __shfl_xor_sync(FULL_MASK, active_size, offset);

  // reduce active size over block
  //int active_size = update_states(mesh, states, cur_time_ms);
  //active_size = reduce_active_size(active_size, tmp);

  //__syncthreads();

  //active_size = blockReduceSum(active_size);
  //active_size =__reduce_add_sync(0xFFFFFFFF, active_size);

  //__syncthreads();

  //if (threadIdx.x == 500 && adx == 0)//1821
  //  printf("active_size: %d\n", active_size);

  //if (threadIdx.x == 0 && active_size > 0)
  //  printf("(A%d: %d),", blockIdx.x, active_size);

  // main loop ================================================================
  while (active_size > 0)
  {
    active_size = 0;

    // solve on elements ------------------------------------------------------
    // many problems: idle threads, heavy load, multiple M computation, if else ...
    for (idx_t bit = batch_idx(threadIdx.x, mesh->Nenc, blockDim.x) ; 
         bit < batch_idx(threadIdx.x + 1, mesh->Nenc, blockDim.x) ; ++bit)
    {
      states->tmp_phi_ms[bit] = DBL_INF;
      idx_t vtx = 0;

      while (bit >= mesh->n2e_dsp[vtx] + mesh->n2e_cnt[vtx]) // -> use warp level reduction? binary search?
        vtx++;

      // make phi -> Nenc and save to first entry? -> what about old value?
      if (states->states[vtx] == ek_narrow)
      {
        states->tmp_phi_ms[bit] = update_per_element(mesh, states, 
                                  vtx, mesh->n2e_con[bit], 
                                  ek_narrow, phi_start_ms, front_width_ms);        
        //solver->tmp_phi_ms[bit] = solver->update_per_element(vtx, mesh->n2e_con[bit], ek_narrow);
        //printf("A%d: %ld %ld\n", blockIdx.x, bit, vtx);
      }
    }

    __syncthreads();

    // reduce phi per vertex (linear) [and assemble neighbours?] ----------------
    for (idx_t vtx = batch_idx(threadIdx.x, mesh->Nn, blockDim.x) ; 
        vtx < batch_idx(threadIdx.x + 1, mesh->Nn, blockDim.x) ; ++vtx)
    {
      states->tmp_state[vtx] = ek_far;

      if (states->states[vtx] != ek_narrow)
        continue;

      const idx_t edsp = mesh->n2e_dsp[vtx];
      const dbl_t old_phi_ms = states->phi_ms[vtx];
      dbl_t new_phi_ms = old_phi_ms;

      for (cnt_t ecnt = 0 ; ecnt < mesh->n2e_cnt[vtx] ; ++ecnt)
      {
        const dbl_t cur_phi_ms = states->tmp_phi_ms[edsp+ecnt];

        if (cur_phi_ms < new_phi_ms)
          new_phi_ms = cur_phi_ms;
      }

      states->phi_ms[vtx] = new_phi_ms;

      if (new_phi_ms - phi_start_ms > dt_ms)
      {
        //solver->tmp_state[vtx] = ek_next;
        //printf("A%d: %ld next\n", blockIdx.x, vtx);
        states->states[vtx] = ek_next;
      }
      else if (fabs(old_phi_ms - new_phi_ms) < (1e-6*(1 + (old_phi_ms + new_phi_ms)*.5)))
      {
        states->tmp_state[vtx] = ek_frozen;
        //printf("A%d: %ld frozen\n", blockIdx.x, vtx);
      }
      else
      {
      //  solver->tmp_state[vtx] = ek_narrow;
      //  //printf("A%d: %ld narrow\n", blockIdx.x, vtx);
      }
    }

    __syncthreads();

    //if (threadIdx.x == 0 && adx == 1)
    //  printf("=== mark neighbors ===\n");
    //__syncthreads();
    
    // mark neighbours ----------------------------------------------------------
    for (idx_t vtx = batch_idx(threadIdx.x, mesh->Nn, blockDim.x) ; 
        vtx < batch_idx(threadIdx.x + 1, mesh->Nn, blockDim.x) ; ++vtx)
    {
      //if (adx == 1 && vtx == 67)
      //  printf("here\n");

      if (states->states[vtx] == ek_narrow)
        continue;
      
      //if (adx == 1 && vtx == 67)
      //  printf("yea\n");

      const idx_t ndsp = mesh->n2n_dsp[vtx];

      for (cnt_t ncnt = 0 ; ncnt < mesh->n2n_cnt[vtx] ; ++ncnt)
      {
        const idx_t nvtx = mesh->n2n_con[ndsp+ncnt];

        //if (adx == 1 && vtx == 67)
        //{
        //  printf("nvtx%ld: state: %d tmp_state: %d phi: %lf\n", nvtx, states->states[nvtx], solver->tmp_state[nvtx], states->phi_ms[nvtx]);
        //  printf("vtx%ld: state: %d tmp_state: %d phi: %lf\n", vtx, states->states[vtx], solver->tmp_state[vtx], states->phi_ms[vtx]);
        //}

        if (states->tmp_state[nvtx] == ek_frozen && states->phi_ms[nvtx] < states->phi_ms[vtx]) // downwind
        {
          //if (adx == 1 && vtx == 67)
          //  printf("  wuhu %ld\n", nvtx);
          states->tmp_state[vtx] = states->states[vtx]; //states->states[nvtx]
          states->states[vtx] = ek_neig;
          break;
        }
      }
    }

    __syncthreads();
    //if (threadIdx.x == 0 && adx == 1)
    //{
    //  printf("=== solve elements for neighbours === (67 state: %d)\n", states->states[67]);
    //}
    //__syncthreads();

    // solve elements for neighbours ------------------------------------------
    for (idx_t bit = batch_idx(threadIdx.x, mesh->Nenc, blockDim.x) ; 
         bit < batch_idx(threadIdx.x + 1, mesh->Nenc, blockDim.x) ; ++bit)
    {
      states->tmp_phi_ms[bit] = DBL_INF;
      idx_t vtx = 0;

      //while (bit >= mesh->n2e_dsp[vtx+1]) // -> use warp level reduction? binary search?
      //{
      //  vtx++;
      //  if (vtx > mesh->Nn)
      //    break;
      //}
      while (bit >= mesh->n2e_dsp[vtx] + mesh->n2e_cnt[vtx]) // -> use warp level reduction? binary search?
        vtx++;
        
      //if (adx == 1 && vtx == 67)
      //  printf("hello\n");
      //if (adx == 1)
      //  printf("%ld,", vtx);

      // make phi -> Nenc and save to first entry? -> what about old value?
      if (states->states[vtx] == ek_neig)
      {
        //if (adx == 1 && vtx == 67)
        //  printf("aye\n");
        states->tmp_phi_ms[bit] = update_per_element(mesh, states, 
                                  vtx, mesh->n2e_con[bit], 
                                  ek_narrow, phi_start_ms, front_width_ms);
        //solver->tmp_phi_ms[bit] = solver->update_per_element(vtx, mesh->n2e_con[bit], ek_narrow);
        //if (vtx == 48)// && solver->tmp_phi_ms[bit] < 1e20
        //  printf("A%d: %ld %ld %lf %ld\n", blockIdx.x, bit, vtx, solver->tmp_phi_ms[bit], mesh->n2e_dsp[vtx]);
      }
    }

    __syncthreads();
    //      if (threadIdx.x == 0 && adx == 1)
    //printf("=== reduce neighbours ===\n");
    //__syncthreads();

    // reduce neighbours ------------------------------------------------------
    for (idx_t vtx = batch_idx(threadIdx.x, mesh->Nn, blockDim.x) ; 
        vtx < batch_idx(threadIdx.x + 1, mesh->Nn, blockDim.x) ; ++vtx)
    {
      if (states->states[vtx] != ek_neig)
        continue;

      const idx_t edsp = mesh->n2e_dsp[vtx];
      const dbl_t old_phi_ms = states->phi_ms[vtx];
      dbl_t new_phi_ms = old_phi_ms;

      for (cnt_t ecnt = 0 ; ecnt < mesh->n2e_cnt[vtx] ; ++ecnt)
      {
        const dbl_t cur_phi_ms = states->tmp_phi_ms[edsp+ecnt];

        if (cur_phi_ms < new_phi_ms)
          new_phi_ms = cur_phi_ms;

        //if (adx == 1 && vtx == 67)
        //  printf("67 cur phi: %lf\n", cur_phi_ms);
        //if (vtx == 48)// && cur_phi_ms < 1e20
        //  printf("A%d: %ld %lf %ld\n", blockIdx.x, vtx, cur_phi_ms, edsp+ecnt);
      }

      states->phi_ms[vtx] = new_phi_ms;
      //printf("A%d: vtx: %ld phi: %lf\n", blockIdx.x, vtx, states->phi_ms[vtx]);
      //if (vtx == 48)
      //  printf("A%d: %ld %lf\n", blockIdx.x, vtx, new_phi_ms);
      //if (adx == 1 && vtx == 67) 
      //  printf("67 new phi: %lf\n", new_phi_ms);

      if (new_phi_ms < old_phi_ms)
        states->states[vtx] = ek_narrow;
      else
        states->states[vtx] = states->tmp_state[vtx];
    }

    __syncthreads();

    // update neighbour states ------------------------------------------------
    for (idx_t vtx = batch_idx(threadIdx.x, mesh->Nn, blockDim.x) ; 
        vtx < batch_idx(threadIdx.x + 1, mesh->Nn, blockDim.x) ; ++vtx)
    {
      if (states->states[vtx] == ek_narrow)
      {
        if (states->tmp_state[vtx] == ek_frozen)
          states->states[vtx] = ek_frozen;
        else
          active_size++;
      }
      else if (states->states[vtx] == ek_neig)
      {
        states->states[vtx] = ek_narrow;
        active_size++;
      }
    }

    //active_size = 0;
    __syncthreads();
    
    // update states and active list ------------------------------------------
    //for (int offset = warpSize/2; offset > 0; offset >>= 1)
    //  active_size += __shfl_xor_sync(FULL_MASK, active_size, offset);
    active_size = reduce_active_size(active_size, tmp);
    //__syncthreads();

    //break;
  }
  
  // batch copy states back to global memory ----------------------------------
  bit_start = batch_idx(threadIdx.x, states_bytes, blockDim.x);
  bit_end = batch_idx(threadIdx.x + 1, states_bytes, blockDim.x);

  for (size_t bit = bit_start ; bit < bit_end ; ++bit)
    states_ptr[bit] = sptr[mesh_bytes + bit];

  __syncthreads();

  if (threadIdx.x == 0) // maybe one thread assigns one pointer?
    d_esvd[adx].assemble();
}












































// ----------------------------------------------------------------------------
// could be a single cuda kernel ... (but then you always have to use many blocks .. [UNLIMITED POWER!])
// can the cpu do work in parallel? poll using DMA? (for example: number of processed steps)
// if too many ressources:
//   - Too large a block or grid size
//   - Too many registers used
//   - Too much shared memory used
double CudaSolverFIM::solve(std::vector<std::vector<dbl_t>> &act_ms,
                          std::vector<std::vector<dbl_t>> &apd_ms,
                          const std::string file, const int num_reps)
{
  double sum_exec_time_s = 0.0;
  log_info("CudaSolverFIM::solve");
  std::cout<<"num_submeshes: "<<h_smsh.N<<", TPB: "<<tpb<<"\n";
  //checkCudaErrors(cudaFuncSetAttribute(compute_step, cudaFuncAttributeMaxDynamicSharedMemorySize, 160768));
  //gpuErrchk(cudaPeekAtLastError());

  /*indicators::show_console_cursor(false);

  indicators::BlockProgressBar bar{
    indicators::option::BarWidth{50},
    indicators::option::Start{"["},
    indicators::option::End{"]"},
    indicators::option::ForegroundColor{indicators::Color::white},
    indicators::option::ShowElapsedTime{true},
    indicators::option::ShowRemainingTime{true},
    indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
  };

  indicators::show_console_cursor(true);*/

  for (int rid = 0 ; rid < num_reps ; ++rid)
  {
    //log_info("reset_states_gpu");
    reset_states_gpu<<<h_smsh.N, tpb>>>(d_eksv);
    gpuErrchk(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //log_info("update_events_gpu");
    update_events_gpu<<<h_smsh.N, tpb>>>(d_eksv, d_events, 0.0);
    gpuErrchk(cudaPeekAtLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    //log_info("main loop");
    cur_time = 0.0; phi_start = 0.0;
    start = std::chrono::steady_clock::now();

    for (size_t step = 0 ; step < opts.n_steps ; ++step)
    {
      phi_start = cur_time;
      cur_time += opts.dt;

      sync_step2<<<h_smsh.N, tpb>>>(d_smsh, d_eksv, d_inter);
      gpuErrchk(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());

      compute_step<<<h_smsh.N, tpb, max_shared_bytes>>>(d_smsh, d_eksv, d_inter, wavefront_width, cur_time, phi_start, opts.dt);//, update_stims_flag // generates 1 act/apd -> (phi, di)
      gpuErrchk(cudaPeekAtLastError());
      checkCudaErrors(cudaDeviceSynchronize());
      // get act/apd from narrow aggs
      // assemble act and apd on CPU

      //if (stims.front().start >= phi_start)
      //  update_stims_flag = true;
      //  stims.pop_front();
    }

    end = std::chrono::steady_clock::now();
    const double exec_time_s = std::chrono::duration<double>(end - start).count();
    sum_exec_time_s += exec_time_s;
    printf("run %2d took %.5f s.\n", rid + 1, exec_time_s);
  }

  // copy states back
  d_eksv.copy_to(h_eksv);
  assemble_pointers(h_eksv);

  for (int it = 0 ; it < h_eksv.N ; ++it)
  {
    unsigned char far_state = h_eksv[it].agg_state == ek_far; //& EKMS_FAR
    unsigned char nar_state = h_eksv[it].agg_state == ek_narrow; //& EKMS_NARROW
    unsigned char fro_state = h_eksv[it].agg_state == ek_frozen; // & EKMS_FROZEN
    //printf("agg%d: %x (far: %x narrow: %x frozen: %x)\n", it, h_eksv[it].agg_state, far_state > 0, nar_state > 0, fro_state > 0);
  }

  //log_info("move data from local to global");
  // move data from local to global
  for (int adx = 0 ; adx < h_eksv.N ; ++adx)
  {
    eksv_t &sv = h_eksv[adx];

    for (int lidx = 0 ; lidx < sv.nNodes ; ++lidx)
    {
      int gidx = l2g[adx][lidx];

      if (sv.phi_ms[lidx] < phi[gidx])
      {
        phi[gidx] = sv.phi_ms[lidx];
        //states[gidx] = static_cast<ek_state>(sv.states[lidx]);
      }
      //phi[gidx] = std::min(phi[gidx], sv.phi_ms[lidx]);
      //states[gidx] = static_cast<ek_state>(sv.states[lidx]);

      if (sv.states[lidx] > states[gidx])
        states[gidx] = static_cast<ek_state>(sv.states[lidx]);
    }
  }

  //log_info("save vtk");
  std::vector<dbl_t> states_dbl;
  states_dbl.assign(states.begin(), states.end());
  save_vtk(file, mesh, {phi, states_dbl}, {"phi", "states"}, {}, {});

  //transpose_solution(act_ms, apd_ms); // many memcpys or just one?

  // ALTERNATE VERSION - SINGLE KERNEL CALL
  // reset states or read state file?
  //solve_fim_gpu<<<(num_agg, max_agg), threads_per_block, max_agg_bytes>>>(
  //  d_smsh, d_eksv, d_ekfd, dt, n_steps);
  //cudaMemcpy(h_eksv, d_eksv);
  // assemble and apply submesh data to global data
  //log_info("CudaSolverFIM::solve finished");

  return sum_exec_time_s/num_reps;
}






















// ----------------------------------------------------------------------------
__host__ void compute_Mp_regular(const mesh_t &mesh, const idx_t idx, const idx_t edx, const dbl_t *M, dbl_t *Mp)
{
  const idx_t dsp = mesh.e2n_dsp[edx];
  idx_t ndxs[3]; 
  const idx_t vtx = mesh.e2n_con[dsp+idx];

  for (cnt_t ncnt = 0 ; ncnt < 3 ; ++ncnt)
    ndxs[ncnt] = mesh.e2n_con[dsp+((idx + 1 + ncnt) % 4)];

  dbl3_t e13; load_edge(ndxs[0], ndxs[2], mesh.xyz.data(), e13);
  dbl3_t e23; load_edge(ndxs[1], ndxs[2], mesh.xyz.data(), e23);
  dbl3_t e34; load_edge(ndxs[2], vtx, mesh.xyz.data(), e34);

  Mp[0] = aMb_sym(e13, M, e13);
  Mp[1] = aMb_sym(e23, M, e13);
  Mp[2] = aMb_sym(e34, M, e13);
  Mp[3] = aMb_sym(e23, M, e23);
  Mp[4] = aMb_sym(e34, M, e23);
  Mp[5] = aMb_sym(e34, M, e34);
}
__host__ char gray2idx(const char gray)
{
  return gray/2 - 1;
}
__host__ char f_sign(const char k_gray, const char l_gray)
{
  return ((2*((k_gray^l_gray) > k_gray) - 1)*(2*((k_gray^l_gray) > l_gray) - 1));
}
__host__ dbl_t compute_mix(const char k_gray, const char l_gray, const dbl_t *geo) // maybe compute mappings before! -> constant
{
  const char s_gray = k_gray^l_gray;
  const int esgn = ((2*(s_gray > k_gray) - 1)*(2*(s_gray > l_gray) - 1));
  const char k_idx = gray2idx(k_gray), l_idx = gray2idx(l_gray), s_idx = gray2idx(s_gray);
  printf("mix:        k_gray: %2d, l_gray: %2d, s_gray: %2d\n", k_gray, l_gray, s_gray);
  printf("  esgn: %2d, k_idx:  %2d, l_idx:  %2d, s_idx:  %2d\n", esgn, k_idx, l_idx, s_idx);
  return esgn*0.5*(geo[k_idx] + geo[l_idx] - geo[s_idx]);
}
__host__ void compute_Mp_gray(const dbl_t *geo, const idx_t idx, dbl_t *Mp,
  const char *idx_adf, const char *idx_bce) //, const char *ssigns , const char *esigns, const char *asigns
{
  //    0        1        2        3         4        5     index
  //    3        5        6        9        10       12     gray code (decimal)
  //   0011     0101     0110     1001     1010     1100    gray code (binary)
  // (e12Me12, e13Me13, e23Me23, e14Me14, e24Me24, e34Me34) geo
  // 
  // ekMel = esgn*0.5*(ekMek + elMel - esMes)
  // s = xor(k, l) -> 3 = xor(6, 5) -> 0011 -> (e12Me12)
  // esgn = ((2∗(s > k) - 1)*(2*(s > l) - 1)) -> (2*(0) - 1)*(2*(0) - 1) = 1
  // e23Me13 = esgn*0.5*(e23Me23 + e13Me13 - e12Me12)
  // 
  // M[0] = e13Me13; M[1] = e23Me13; M[2] = e34Me13;
  //                 M[3] = e23Me23; M[4] = e34Me23;
  //                                 M[5] = e34Me34;

  const char i0 = idx_adf[3*idx + 0], i1 = idx_adf[3*idx + 1], i2 = idx_adf[3*idx + 2],
             i3 = idx_bce[3*idx + 0], i4 = idx_bce[3*idx + 1], i5 = idx_bce[3*idx + 2];
  
  const dbl_t e13Me13 = geo[i0], e23Me23 = geo[i1], e34Me34 = geo[i2],
              e12Me12 = geo[i3], e14Me14 = geo[i4], e24Me24 = geo[i5];

  // a, b, c, d, e, f
  Mp[0] =       e13Me13;
  Mp[1] =  0.5*(e23Me23 + e13Me13 - e12Me12);
  Mp[2] = -0.5*(e34Me34 + e13Me13 - e14Me14);
  Mp[3] =       e23Me23;
  Mp[4] = -0.5*(e34Me34 + e23Me23 - e24Me24);
  Mp[5] =       e34Me34;
}
__host__ void test_gray_code(const mesh_t &mesh)
{
  dbl3_t edg;
  dbl_t M[6], geo[6], Mp_reg[6], Mp_gray[6]; 
  const vec3_t velo = {0.6, 0.6, 0.6};

  for (idx_t edx = 0 ; edx < static_cast<idx_t>(mesh.e2n_cnt.size()) ; ++edx)
  {
    const idx_t dsp = mesh.e2n_dsp[edx];
    const idx_t *con = mesh.e2n_con.data() + dsp;
    const idx_t x1 = con[0], x2 = con[1], x3 = con[2], x4 = con[3];

    compute_velocity_tensor_f3(mesh.lon.data(), edx, velo, M);

    // precompute squared times
    load_edge(x1, x2, mesh.xyz.data(), edg); geo[0] = aMb_sym(edg, M, edg); // e12
    load_edge(x1, x3, mesh.xyz.data(), edg); geo[1] = aMb_sym(edg, M, edg); // e13
    load_edge(x2, x3, mesh.xyz.data(), edg); geo[2] = aMb_sym(edg, M, edg); // e23
    load_edge(x1, x4, mesh.xyz.data(), edg); geo[3] = aMb_sym(edg, M, edg); // e14
    load_edge(x2, x4, mesh.xyz.data(), edg); geo[4] = aMb_sym(edg, M, edg); // e24
    load_edge(x3, x4, mesh.xyz.data(), edg); geo[5] = aMb_sym(edg, M, edg); // e34

    //std::cout<<"\nPRE TEST\n";
    //printf("e13: %d e23: %d e34: %d\n", gray2idx(5), gray2idx(6), gray2idx(12)); 
    //printf("e23M13: %d e34M13: %d e34M23: %d\n", gray2idx(6^5), gray2idx(12^5), gray2idx(12^6)); 
    //printf("s23M13: %d s34M13: %d s34M23: %d\n", f_sign(6, 5), f_sign(12, 5), f_sign(12, 6)); 

    // first test ---------------------------------------------------------------
    //std::cout<<"\nFIRST TEST\n";
    /*dbl_t gres, gref;
    dbl3_t e12, e13, e23;
    load_edge(con[0], con[1], mesh.xyz.data(), e12);
    load_edge(con[0], con[2], mesh.xyz.data(), e13);
    load_edge(con[1], con[2], mesh.xyz.data(), e23);

    gres = compute_mix(3, 5, geo); gref = aMb_sym(e12, M, e13);
    //printf("g12, g13: %lf == %lf (%lf)\n", gres, gref, gres - gref);

    gres = compute_mix(3, 6, geo); gref = aMb_sym(e12, M, e23); 
    //printf("g12, g23: %lf == %lf (%lf)\n", gres, gref, gres - gref);

    gres = compute_mix(5, 6, geo); gref = aMb_sym(e13, M, e23); 
    //printf("g13, g23: %lf == %lf (%lf)\n", gres, gref, gres - gref);
    */

    // second test --------------------------------------------------------------
    /*std::cout<<"\nSECOND TEST\n";

    for (idx_t idx = 0 ; idx < 4 ; ++idx)
    {
      printf("idx: %ld\n", idx);
      printf("%ld %ld %ld %ld\n", idx, (idx + 1)&3, (idx + 2)&3, (idx + 3)&3);
      printf("%ld %ld %ld %ld\n", con[idx], con[(idx + 1)&3], con[(idx + 2)&3], con[(idx + 3)&3]);
    }*/

    // third test ---------------------------------------------------------------
    //std::cout<<"\nTHIRD TEST\n";
    // precompute edge numbers and signs -> 48 bytes -> fits in one cache line!
    // 4*[(eidxs[3] + esigns[3]) + (aidxs[3] + asigns[3])] 
    //         5    6    12
    //       (e13, e23, e34) (e23M13, e34M13, e34M23) [s]
    // phi4: 1,2,5 : +,+,+ |  
    // phi3: 2,4,0 : +,-,+ |  
    // phi2: 0,1,3 : +,-,- |  
    // phi1: 3,4,5 : -,+,+ |  

    // cache edge codes and signs in constant memory!
    const char idx_adf[12] = {4, 5, 3, 1, 3, 0, 4, 0, 2, 1, 2, 5};
    const char idx_bce[12] = {2, 0, 1, 5, 2, 4, 3, 5, 1, 0, 3, 4};
    dbl_t error_M = 0.0;

    for (idx_t idx = 0 ; idx < 4 ; ++idx)
    {
      //std::cout<<"idx: "<<idx<<":\n";
      compute_Mp_regular(mesh, idx, edx, M, Mp_reg);
      compute_Mp_gray(geo, idx, Mp_gray, idx_adf, idx_bce);
    
      for (int it = 0 ; it < 6 ; ++it)
        error_M += fabs(Mp_gray[it] - Mp_reg[it]); //printf("%lf == %lf (%lf)\n", Mp_gray[it], Mp_reg[it], Mp_gray[it] - Mp_reg[it]);
    }

    //printf("edx %ld error: %lf\n", edx, error_M);
    assert(error_M < 1e-5);

    break;
  }
}
















/*
// correctness testing --------------------------------------------------------
dbl_t solve_tri_cpu(const dbl_t &e12Me12, const dbl_t &e13Me13, const dbl_t &e23Me23,
                    const dbl_t &phi1,    const dbl_t &phi2)
{
  const dbl_t a =       e23Me23;
  const dbl_t b = -0.5*(e12Me12 + e23Me23 - e13Me13);
  const dbl_t c =       e12Me12;
  const dbl_t phi12 = phi2 - phi1;
  printf("a: %lf b: %lf c: %lf\n", c, b, a);

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
dbl_t solve_tet_cpu(const dbl_t &e13Me13, const dbl_t &e23Me23, const dbl_t &e34Me34,
                    const dbl_t &e12Me12, const dbl_t &e14Me14, const dbl_t &e24Me24,
                    const dbl_t &phi1,    const dbl_t &phi2,    const dbl_t &phi3)
{
  const dbl_t a =       e13Me13;
  const dbl_t b =  0.5*(e23Me23 + e13Me13 - e12Me12);
  const dbl_t c = -0.5*(e34Me34 + e13Me13 - e14Me14);
  const dbl_t d =       e23Me23;
  const dbl_t e = -0.5*(e34Me34 + e23Me23 - e24Me24);
  const dbl_t f =       e34Me34;

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
dbl_t solve_local(const mesh_t &mesh, const ek_data &ekdata, const idx_t vtx, const idx_t edx, 
                  const ek_state threshold, const dbl_t phi_start_ms, const dbl_t front_width_ms)
{
  log_info("solve_local");
  cnt_t idx = 0; // array -> local
  dbl_t phi_ms[4];
  const idx_t dsp = mesh.e2n_dsp[edx]; // move this to loop (or make reference) and check if it makes a difference in register count
  const dbl_t *geo = mesh.geo.data() + 6*edx; // reference?
  const gray_symbols gidx = gray_symbols();// = d_gray_mappings;

  for (cnt_t cnt = 0 ; cnt < mesh.e2n_cnt[edx] ; ++cnt)
  {
    const idx_t nvtx = mesh.e2n_con[dsp+cnt]; // move to if?
    std::cout<<"nidx: "<<nvtx<<" ("<<ekdata.states[nvtx]<<")"<<":";

    if (nvtx == vtx)
    {
      idx = cnt;
      std::cout<<" main!\n";
    }
    else if ((ekdata.states[nvtx] >= threshold) && (ekdata.phi[nvtx] > (phi_start_ms - front_width_ms)))
    {
      phi_ms[cnt] = ekdata.phi[nvtx];
      std::cout<<" valid!\n";
    }
    else
    {
      phi_ms[cnt] = DBL_INF;
      std::cout<<" invalid!\n";
    }
  }

  std::cout<<"idx: "<<idx<<"\n";
  
  const idx_t i0 = gidx.adf[3*idx + 0], i1 = gidx.adf[3*idx + 1], i2 = gidx.adf[3*idx + 2], // and here
              i3 = gidx.bce[3*idx + 0], i4 = gidx.bce[3*idx + 1], i5 = gidx.bce[3*idx + 2];

  const dbl_t e13Me13 = geo[i0], e23Me23 = geo[i1], e34Me34 = geo[i2], // also references here
              e12Me12 = geo[i3], e14Me14 = geo[i4], e24Me24 = geo[i5];

  const dbl_t phi1 = phi_ms[gidx.phi[3*idx + 0]]; // use reference here, check if register count changes
  const dbl_t phi2 = phi_ms[gidx.phi[3*idx + 1]];
  const dbl_t phi3 = phi_ms[gidx.phi[3*idx + 2]];

  printf("phis: %lf %lf %lf\n", phi1, phi2, phi3);
  printf("edxs: %ld %ld %ld %ld %ld %ld\n", i0, i1, i2, i3, i4, i5);
  printf("edgs: %lf %lf %lf %lf %lf %lf\n", e13Me13, e23Me23, e34Me34, e12Me12, e14Me14, e24Me24);

  dbl_t phi4 = DBL_INF, res1, res2, res3;

  // tetra solutions:
  res1 = solve_tet_cpu(e13Me13, e23Me23, e34Me34, e12Me12, e14Me14, e24Me24, phi1, phi2, phi3); phi4 = fmin(phi4, res1);
  printf("tet1: %lf\n", res1);

  // triangle solutions:
  res1 = solve_tri_cpu(e12Me12, e14Me14, e24Me24, phi1, phi2); phi4 = fmin(phi4, res1);
  res2 = solve_tri_cpu(e23Me23, e24Me24, e34Me34, phi2, phi3); phi4 = fmin(phi4, res2);
  res3 = solve_tri_cpu(e13Me13, e14Me14, e34Me34, phi1, phi3); phi4 = fmin(phi4, res2);
  printf("tri1: %lf tri2: %lf tri3: %lf\n", res1, res2, res3);

  // edge solutions
  res1 = phi1 + sqrt(e14Me14); phi4 = fmin(phi4, res1);
  res2 = phi2 + sqrt(e24Me24); phi4 = fmin(phi4, res2);
  res3 = phi3 + sqrt(e34Me34); phi4 = fmin(phi4, res3);
  printf("edg1: %lf edg2: %lf edg3: %lf\n", res1, res2, res3);

  return phi4;
}
// ----------------------------------------------------------------------------
void read_xyz_ref(const mesh_t &mesh, const idx_t idx, vec3_t& pt) // to mesh class?
{
  const dbl_t* xp = mesh.xyz.data() + idx*3;
  pt.x = *xp++, pt.y = *xp++, pt.z = *xp++;
}
dbl_t aMb_sym_ref(const vec3_t &a, const dbl_t *M, const vec3_t &b)
{
  return (M[0]*a.x + M[1]*a.y + M[2]*a.z)*b.x + 
        (M[1]*a.x + M[3]*a.y + M[4]*a.z)*b.y + 
        (M[2]*a.x + M[4]*a.y + M[5]*a.z)*b.z;
}
void compute_velocity_tensor_ref(const vec3_t &v, dbl_t *M)//const dbl_t eidx, 
{
  //const dbl_t *fdat = mesh.lon.data() + eidx*6;
  //const dbl_t *sdat = fdat+3;
  //const dbl_t f0 = fdat[0], f1 = fdat[1], f2 = fdat[2];
  //const dbl_t s0 = sdat[0], s1 = sdat[1], s2 = sdat[2];
  const dbl_t f0 = 1., f1 = 0., f2 = 0.;
  const dbl_t s0 = 1., s1 = 0., s2 = 0.;
  const dbl_t fa = f0*f0, fb = f0*f1, fc = f0*f2;
  const dbl_t fd = f1*f1, fe = f1*f2, ff = f2*f2;
  const dbl_t sa = s0*s0, sb = s0*s1, sc = s0*s2;
  const dbl_t sd = s1*s1, se = s1*s2, sf = s2*s2;
  const dbl_t vf = 1./v.x/v.x, vs = 1./v.y/v.y, vn = 1./v.z/v.z;

  M[0] = fa*vf + sa*vs - (fa + sa - 1.)*vn;
  M[1] = fb*vf + sb*vs - (fb + sb)*vn;
  M[2] = fc*vf + sc*vs - (fc + sc)*vn;
  M[3] = fd*vf + sd*vs - (fd + sd - 1.)*vn; // M4
  M[4] = fe*vf + se*vs - (fe + se)*vn;      // M5
  M[5] = ff*vf + sf*vs - (ff + sf - 1.)*vn; // M8
}
dbl_t solve_edg_ref(idx_t idx, idx_t* nidxs, dbl_t* M, const mesh_t &mesh, const std::vector<dbl_t> &phi)
{
  dbl_t tA;
  vec3_t A, B, AB;
  tA = phi[nidxs[0]];
  read_xyz_ref(mesh, nidxs[0], A);
  read_xyz_ref(mesh, idx, B);
  AB = B - A;

  return tA + sqrt(aMb_sym_ref(AB, M, AB));
}
// --------------------------------------------------------------------------
dbl_t solve_tri_ref(idx_t idx, idx_t* nidxs, dbl_t* M, const mesh_t &mesh, const std::vector<dbl_t> &phi)
{
  vec3_t x1, x2, x3, e12, e23; // const
  
  read_xyz_ref(mesh, nidxs[0], x1);
  read_xyz_ref(mesh, nidxs[1], x2);
  read_xyz_ref(mesh, idx, x3);

  e12 = x2 - x1; //eAB = B - A;
  e23 = x3 - x2; //eAC = C - A;

  dbl_t tA, tB;
  tA = phi[nidxs[0]];
  tB = phi[nidxs[1]];
  const dbl_t u = tB - tA;
  const dbl_t u2 = u*u;

  const double a = aMb_sym_ref(e23, M, e23); // aMb_sym(eAC, M, eAC);
  const double b = aMb_sym_ref(e23, M, e12); // aMb_sym(eAC, M, eAB);
  const double c = aMb_sym_ref(e12, M, e12); // aMb_sym(eAB, M, eAB);
  printf("a: %lf b: %lf c: %lf\n", c, b, a);

  const dbl_t x = c*(u2 - c);
  const dbl_t y = b*(u2 - c);
  const dbl_t z = u2*a - b*b;

  const dbl_t p = y/x;
  const dbl_t q = z/x;
  const dbl_t sqt = p*p - q;

  dbl_t tC = DBL_INF, l1, l2, res;

  if(!(sqt < 0.0))
  {
    l1 = -p + sqrt(sqt);
    l2 = -p - sqrt(sqt);

    if((l1 > 0.0) && (l1 < 1.0))
    {
      res = l1*tA + (1.0 - l1)*tB + sqrt(l1*l1*c + 2*b*l1 + a);
      tC = std::min(tC, res);
    }

    if((l2 > 0.0) && (l2 < 1.0))
    {
      res = l2*tA + (1.0 - l2)*tB + sqrt(l2*l2*c + 2*b*l2 + a);
      tC = std::min(tC, res);
    }
  }

  //if(tC == DBL_INF)
  //{
    //std::vector<idx_t> nidxs_edg = {nidxs[0], nidxs[1]};
    //tC = solve_edg(idx, nidxs, M);
  //  tC = std::min(tC, solve_edg(idx, nidxs, M));
  //  tC = std::min(tC, solve_edg(idx, &nidxs[1], M));
  //}
  
  return tC;
}
dbl_t solve_tet_ref(idx_t idx, idx_t* nidxs, dbl_t* M, const mesh_t &mesh, const std::vector<dbl_t> &phi)
{
  vec3_t x1, x2, x3, x4, e34, e13, e23, tmp; // const
  read_xyz_ref(mesh, nidxs[0], x1);
  read_xyz_ref(mesh, nidxs[1], x2);
  read_xyz_ref(mesh, nidxs[2], x3);
  read_xyz_ref(mesh, idx, x4);

  e34 = x4 - x3;
  e13 = x3 - x1;
  e23 = x3 - x2;

  dbl_t tA, tB, tC;

  tA = phi[nidxs[0]]; //phi1
  tB = phi[nidxs[1]]; //phi2
  tC = phi[nidxs[2]]; //phi3

  const dbl_t u = tC - tA;//abs(phi1 - phi3);//e13.length();// has to be > 0
  const dbl_t v = tC - tB;
  const dbl_t u2 = u*u;

  const dbl_t a = aMb_sym_ref(e34, M, e34);// e34 e34 //computeM(nidxs[2], eidx, M);
  const dbl_t b = aMb_sym_ref(e13, M, e34);// e13 e34
  const dbl_t c = aMb_sym_ref(e13, M, e13);// e13 e13
  const dbl_t d = aMb_sym_ref(e13, M, e23);// e13 e23
  const dbl_t e = aMb_sym_ref(e23, M, e34);// e23 e34
  const dbl_t f = aMb_sym_ref(e23, M, e23);// e23 e23

  const dbl_t g = -(u*d - v*c)/(u*f - v*d);
  const dbl_t h = -(u*e - v*b)/(u*f - v*d);

  const dbl_t x = u2*(c + 2.0*d*g + f*g*g) - (c + d*g)*(c + d*g);
  const dbl_t y = u2*(d*h + b + f*g*h + e*g) - (c*d*h + b*c + d*d*g*h + b*d*g);
  const dbl_t z = u2*(f*h*h + 2.0*e*h + a) - (d*h + b)*(d*h + b);

  const dbl_t p = y/x;
  const dbl_t q = z/x;
  const dbl_t sqt = p*p - q;

  dbl_t tD = DBL_INF, l1, l2, l3, res;

  if(!(sqt < 0.0))
  {
    l1 = -p + sqrt(sqt);
    l2 = l1*g + h;
    l3 = 1.0 - l1 - l2;

    if((l1 > 0.0) && (l1 < 1.0) && (l2 > 0.0) && (l2 < 1.0) && (l3 > 0.0))
    {
      tmp = x4 - (x1*l1 + x2*l2 + x3*l3); //e54
      res = l1*tA + l2*tB + l3*tC + sqrt(aMb_sym_ref(tmp, M, tmp)); //phi4 = phi5 + phi54
      tD = std::min(tD, res);
    }

    l1 = -p - sqrt(sqt);
    l2 = l1*g + h;
    l3 = 1.0 - l1 - l2;

    if((l1 > 0.0) && (l1 < 1.0) && (l2 > 0.0) && (l2 < 1.0) && (l3 > 0.0))
    {
      tmp = x4 - (x1*l1 + x2*l2 + x3*l3); //e54
      res = l1*tA + l2*tB + l3*tC + sqrt(aMb_sym_ref(tmp, M, tmp)); //phi4 = phi5 + phi54
      tD = std::min(tD, res);
    }
  }
  
  // resort to triangles
  //if(tD ==  DBL_INF)
  //{
  //  
  //}

  return tD;
}
// --------------------------------------------------------------------------
dbl_t update_phi_ref(const mesh_t &mesh, const ek_data &ekdata, const idx_t vtx, const idx_t edx, const ek_state threshold)
{
  log_info("update_phi_ref");
  dbl_t M[6];
  const vec3_t velo = {0.6, 0.6, 0.6};
  compute_velocity_tensor_ref(velo, M);
  idx_t ndxs[3] = {5, 51, 130};
  idx_t tdxs1[2] = {5, 51};
  idx_t tdxs2[2] = {51, 130};
  idx_t tdxs3[2] = {5, 130};

  dbl_t phi4 = DBL_INF, res1, res2, res3;
  res1 = solve_tet_ref(vtx, ndxs, M, mesh, ekdata.phi); phi4 = fmin(phi4, res1);
  printf("tet1: %lf\n", res1);

  res1 = solve_tri_ref(vtx, tdxs1, M, mesh, ekdata.phi); phi4 = fmin(phi4, res1);
  res2 = solve_tri_ref(vtx, tdxs2, M, mesh, ekdata.phi); phi4 = fmin(phi4, res2);
  res3 = solve_tri_ref(vtx, tdxs3, M, mesh, ekdata.phi); phi4 = fmin(phi4, res3);
  printf("tri1: %lf tri2: %lf tri3: %lf\n", res1, res2, res3);

  res1 = solve_edg_ref(vtx, &ndxs[0], M, mesh, ekdata.phi); phi4 = fmin(phi4, res1);
  res2 = solve_edg_ref(vtx, &ndxs[1], M, mesh, ekdata.phi); phi4 = fmin(phi4, res2);
  res3 = solve_edg_ref(vtx, &ndxs[2], M, mesh, ekdata.phi); phi4 = fmin(phi4, res3);
  printf("edg1: %lf edg2: %lf edg3: %lf\n", res1, res2, res3);

  return phi4;
}
*/













/*
// ----------------------------------------------------------------------------
__host__ void cuda_fim_test2()
{
  std::cout<<"cuda_fim_test2\n";
  //checkCudaErrors(cudaSetDevice(gpuid[0]));

  // advanced test -> load mesh data into agglomerate -> requires compute_full_connectivity() ..
  const std::string mesh_path = "data/ut_mesh/ut_mesh.vtk";
  const std::string plan_path = "data/ut_mesh/ut_studio_plan.json";
  const std::string cfg_path  = "data/default.cfg";
  //const std::string ref_path  = "test/data/";
  //const std::string out_path  = "test/";
  const std::string part_path = "data/ut_mesh/ut_mesh.tags";
  
  //std::cout<<std::filesystem::current_path()<<"\n";
  // load mesh
  mesh_t mesh;
  load_vtk(mesh_path, mesh); // load_mesh
  read_vector_txt(part_path, mesh.atags); // load atags

  mesh.lon.assign(3*mesh.e2n_cnt.size(), 0.0); // !!!!
  for (size_t it = 0 ; it < mesh.e2n_cnt.size() ; ++it)
    mesh.lon[3*it] = 1.0;

  mesh.compute_base_connectivity(true);
  mesh.compute_geo();
  test_gray_code(mesh);

  //for (int it = 0 ; it < static_cast<int>(mesh.e2n_cnt.size()) ; ++it)
  //  std::cout<<"edx"<<it<<": "<<mesh.geo[6*it+0]<<" "<<mesh.geo[6*it+1]<<" "<<mesh.geo[6*it+2]<<" "<<mesh.geo[6*it+3]<<" "<<mesh.geo[6*it+4]<<" "<<mesh.geo[6*it+5]<<" "<<"\n";

  ek_data ekdata(mesh);
  load_options(cfg_path, ekdata.opts);
  load_plan(plan_path, ekdata); // check if plan <-> mesh
  ekdata.opts.dt = 150e3;
  ekdata.opts.n_steps = 3;

  // test: elm 168 idx 159
  //ekdata.phi[5] = 75e3; ekdata.states[5] = ek_narrow; 
  //ekdata.phi[51] = 60e3; ekdata.states[51] = ek_narrow;
  //ekdata.phi[130] = 50e3; ekdata.states[130] = ek_narrow;

  //const dbl_t res = solve_local(mesh, ekdata, 159, 168, ek_narrow, 0.0, DBL_INF);
  //const dbl_t ref = update_phi_ref(mesh, ekdata, 159, 168, ek_narrow);
  //std::cout<<"phi res: "<<res<<" vs "<<ref<<"\n";
  

  std::vector<std::vector<dbl_t>> act_ms, apd_ms;
  CudaSolverFIM solver(ekdata);
  solver.solve(act_ms, apd_ms, "cuda_fim_res.vtk");

  // write output!

  std::cout<<"cuda_fim_test2 end\n";
}
// ----------------------------------------------------------------------------
__host__ void cuda_fim_test3()
{
  const std::string mesh_path = "data/heart/S62.vtk";
  const std::string plan_path = "data/heart/S62.plan.json";
  const std::string cfg_path  = "data/default.cfg";
  const std::string part_path = "data/heart/S62.7560.tags";//6050

  //const std::string mesh_path = "/home/tom/workspace/masc-experiments/heart2/S62_500um.vtk";
  //const std::string part_path = "/home/tom/workspace/masc-experiments/heart2/S62_500um.29135.tags";

  //cudaSetDevice(0);                   // Set device 0 as current
  //float* p0;
  //size_t size = 1024 * sizeof(float);
  //cudaMalloc(&p0, size);              // Allocate memory on device 0
  //MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
  //cudaSetDevice(1);                   // Set device 1 as current
  //cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                      // with device 0

  // Launch kernel on device 1
  // This kernel launch can access memory on device 0 at address p0
  //MyKernel<<<1000, 128>>>(p0);

  // load mesh
  mesh_t mesh;
  load_vtk(mesh_path, mesh); // load_mesh
  read_vector_txt(part_path, mesh.atags); // load atags

  mesh.lon.assign(3*mesh.e2n_cnt.size(), 0.0); // !!!!
  for (size_t it = 0 ; it < mesh.e2n_cnt.size() ; ++it)
    mesh.lon[3*it] = 1.0;

  log_info("compute_base_connectivity");
  std::chrono::time_point<std::chrono::steady_clock> start, end;
  start = std::chrono::steady_clock::now();
  mesh.compute_base_connectivity(true);
  end = std::chrono::steady_clock::now();
  std::cout<<"compute_base_connectivity: "<<std::chrono::duration<double>(end-start).count()<<"s\n";

  ek_data ekdata(mesh);
  load_options(cfg_path, ekdata.opts);
  load_plan(plan_path, ekdata); // check if plan <-> mesh
  ekdata.opts.dt = 1e3;
  ekdata.opts.n_steps = 100;

  std::vector<std::vector<dbl_t>> act_ms, apd_ms;
  CudaSolverFIM solver(ekdata);
  solver.solve(act_ms, apd_ms, "cuda_fim_heart.vtk");

  //const std::string ref_path = "/home/tom/workspace/masc-experiments/heart/S62_ref.dat";
  //if (!std::filesystem::exists(ref_path))
}
*/
// ----------------------------------------------------------------------------
} // namespace tw =======================================================