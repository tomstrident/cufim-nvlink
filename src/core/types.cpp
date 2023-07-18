
#include "types.h"


namespace tw
{

void mesh_t::clear()
{
  xyz.clear();
  lon.clear();
  geo.clear();

  etags.clear();
  etype.clear();

  e2n_cnt.clear(); n2e_cnt.clear(); n2n_cnt.clear();
  e2n_dsp.clear(); n2e_dsp.clear(); n2n_dsp.clear();
  e2n_con.clear(); n2e_con.clear(); n2n_con.clear();
}

void mesh_t::compute_dsp(const std::vector<cnt_t>& cnt, std::vector<idx_t>& dsp) // wrong
{
  //dsp.resize(cnt.size());
  //idx_t* dsp_p = dsp.data();

  //for(size_t eidx = 0 ; eidx < cnt.size() ; eidx++)
  //{
  //  *dsp_p++ = dsp[eidx] + cnt[eidx];
  //}

  dsp.resize(cnt.size());
  idx_t tmp = 0;

  for(size_t eidx = 0 ; eidx < cnt.size() ; eidx++)
  {
    dsp[eidx] = tmp;
    tmp += cnt[eidx];
  }
}

// big parallel loop?
void mesh_t::compute_base_connectivity(const bool verbose) // outside of struct -> applied for mesh and agglomerates
{
  log_info("compute_base_connectivity");
  //log_info("compute_base_connectivity"); //log_info("compute_base_connectivity");
  std::chrono::time_point<std::chrono::steady_clock> start, end;
  start = std::chrono::steady_clock::now();

  // e2n
  const size_t num_elem = e2n_cnt.size();
  e2n_dsp.resize(num_elem);
  idx_t tmp = 0;

  // #pragma omp parallel for
  compute_dsp(e2n_cnt, e2n_dsp); // linear
  //for(size_t eidx = 0 ; eidx < num_elem ; eidx++)
  //{
  //  e2n_dsp[eidx] = tmp; // might already be computed
  //  tmp += e2n_cnt[eidx];
  //}

  //#pragma omp parallel for
  for(size_t eidx = 0 ; eidx < num_elem ; eidx++)
  {
    // we use insertion sort here since it performs well on small sets
    insertion_sort(e2n_con.data() + e2n_dsp[eidx], e2n_cnt[eidx]); 
  }

  end = std::chrono::steady_clock::now();
  if (verbose)
    std::cout<<"e2n: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  //for(const auto &cnt : e2n_cnt) std::cout<<cnt<<" ";
  //std::cout<<std::endl;

  //for(const auto &dsp : e2n_dsp) std::cout<<dsp<<" ";
  //std::cout<<std::endl;

  //for(const auto &con : e2n_con) std::cout<<con<<" ";
  //std::cout<<std::endl;

  //log_info("n2e");
  // n2e (transpose)
  const size_t num_pts = *(std::max_element(e2n_con.begin(), e2n_con.end())) + 1;
  n2e_cnt.assign(num_pts, 0);
  n2e_dsp.assign(num_pts, 0);
  n2e_con.resize(e2n_con.size()); // always true?

  //#pragma omp parallel for
  for(size_t eidx = 0 ; eidx < num_elem ; eidx++) // distributed
  {
    const idx_t edsp = e2n_dsp[eidx];

    for(idx_t it = 0 ; it < e2n_cnt[eidx] ; it++)
    {
      const idx_t idx = e2n_con[edsp+it];
      //#pragma omp atomic update
      n2e_cnt[idx]++; // atomic
    }
  }

  // n2e_con size test
  //idx_t testval = 0;
  //for(const auto &it : n2e_cnt)
  //  testval += it;
  //std::cout<<"TESTVAL: "<<testval<<" "<<e2n_con.size()<<std::endl;

  compute_dsp(n2e_cnt, n2e_dsp); // linear
  n2e_cnt.assign(num_pts, 0); // set 0

  // block based? (each thread goes through all elements, only computes results for its idx block)
  for(idx_t eidx = 0 ; eidx < static_cast<idx_t>(e2n_cnt.size()) ; eidx++) // distributed
  {
    const idx_t edsp = e2n_dsp[eidx];

    for(idx_t it = 0 ; it < e2n_cnt[eidx] ; it++)
    {
      const idx_t idx = e2n_con[edsp+it];
      const idx_t dsp = n2e_dsp[idx];
      cnt_t& cnt = n2e_cnt[idx];
      n2e_con[dsp+cnt] = eidx;
      cnt++;
    }
  }

  end = std::chrono::steady_clock::now();
  if (verbose)
    std::cout<<"n2e: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  //for(const auto &cnt : n2e_cnt) std::cout<<cnt<<" ";
  //std::cout<<std::endl;

  //for(const auto &dsp : n2e_dsp) std::cout<<dsp<<" ";
  //std::cout<<std::endl;

  //for(const auto &con : n2e_con) std::cout<<con<<" ";
  //std::cout<<std::endl;

  //log_info("n2n");
  size_t n2n_con_sz = 0;//, n2n_max = 0;
  std::vector<std::set<idx_t>> n2n(num_pts);

  #pragma omp parallel for num_threads(16)
  for(size_t idx = 0 ; idx < num_pts ; idx++)
  {
    auto &iset = n2n[idx];
    const idx_t edsp = n2e_dsp[idx];

    for(cnt_t eit = 0 ; eit < n2e_cnt[idx] ; eit++)
    {
      const idx_t eidx = n2e_con[edsp+eit];
      const idx_t ndsp = e2n_dsp[eidx];

      for(cnt_t nit = 0 ; nit < e2n_cnt[eidx] ; ++nit)
      {
        const idx_t nidx = e2n_con[ndsp+nit];
        iset.insert(nidx);
      }
    }

    //n2n_con_sz += iset.size();
    //n2n_max = std::max(n2n_max, iset.size());
  }

  for (const auto &item : n2n)
    n2n_con_sz += item.size();

  //std::cout<<"n2n_con_sz: "<<n2n_con_sz<<" n2n_max: "<<n2n_max<<std::endl;

  end = std::chrono::steady_clock::now();
  if (verbose)
    std::cout<<"n2n prep: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  // n2n (either over idx or eidx)
  tmp = 0;
  n2n_cnt.assign(num_pts, 0);
  n2n_dsp.assign(num_pts, 0);
  n2n_con.resize(n2n_con_sz);// approximation, might fail, also 32 might be too low
  idx_t buf[1024], eit; // local in each loop if parallel

  // #pragma omp parallel for
  for(size_t idx = 0 ; idx < num_pts ; idx++)
  {
    const idx_t edsp = n2e_dsp[idx];
    n2n_cnt[idx] = 0;
    n2n_dsp[idx] = tmp;

    for(eit = 0 ; eit < n2e_cnt[idx] ; eit++)
    {
      const idx_t eidx = n2e_con[edsp+eit];
      const idx_t ndsp = e2n_dsp[eidx];

      if(eit % 2 == 0) // switch between buffers to avoid memcpy
      {
        set_union(n2n_con.data() + tmp, n2n_cnt[idx],
                  e2n_con.data() + ndsp, e2n_cnt[eidx],
                  buf, n2n_cnt[idx]);
      }
      else
      {
        set_union(buf, n2n_cnt[idx],
                  e2n_con.data() + ndsp, e2n_cnt[eidx],
                  n2n_con.data() + tmp, n2n_cnt[idx]);
      }
    }

    if(eit % 2 == 1)
    {
      memcpy(n2n_con.data() + tmp, buf, sizeof(idx_t)*n2n_cnt[idx]);
    }

    tmp += n2n_cnt[idx];
    // insert into n2n_con -> omp barrier
  }

  n2n_con.resize(tmp);

  end = std::chrono::steady_clock::now();
  if (verbose)
    std::cout<<"n2n: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;
}
// ----------------------------------------------------------------------------
void mesh_t::compute_geo()
{
  //log_info("compute_geo");
  geo.assign(6*e2n_cnt.size(), 0.0);
  vec3_t velo = {0.6, 0.6, 0.6};

  //#ifdef TW_OMP_EN
  //#pragma omp parallel num_threads(TW_OMP_NT)
  //#endif
  #pragma omp parallel for num_threads(16)
  for (size_t edx = 0 ; edx < e2n_cnt.size() ; ++edx)
  {
    const size_t pos = 6*edx; dbl3_t edg; dbl_t M[6];
    const idx_t dsp = e2n_dsp[edx];
    const idx_t x1 = e2n_con[dsp+0], x2 = e2n_con[dsp+1], 
                x3 = e2n_con[dsp+2], x4 = e2n_con[dsp+3];

    // how about computing square root before?!!!!!
    compute_velocity_tensor_f3(lon.data(), edx, velo, M);
    load_edge(x1, x2, xyz.data(), edg); geo[pos+0] = aMb_sym(edg, M, edg); // e12
    load_edge(x1, x3, xyz.data(), edg); geo[pos+1] = aMb_sym(edg, M, edg); // e13
    load_edge(x2, x3, xyz.data(), edg); geo[pos+2] = aMb_sym(edg, M, edg); // e23
    load_edge(x1, x4, xyz.data(), edg); geo[pos+3] = aMb_sym(edg, M, edg); // e14
    load_edge(x2, x4, xyz.data(), edg); geo[pos+4] = aMb_sym(edg, M, edg); // e24
    load_edge(x3, x4, xyz.data(), edg); geo[pos+5] = aMb_sym(edg, M, edg); // e34

    //if (edx == 167)
    //  printf("edx %ld: (%ld %ld %ld %ld) %f %f %f %f %f %f\n", edx, x1, x2, x3, x4, sqrt(geo[pos+0]), sqrt(geo[pos+1]), sqrt(geo[pos+2]), sqrt(geo[pos+3]), sqrt(geo[pos+4]), sqrt(geo[pos+5]));
      //printf("edx %ld: %f %f %f %f %f %f\n", edx, sqrt(geo[pos+0]), sqrt(geo[pos+1]), sqrt(geo[pos+2]), sqrt(geo[pos+3]), sqrt(geo[pos+4]), sqrt(geo[pos+5]));
  }
}

// partitioning -> n_elem entries, maps to agglomerates
// applied before compute_base_connectivity!
// shift everything according to partitioning
/*  void apply_partitioning(const std::vector<std::vector<idx_t>> &partitions) // e2n_partitions
{
  const std::vector<cnt_t> old_cnt(e2n_cnt.begin(), e2n_cnt.end());
  const std::vector<idx_t> old_dsp(e2n_dsp.begin(), e2n_dsp.end()), old_con(e2n_con.begin(), e2n_con.end());
  idx_t dsp = 0, eidx = 0;

  submesh_offsets.assign(partitions.size(), 0);
  //ghost_offsets.assign(N_partitions, 0);
  
  //for(size_t eidx = 0 ; eidx < n_cells ; eidx++)
  //{
    // maybe order based on additional information (connectivity/proximity)?
  //  const idx_t old_eidx = partitioning[eidx];
  //  e2n_cnt[eidx] = old_cnt[old_eidx];
  //  e2n_dsp[eidx] = dsp;
  //  memcpy(e2n_con.data() + dsp, old_con.data() + old_dsp[old_eidx], e2n_cnt[eidx]);
  //  dsp += e2n_cnt[eidx];
  //}

  // maybe order based on additional information (connectivity/proximity)?
  for(const auto &p : partitions)
  {
    for(const auto &old_eidx : p)
    {
      e2n_cnt[eidx] = old_cnt[old_eidx];
      e2n_dsp[eidx] = dsp;
      memcpy(e2n_con.data() + dsp, old_con.data() + old_dsp[old_eidx], e2n_cnt[eidx]);
      dsp += e2n_cnt[eidx];
      eidx++;
    }

    //e2n_cnt_submesh_offsets[it] = p.size();
    //e2n_dsp_submesh_offsets[it] = p.size();
    //e2n_con_submesh_offsets[it] = dsp;
  }

  // n2e still needs care since N_pts < N_pts_agg
  // for all agglomerates
  //   compute n2e and n2n
}*/

// ----------------------------------------------------------------------------
// mesh.etags.size() -> what if empty? -> all elements of mesh have same tag
// resize instead of assign?
int extract_submeshes2(const mesh_t &mesh, const std::vector<tag_t> &atags, std::vector<mesh_t> &submeshes,
                       std::vector<std::vector<idx_t>> &l2g, std::vector<idx_t> &g2l,
                       std::vector<std::vector<idx_t>> &n2a, std::vector<std::map<idx_t, idx_t>> &a2n,
                       std::vector<std::vector<idx_t>> &a2a,
                       std::vector<std::map<idx_t,std::vector<idx_t>>> &a2i,
                       //std::vector<std::vector<std::vector<ipair_t>>> &n2i,
                       std::vector<std::vector<ipair_t>> &n2i)
{
  log_info("extract_submeshes2"); //log_info("extract_submeshes2");
  std::chrono::time_point<std::chrono::steady_clock> start, end;

  // clear submeshes if allocated
  if(!submeshes.empty())
  {
    for(auto &submesh : submeshes)
      submesh.clear();

    submeshes.clear();
  }

  // convert atags to e2a, compute Naggs
  start = std::chrono::steady_clock::now();
  idx_t aidx = 0;
  std::vector<idx_t> e2a(mesh.etags.size());
  std::map<tag_t,idx_t> atag2aidx;

  for(size_t eidx = 0 ; eidx < mesh.etags.size() ; ++eidx)
  {
    const tag_t atag = atags[eidx];
    auto item = atag2aidx.find(atag);

    if(item != atag2aidx.end())
      e2a[eidx] = item->second;
    else
    {
      e2a[eidx] = aidx;
      atag2aidx[atag] = aidx++;
    }

    //e2a[eidx] = aidx;
  }

  const size_t Nagg = atag2aidx.size();

  // clear and resize n2a and a2n
  for (auto &item : n2a)
    item.clear();
  
  for (auto &item : a2n)
    item.clear();

  for (auto &item : a2a)
    item.clear();

  for (auto &item : a2i)
  {
    for (auto jtem : item)
      jtem.second.clear();

    item.clear();
  }

  for (auto &item : n2i)
    item.clear();

  n2i.clear();

  n2a.resize(mesh.n2e_cnt.size());
  a2n.resize(Nagg);
  a2a.resize(Nagg);
  a2i.resize(Nagg);
  n2i.resize(mesh.n2e_cnt.size());

  end = std::chrono::steady_clock::now();
  std::cout<<"prepare: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  // compute a2e and a2n (you could also compute a2a here ..)
  std::vector<idx_t> idx2(Nagg, 0);
  std::vector<std::vector<idx_t>> a2e(Nagg); // -> global -> etags.size

  submeshes.resize(Nagg);
  g2l.assign(mesh.n2e_cnt.size(), -1); // n2a
  l2g.assign(Nagg, {});

  for(idx_t eidx = 0 ; eidx < static_cast<idx_t>(mesh.etags.size()) ; ++eidx)
  {
    const idx_t aidx = e2a[eidx];
    const idx_t edsp = mesh.e2n_dsp[eidx];
    auto &amap = a2n[aidx];

    a2e[aidx].push_back(eidx);

    for(idx_t eit = 0 ; eit < mesh.e2n_cnt[eidx] ; ++eit)
    {
      // a2n
      const idx_t idx = mesh.e2n_con[edsp+eit];
      const auto &it = amap.find(idx);

      if(it == amap.end())
        amap[idx] = idx2[aidx]++;
    }
  }

  end = std::chrono::steady_clock::now();
  std::cout<<"a2e and a2n: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  // assemble submesh
  std::vector<std::set<idx_t>> a2a_tmp(Nagg);

  //#pragma omp parallel for
  for(idx_t aidx = 0 ; aidx < static_cast<idx_t>(Nagg) ; ++aidx)
  {
    const size_t Apts = a2n[aidx].size(), Aelm = a2e[aidx].size();
    const auto &amap = a2n[aidx];
    mesh_t &submesh = submeshes[aidx];
    idx_t eidx2 = 0, dsp2 = 0;

    submesh.e2n_cnt.resize(Aelm);
    submesh.e2n_dsp.resize(Aelm);
    submesh.e2n_con.resize(4*Aelm); // fix: determine size for different element types!

    submesh.etype.resize(Aelm);
    submesh.etags.resize(Aelm);
    submesh.xyz.resize(3*Apts);
    submesh.lon.resize(3*Aelm);
    submesh.geo.resize(6*Aelm);

    // assemble l2g
    l2g[aidx].assign(amap.size(), -1);
    //size_t jt = 0;
    //for(const auto &it : amap)
    //  l2g[aidx][jt++] = it.first;

    for(size_t it = 0 ; it < a2e[aidx].size() ; ++it)
    {
      const idx_t eidx = a2e[aidx][it];
      const idx_t edsp = mesh.e2n_dsp[eidx];

      submesh.etype[eidx2] = mesh.etype[eidx];
      submesh.etags[eidx2] = mesh.etags[eidx];
      submesh.e2n_cnt[eidx2] = mesh.e2n_cnt[eidx];
      submesh.e2n_dsp[eidx2] = dsp2;
      //submesh.geo[eidx2] = mesh.geo[eidx];
      //memcpy(submesh.lon.data() + 3*idx2, mesh.lon.data() + 3*idx, 3*sizeof(dbl_t));
      memcpy(submesh.lon.data() + 3*eidx2, mesh.lon.data() + 3*eidx, 3*sizeof(dbl_t));
      //memcpy(submesh.geo.data() + 6*eidx2, mesh.geo.data() + 6*eidx, 6*sizeof(dbl_t));

      for(idx_t eit = 0 ; eit < mesh.e2n_cnt[eidx] ; ++eit)
      {
        const idx_t idx = mesh.e2n_con[edsp+eit];
        const idx_t idx2 = amap.at(idx);
        l2g[aidx][idx2] = idx;
        submesh.e2n_con[dsp2+eit] = idx2;
        memcpy(submesh.xyz.data() + 3*idx2, mesh.xyz.data() + 3*idx, 3*sizeof(dbl_t));
        //memcpy(submesh.lon.data() + 3*idx2, mesh.lon.data() + 3*idx, 3*sizeof(dbl_t));
      }

      dsp2 += submesh.e2n_cnt[eidx2];
      eidx2++;
    }

    // compute n2a
    for (const auto &it : amap)
      n2a[it.first].push_back(aidx); // idx -> global, idx2 -> local

    submesh.compute_base_connectivity(false);
    //save_vtk("/home/tom/dev/tw/test/submesh_" + std::to_string(aidx) + ".vtk", submesh, {}, {}, {}, {});
  }

  end = std::chrono::steady_clock::now();
  std::cout<<"assemble : "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  // a2a and a2i + n2i
  std::cout<<"n2a size: "<<n2a.size()<<", a2e size: "<<a2e.size()<<", e2a size: "<<e2a.size()<<" a2a_tmp size: "<<a2a_tmp.size()<<"\n";

  for (idx_t adx = 0 ; adx < static_cast<idx_t>(Nagg) ; ++adx)
  {
    for (const auto &item : a2n[adx])
    {
      const auto &jtem = n2a.at(item.first);

      for (const auto &nadx : jtem)
        a2a_tmp[adx].insert(nadx);

      if (jtem.size() < 2)
        continue;

      for (const auto &nadx : jtem)
      {
        if (adx != nadx)
          a2i[adx][nadx].push_back(item.second);
      }
    }
  }

  for (idx_t adx = 0 ; adx < static_cast<idx_t>(Nagg) ; ++adx)
  {
    a2a[adx].assign(a2a_tmp[adx].begin(), a2a_tmp[adx].end());
    //std::cout<<"  "<<a2a[adx].size()<<", "<<a2a_tmp[adx].size()<<"\n";
  }

  end = std::chrono::steady_clock::now();
  std::cout<<"a2a and a2i: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  /*
  for (idx_t adx = 0 ; adx < static_cast<idx_t>(Nagg) ; ++adx)
  {
    n2i[adx].resize(submeshes[adx].n2e_cnt.size());
    //std::cout<<"Nn: "<<submeshes[adx].n2e_cnt.size()<<"\n";

    for (const auto &item : a2i[adx])
    {
      const idx_t nadx = item.first;

      //std::cout<<"adx"<<adx<<": "<<a2i[nadx][adx].size()<<" nadx"<<nadx<<": "<<a2i[adx][nadx].size()<<"\n";
      assert(a2i[nadx][adx].size() == a2i[adx][nadx].size());

      for (size_t it = 0 ; it < item.second.size() ; ++it)
      {
        const idx_t nvtx = a2i[nadx][adx][it];
        const idx_t vtx = a2i[adx][nadx][it];
        //std::cout<<"  "<<vtx<<" "<<nvtx<<"\n";
        n2i[adx][vtx].push_back({nvtx, nadx});
      }
    }
  }
  */

  for (idx_t vtx = 0 ; vtx < static_cast<idx_t>(mesh.n2e_cnt.size()) ; ++vtx)
  {
    for (const idx_t adx : n2a[vtx])
    {
      const idx_t ldx = a2n[adx][vtx];
      n2i[vtx].push_back({ldx, adx});
    }
  }

  end = std::chrono::steady_clock::now();
  std::cout<<"n2i: "<<std::chrono::duration<double>(end-start).count()<<"s\n";
  start = end;

  // inter2:
  /*for (idx_t adx = 0 ; adx < static_cast<idx_t>(Nagg) ; ++adx)
  {
    mesh_t &submesh = submeshes[adx];
    i2i[adx].resize(submesh.n2e_cnt.size());

    for (idx_t vtx = 0 ; vtx < submesh.n2e_cnt.size() ; ++vtx)
    {
      for (const auto &nadx : n2a[vtx])
      {
        i2i[adx][vtx] = {nadx, nvtx};
      }
    }
  }*/

  for (auto &submesh : submeshes)
    submesh.compute_geo();

  return submeshes.size();
}
// ----------------------------------------------------------------------------
// e2a is always complete, no virtual agglomerates neccessary
/*void arrange_mesh(const std::vector<tag_t> &e2a, mesh_t &mesh)
{
  assert(e2a.size() == mesh.etags.size());

  std::vector<std::vector<idx_t>> a2n(Nagg);

  for(size_t eidx = 0 ; eidx < mesh.etags.size() ; ++eidx)
  {
    const idx_t aidx = e2a[eidx];
    const idx_t dsp = mesh.e2n_dsp[eidx];

    for(size_t it = 0 ; it < mesh.e2n_cnt[it] ; ++it)
    {
      const idx_t idx = mesh.e2n_con[dsp+it];
      a2n[aidx].push_back(idx);
    } // split into neighbours and non-neighbours?
    // a2a und dann set union?
  }

  for(size_t idx = 0 ; idx < mesh.n2e_cnt.size() ; ++idx)
  {
    const idx_t edsp = mesh.n2e_dsp[idx];

    for(size_t eit = 0 ; eit < mesh.n2e_cnt[idx] ; ++eit)
    {
      const idx_t eidx = mesh.n2e_con[edsp+eit];
      const idx_t aidx = e2a[eidx];
      // add to alist
    }

    //if(alist contains only aidx) // same aidx
    //{
      // add to inner region
    //  a2n[aidx].push_back(idx);
    //}
    //else
    //{
      // add to border region
    //  for(aidx : alist) // add each combination, lower idx first
    //  {
    //    for(ait = aidx ; ait < amax ; ...)
    //    {
          // I[..,..].push_back(idx);
    //    }
    //  }
    //}
  }

  // find all unique agglomerate ids
  //std::set<tag_t> atags(e2a.begin(), e2a.end()); // not very efficient but short
  //std::vector<tag_t> atags(e2a.begin(), e2a.end());
  //sort(atags.begin(), atags.end());
  //atags.erase(unique(atags.begin(), atags.end()), atags.end());
  // does not return agg sizes!

  std::vector<tag_t> tag_map;
  std::map<tag_t,std::vector<idx_t>> a2e_map;
  std::map<tag_t,std::set<idx_t>> a2n_map;
  std::map<tag_t,std::vector<idx_t>> a2n_vec;

  for(size_t eidx = 0 ; eidx < e2a.size() ; ++eidx)
  {
    const idx_t aidx = e2a[eidx];
    a2e_map[aidx].push_back(eidx);

    const idx_t dsp = mesh.e2n_dsp[eidx];

    for(size_t it = 0 ; it < mesh.e2n_cnt[it] ; ++it) // binary heap insert?
      a2n_map[aidx].insert(mesh.e2n_con[dsp+it]);
  } // -> each entry is already sorted after this loop!

  for(const auto &it : a2e_map)
  {
    tag_map.push_back(it.first);
    a2n_vec[it.first].assign(a2n_map[it.first].begin(), a2n_map[it.first].end());
  }

  const size_t N_agg = tag_map.size();
  std::vector<cnt_t> cnt_new(mesh.e2n_cnt.size());
  std::vector<idx_t> dsp_new(mesh.e2n_dsp.size());
  std::vector<idx_t> con_new(mesh.e2n_con.size());

  for(const auto &it : a2e_map)
  {
    idx_t new_idx = 0, new_eidx = 0, new_dsp = 0;

    for(auto &eidx : it.second)
    {
      const cnt_t cnt = mesh.e2n_cnt[eidx];
      const idx_t dsp = mesh.e2n_dsp[eidx];

      cnt_new[new_eidx] = cnt;
      dsp_new[new_eidx] = new_dsp;

      for(size_t jt = 0 ; jt < cnt ; ++jt)
      {
        const idx_t idx = mesh.e2n_con[dsp+jt];
        con_new[new_dsp+jt] = idx_map[idx];
      }

      new_eidx++;
      new_dsp += cnt;
    }
  }

  mesh.e2n_cnt.assign(cnt_new.begin(), cnt_new.end());
  mesh.e2n_dsp.assign(dsp_new.begin(), dsp_new.end());
  mesh.e2n_con.assign(con_new.begin(), con_new.end());
  mesh.compute_base_connectivity();
}*/
// ----------------------------------------------------------------------------
void extract_submeshes(const mesh_t &mesh, const std::vector<tag_t> &e2a, std::vector<mesh_t> &submeshes)
{
  log_info("extract_submeshes"); //log_info("extract_submeshes");

  assert(mesh.etags.size() == e2a.size());

  std::vector<tag_t> tag_map;
  std::map<tag_t,std::vector<idx_t>> a2e_map;

  for(size_t eidx = 0 ; eidx < e2a.size() ; ++eidx)
  {
    const idx_t aidx = e2a[eidx];
    a2e_map[aidx].push_back(eidx);
    // a2n_map?
  }

  for(const auto &it : a2e_map)
  {
    tag_map.push_back(it.first);
  }

  const size_t Nagg = a2e_map.size();
  submeshes.resize(Nagg);

  // extract connections
  #ifdef TW_OMP_EN
  #pragma omp parallel num_threads(TW_OMP_NT)
  #endif
  {
    #ifdef TW_OMP_EN
    int tid = omp_get_thread_num(), N_threads = omp_get_num_threads();
    #else
    int tid = 0, N_threads = 1;
    #endif

    const int begin = (tid)*Nagg/N_threads;
    const int end   = (tid+1)*Nagg/N_threads; // last -> Nagg % N_threads -> not evenly distributed

    std::cout<<begin<<" "<<end<<std::endl;

    // parallel?
    //for(tag_t aidx = aStart ; aidx < aEnd ; ++aidx)
    for(int it = begin ; it < end ; ++it)
    {
      const tag_t aidx = tag_map[it];
      mesh_t &submesh = submeshes[it];
      const size_t Ne = a2e_map.at(aidx).size();

      std::cout<<"Ne: "<<Ne<<std::endl;

      submesh.e2n_cnt.resize(Ne);
      submesh.e2n_dsp.resize(Ne);
      submesh.e2n_con.resize(4*Ne); // tets only for now ..

      idx_t eidx_new = 0;
      idx_t dsp_new = 0;

      std::cout<<"iter through elems (tag = "<<aidx<<"): "<<std::endl;

      for(const auto &eidx : a2e_map[aidx])
      {
        const cnt_t cnt_new = mesh.e2n_cnt[eidx];
        submesh.e2n_dsp[eidx_new] = dsp_new;
        memcpy(submesh.e2n_con.data() + dsp_new, mesh.e2n_con.data() + mesh.e2n_dsp[eidx], cnt_new*sizeof(idx_t));
        // idx -> re-indexing
        submesh.e2n_cnt[eidx_new] = cnt_new;
        dsp_new += cnt_new;
        eidx_new++;
      }

      for(const auto &cnt : submesh.e2n_cnt) std::cout<<cnt<<" ";
      std::cout<<std::endl;

      for(const auto &dsp : submesh.e2n_dsp) std::cout<<dsp<<" ";
      std::cout<<std::endl;

      for(const auto &con : submesh.e2n_con) std::cout<<con<<" ";
      std::cout<<std::endl;

      submesh.compute_base_connectivity(false);

      assert(false);
    }
  }
}

// ----------------------------------------------------------------------------
ek_data::ek_data(const mesh_t &mesh) : mesh(mesh), 
  Npts(mesh.n2e_cnt.size()), Nelm(mesh.e2n_cnt.size()), Ndat(mesh.xyz.size()/3),
  phi(Ndat), states(Ndat), di(Ndat), curv(Ndat), act(Ndat), apd(Ndat)
{

}
// ----------------------------------------------------------------------------
ek_data::~ek_data()
{
  phi.clear();
  states.clear();
  di.clear();
  //pcl.clear();
  curv.clear();

  for(size_t it = 0 ; it < Ndat; ++it)
  {
    act[it].clear();
    apd[it].clear();
  }
}
void ek_data::reset_states()
{
  phi.assign(Ndat, DBL_INF);
  states.assign(Ndat, ek_far);
  di.assign(Ndat, DBL_INF);
  curv.assign(Ndat, 0.);
  
  for(size_t it = 0 ; it < Ndat ; ++it)
  {
    act[it].clear();
    apd[it].clear();
  }

  act.assign(Ndat, {});
  apd.assign(Ndat, {});
}

}