/**
 * Copyright (c) 2022 Thomas Schrotter. All rights reserved.
 * @file types.h -> structs.h
 * @brief Provides fundamental types and structs.
 * @author Thomas Schrotter
 * @version 0.0.0
 * @date 2022-12
 */
// CODE =======================================================================
#ifndef TW_TYPES_H_
#define TW_TYPES_H_

// tw includes
//#include "io_base.h"

// stdlib includes
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include <iostream>
#include <map>
#include <set>
#include <chrono>
#include <unordered_set>

#include "excepts.h"
#include "sorts.h"
#include "logging.h"

#define DBL_INF __DBL_MAX__

//#if __cplusplus >= 201703L
//#define EK_CPP17
//#endif

// agg multistates (1 byte) [use counters for each state? -> yields distribution]
#define EKMS_FAR 0x1
#define EKMS_NARROW 0x2
#define EKMS_FROZEN 0x4

namespace tw
{

typedef int64_t idx_t;
typedef int32_t tag_t;
//typedef int16_t cnt_t;
typedef int16_t cnt_t;
typedef double  dbl_t;
//typedef float dbl_t;

struct dbl3_t
{
  dbl_t x, y, z;

public:
  dbl3_t() {}
  dbl3_t(const dbl_t x, const dbl_t y, const dbl_t z) : 
    x(x), y(y), z(z) {}
  friend dbl3_t operator+(const dbl3_t &v1, const dbl3_t &v2)
  {
    return dbl3_t(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z);
  }

  friend dbl3_t operator-(const dbl3_t &v1, const dbl3_t &v2)
  {
    return dbl3_t(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z);
  }

  friend dbl3_t operator*(const dbl3_t &v, dbl_t scalar)
  {
    return dbl3_t(v.x*scalar, v.y*scalar, v.z*scalar);
  }

  friend dbl3_t operator*(dbl_t scalar, const dbl3_t &v)
  {
    return dbl3_t(v.x*scalar, v.y*scalar, v.z*scalar);
  }
};
typedef dbl3_t vec3_t;


typedef unsigned char ek_multistate;

enum ek_state {ek_far, ek_neig, ek_narrow, ek_next, ek_frozen, ek_tail, ek_block};
enum ek_method {ek_fmm, ek_sfmm, ek_pfmm, ek_fim, ek_fsm, ek_lsm, ek_ref};
enum ek_solve_type {Lin, Vol};//Sur oVol -> obtuse cases, ... or flag
enum elm_e { Tetra };

struct ipair_t
{
  idx_t vtx;
  idx_t adx;
};

template<class T>
struct ek_coeff
{
  // tet
  T a, b, c, d, e, f;

  // tri
  T a1, a2, a3, b1, b2, b3, c1, c2, c3;

  // line
  T l1, l2, l3;
};

struct ek_event // active/blocked
{
  int id, active = 0;
  dbl_t tstart = 0., dur = 0.;
  std::vector<idx_t> vtx;
  //size_t N;
  //idx_t *vtx;

  ek_event()
  {

  }

  ek_event(const int &id, const int active, const dbl_t tstart, const dbl_t dur,
                const std::vector<idx_t> &vtx) : id(id), active(active), tstart(tstart),
                dur(dur), vtx(vtx)
  {
  
  }
};

// https://en.cppreference.com/w/cpp/algorithm/set_union
// todo: same for intersection, move to vector/container ops
template<typename T1, typename T2> inline
void set_union(const T1* A, const T2 sA, const T1* B, const T2 sB, 
               T1* Res, T2& sR)
{
  const T1 *Ae = A + sA, *Be = B + sB;
  
  for(sR = 0 ; A != Ae ; ++Res, ++sR)
  {
    if(B == Be)
    {
      memcpy(Res, A, sizeof(T1)*(Ae - A));
      sR += (Ae - A);
      return;
    }

    if(*B < *A) *Res = *B++;
    else
    {
      *Res = *A;
      if(!(*A < *B)) ++B;
      ++A;
    }
  }

  memcpy(Res, B, sizeof(T1)*(Be - B));
  sR += (Be - B);
}






inline 
dbl_t aMb_sym(const dbl3_t &a, const dbl_t *M, const dbl3_t &b)
{
  return (M[0]*a.x + M[1]*a.y + M[2]*a.z)*b.x + 
         (M[1]*a.x + M[3]*a.y + M[4]*a.z)*b.y + 
         (M[2]*a.x + M[4]*a.y + M[5]*a.z)*b.z;
}
// --------------------------------------------------------------------------
inline 
void compute_velocity_tensor_f3(const dbl_t *lon, const idx_t edx, const vec3_t &v, dbl_t *M)
{
  const dbl_t* fdat = lon + edx*3;
  const dbl_t f0 = fdat[0], f1 = fdat[1], f2 = fdat[2];
  const dbl_t fa = f0*f0, fb = f0*f1, fc = f0*f2;
  const dbl_t fd = f1*f1, fe = f1*f2, ff = f2*f2;
  const dbl_t vf = 1./v.x/v.x, vs = 1./v.y/v.y;//, vn = 1./v.z/v.z;

  M[0] = vf*fa + vs*(1. - fa);
  M[1] = (vf - vs)*fb;
  M[2] = (vf - vs)*fc;
  M[3] = vf*fd + vs*(1. - fd); // M4
  M[4] = (vf - vs)*fe;         // M5
  M[5] = vf*ff + vs*(1. - ff); // M8
}
// --------------------------------------------------------------------------
inline 
void compute_velocity_tensor_f6(const dbl_t *lon, const idx_t edx, const vec3_t &v, dbl_t *M)
{
  const dbl_t *fdat = lon + edx*6;
  const dbl_t *sdat = fdat+3;
  const dbl_t f0 = fdat[0], f1 = fdat[1], f2 = fdat[2];
  const dbl_t s0 = sdat[0], s1 = sdat[1], s2 = sdat[2];
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
inline 
void load_edge(const idx_t vtx, const idx_t ndx, const dbl_t *xyz, dbl3_t &eAB)
{
  const dbl_t *xA = xyz + ndx*3;
  const dbl_t *xB = xyz + vtx*3;
  eAB = {xB[0] - xA[0], xB[1] - xA[1], xB[2] - xA[2]}; // B - A
}

// data structs ===============================================================
// ----------------------------------------------------------------------------
// https://arxiv.org/pdf/1205.3081.pdf
struct mesh_t
{
  size_t n_points, n_cells, n_edges;

  // node data
  std::vector<dbl_t> xyz;
  std::vector<dbl_t> lon;
  std::vector<dbl_t> geo;

  // cell data / element data
  std::vector<tag_t> atags;
  std::vector<tag_t> etags;
  std::vector<elm_e> etype;
  //std::vector<dbl_t> dir;

  // agglomerates
  //std::vector<agglomerate_t> aggs;
  // a2n, n2a, a2a? -> agglomerates
  // unit [m, mm, um, ...]

  // topology
  std::vector<cnt_t> e2n_cnt, n2e_cnt, n2n_cnt;
  std::vector<idx_t> e2n_dsp, n2e_dsp, n2n_dsp;
  std::vector<idx_t> e2n_con, n2n_con, n2e_con;
  //size_t e2n_max, n2e_max, n2n_max;

  void clear();
  void compute_dsp(const std::vector<cnt_t>& cnt, std::vector<idx_t>& dsp);
  void compute_base_connectivity(const bool verbose);
  void compute_geo();
  void compute_edges();

  ~mesh_t() { clear(); }
};

//struct multimesh_t?

// ----------------------------------------------------------------------------
struct ek_options
{
  int verbose = 0;    // toggles verbose output
  int rent_en = 1;    // toggles reentrant (depricated, choose based on input)
  int apdr_en = 1;    // toggles apd restitution
  int  cvr_en = 1;    // toggles cv restitution
  int curv_en = 0;    // toggles curvature computation
  int diff_en = 0;    // toggles diffusion computation

  // parallelization flags -> single, omp, cuda -> makefile option
  //int gpuc_en = 0;    // toggles gpu computation if available

  int method_id = 0;  // selects eikonal method    ek_mth -> [ek_fmm, ek_fim, ek_lsm, ek_pfmm]
  int solvetype = 2;  // selects solver complexity ek_svt -> [ek_1D, ek_2D, ek_3D]
  std::string mesh_id = "";

  dbl_t dt = 1.;    // defines time step size
  size_t n_steps = 3; // defines number of steps (tend?)
  // N_steps

  ek_options(const int verbose=0, const int rent_en=1, const int apdr_en=1,
             const int cvr_en=1, const int curv_en=0, const int diff_en=0,
             const int method_id=0, const int solvetype=2, const std::string mesh_id="", 
             const dbl_t dt=1., const size_t n_steps=0) : 
    verbose(verbose), rent_en(rent_en), apdr_en(apdr_en), cvr_en(cvr_en), 
    curv_en(curv_en), diff_en(diff_en), method_id(method_id), solvetype(solvetype), 
    mesh_id(mesh_id), dt(dt), n_steps(n_steps) {}

  void print() const
  {
    printf("EK options:\n  verbose: %d\n  rent_en: %d\n  apdr_en: %d\n  cvr_en: %d\n  curv_en: %d\n  diff_en: %d\n  method_id: %d\n  solvetype: %d\n  dt: %lfus\n  n_steps: %lu\n",
    verbose, rent_en, apdr_en, cvr_en, curv_en, diff_en, method_id, solvetype, dt, n_steps);
  }
};
// ----------------------------------------------------------------------------
// holds various properties per region
struct region_property // ek_region_properties
{
  int id;
  std::vector<tag_t> tags;
  int active;
  vec3_t velo;

  region_property() {}
  region_property(const int id, const std::vector<tag_t> &tags,
                  const int active, const vec3_t &velo) : 
                  id(id), tags(tags), active(active), velo(velo) {}
};

// ----------------------------------------------------------------------------
// cpu side, consider doing this on gpu side?
void extract_submeshes(const mesh_t &mesh, const std::vector<tag_t> &e2a, std::vector<mesh_t> &submeshes);
int extract_submeshes2(const mesh_t &mesh, const std::vector<tag_t> &atags, std::vector<mesh_t> &submeshes,
                       std::vector<std::vector<idx_t>> &l2g, std::vector<idx_t> &g2l,
                       std::vector<std::vector<idx_t>> &n2a, std::vector<std::map<idx_t, idx_t>> &a2n,
                       std::vector<std::vector<idx_t>> &a2a,
                       std::vector<std::map<idx_t,std::vector<idx_t>>> &a2i,
                       std::vector<std::vector<ipair_t>> &n2i);

/*
struct agg_mesh_t
{
  // ravel data for gpu loads?
  mesh_t mesh;
  //std::vector<agg> aggs;

  std::vector<cnt_t> a2e_cnt; // cuda vector?
  std::vector<idx_t> a2e_dsp;
  std::vector<idx_t> a2e_con;

  std::vector<cnt_t> e2a_cnt;
  std::vector<idx_t> e2a_dsp;
  std::vector<idx_t> e2a_con;

  std::vector<cnt_t> a2a_cnt;
  std::vector<idx_t> a2a_dsp;
  std::vector<idx_t> a2a_con;

  void clear()
  {
    a2e_cnt.clear();
    a2e_dsp.clear();
    a2e_con.clear();

    e2a_cnt.clear();
    e2a_dsp.clear();
    e2a_con.clear();

    a2a_cnt.clear();
    a2a_dsp.clear();
    a2a_con.clear();
  }

  ~agg_mesh_t() { clear(); }
};

// copy: -> void* , sizeof(agglomerate)? heap not aligned ..
// agglomerate_mesh -> holds mesh + agglomerates
struct agglomerate // located on host and device
{
  size_t N_vtx, N_elm;
  dbl_t *xyz;
  tag_t *etags;
  elm_e *etype;

  std::vector<cnt_t> n2n_cnt;
  std::vector<idx_t> n2n_dsp;
  std::vector<idx_t> n2n_con;
  size_t n2n_max;

  std::vector<cnt_t> n2e_cnt;
  std::vector<idx_t> n2e_dsp;
  std::vector<idx_t> n2e_con;
  size_t n2e_max;

  // base connectivity information
  cnt_t *e2n_cnt;
  std::vector<idx_t> *e2n_dsp;
  std::vector<idx_t> *e2n_con;
};

*/

struct ek_data
{
  const mesh_t &mesh;
  size_t Npts, Nelm, Ndat;
  //int Nint = -1;
  //dbl_t dt = 0.0;
  //size_t n_steps = 0;

  ek_options opts;
  std::vector<ek_event> events;
  std::vector<region_property> rprops;

  dbl_t wavefront_width, phi_start;

  std::vector<dbl_t>  phi;      // activation times
  std::vector<ek_state> states; // state of node
  std::vector<dbl_t>  di;       // diastolic interval
  std::vector<dbl_t>  curv;     // wavefront curvature

  std::vector<std::vector<dbl_t>> act, apd; // const size

  ek_data(const mesh_t &mesh);
  ~ek_data();
  void reset_states();
};

} // namespace tw
#endif // TW_TYPES_H_