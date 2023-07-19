/**
 * Copyright (c) 2022 Thomas Schrotter. All rights reserved.
 * @file main.cpp
 * @author Thomas Schrotter
 * @version 0.0.0
 * @date 2022-12
 */


// STRUCTURE ==================================================================
// 
//        main
// 
// TODO =======================================================================
// main:
// - for loops -> signed, check what compiler creates, avoid implicit casts!
// - cmake at the end (https://cliutils.gitlab.io/modern-cmake/chapters/basics/example.html)
// active:
// - virtual filesystem?
// - make list of features (read mesh, spawn it on screen, texture?)
// - move imgui to lib, add lib path, custom backend
// - cuda valgrind pseudo leaks ..
// - tests should be top level ..
// - rule: each agg has only one etag! -> constant velocity per agg
// - aggl shared memory allocation
// - compute_connectivity for both mesh and aggl?
// - valgrind cuda_print_info!
// - revised fim + cuda fim
// future:
// - json cuda compability
// - high performance umap and uset
// - cuda vector + functionality?
// optional:
// - argument parser
// - structure ascii generator based on project files (python or cpp?)
// CODING STANDARD ============================================================
// - utils file with base things like exceptions
// - performance functions (which are called often) -> inline, else cpp
// - unit test in each cpp file, as independent as possible (functions -> external_test(), classes -> class.internal_test(), enable using macro? strong references like python always possible?)
// - templates at the end when everything is finished
// - struct = data + functions? not always ..
// - .h .cpp .cu

// Convention:
// .cu -> kernels only, .cpp + .h files which use kernels (wrappers)

// .cu -> kernels + wrappers

// Testing:
// internal: own test member function for internal testing? -> tests single functionality (enable flag if debug)
//     +
// external: tests combined whole functionality

// remarks on fim efficiency and memory footprint:
// footprint should be low, paper avoids computing m during fim
// -> can i too? -> M changes with di, apd and curvature ... 
// -> BUT if we employ averaging based on di, apd and curvature, we can precompute M
// -> (ALL POSSIBLE COMBINATIONS)

// bit fields: https://en.cppreference.com/w/cpp/language/bit_field
// c++ Guide: https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines
// exceptions: https://learn.microsoft.com/en-us/cpp/cpp/errors-and-exception-handling-modern-cpp?view=msvc-160
// imgui: https://github.com/ocornut/imgui

// CODE =======================================================================

// cufim-nvlink includes
#include "core/types.h"
#include "core/logging.h"
#include "core/cuda_base.h"
#include "io/vtk.h"
#include "eikonal/cuda_fim.h"

// libs (Copyright (c) 2018 Pranav Srinivas Kumar)
#include "../lib/argparse.hpp"

// stdlib includes
#include <string>
#include <vector>
#include <iostream>

namespace tw { // =============================================================
// ----------------------------------------------------------------------------
double run(const std::string &mesh_path, const std::string &part_path,
           const std::string &plan_path, const std::string &out_path,
           const int num_reps)
{
  // load mesh
  mesh_t mesh;
  load_vtk(mesh_path, mesh); // load_carp_txt(); load_mesh();
  read_vector_txt(part_path, mesh.atags); // load_part();
  mesh.compute_base_connectivity(true);

  ek_data ekdata(mesh);
  load_options("data/default.cfg", ekdata.opts);
  load_plan(plan_path, ekdata);
  //ekdata.opts.print();

  std::vector<std::vector<dbl_t>> act_ms, apd_ms;
  CudaSolverFIM solver(ekdata);
  const double avg_exec_time_s = solver.solve(act_ms, apd_ms, out_path, num_reps);//"cuda_fim_heart.vtk"

  return avg_exec_time_s;
}
// ----------------------------------------------------------------------------
void run_unit_tests()
{
  //const std::string mesh_path = "/home/tom/workspace/masc-experiments/heart2/S62_500um.vtk";
  //const std::string part_path = "/home/tom/workspace/masc-experiments/heart2/S62_500um.29135.tags";

  /*
  cudaSetDevice(0);                   // Set device 0 as current
  float* p0;
  size_t size = 1024 * sizeof(float);
  cudaMalloc(&p0, size);              // Allocate memory on device 0
  MyKernel<<<1000, 128>>>(p0);        // Launch kernel on device 0
  cudaSetDevice(1);                   // Set device 1 as current
  cudaDeviceEnablePeerAccess(0, 0);   // Enable peer-to-peer access
                                      // with device 0

  // Launch kernel on device 1
  // This kernel launch can access memory on device 0 at address p0
  MyKernel<<<1000, 128>>>(p0);
  */

  test_logging();
  //cuda_fim_test2();
  //cuda_fim_test3();

  //const dim3 block(threads_per_block);
  //const dim3 grid(1);//(kNum + block.x - 1)/block.x
  //void *params[] = {&d_smsh};//&d_values
  //checkCudaErrors(cudaLaunchCooperativeKernel((void*)coop_kernel_test, grid, block, params));
  //return;

  /*
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
  */

#if true
  run("data/ut_mesh/ut_mesh.vtk",       "data/ut_mesh/ut_mesh.tags", 
      "data/ut_mesh/ut_mesh.plan.json", "data/output/cufim-ut_mesh-out.vtk", 1);
#elif
  run("data/heart/S62.vtk", "data/heart/S62.7560.tags", 
      "data/heart/S62.plan.json", "data/output/cufim-heart-out.vtk", 1);
#endif
}
// ----------------------------------------------------------------------------
} // MAIN =====================================================================
// ----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int exit_code = EXIT_SUCCESS;
  // make clean && make && ./cufim-nvlink -mesh=data/ut_mesh/ut_mesh.vtk -part=data/ut_mesh/ut_mesh.tags -plan=data/ut_mesh/ut_studio_plan.json -outfile=data/output/out.vtk
  // ./cufim-nvlink --mesh=/home/tom/workspace/masc-experiments/heart2/S62_1200um.vtk --part=/home/tom/workspace/masc-experiments/heart2/S62_1200um.7560.tags --plan=/home/tom/workspace/masc-experiments/heart2/S62_1200um.plan.json --odir=data/output/cufim-heart_1200um-out.vtk
  // ./cufim-nvlink --mesh=/home/schrottert/projects/heart-meshes/S62_1200um.vtk --part=/home/schrottert/projects/heart-meshes/S62_1200um.7560.tags --plan=/home/schrottert/projects/heart-meshes/S62_1200um.plan.json --odir=data/output/cufim-heart_1200um-out.vtk

  argparse::ArgumentParser program("cufim-nvlink", "0.5.0"); tw::print_head();
  program.add_description("Forward a thing to the next member.");
  program.set_assign_chars("=");

  program.add_argument("-m", "--mesh").help("mesh file (vtk or carp_txt)").metavar("STR");
  program.add_argument("-p", "--part").help("metis partitioning file    ").metavar("STR");
  program.add_argument("-s", "--plan").help("plan file                  ").metavar("STR");
  program.add_argument("-o", "--odir").help("output directory           ").metavar("STR");

  program.add_argument("-i", "--info").help("outputs gpu information").default_value(false).implicit_value(true);
  program.add_argument("-t", "--test").help("run unit test configuration").default_value(false).implicit_value(true);
  program.add_argument("-r", "--reps").help("set number of runs/repetitions").default_value(1).metavar("INT").scan<'i', int>();;
  program.add_argument("-np", "--num_threads").help("specifies how many threads are used").default_value(8).metavar("INT").scan<'i', int>();;
  
  //program.add_argument("-v", "--verbose").help("increases output verbosity").default_value(false).implicit_value(true);
  // gpuids (default 0) 0,1,2,3

  program.add_epilog("Possible things include betingalw, chiz, and res.");

  try
  {
    program.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err)
  {
    std::cerr<<err.what()<<std::endl;
    std::cerr<<program;
    return EXIT_FAILURE;
  }

  try
  {
    tw::cuda_info cudainfo;
    const int num_reps = program.get<int>("--reps");

    if (program.is_used("--help"))
      std::cout<<program<<std::endl;
    else if (program.is_used("--info"))
      cudainfo.print();
    else if (program.is_used("--test"))
      tw::run_unit_tests();
    else if (num_reps < 1)
      std::cout<<"invalid number of reps!\n";
    else if (program.is_used("--mesh") && program.is_used("--part") && program.is_used("--plan") && program.is_used("--odir"))
    {
      const double avg_exec_time_s = tw::run(program.get("--mesh"), program.get("--part"), program.get("--plan"), program.get("--odir"), num_reps);
      printf("Average execution time for %d runs: %f s\n", num_reps, avg_exec_time_s);
    }
    else
      std::cout<<program<<std::endl;
  } 
  catch(const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    exit_code = EXIT_FAILURE;
  }

  checkCudaErrors(cudaDeviceReset());

  return exit_code;
}
// ----------------------------------------------------------------------------
// ============================================================================
