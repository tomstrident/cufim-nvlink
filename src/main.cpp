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

//#define STB_IMAGE_IMPLEMENTATION
//#define TINYOBJLOADER_IMPLEMENTATION
//#define NK_IMPLEMENTATION
//#include "nuklear.h"

// cufim-nvlink includes
#include "core/types.h"
#include "io/vtk.h"
//#include "eikonal/fim.h"
#include "core/cuda_base.h"
#include "eikonal/cuda_fim.h"
//#include "eikonal/cuda_fim2.h"

#include "core/logging.h"

// stdlib includes
#include <string>
#include <vector>
#include <iostream>

namespace tw { // =============================================================

/*
struct teststruct
{
  int N = 1337;
  const bool shallow;
  double *data=nullptr;
  teststruct() : shallow(false) { std::cout<<"const: "<<shallow<<"\n"; }
  ~teststruct() { std::cout<<"dest: "<<shallow<<"\n"; }
  teststruct(const teststruct &other) : shallow(true) { std::cout<<"copy: "<<shallow<<"\n"; }
};

void testfunc(teststruct sadf)
{
  std::cout<<sadf.N<<"\n";
}*/

// ----------------------------------------------------------------------------
void run_unit_tests()
{
  tw::cuda_info cinfo;
  cinfo.print();
}
// ----------------------------------------------------------------------------
void run()
{
  test_logging();

  //fim_test();
  //cuda_base_test();
  //cuda_fim_test();
  cuda_fim_test2();
  //cuda_fim_test3();

  //mem_t<size_t, float> test(2, shallow_e);

  //teststruct test;
  //testfunc(test);

  //int* asdfasd;
  //std::cout<<sizeof(decltype(&asdfasd))<<std::endl;
  //std::cout<<has_size<int>::value<<std::endl;
  //std::cout<<has_size<std::vector<int>>::value<<std::endl;
  //std::cout<<sizeof(*std::declval<int*>())<<std::endl;
  //std::cout<<sizeof(typeid(asdfasd))<<std::endl; 
  //std::cout<<sizeof(*std::declval<decltype(asdfasd)>())<<std::endl;

  checkCudaErrors(cudaDeviceReset());
}
// ----------------------------------------------------------------------------
} // ==========================================================================
// ----------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  int exit_code = EXIT_SUCCESS;

  try
  {
    const std::vector<std::string> args(argv, argv + argc);
    tw::print_head(); // static and only here
    //std::cout<<"TW_OMP_NT: "<<TW_OMP_NT<<std::endl;
    tw::run_unit_tests();

    switch(argc)
    {
      case 1: { tw::run(); break; }
      // case help: parser.print_help();
      default: break;
    }
  } 
  catch(const std::exception& e)
  {
    std::cerr<<e.what()<<std::endl;
    return EXIT_FAILURE;
  }

  return exit_code;
}
// ----------------------------------------------------------------------------
// ============================================================================