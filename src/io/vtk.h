/**
* Copyright (c) 2022 Thomas Schrotter. All rights reserved.
* @file vtk.h
* @brief A collection of routines to read and write .vtk files.
* @author Thomas Schrotter
* @version 0.0.0
* @date 2022-12
*/
// CODE =======================================================================

#ifndef TW_VTK_H_
#define TW_VTK_H_

// tw includes
#include "io_base.h"
#include "../core/types.h"

// stdlib
#include <string>
#include <fstream>
#include <cstdio>

namespace tw // ===============================================================
{
// ----------------------------------------------------------------------------
void load_vtk(const std::string& vtk_path, mesh_t& mesh);
// ----------------------------------------------------------------------------
void save_vtk(const std::string &vtk_path, const mesh_t &mesh,
              const std::vector<std::vector<dbl_t>> &data,
              const std::vector<std::string> &data_names,
              const std::vector<std::vector<vec3_t>> &vdata,
              const std::vector<std::string> &vdata_names);
} // namespace tw // ====================================================
#endif // TW_VTK_H_