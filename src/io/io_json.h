/**
* Copyright (c) 2022 Thomas Schrotter. All rights reserved.
* @file vtk.h
* @brief A collection of routines to read and write .json files.
* @author Thomas Schrotter
* @version 0.0.0
* @date 2022-12
*/
// CODE =======================================================================

#ifndef TW_IO_JSON_H_
#define TW_IO_JSON_H_

// tw includes
#include "io_base.h"
#include "../core/types.h"

// libs
#include "../../lib/json.hpp"

// stdlib


namespace tw // ===============================================================
{
// JSON IO ====================================================================
// ----------------------------------------------------------------------------
void read_jdata(const std::string& json_path, nlohmann::json& root);
// ----------------------------------------------------------------------------
void read_configs_from_json(const nlohmann::json& root,
                            std::map<const std::string, std::vector<tag_t>>& configs);
// ----------------------------------------------------------------------------
//void read_elecs_from_json(const nlohmann::json& root,
//                          std::map<const std::string, Electrode>& elecs);
// ----------------------------------------------------------------------------
void read_funcs_from_json(//const std::string& root_path,
                          const nlohmann::json& root,
                          //const mesh_t& mesh,
                          const std::map<const std::string, std::vector<tag_t>>& configs,
                          //std::vector<ek_event>& blocks,
                          std::vector<region_property>& rprops);
// ----------------------------------------------------------------------------
void read_sim_from_json(const nlohmann::json& root, ek_data &ekdata);
// ----------------------------------------------------------------------------
//void stim_from_elecs(const mesh_t& mesh,
//                     std::map<const std::string,Electrode>& elecs,
//                     std::vector<temp_item>& stims);
// ----------------------------------------------------------------------------
/**
* @brief loads options specified by cfg_file
*
* @param [in]  cfg_file file path to mesh
* @param [out] opts     struct that contains options
*/
void load_options(const std::string &cfg_file, ek_options& opts);
// ----------------------------------------------------------------------------
/**
* @brief loads configurations and data from .json files
*
* @param [in]  paths   path(s) to one or more json files
* @param [out] ek_data data for eikonal solvers
*/
void load_plan(const std::string &path, ek_data &ekdata);
} // namespace tw // ====================================================
#endif // TW_IO_JSON_H_