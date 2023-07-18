/**
* Copyright (c) 2022 Thomas Schrotter. All rights reserved.
* @file io_base.h
* @brief Base file for I/O based routines.
* @author Thomas Schrotter
* @version 0.0.0
* @date 2022-12
*/
// TODO =======================================================================
// - check dir / file
// CODE =======================================================================

#ifndef TW_IO_BASE_H_
#define TW_IO_BASE_H_

// tw includes
#include "../core/excepts.h"
#include "../core/types.h"

// lib includes
//#include "../../lib/json.hpp"

// stdlib includes
#include <filesystem>
#include <string>
#include <fstream>
#include <iterator>

namespace tw // ===============================================================
{
// ----------------------------------------------------------------------------
void string_split(const std::string str, const std::string del, std::vector<std::string>& res);
// ----------------------------------------------------------------------------
/**
* @brief Throws a tw exception if given pile path does not exist.
*
* @param [in] file_path file path of a file
*
*/
int check_file_path(const std::string &file_path);
// ----------------------------------------------------------------------------
/*void load_json(const std::string &file_path, nlohmann::json &jfile);
// ----------------------------------------------------------------------------
void load_plan_from_json(const std::string &file_path, ek_options &opts, 
                         std::vector<region_property> &rps, 
                         std::vector<node_activity> &nas);*/
// ----------------------------------------------------------------------------
void write_node_data(const std::string &outfile_path, const std::vector<dbl_t> &phi);
// ----------------------------------------------------------------------------
void load_ek_reference(const std::string &filepath, 
                       const size_t Nslices, const size_t Ndata,
                       std::vector<std::vector<dbl_t>> &slice_data);
// ----------------------------------------------------------------------------
void save_ek_reference(const std::string &filepath, 
                       const std::vector<std::vector<dbl_t>> &slice_data);
// ----------------------------------------------------------------------------
template<typename T>
void write_vector_to_dat(const std::string &path, const std::vector<T> &vec, const bool append=false)
{
  //check_file_path(path);

  const auto flags = std::ofstream::binary | (append?std::ios_base::app:std::ios::out);
  std::ofstream ofs(path, flags); std::ostream_iterator<char> osi{ofs};
  std::copy((char*)&vec[0], (char*)&vec.back() + sizeof(T), osi); ofs.close();
}
// ----------------------------------------------------------------------------
template<typename T>
void read_vector_from_dat(const std::string &path, std::vector<T> &vec)
{
  check_file_path(path);

  std::vector<char> tmp;
  const auto flags = std::ifstream::binary | std::ios::in;
  std::ifstream ifs(path, flags); std::istreambuf_iterator<char> isi{ifs};
  //tmp.insert(tmp.begin(), isi, std::istream_iterator<char>());
  std::copy(isi, std::istreambuf_iterator<char>(), std::back_inserter(tmp));
  vec.resize(tmp.size()/sizeof(T));
  memcpy(&vec[0], &tmp[0], tmp.size());
}
// ----------------------------------------------------------------------------
template<typename T>
inline void read_vector_txt(const std::string& path, std::vector<T>& vec)
{
  check_file_path(path);
  std::ifstream ifs(path); std::istream_iterator<T> isi_start{ifs}, isi_end;
	vec.assign(isi_start, isi_end); //ifs.close(); automatically in destructor
}















} // namespace tw =============================================================
#endif // TW_IO_BASE_H_
// END ========================================================================