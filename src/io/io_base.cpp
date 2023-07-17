// CODE =======================================================================
#include "io_base.h"

namespace tw // ===============================================================
{
// ----------------------------------------------------------------------------
void string_split(const std::string str, const std::string del, std::vector<std::string>& res)
{
  size_t pos = 0, last_pos = 0; res.clear();
  
  while ((pos = str.find(del, last_pos)) != std::string::npos)
  {
    if (pos == last_pos)
    {
      last_pos++;
      continue;
    }

    res.push_back(str.substr(last_pos, pos-last_pos));
    //token = str.substr(last_pos, pos-last_pos);
    //std::cout<<token<<std::endl;
    last_pos = pos;
  }

  if (last_pos != str.size())
  {
    res.push_back(str.substr(last_pos, str.size() - last_pos));
    //token = str.substr(last_pos, str.size() - last_pos);
    //std::cout<<token<<std::endl;
  }
}
// ----------------------------------------------------------------------------
int check_file_path(const std::string &file_path)
{
  if(!std::filesystem::exists(file_path))
    throw_tw_error("file not found: " + file_path);
  
  return EXIT_SUCCESS;
}
// ----------------------------------------------------------------------------
/*void load_json(const std::string &file_path, nlohmann::json &jfile)
{
  check_file_path(file_path);
  std::ifstream ifs(file_path, std::ios::in); ifs>>jfile; ifs.close();
}
// ----------------------------------------------------------------------------
void load_plan_from_json(const std::string &file_path, ek_options &opts,
                         std::vector<region_property> &rps,
                         std::vector<node_activity> &nas)
{
  nlohmann::json jfile; load_json(file_path, jfile);

  // load options
  jfile["verbose"].get_to(opts.verbose);
  jfile["rent_en"].get_to(opts.rent_en);
  jfile["apdr_en"].get_to(opts.apdr_en);
  jfile["cvr_en"].get_to(opts.cvr_en);
  jfile["curv_en"].get_to(opts.curv_en);
  jfile["diff_en"].get_to(opts.diff_en);

  jfile["method_id"].get_to(opts.method_id);
  jfile["solvetype"].get_to(opts.solvetype);

  jfile["dt"].get_to(opts.dt); opts.dt *= 1e6;
  jfile["n_steps"].get_to(opts.n_steps);

  // load region properties
  rps.clear(); int id = 0;
  for(const auto& region : jfile["regions"])
  {
    //const auto& cv = region["cv"];
    if(region["active"] == 1)
    {
      std::vector<dbl_t> veldat;
      region["velocity"].get_to(veldat);
      rps.push_back({id, region["tags"], region["active"], {veldat[0], veldat[1], veldat[2]}});//{.6, .6, .6}
    }
  }

  // load node activities
  nas.clear();
  for(const auto& nact : jfile["nacts"])
  {
    nas.push_back({id, nact["active"], nact["start"], nact["duration"], nact["vtx"]});
  }
}*/
// ----------------------------------------------------------------------------
void write_node_data(const std::string &outfile_path, const std::vector<dbl_t> &phi)
{
  throw_tw_error("not implemented yet! " + outfile_path + " " + std::to_string(phi.size()));
}
// ----------------------------------------------------------------------------
void load_ek_reference(const std::string &filepath, 
                       const size_t Nslices, const size_t Ndata,
                       std::vector<std::vector<dbl_t>> &slice_data)
{
  //std::ifstream ifs(filepath);
  //std::istream_iterator<mt_real> isi_start{ifs}, isi_end;
  //std::vector<mt_real> flattened_data(isi_start, isi_end);
  std::vector<dbl_t> flattened_data;
  read_vector_from_dat(filepath, flattened_data);

  slice_data.assign(Nslices, std::vector<dbl_t>(Ndata));

	for(size_t it = 0 ; it < Nslices ; it++)
  {
    //slice_data[it].assign(Ndata, &(flattened_data[it*Ndata]));
    //slice_data[it].assign(Ndata);
    memcpy(slice_data[it].data(), flattened_data.data() + it*Ndata, sizeof(dbl_t)*Ndata);
  }
}
void save_ek_reference(const std::string &filepath, 
                       const std::vector<std::vector<dbl_t>> &slice_data)
{
  if(slice_data.empty()) return;

  const size_t Nslices = slice_data.size(), Ndata = slice_data[0].size();
  std::vector<dbl_t> flattened_data(Nslices*Ndata);
  dbl_t* ptr_start = flattened_data.data();

  //std::cout<<"Nslices: "<<Nslices<<" Ndata: "<<Ndata<<std::endl;

  for(size_t it = 0 ; it < Nslices ; it++)
  {
    memcpy(ptr_start, slice_data[it].data(), sizeof(dbl_t)*Ndata);
    ptr_start += Ndata;
  }

  //std::ofstream ofs(filepath); 
  //std::ostream_iterator<mt_real> osi{ofs};
  //std::copy(&(flattened_data.front()), &(flattened_data.back()), osi);
  write_vector_to_dat(filepath, flattened_data);

  flattened_data.clear();
}
} // namespace tw =============================================================
// END ========================================================================