
#include "io_json.h"


namespace tw
{

// JSON IO ====================================================================
// ----------------------------------------------------------------------------
void read_jdata(const std::string& json_path, nlohmann::json& root)
{
  if (check_file_path(json_path) == EXIT_FAILURE)
    throw_tw_error("read_jdata: file not found: " + json_path);

  std::ifstream ifs(json_path, std::ios::in); ifs>>root; ifs.close();
}
// ----------------------------------------------------------------------------
void read_configs_from_json(const nlohmann::json& root,
                            std::map<const std::string, std::vector<tag_t>>& configs)
{
  for (auto& config : configs)
    config.second.clear();
  
  configs.clear();
  
  for (const auto& entry : root)
  {
    const std::string& id = entry["func"];

    for (const auto& tag : entry["tags"])
      configs[id].push_back(tag);
  }
}
// ----------------------------------------------------------------------------
/*
void read_elecs_from_json(const nlohmann::json& root,
                          std::map<const std::string,Electrode>& elecs)
{
  mt_real epos[3]; auto& edata = root["electrodes"]; elecs.clear();

  for (nlohmann::json::const_iterator jt = edata.begin(); jt != edata.end(); jt++)
  {
    size_t it = 0;
    for (const auto& p0_crd : jt.value()["p0"]) { epos[it] = p0_crd; it++; }
    const mt_real rad = jt.value()["radius"];
    elecs[jt.key()] = {false, epos[0], epos[1], epos[2], rad};
  }
}
// ----------------------------------------------------------------------------
void stim_from_elecs(const mesh_t& mesh,
                     std::map<const std::string,Electrode>& elecs,
                     std::vector<ek_event>& stims)
{
  vec3r vpos;

  for (const auto& eitem : elecs)
  {
    const Electrode& e = eitem.second;

    if (!e.active) continue;

    vec3r epos(e.x, e.y, e.z); //epos *= 1e-4;
    const mt_real r2 = e.rad*e.rad;// *1e-8;

    for (const auto& t : e.t)
    {
      temp_item& item = stims.push_back(temp_item()); // name missing but not neccessary
      item.active = 1;
      item.start = t;

      for (size_t idx = 0 ; idx < mesh.xyz.size()/3 ; idx++)
      {
        getPointFromMesh(idx, mesh.xyz.data(), vpos);
        if ((vpos - epos).length2() > r2) continue;
        item.vtx.push_back(idx);
      }
    }
  }
}
*/
// ----------------------------------------------------------------------------
void read_funcs_from_json(//const std::string& root_path,
                          const nlohmann::json& root,
                          //const mesh_t& mesh,
                          const std::map<const std::string, std::vector<tag_t>>& configs,
                          //std::vector<temp_item>& blocks,
                          std::vector<region_property>& rprops)
{
  std::vector<dbl_t> di_f, di_t, cv_f, cv_t, di_apd, apd;
  rprops.clear(); rprops.reserve(configs.size());

  for (const auto& config : configs)
  {
    const auto& fnc = root[config.first];
    const auto& EP = fnc["EP"]; // DI, APD?

    if (fnc["EP"].size() == 0) // inactive region, add to block
    {
      //for (const auto& tag : config.second)
      //{
      //  ek_event& item = blocks.push_back(temp_item());
      //  block_label(mesh, tag, config.first, 0., 0., 1, item);//MT_REAL_INF
      //}
    }
    else // active region
    {
      const int id = static_cast<int>(rprops.size());
      const auto& CV = fnc["CV"];
      // "dx": 500.0!!! 
      const auto& CVref = CV["CVref"]; // CVmeas?
      const vec3_t default_cv({CVref["vf"], CVref["vs"], CVref["vn"]});

      dbl_t default_apd;
      if (EP.contains("apd"))
      {
        default_apd = EP["apd"]; default_apd *= 1000.;
      }
      else
      {
        log_warning("no default apd given, selecting 300ms as default.");
        default_apd = 3e5;
      }

      rprops.push_back({id, config.second, 
                        true, {default_cv.x, default_cv.y, default_cv.z}});
    }
  }
}
// ----------------------------------------------------------------------------
void read_sim_from_json(const nlohmann::json& root, ek_data &ekdata)
{
  //auto &mesh = ekdata.mesh;
  auto &opts = ekdata.opts;
  //auto &stims = ekdata.stims;
  //auto &blocks = ekdata.blocks;
  auto &events = ekdata.events;

  //stims.reserve(16);//prevent pointer change on push_back .. fix later
  //blocks.reserve(16);
  events.reserve(16);

  int npls = 0, active = 0; dbl_t bcl = 0.;
  std::vector<ek_event> *temp_items;
  std::vector<dbl_t> tmp;
  opts.dt = root["deltatime"]; opts.dt *= 1000.; // to seconds
  opts.n_steps = root["timesteps"];

  if (!root.contains("stimuli"))
  {
    log_warning("no stimuli found in plan file. (1)");
    return;
  }

  for (const auto& block_or_stim : root["stimuli"])
  {
    block_or_stim["active"].get_to(active);
    const std::string type = block_or_stim["type"];

    if (!active) // todo: fix
      continue;

    if ((type != "Transmembrane Current") && (type != "Transmembrane Voltage clamp"))
      throw_tw_error("read_sim_from_json: unknown stim type: " + type);

    //temp_items = (type == "Transmembrane Current")?&stims:&blocks;
    temp_items = &events;

    temp_items->push_back({});
    ek_event& item = temp_items->back(); 
    //item.name = "user_input";
    block_or_stim["active"].get_to(item.active);
    block_or_stim["protocol"]["start"].get_to(item.tstart); item.tstart *= 1000.;
    block_or_stim["protocol"]["duration"].get_to(item.dur); item.dur *= 1000.;
    block_or_stim["protocol"]["bcl"].get_to(bcl); bcl *= 1000.;
    block_or_stim["protocol"]["npls"].get_to(npls);
    const auto &elec = block_or_stim["electrode"];
    const std::string elec_type = elec["type"];

    if (elec_type == "vtxdata")
    {
      log_info("vtxdata stim");
      block_or_stim["electrode"]["vtxdata"].get_to(tmp); // future: differentiate between elecs and vtxdata!
      item.vtx.assign(tmp.begin(), tmp.end());
    }
    else if (elec_type == "block")
    {
      log_info("block stim (not implemented!)");
      /*std::set<idx_t> svtx;
      const std::vector<float> p0 = elec["p0"].get<std::vector<float>>();
      const std::vector<float> p1 = elec["p1"].get<std::vector<float>>();
      const box_check check(vec3r(p0[0], p0[1], p0[2]), vec3r(p1[0], p1[1], p1[2]));
      std::vector<idx_t> nod_vec = mesh.e2n_con;
      binary_sort(nod_vec); 
      unique_resize(nod_vec);
      elec_select_vertices(nod_vec, mesh.xyz, check, svtx);
      item.vtx.assign(svtx.begin(), svtx.end());*/
    }

    //std::cout<<item.name<<"\n";
    //std::cout<<item.tstart<<"\n";
    //std::cout<<item.dur<<"\n";
    //std::cout<<item.active<<"\n";
    //std::cout<<item.vtx.size()<<"\n";

    for (int it = 1 ; it < npls ; ++it)
    {
      temp_items->push_back({});
      ek_event& item2 = temp_items->back();
      //item2.name = "user_input";
      item2.tstart = item.tstart + it*bcl;
      item2.dur = item.dur;
      item2.vtx.assign(item.vtx.begin(), item.vtx.end());
      item2.active = item.active;
    }
  }

  std::cout<<"num_stims: "<<events.size()<<"\n";
  for (const auto &stim : events)
    std::cout<<stim.tstart<<"\n";

  if (events.size() == 0)
    log_warning("no stimuli found in plan file. (2)");
}
// ----------------------------------------------------------------------------
// pls make guards tom ..
// also relocate this struct + can be written better!
// also consider using numbers in cfg file + description instead of string compare
void load_options(const std::string& cfg_file, ek_options& opts)
{
  if (check_file_path(cfg_file) == EXIT_FAILURE)
    throw_tw_error("load_params: file not found: " + cfg_file);

  // check_file_path(ref_filename) == EXIT_FAILURE
  std::ifstream file_stream(cfg_file);
  std::stringstream buffer;
  buffer<<file_stream.rdbuf();

  // parse
  std::string line; std::vector<std::string> cfg;
  string_split(buffer.str(), "\n", cfg);
  char* ret = nullptr;

  // print cfg
  //for (const auto& st : cfg) std::cout<<st<<std::endl;
  
  // verbose
  if (cfg[0] == "false") opts.verbose = false;
  else if (cfg[0] == "true") opts.verbose = true;
  else { throw_tw_error("invalid value for verbose"); }

  // method
  line = cfg[1];

  for (auto& c: line) 
    c = tolower(c);

  opts.method_id = 1;//"fim"; //str2mth_.at(line);

  // solve_type
  if (cfg[2] == "lin") opts.solvetype = Lin;
  else if (cfg[2] == "vol") opts.solvetype = Vol;
  else { throw_tw_error("invalid value for slvtype"); }

  /*// compute_select
  if (cfg[3] == "const") opts.cs = cs_const;
  else if (cfg[3] == "apd") opts.cs = cs_apd;
  else if (cfg[3] == "curv") opts.cs = cs_curv;
  else if (cfg[3] == "full") opts.cs = cs_full;
  else { print_error("invalid value for slvtype"); assert(false); }*/

  // re_flag
  if (cfg[4] == "false") opts.rent_en = false;
  else if (cfg[4] == "true") opts.rent_en = true;
  else { throw_tw_error("invalid value for re_flag"); }

  /*
  // bvn_enable
  if (cfg[5] == "false") opts.bvn_enable = false;
  else if (cfg[5] == "true") opts.bvn_enable = true;
  else { print_error("invalid value for bvn_enable"); assert(false); }

  // act_override_threshold
  if (cfg[6] == "narrow") opts.act_threshold = ek_narrow;
  else if (cfg[6] == "frozen") opts.act_threshold = ek_frozen;
  else { print_error("invalid value for slvtype"); assert(false); }*/

  // dt
  opts.dt = strtod(cfg[7].c_str(), &ret)*1000;
  if (*ret) { throw_tw_error("invalid value for dt"); }

  // tmax
  opts.n_steps = strtol(cfg[8].c_str(), &ret, 10);
  if (*ret) { throw_tw_error("invalid value for t_max"); }

  //if (params.verbose) params.print_info();
}
// ----------------------------------------------------------------------------
// config -> 0, elecs -> 1, functions -> 2
void load_plan(const std::string &path, ek_data &ekdata)
{
  nlohmann::json root; 
  std::map<const std::string,std::vector<tag_t>> configs;
  const std::string file_path = path.substr(0, path.find_last_of('/') + 1);
  
  //ekdata.stims.clear(); ekdata.blocks.clear();
  ekdata.events.clear();

  read_jdata(path, root);
  read_configs_from_json(root["config"], configs);
  read_funcs_from_json(root["functions"], configs, ekdata.rprops);
  read_sim_from_json(root["simulation"], ekdata);
  
  for (auto& config : configs)
    config.second.clear();

  configs.clear();
}

}