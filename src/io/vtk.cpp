

#include "vtk.h"


namespace tw
{

// ----------------------------------------------------------------------------
// reads only tet meshes
void load_vtk(const std::string& vtk_path, mesh_t& mesh)
{
  log_info("load_vtk");
  printf("loading %s\n", vtk_path.c_str());
  const auto start = std::chrono::steady_clock::now();
  check_file_path(vtk_path); std::ifstream vtk_file(vtk_path);
  //std::stringstream vtk_buf; vtk_buf<<vtk_file.rdbuf();

  std::string header, format, structure, line;

  std::getline(vtk_file, line);
  const float version = stof(line.substr(23)); //sscanf_s(line.c_str(), "# vtk DataFile Version %f", &version);
  printf("Version: %f\n", version);
  
  std::getline(vtk_file, header);
  std::getline(vtk_file, format);
  std::getline(vtk_file, line); structure = line.substr(8);

  while(std::getline(vtk_file, line))
  {
    const size_t pos = line.find(" ");
    const std::string key = line.substr(0, pos);

    if(key == "POINTS")
    {
      const idx_t n_points = stoll(line.substr(pos));
      const std::string dtype = line.substr(pos);

      // assume double for now
      mesh.n_points = n_points;
      mesh.xyz.resize(n_points*3);
      dbl_t* xyz_p = mesh.xyz.data();

      for(idx_t idx = 0 ; idx < n_points ; idx++)
      {
        std::getline(vtk_file, line);
        std::sscanf(line.c_str(), "%lf %lf %lf", xyz_p, xyz_p+1, xyz_p+2);
        xyz_p += 3;
      }
      //std::istream_iterator<dbl_t> isi_start{vtk_file}, isi_end{vtk_file}; 
      //std::advance(isi_end, n_points); mesh.xyz.assign(isi_start, isi_end);
    }
    else if(key == "CELLS")
    {
      const idx_t n_cells = stoll(line.substr(pos));

      mesh.n_cells = n_cells;
      mesh.e2n_cnt.resize(n_cells); mesh.e2n_dsp.resize(n_cells); mesh.e2n_con.resize(4*n_cells);
      idx_t *dsp =  mesh.e2n_dsp.data(), *con =  mesh.e2n_con.data();

      for(idx_t eidx = 0 ; eidx < n_cells ; eidx++)
      {
        std::getline(vtk_file, line);
        std::sscanf(line.c_str(), "%hd %ld %ld %ld %ld", &mesh.e2n_cnt[eidx], con, con+1, con+2, con+3);
        dsp[eidx] = eidx*4;
        con += mesh.e2n_cnt[eidx]; // 4
      }
    }
    else if(key == "CELL_TYPES")
    {
      const idx_t n_cells = stoll(line.substr(pos));
      mesh.etype.resize(n_cells);

      for(idx_t eidx = 0 ; eidx < n_cells ; eidx++)
      {
        std::getline(vtk_file, line);
        //std::sscanf(line.c_str(), "%d", &mesh.etype[eidx]);
      }
    }
    else if(key == "CELL_DATA")
    {
      const idx_t n_cells = stoll(line.substr(pos));
      mesh.etags.resize(n_cells);

      // only of elemTag
      // scalar, vector, ...
      std::getline(vtk_file, line);
      std::getline(vtk_file, line);

      for(idx_t eidx = 0 ; eidx < n_cells ; eidx++)
      {
        std::getline(vtk_file, line);
        std::sscanf(line.c_str(), "%d", &mesh.etags[eidx]);
      }
    }
  }

  printf("assign fibers\n");
  mesh.lon.assign(3*mesh.e2n_cnt.size(), 0.0); // !!!! -> put into mesh loading
  for (size_t it = 0 ; it < mesh.e2n_cnt.size() ; ++it)
    mesh.lon[3*it] = 1.0;

  const auto end = std::chrono::steady_clock::now();
  const double dt = std::chrono::duration<double>(end - start).count();
  printf("took %lf s\n", dt);
}
// ----------------------------------------------------------------------------
// not original!
void save_vtk(const std::string &vtk_path, const mesh_t &mesh,
              const std::vector<std::vector<dbl_t>> &data,
              const std::vector<std::string> &data_names,
              const std::vector<std::vector<vec3_t>> &vdata,
              const std::vector<std::string> &vdata_names)
{
  FILE* vtk_file = fopen(vtk_path.c_str(), "w");
  if(vtk_file == NULL) throw_tw_error("cannot open file: " + vtk_path);

  // if possible we use the length of the data vector as the number of nodes
  //int numnodes = data.size() ? data[0].size() : mesh.n2n_cnt.size();
  const int numnodes = *(std::max_element(mesh.e2n_con.begin(), mesh.e2n_con.end())) + 1;
  int numelems = mesh.e2n_cnt.size();

  fprintf (vtk_file, "# vtk DataFile Version 3.0\n");
  fprintf (vtk_file, "vtk output\n");
  fprintf (vtk_file, "ASCII\n");
  fprintf (vtk_file, "DATASET UNSTRUCTURED_GRID\n\n");
  fprintf (vtk_file, "POINTS %d     float\n", numnodes);

  float pts[3];
  const dbl_t* p = mesh.xyz.data();

  for (int i=0; i<numnodes; i++ ) {
    pts[0] = p[0], pts[1] = p[1], pts[2] = p[2];
    fprintf(vtk_file, "%.3f %.3f %.3f\n", pts[0], pts[1], pts[2]);
    p += 3;
  }

  fprintf (vtk_file, "CELL_TYPES %d\n", numelems);
  size_t valcount = 0;
  for(int i=0; i<numelems; i++) {
    cnt_t actnodes = mesh.e2n_cnt[i];
    switch(actnodes) {
      // Line for Purkinje
      case 2:
        fprintf(vtk_file, "%d\n", 3); // Lines are encoded as index 3
        break;

      // Triangle
      case 3:
        fprintf(vtk_file, "%d\n", 5); // Tetras are encoded as index 5
        break;

      // Tetra
      case 4:
        fprintf(vtk_file, "%d\n", 10); // Tetras are encoded as index 10
        break;
    }
    valcount += actnodes+1;
  }

  fprintf(vtk_file, "CELLS %d %lu\n", numelems, valcount);

  const idx_t* elem = mesh.e2n_con.data();
  for(int i=0; i<numelems; i++)
  {
    switch(mesh.e2n_cnt[i]) {
      case 2:
        fprintf(vtk_file, "%d %ld %ld\n", 2, elem[0], elem[1]);
        elem += 2;
        break;

      case 3:
        fprintf(vtk_file, "%d %ld %ld %ld\n", 3, elem[0], elem[1], elem[2]);
        elem += 3;
        break;

      case 4:
        fprintf(vtk_file, "%d %ld %ld %ld %ld\n", 4, elem[0], elem[1], elem[2], elem[3]);
        elem += 4;
        break;
    }
  }
  fprintf (vtk_file, "CELL_DATA %d \n", numelems);
  fprintf (vtk_file, "SCALARS %s int %d\n", "elemTag", 1);
  fprintf (vtk_file, "LOOKUP_TABLE default\n");
  for (int i=0; i<numelems; i++ ) {
    fprintf(vtk_file, "%d \n", mesh.etags[i]);
  }

  if(!data.empty())
  {
    fprintf (vtk_file, "POINT_DATA %d \n", numnodes);
    for(size_t n=0; n<data.size(); n++) {
      fprintf (vtk_file, "SCALARS %s double %d\n", data_names[n].c_str(), 1);
      fprintf (vtk_file, "LOOKUP_TABLE default\n");
      for (int i=0; i<numnodes; i++ ) {
        fprintf(vtk_file, "%lf \n", (data[n][i] < 1e50)?data[n][i]:-1);
      }
    }
  }
  

  for(size_t n=0; n<vdata.size(); n++) {
    fprintf (vtk_file, "VECTORS %s double\n", vdata_names[n].c_str());
    for (int i=0; i<numnodes; i++ ) {
      fprintf(vtk_file, "%lf %lf %lf \n", vdata[n][i].x, vdata[n][i].y, vdata[n][i].z);
    }
  }

  fclose(vtk_file);
}

}
