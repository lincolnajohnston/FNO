#ifndef _P3_SOLVER_HEADER_
#define _P3_SOLVER_HEADER_

#include <vector>

#include "material.hpp"
#include "slab_geom.hpp"

/*
class SlabGeometry {
  public:
    std::vector<std::shared_ptr<Material>> mat;
    std::vector<double>   xcoords;
    uint32_t              ints;

    size_t size() { return mat.size(); }
};
*/

class P3Solver {
  private:
    std::shared_ptr<SlabGeometry> geom;

    std::vector<std::vector<double>> f0, f2;
    std::vector<std::vector<double>> q;
    std::vector<std::vector<double>> Sigma_t;

    std::vector<std::vector<std::vector<double>>>
      Sigma_s0, Sigma_s2, D0, D2;

    uint32_t xints, Egroups;
    uint32_t niter;
    uint32_t nsolves;
    double   l2_error;
    double   dx;
  public:
    P3Solver() {};
    P3Solver( uint32_t groups, std::shared_ptr<SlabGeometry> slab_geom );

    void solve( std::vector<std::vector<double>> source, std::string output_file_name,
                double error_tol = 1.e-9, double sor_omega = 1.0 );

    void update_xs();

    std::vector<std::vector<double>> flux();
    uint32_t num_iterations() { return niter; }
    double   error() { return l2_error; }
};

#endif
