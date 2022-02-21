#ifndef _DEPLETION_SOLVER_HEADER_
#define _DEPLETION_SOLVER_HEADER_

#include <vector>

#include "material.hpp"
#include "slab_geom.hpp"
#include "p3_solver.hpp"
#include "eigen_solver.hpp"

class DepletionSolver {
  private:
    uint32_t     Egroups, xints;
    std::shared_ptr<SlabGeometry> geom;
    std::shared_ptr<EigenSolver>  eigen_solver;

    std::vector<uint32_t> zone_index;

    std::vector<std::vector<double>> Sigma_f;
    std::vector<std::vector<double>> phi;
    double dx;
    double keff;

    double power_normalize( double power );
    std::vector<double> zone_average_flux( uint32_t zone );

  public:
    DepletionSolver( uint32_t groups, std::shared_ptr<SlabGeometry> geom,
                     std::shared_ptr<EigenSolver> eigsol );

    void solve( std::vector<std::vector<double>>& source_guess, std::string output_file_name, const double dt, const double power );
};

#endif
