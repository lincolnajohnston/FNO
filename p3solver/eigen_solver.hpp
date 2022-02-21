#ifndef _EIGEN_SOLVER_HEADER_
#define _EIGEN_SOLVER_HEADER_

#include <vector>

#include "material.hpp"
#include "slab_geom.hpp"
#include "p3_solver.hpp"

class EigenSolver {
  private:
    std::shared_ptr<SlabGeometry> geom;
    std::shared_ptr<P3Solver>     p3_solver;
    uint32_t     Egroups, xints;

    std::vector<std::vector<double>> nuSigma_f, chi;
    std::vector<std::vector<double>> phi;
    std::vector<std::vector<double>> fission_source;
    double dx;
    double keff;

  public:
    EigenSolver() {};
    EigenSolver( uint32_t groups, std::shared_ptr<SlabGeometry> geom, std::shared_ptr<P3Solver> p3sol );

    void update_xs();

    std::vector<std::vector<double>> flux()   { return phi; }
    std::vector<std::vector<double>> source() { return fission_source; }

    void solve( std::vector<std::vector<double>> source_guess, std::string output_file_name, double error_tol = 1.0e-8 );
};

#endif
