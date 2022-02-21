#include <fstream>
#include <chrono>
#include "eigen_solver.hpp"

EigenSolver::EigenSolver( uint32_t groups, std::shared_ptr<SlabGeometry> slab_geom,
                          std::shared_ptr<P3Solver> p3sol ) {
  Egroups   = groups;
  geom      = slab_geom;
  p3_solver = p3sol;

  xints   = geom->ints;
  nuSigma_f.resize(xints+1, std::vector<double>(Egroups) );
  chi.resize(xints+1, std::vector<double>(Egroups) );

  const auto nzones = geom->size();
  dx = geom->xcoords[nzones-1] / xints;

  update_xs();
}

void EigenSolver::update_xs() {
  const auto nzones = geom->size();
  auto zone = 0;
  double x = 0.0;
  auto xs = geom->mat[0]->gen_xs();
  for ( auto i = 0 ; i <= xints ; ++i ) {
    nuSigma_f[i] = static_cast<std::vector<double>>( xs.nuSigmaF );
    chi[i]       = static_cast<std::vector<double>>( xs.chi );

    x += dx;
    if ( x > geom->xcoords[zone] && zone < nzones-1 ) {
      ++zone;
      xs = geom->mat[zone]->gen_xs();
    }
  }

  p3_solver->update_xs();
}

void EigenSolver::solve( std::vector<std::vector<double>> source_guess, std::string output_file_name, double error_tol) {

  fission_source = source_guess;
  auto etol      = fmax( 1.0e-12, error_tol );
  auto p3_etol   = fmax( 1.0e-14, 0.001*error_tol );
  auto sor_omega = 1.985;

  auto l2_error = 1.0;
  auto niter = 0;
  std::ofstream k_eff_file;
  k_eff_file.open(output_file_name + "_k-eff.txt");
  do {
    //std::chrono::time_point<std::chrono::system_clock> start_time = std::chrono::system_clock::now();
    auto phi = std::vector<std::vector<double>>(xints + 1, std::vector<double>(Egroups));
    if(niter == 0) {
        for (size_t i = 0; i < xints; ++i) {
            for (size_t g = 0; g < Egroups; ++g) {
                phi[i][g] = 0.2;
            }
        }
        std::ofstream outfile;
        outfile.open(output_file_name + "-0.txt");
        for (auto i = 0; i <= xints; ++i) {
            const auto x = i * dx;
            outfile << x;
            for (auto g = 0; g < Egroups; ++g) {
                outfile << "   " << phi[i][g];
            }
            outfile << '\n';
        }
        outfile.close();
    }
    else {
        p3_solver->solve(fission_source, output_file_name, p3_etol, sor_omega);

        phi = p3_solver->flux();
    }
    
    keff = 0.0;
    //std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() << std::endl;
    // compute keff by integrating fission source with trapezoid rule
    for ( auto i = 0 ; i < xints ; ++i ) {
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        keff += nuSigma_f[i][g] * phi[i][g] + nuSigma_f[i+1][g] * phi[i+1][g];
      }
    }
    keff *= 0.5*dx;
    // construct new normalized fission source
    std::vector<std::vector<double>> new_source( xints+1, std::vector<double>( Egroups, 0.0 ));
    for ( auto i = 0 ; i <= xints ; ++i ) {
      auto f = 0.0;
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        f += nuSigma_f[i][g] * phi[i][g];
      }
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        new_source[i][g] = f * chi[i][g] / keff;
      }
    }
    auto e2_diff  = 0.0;
    auto e2_norm  = 0.0;
    for ( auto i = 0 ; i <= xints ; ++i ) {
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        e2_diff += pow( (new_source[i][g] - fission_source[i][g]), 2 );
        e2_norm += pow( fission_source[i][g], 2);
      }
    }
    l2_error = sqrt( e2_diff / e2_norm );
    std::cout << niter << "   " << keff << "   " << l2_error << "   " << p3_solver->num_iterations() << '\n';
    fission_source = new_source;
    ++niter;
    k_eff_file << keff << std::endl;

  } while ( l2_error > etol && niter < 6);

}
