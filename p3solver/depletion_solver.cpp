
#include "depletion_solver.hpp"

DepletionSolver::DepletionSolver( uint32_t groups, std::shared_ptr<SlabGeometry> slab_geom,
                                  std::shared_ptr<EigenSolver> eigsol ) {
  Egroups      = groups;
  eigen_solver = eigsol;
  geom         = slab_geom;

  xints  = geom->ints;
  Sigma_f.resize(xints+1, std::vector<double>(Egroups) );

  const auto nzones = geom->size();
  auto xcoords = geom->xcoords;
  dx = xcoords[nzones-1] / xints;

  zone_index.resize( nzones+1, 0 );
  zone_index[nzones] = xints;

  auto zone = 0;
  double x = 0.0;
  auto xs = geom->mat[0]->gen_xs();
  for ( auto i = 0 ; i <= xints ; ++i ) {
    Sigma_f[i] = static_cast<std::vector<double>>( xs.SigmaF );

    x += dx;
    if ( x > xcoords[zone] && zone < nzones-1 ) {
      ++zone;
      xs = geom->mat[zone]->gen_xs();

      zone_index[zone] = (i+1);
    }
  }
}

// normalize flux to match input power in Watts with flux units of b^-1 s^-1
double DepletionSolver::power_normalize( double power ) {

  constexpr double joule_per_fission = 3.2044e-11;
  constexpr double cm2_per_barn      = 1.0e-24;
  double integral = 0.0;
  for ( auto i = 0 ; i < xints ; ++i ) {
    for ( auto g = 0 ; g < Egroups ; ++g ) {
      integral += Sigma_f[i][g] * phi[i][g] + Sigma_f[i+1][g] * phi[i+1][g];
    }
  }
  integral *= 0.5*dx * joule_per_fission / cm2_per_barn;
  return power / integral;
}

std::vector<double> DepletionSolver::zone_average_flux( uint32_t zone ) {

  const auto xcoords = geom->xcoords;
  const auto xL = ( zone > 0 ? xcoords[zone-1] : 0.0 );
  const auto xR = xcoords[zone];
  const auto c  = 0.5*dx/(xR - xL);

  std::vector<double> integral( Egroups, 0.0 );
  for ( auto i = zone_index[zone] ; i < zone_index[zone+1] ; ++i ) {
    for ( auto g = 0 ; g < Egroups ; ++g ) {
      integral[g] += c*( phi[i][g] + phi[i+1][g] );
    }
  }
  return integral;
}

void DepletionSolver::solve( std::vector<std::vector<double>>& source_guess, std::string output_file_name,
                             const double dt, const double power ) {

  //-----> predictor step
  eigen_solver->solve( source_guess, output_file_name );

  phi = eigen_solver->flux();
  auto pnorm = power_normalize( power );

  auto nzones    = geom->size();
  auto atom_den0 = std::vector<xs1d_table>( nzones );
  auto f0        = std::vector<xs1d_table>( nzones );
  for ( auto i = 0 ; i < nzones ; ++i ) {
    if ( geom->mat[i]->deplete ) {
      atom_den0[i] = xs1d_table( geom->mat[i]->atom_den );

      auto xs_data   = geom->mat[i]->gen_xs();
      auto depl_data = geom->mat[i]->gen_deplete();
      auto phi_bar   = zone_average_flux(i);

      f0[i] = depl_data.decay * atom_den0[i];
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        f0[i] += pnorm * phi_bar[g] * ( depl_data.rxn_xs[g] * atom_den0[i] + xs_data.SigmaF(g) * depl_data.fyield );
      }
      geom->mat[i]->atom_den = static_cast<std::vector<double>>( dt * f0[i] + atom_den0[i] );

      for ( auto j = 0 ; j < geom->mat[i]->atom_den.size() ; ++j ) {
        std::cout << geom->mat[i]->isotope[j]->name << "   " << geom->mat[i]->atom_den[j] << '\n';
      }
    }
  }
  source_guess = eigen_solver->source();
  eigen_solver->update_xs();

  //-----> corrector step
  eigen_solver->solve( source_guess, output_file_name );

  phi = eigen_solver->flux();
  pnorm = power_normalize( power );

  for ( auto i = 0 ; i < nzones ; ++i ) {
    if ( geom->mat[i]->deplete ) {
      auto atom_den1 = xs1d_table( geom->mat[i]->atom_den );

      auto xs_data   = geom->mat[i]->gen_xs();
      auto depl_data = geom->mat[i]->gen_deplete();
      auto phi_bar   = zone_average_flux(i);

      auto f1 = depl_data.decay * atom_den1;
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        f1 += pnorm * phi_bar[g] * ( depl_data.rxn_xs[g] * atom_den1 + xs_data.SigmaF(g) * depl_data.fyield );
      }
      geom->mat[i]->atom_den = static_cast<std::vector<double>>( 0.5*dt*( f0[i] + f1 ) + atom_den0[i] );

      for ( auto j = 0 ; j < geom->mat[i]->atom_den.size() ; ++j ) {
        std::cout << geom->mat[i]->isotope[j]->name << "   " << geom->mat[i]->atom_den[j] << '\n';
      }
    }
  }
  source_guess = eigen_solver->source();
  eigen_solver->update_xs();
}
