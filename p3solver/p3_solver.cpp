#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "p3_solver.hpp"

P3Solver::P3Solver( uint32_t energy_groups, std::shared_ptr<SlabGeometry> slab_geom ) {
  geom = slab_geom;

  xints   = geom->ints;
  Egroups = energy_groups;
  nsolves = 0;

  f0.resize(xints+1, std::vector<double>(Egroups, 0.0) );
  f2.resize(xints+1, std::vector<double>(Egroups, 0.0) );
  q.resize(xints+1, std::vector<double>(Egroups) );

  Sigma_t.resize(xints+1, std::vector<double>(Egroups) );

  Sigma_s0.resize(xints+1, std::vector<std::vector<double>>( Egroups, std::vector<double>(Egroups, 0.0) ) );
  Sigma_s2.resize(xints+1, std::vector<std::vector<double>>( Egroups, std::vector<double>(Egroups, 0.0) ) );

  D0.resize(xints+1, std::vector<std::vector<double>>( Egroups, std::vector<double>(Egroups, 0.0) ) );
  D2.resize(xints+1, std::vector<std::vector<double>>( Egroups, std::vector<double>(Egroups, 0.0) ) );

  auto nzones = geom->size();
  dx = geom->xcoords[nzones-1] / xints;

  update_xs();
}

void P3Solver::update_xs() {
  auto nzones = geom->size();

  auto zone = 0;
  double x = 0.0;
  auto xs = geom->mat[0]->gen_xs();
  for ( auto i = 0 ; i <= xints ; ++i ) {
    Sigma_t[i] = static_cast<std::vector<double>>( xs.SigmaT );

    Sigma_s0[i] = static_cast<std::vector<std::vector<double>>>( xs.SigmaS0 );
    Sigma_s2[i] = static_cast<std::vector<std::vector<double>>>( xs.SigmaS2 );

    D0[i] = static_cast<std::vector<std::vector<double>>>( xs.D0 );
    D2[i] = static_cast<std::vector<std::vector<double>>>( xs.D2 );

    x += dx;
    if ( x > geom->xcoords[zone] && zone < nzones-1 ) {
      ++zone;
      xs = geom->mat[zone]->gen_xs();
    }
  }

}

void P3Solver::solve( std::vector<std::vector<double>> source,
                      std::string output_file_name, const double error_tol, const double sor_omega ) {

  q = source;

  auto kron = []( uint32_t i, uint32_t j ) { return i == j ? 1.0 : 0.0; };

  auto dx2 = dx*dx;

  niter    = 0;
  l2_error = 1.0;
  do {
    const auto f0_old = f0;
    const auto f2_old = f2;

    // n = 0 equation
    {
    // left marshak boundary condition
    for ( auto g = 0 ; g < Egroups ; ++g ) {
      auto L = 3*D0[0][g][g] + D0[1][g][g]
             + 2*dx2*( Sigma_t[0][g] - Sigma_s0[0][g][g] + 1.0/dx );
      auto R = 2*dx2*q[0][g];
      for ( auto h = 0 ; h < Egroups ; ++h ) {
        R += ( 3*D0[0][g][h] + D0[1][g][h] ) * f0[1][h]
           - (1-kron(g,h))*( 3*D0[0][g][h] + D0[1][g][h] - 2*dx2*Sigma_s0[0][g][h] )*f0[0][h]
           + 4*dx2*( ( Sigma_t[0][h] + 3/(8*dx) )*kron(g,h) - Sigma_s0[0][g][h] )*f2[0][h];
      }
      f0[0][g] = sor_omega * R/L + (1-sor_omega)*f0[0][g];
    }
    // internal zones
    for ( auto i = 1 ; i < xints ; ++i ) {
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        auto L = D0[i-1][g][g] + 2*D0[i][g][g] + D0[i+1][g][g]
               + 2*dx2*( Sigma_t[i][g] - Sigma_s0[i][g][g] );
        auto R = 2*dx2*q[i][g];
        for ( auto h = 0 ; h < Egroups ; ++h ) {
          R += ( D0[i-1][g][h] + D0[i][g][h] ) * f0[i-1][h]
             + ( D0[i+1][g][h] + D0[i][g][h] ) * f0[i+1][h]
             - (1-kron(g,h))*( D0[i-1][g][h] + 2*D0[i][g][h] + D0[i+1][g][h] - 2*dx2*Sigma_s0[i][g][h] )*f0[i][h]
             + 4*dx2*( Sigma_t[i][h]*kron(g,h) - Sigma_s0[i][g][h] )*f2[i][h];
        }
        f0[i][g] = sor_omega * R/L + (1-sor_omega)*f0[i][g];
      }
    }
    // right marshak boundary condition
    auto I = xints;
    for ( auto g = 0 ; g < Egroups ; ++g ) {
      auto L = 3*D0[I][g][g] + D0[I-1][g][g]
             + 2*dx2*( Sigma_t[I][g] - Sigma_s0[I][g][g] + 1.0/dx );
      auto R = 2*dx2*q[0][g];
      for ( auto h = 0 ; h < Egroups ; ++h ) {
        R += ( 3*D0[I][g][h] + D0[I-1][g][h] ) * f0[I-1][h]
           - (1-kron(g,h))*( 3*D0[I][g][h] + D0[I-1][g][h] - 2*dx2*Sigma_s0[I][g][h] )*f0[I][h]
           + 4*dx2*( ( Sigma_t[I][h] + 3/(8*dx) )*kron(g,h) - Sigma_s0[I][g][h] )*f2[I][h];
      }
      f0[I][g] = sor_omega * R/L + (1-sor_omega)*f0[I][g];
    }}

    // n = 2 equation
    {
    // left marshak boundary condition
    for ( auto g = 0 ; g < Egroups ; ++g ) {
      auto L = 3*D2[0][g][g] + D2[1][g][g]
             + 2*dx2*( 9.0/5.0*Sigma_t[0][g] - Sigma_s2[0][g][g] - 4.0/5.0*Sigma_s0[0][g][g] + 21.0/(40.0*dx) );
      auto R = -4.0/5.0*dx2*q[0][g];
      for ( auto h = 0 ; h < Egroups ; ++h ) {
        R += ( 3*D2[0][g][h] + D2[1][g][h] ) * f2[1][h]
           - (1-kron(g,h))*( 3*D2[0][g][h] + D2[1][g][h]
                           - 2*dx2*( Sigma_s2[0][g][h] + 4.0/5.0*Sigma_s0[0][g][h] ))*f2[0][h]
           + 4.0/5.0*dx2*( ( Sigma_t[0][h] + 3/(16*dx) )*kron(g,h) - Sigma_s0[0][g][h] )*f0[0][h] ;
      }
      f2[0][g] = sor_omega * R/L + (1-sor_omega)*f2[0][g];
    }
    // internal zones
    for ( auto i = 1 ; i < xints ; ++i ) {
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        auto L = D2[i-1][g][g] + 2*D2[i][g][g] + D2[i+1][g][g]
               + 2*dx2*( 9.0/5.0*Sigma_t[i][g] - Sigma_s2[i][g][g] - 4.0/5.0*Sigma_s0[i][g][g] );
        auto R = -4.0/5.0*dx2*q[i][g];
        for ( auto h = 0 ; h < Egroups ; ++h ) {
          R += ( D2[i-1][g][h] + D2[i][g][h] ) * f2[i-1][h]
             + ( D2[i+1][g][h] + D2[i][g][h] ) * f2[i+1][h]
             - (1-kron(g,h))*( D2[i-1][g][h] + 2*D2[i][g][h] + D2[i+1][g][h]
                             - 2*dx2*( Sigma_s2[i][g][h] + 4.0/5.0*Sigma_s0[i][g][h] ))*f2[i][h]
             + 4.0/5.0*dx2*( Sigma_t[i][h]*kron(g,h) - Sigma_s0[i][g][h] )*f0[i][h];
        }
        f2[i][g] = sor_omega * R/L + (1-sor_omega)*f2[i][g];
      }
    }
    // right marshak boundary condition
    auto I = xints;
    for ( auto g = 0 ; g < Egroups ; ++g ) {
      auto L = 3*D2[I][g][g] + D2[I-1][g][g]
             + 2*dx2*( 9.0/5.0*Sigma_t[I][g] - Sigma_s2[I][g][g] - 4.0/5.0*Sigma_s0[I][g][g] + 21.0/(40.0*dx) );
      auto R = -4.0/5.0*dx2*q[I][g];
      for ( auto h = 0 ; h < Egroups ; ++h ) {
        R += ( 3*D2[I][g][h] + D2[I-1][g][h] ) * f2[I-1][h]
           - (1-kron(g,h))*( 3*D2[I][g][h] + D2[I-1][g][h]
                           - 2*dx2*( Sigma_s2[I][g][h] + 4.0/5.0*Sigma_s0[I][g][h] ))*f2[I][h]
           + 4.0/5.0*dx2*( ( Sigma_t[I][h] + 3/(16*dx) )*kron(g,h) - Sigma_s0[I][g][h] )*f0[I][h];
      }
      f2[I][g] = sor_omega * R/L + (1-sor_omega)*f2[I][g];
    }}

    l2_error = 0.0;
    auto e2_diff = 0.0;
    auto e2_norm = 0.0;
    for ( auto i = 0 ; i <= xints ; ++i ) {
      for ( auto g = 0 ; g < Egroups ; ++g ) {
        e2_diff += pow( (f0_old[i][g] - f0[i][g]), 2 )
                 + pow( (f2_old[i][g] - f2[i][g]), 2 );
        e2_norm += pow( f0_old[i][g], 2) + pow( f2_old[i][g], 2);
      }
    }
    l2_error = sqrt( e2_diff / e2_norm );
    ++niter;
//    std::cout << niter << "  " << l2_error << '\n';
  } while ( l2_error > error_tol );

  // write out scalar flux
  std::ofstream outfile;
  outfile.open(output_file_name + "-" + std::to_string(nsolves + 1) + ".txt");
  for (auto i = 0; i <= xints; ++i) {
      const auto x = i * dx;
      outfile << x;
      for (auto g = 0; g < Egroups; ++g) {
          outfile << "   " << f0[i][g] - 2 * f2[i][g];
      }
      outfile << '\n';
  }
  outfile.close();
  ++nsolves;
}

std::vector<std::vector<double>> P3Solver::flux() {
  auto phi = std::vector<std::vector<double>>( xints+1, std::vector<double>( Egroups ));
  for ( auto i = 0 ; i <= xints ; ++i ) {
    for ( auto g = 0 ; g < Egroups ; ++g ) {
      phi[i][g] = f0[i][g] - 2*f2[i][g];
    }
  }
  return phi;
}

