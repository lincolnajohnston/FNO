#include <algorithm>

#include "material.hpp"

xs1d_table::xs1d_table( std::initializer_list<double> vals ) {
  _data.assign( vals );
}

std::ostream& operator<<( std::ostream& os, const xs1d_table& xs ) {
  for ( auto i = 0 ; i < xs.size() ; ++i ) {
    os << xs(i) << "  ";
  }
  return os;
}

xs1d_table operator+( const xs1d_table& u, const xs1d_table& v ) {
  auto n = v.size();
  auto R = static_cast<std::vector<double>>( u );
  for ( auto i = 0 ; i < n ; ++i ) {
    R[i] += v(i);
  }
  return xs1d_table( R );
}
xs1d_table operator-( const xs1d_table& u, const xs1d_table& v ) {
  auto n = v.size();
  auto R = static_cast<std::vector<double>>( u );
  for ( auto i = 0 ; i < n ; ++i ) {
    R[i] -= v(i);
  }
  return xs1d_table( R );
}
xs1d_table operator*( const double& a, const xs1d_table& v ) {
  auto n = v.size();
  auto R = static_cast<std::vector<double>>( v );
  for ( auto i = 0 ; i < n ; ++i ) {
    R[i] *= a;
  }
  return xs1d_table( R );
}
// * operator between two 1d tables is elementwise multipliction
xs1d_table operator*( const xs1d_table& u, const xs1d_table& v ) {
  auto n = v.size();
  auto R = static_cast<std::vector<double>>( u );
  for ( auto i = 0 ; i < n ; ++i ) {
    R[i] *= v(i);
  }
  return xs1d_table( R );
}
// * operator between a 2d and a 1d table is matrix vector multiplication
xs1d_table operator*( const xs2d_table& A, const xs1d_table& v ) {
  auto n = v.size();
  auto R = std::vector<double>( n, 0.0 );
  for ( auto i = 0 ; i < n ; ++i ) {
    for ( auto j = 0 ; j < n ; ++j ) {
      R[i] += A(i,j) * v(j);
    }
  }
  return xs1d_table( R );
}
// / operator is elementwise multipliction where 0/0 = 0
xs1d_table operator/( const xs1d_table& u, const xs1d_table& v ) {
  auto n = v.size();
  auto R = static_cast<std::vector<double>>( u );
  for ( auto i = 0 ; i < n ; ++i ) {
    if ( R[i] != 0.0 ) {
      R[i] /= v(i);
    }
  }
  return xs1d_table( R );
}

xs1d_table& xs1d_table::operator+=( const xs1d_table& v ) {
  auto& R = *this;

  // if lhs is an empty matrix prior to operation, return a copy of v
  // this permits use of +=, -+, etc in a loop without needing to appropriately size
  if ( R.size() == 0 ) {
    R._data = static_cast<std::vector<double>>( v );
    return R;
//    return const_cast<xs1d_table&>(v);
  }
  R = R + v;
  return R;
}

xs2d_table::xs2d_table( std::initializer_list<double> vals ) {
  size_t n = std::sqrt( vals.size() );
  auto aij = vals.begin();
  _data.resize( n, std::vector<double>( n, 0.0 ));
  for ( auto i = 0 ; i < n ; ++i ) {
    for ( auto j = 0 ; j < n ; ++j ) {
      _data[i][j] = *aij;
      aij++;
    }
  }
}

// +/- operators for 1d table, which gets turned into a diagonal 2d matrix, and 2d table (matrix)
xs2d_table operator+( const xs1d_table& v, const xs2d_table& A ) {
  auto n = v.size();
  auto R = static_cast<std::vector<std::vector<double>>>( A );
  for ( auto i = 0 ; i < n ; ++i ) {
    R[i][i] += v(i);
  }
  return xs2d_table( R );
}
xs2d_table operator-( const xs1d_table& v, const xs2d_table& A ) {
  auto n = v.size();
  auto R = static_cast<std::vector<std::vector<double>>>( -1.0*A );
  for ( auto i = 0 ; i < n ; ++i ) {
    R[i][i] += v(i);
  }
  return xs2d_table( R );
}

// +/- operators for 2d tables
xs2d_table operator+( const xs2d_table& A, const xs2d_table& B ) {
  auto n = A.size();
  auto R = static_cast<std::vector<std::vector<double>>>( A );
  for ( auto i = 0 ; i < n ; ++i ) {
    for ( auto j = 0 ; j < n ; ++j ) {
      R[i][j] += B(i,j);
    }
  }
  return xs2d_table( R );
}
xs2d_table operator-( const xs2d_table& A, const xs2d_table& B ) {
  auto n = A.size();
  auto R = static_cast<std::vector<std::vector<double>>>( A );
  for ( auto i = 0 ; i < n ; ++i ) {
    for ( auto j = 0 ; j < n ; ++j ) {
      R[i][j] -= B(i,j);
    }
  }
  return xs2d_table( R );
}

xs2d_table operator*( const double& a, const xs2d_table& M ) {
  auto n = M.size();
  auto R = static_cast<std::vector<std::vector<double>>>( M );
  for ( auto i = 0 ; i < n ; ++i ) {
    for ( auto j = 0 ; j < n ; ++j ) {
      R[i][j] *= a;
    }
  }
  return xs2d_table( R );
}

xs2d_table& xs2d_table::operator+=( const xs2d_table& A ) {
  auto& R = *this;

  // if lhs is an empty matrix prior to operation, return a copy of A
  // this permits use of += in a loop without needing to appropriately size
  if ( R.size() == 0 ) {
    R._data = static_cast<std::vector<std::vector<double>>>( A );
    return R;
    //return const_cast<xs2d_table&>(A);
  }
  R = R + A;
  return R;
}

std::ostream& operator<<( std::ostream& os, const xs2d_table& xs ) {
  for ( auto i = 0 ; i < xs.size() ; ++i ) {
    for ( auto j = 0 ; j < xs.size() ; ++j ) {
      os << xs(i,j) << "  ";
    }
    os << '\n';
  }
  return os;
}

void xs2d_table::transpose() {
  auto& A = *this;
  auto n = A.size();
  for ( auto i = 0 ; i < n ; ++i ) {
    for ( auto j = i+1 ; j < n ; ++j ) {
      std::swap( _data[i][j], _data[j][i] );
    }
  }
}

double xs2d_table::det() const {
  auto& A = *this;
  auto n = A.size();
  if ( n == 0 ) {
    // by definition, deteriminant of 0x0 matrix is one
    return 1.0;
  }
  else if ( n == 1 ) {
    // for 1x1 matrix, return value of matrix
    return A(0,0);
  }
  else if ( n == 2 ) {
    // return 2x2 determinant formula
    return A(0,0)*A(1,1) - A(0,1)*A(1,0);
  }
  else {
    // decompose into n (n-1)x(n-1) determinants
    double s = 0.0;
    auto parity = 1;
    for ( auto j = 0 ; j < n ; ++j ) {
      // loop over columns
      if ( A(0,j) != 0.0 ) {
        // construct the submatrix
        std::vector<std::vector<double>> V( n-1, std::vector<double>( n-1, 0.0 ) );
	auto m = 0;
	for ( auto l = 0 ; l < n ; ++l ) {
          // copy over 1:n-1 elements of column l, excluding column j
          if ( l != j ) {
            for ( auto k = 1 ; k < n ; ++k ) {
              // loop over rows
              V[k-1][m] = A(k,l);
            }
            m++;
          }
        }
        const xs2d_table B( V );
        s += parity * A(0,j) * B.det();
      }
      parity *= -1;
    }
    return s;
  }
}

xs2d_table xs2d_table::inverse() {
  const auto A = *this;
  const auto n = A.size();
  std::vector<std::vector<double>> R( n, std::vector<double>( n, 0.0 ) );

  // compute normalized transpose of cofactor matrix
  const auto detA = A.det();
  for ( auto i = 0 ; i < n ; ++i ) {
    for ( auto j = 0 ; j < n ; ++j ) {
      std::vector<std::vector<double>> V( n-1, std::vector<double>( n-1, 0.0 ) );
      uint32_t row = 0;
      for ( auto k = 0 ; k < n ; ++k ) {
        if ( k == i ) continue;
        uint32_t col = 0;
        for ( auto l = 0 ; l < n ; ++l ) {
          if ( l == j ) continue;
          V[row][col] = A(k,l);
          ++col;
        }
        ++row;
      }
      xs2d_table C( V );
      R[j][i] = pow(-1,i+j)*C.det()/detA;
    }
  }
  return xs2d_table(R);
}

MaterialXS Material::gen_xs() const {

  MaterialXS xs;
  for ( auto i = 0 ; i < atom_den.size() ; ++i ) {
    xs.SigmaT   += atom_den[i] * isotope[i]->sigmaT;
    xs.SigmaF   += atom_den[i] * isotope[i]->sigmaF;
    xs.nuSigmaF += atom_den[i] * isotope[i]->nubar * isotope[i]->sigmaF;
    xs.chi      += atom_den[i] * isotope[i]->chi * isotope[i]->nubar * isotope[i]->sigmaF;

    xs.SigmaS0 += atom_den[i] * isotope[i]->sigmaS0;
    xs.SigmaS1 += atom_den[i] * isotope[i]->sigmaS1;
    xs.SigmaS2 += atom_den[i] * isotope[i]->sigmaS2;
    xs.SigmaS3 += atom_den[i] * isotope[i]->sigmaS3;
  }
  // diffusion coefficients
  xs.D0 = 1.0/3.0  * ( xs.SigmaT - xs.SigmaS1 ).inverse();
  xs.D2 = 9.0/35.0 * ( xs.SigmaT - xs.SigmaS3 ).inverse();

  // normalize fission spectrum
  xs.chi = xs.chi / xs.nuSigmaF;

  return xs;
}

MaterialDepletionData Material::gen_deplete() const {

  MaterialDepletionData depl_data;

  auto Egroups = isotope[0]->sigmaT.size();
  depl_data.rxn_xs.resize( Egroups );

  // construct fission yield vector
  std::vector<double> fyield_vec( isotope.size() );
  for ( auto i = 0 ; i < isotope.size() ; ++i ) {
    fyield_vec[i] = isotope[i]->fyield;
  }
  depl_data.fyield = xs1d_table( fyield_vec );

  // construct decay matrix
  std::vector<std::vector<double>> dec_mat( isotope.size(), std::vector<double>( isotope.size(), 0.0 ) );
  for ( auto i = 0 ; i < isotope.size() ; ++i ) {
    auto name   = isotope[i]->name;
    if ( isotope[i]->decay.first > 0.0 ) {
      dec_mat[i][i] = -isotope[i]->decay.first;
    }
    for ( auto j = 0 ; j < isotope.size() ; ++j ) {
      if ( isotope[j]->decay.second == name ) {
        dec_mat[i][j] = isotope[j]->decay.first;
        break;
      }
    }
  }
  depl_data.decay = xs2d_table( dec_mat );

  // construct reaction xs matrices
  for ( auto g = 0 ; g < Egroups ; ++g ) {
    std::vector<std::vector<double>> rxn_mat( isotope.size(), std::vector<double>( isotope.size(), 0.0 ) );
    for ( auto i = 0 ; i < isotope.size() ; ++i ) {
      auto name = isotope[i]->name;

      if ( isotope[i]->sigmaGamma.first.size() > 0 ) {
        rxn_mat[i][i] = -isotope[i]->sigmaGamma.first(g);
      }
      for ( auto j = 0 ; j < isotope.size() ; ++j ) {
        if ( isotope[j]->sigmaGamma.second == name ) {
          rxn_mat[i][j] = isotope[j]->sigmaGamma.first(g);
          break;
        }
      }
    }
    depl_data.rxn_xs[g] = xs2d_table( rxn_mat );
  }

  return depl_data;
}


