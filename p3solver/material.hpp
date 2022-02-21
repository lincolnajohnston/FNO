#ifndef _MATERIAL_HEADER_
#define _MATERIAL_HEADER_

#include <cmath>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

// classes for cross section vector and (square) matrix
class xs1d_table {
  private:
    std::vector<double> _data;
  public:
    xs1d_table() {};
    xs1d_table( std::vector<double> data ) : _data(data) {};
    xs1d_table( std::initializer_list<double> vals );

    double operator()( uint32_t g ) const { return _data[g]; };
    operator std::vector<double>() const { return _data; };

    friend xs1d_table operator+( const xs1d_table& u, const xs1d_table& v );
    friend xs1d_table operator-( const xs1d_table& u, const xs1d_table& v );
    friend xs1d_table operator*( const double& a, const xs1d_table& v );
    friend xs1d_table operator*( const xs1d_table& u, const xs1d_table& v );
    friend xs1d_table operator/( const xs1d_table& u, const xs1d_table& v );
    xs1d_table& operator+= ( const xs1d_table& );

    uint32_t size() const { return _data.size(); }
    friend std::ostream& operator<<( std::ostream&, const xs1d_table& );
};
class xs2d_table {
  private:
    std::vector<std::vector<double>> _data;

  public:
    xs2d_table() {};
    xs2d_table( std::vector<std::vector<double>> data ) : _data(data) {};
    xs2d_table( std::initializer_list<double> vals );

    double operator()( uint32_t g, uint32_t h ) const { return _data[g][h]; };
    operator std::vector<std::vector<double>>() const { return _data; };

    friend xs2d_table operator+( const xs1d_table& A, const xs2d_table& B );
    friend xs2d_table operator-( const xs1d_table& A, const xs2d_table& B );

    friend xs2d_table operator+( const xs2d_table& A, const xs2d_table& B );
    friend xs2d_table operator-( const xs2d_table& A, const xs2d_table& B );

    friend xs2d_table operator*( const double& a, const xs2d_table& B );
    friend xs1d_table operator*( const xs2d_table& A, const xs1d_table& v );

    xs2d_table& operator+= ( const xs2d_table& );

    uint32_t size() const { return _data.size(); }
    friend std::ostream& operator<<( std::ostream&, const xs2d_table& );

    double det() const;
    xs2d_table inverse();
    void transpose();
};

// isotopic cross sections
class IsotopeXS {
  public:
    std::string name;

    // transport data
    xs1d_table sigmaT, sigmaF, nubar, chi;
    xs2d_table sigmaS0, sigmaS1, sigmaS2, sigmaS3;

    // depletion data (pairs for data, product)
    double fyield = 0.0;
    std::pair<double,std::string> decay = std::make_pair( 0.0, "" );
    std::pair<xs1d_table,std::string> sigmaGamma;
};

// material cross section
class MaterialXS {
  public:
    xs1d_table SigmaT, SigmaF, nuSigmaF, chi;
    xs2d_table SigmaS0, SigmaS1, SigmaS2, SigmaS3;
    xs2d_table D0, D2;
};

// material depletion data
class MaterialDepletionData {
  public:
    xs1d_table fyield;
    xs2d_table decay;
    std::vector<xs2d_table> rxn_xs;
};

// material structure
class Material {
  public:
    std::vector<double> atom_den;
    std::vector<std::shared_ptr<IsotopeXS>> isotope;

    bool deplete = false;

    // computation of macroscopic xs for material
    MaterialXS gen_xs() const;
    MaterialDepletionData gen_deplete() const;
};

#endif
