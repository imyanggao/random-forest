#ifndef INTEGRATION_H
#define INTEGRATION_H

#define _USE_MATH_DEFINES

#include <math.h>
#include "linearalgebra.h"

typedef std::size_t size_t;
typedef std::size_t index_t;


inline int IsNan(double x)
{
   volatile double temp = x;
   return temp != x;
}

inline int IsInf(double x)
{
   volatile double temp = x;
   if ((temp == x) && ((temp - x) != 0.0))
      return (x < 0.0 ? -1 : 1);
   else return 0;
}

// Implementation of Marsaglia (2004) based on Taylor series expansion.
template<class T>
T UnivariateNormalCDF(const T x)
{
  // use order parameter to achieve desired precision
  const index_t order = 100;
  T sum = x;
  T value = x;
  for (index_t i = 1; i <= order; ++i)
    {
      value = value * x * x / (2 * i + 1);
      sum += value;
    }

  if (IsInf(sum) || IsInf(-sum) || IsInf(exp(x * x)))
    {
      if (x > 0)
        {
          return 1;
        }
      else
        {
          return 0;
        }
    }
  else
    {
      return (0.5 + sum / sqrt(2 * M_PI) * exp((x * x) / -2.0));
    }
}

//// http://home.online.no/~pjacklam/notes/invnorm/
//template<class T>
//T UnivariateNormalCDFInverse(T p)
//{

//  static const double a[] =
//  {
//    -3.969683028665376e+01,
//    2.209460984245205e+02,
//    -2.759285104469687e+02,
//    1.383577518672690e+02,
//    -3.066479806614716e+01,
//    2.506628277459239e+00
//  };

//  static const double b[] =
//  {
//    -5.447609879822406e+01,
//    1.615858368580409e+02,
//    -1.556989798598866e+02,
//    6.680131188771972e+01,
//    -1.328068155288572e+01
//  };

//  static const double c[] =
//  {
//    -7.784894002430293e-03,
//    -3.223964580411365e-01,
//    -2.400758277161838e+00,
//    -2.549732539343734e+00,
//    4.374664141464968e+00,
//    2.938163982698783e+00
//  };

//  static const double d[] =
//  {
//    7.784695709041462e-03,
//    3.224671290700398e-01,
//    2.445134137142996e+00,
//    3.754408661907416e+00
//  };

//  //Define break-points.
//  double p_low =  0.02425;
//  double p_high = 1 - p_low;
//  double q, r;
//  T x;


//  if (p < 0 || p > 1)
//    {
//      x = 0.0;
//    }
//  else if (p == 0)
//    {
//      x = -HUGE_VAL;
//    }
//  else if (p == 1)
//    {
//      x = HUGE_VAL;
//    }

//  //Rational approximation for lower region.
//  else if (p < p_low)
//    {
//      q = sqrt(-2*log(p));
//      x = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
//          ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
//    }

//  //Rational approximation for upper region.
//  else if (p > p_high)
//    {
//      q = sqrt(-2*log(1-p));
//      x = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
//          ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
//    }

//  //Rational approximation for central region.
//  else
//    {
//      q = p - 0.5;
//      r = q*q;
//      x = (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
//          (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
//    }

//  return x;
//}


// Given n and a, the form of the polynomial is:
// p(x) = a[0] + a[1] * x + ... + a[n-2] * x^(n-2) + a[n-1] * x^(n-1)
template<class T>
inline T PolyValue(const int n, const T a[], const T x)
{
  int i;
  T value;

  value = 0.0;
  for ( i = n-1; 0 <= i; i-- )
    {
      value = value * x + a[i];
    }
  return value;
}

// http://people.sc.fsu.edu/~jburkardt/cpp_src/asa241/asa241.html
template<class T>
T UnivariateNormalCDFInverse(const T p)
{
  static double a[8] = {
    3.3871328727963666080,     1.3314166789178437745e+2,
    1.9715909503065514427e+3,  1.3731693765509461125e+4,
    4.5921953931549871457e+4,  6.7265770927008700853e+4,
    3.3430575583588128105e+4,  2.5090809287301226727e+3 };
  static double b[8] = {
    1.0,                       4.2313330701600911252e+1,
    6.8718700749205790830e+2,  5.3941960214247511077e+3,
    2.1213794301586595867e+4,  3.9307895800092710610e+4,
    2.8729085735721942674e+4,  5.2264952788528545610e+3 };
  static double c[8] = {
    1.42343711074968357734,     4.63033784615654529590,
    5.76949722146069140550,     3.64784832476320460504,
    1.27045825245236838258,     2.41780725177450611770e-1,
    2.27238449892691845833e-2,  7.74545014278341407640e-4 };
  static double const1 = 0.180625;
  static double const2 = 1.6;
  static double d[8] = {
    1.0,                        2.05319162663775882187,
    1.67638483018380384940,     6.89767334985100004550e-1,
    1.48103976427480074590e-1,  1.51986665636164571966e-2,
    5.47593808499534494600e-4,  1.05075007164441684324e-9 };
  static double e[8] = {
    6.65790464350110377720,     5.46378491116411436990,
    1.78482653991729133580,     2.96560571828504891230e-1,
    2.65321895265761230930e-2,  1.24266094738807843860e-3,
    2.71155556874348757815e-5,  2.01033439929228813265e-7 };
  static double f[8] = {
    1.0,                        5.99832206555887937690e-1,
    1.36929880922735805310e-1,  1.48753612908506148525e-2,
    7.86869131145613259100e-4,  1.84631831751005468180e-5,
    1.42151175831644588870e-7,  2.04426310338993978564e-15 };
  double q;
  double r;
  static double split1 = 0.425;
  static double split2 = 5.0;
  T value;

  if ( p <= 0 )
  {
    return -HUGE_VAL;
  }

  if ( p >= 1 )
  {
    return HUGE_VAL;
  }

  q = p - 0.5;

  if ( fabs ( q ) <= split1 )
  {
    r = const1 - q * q;
    value = q * PolyValue ( 8, a, r ) / PolyValue ( 8, b, r );
  }
  else
  {
    if ( q < 0.0 )
    {
      r = p;
    }
    else
    {
      r = 1.0 - p;
    }

    if ( r <= 0.0 )
    {
      value = -1.0;
      exit ( 1 );
    }

    r = sqrt ( -log ( r ) );

    if ( r <= split2 )
    {
      r = r - const2;
      value = PolyValue ( 8, c, r ) / PolyValue ( 8, d, r );
     }
     else
     {
       r = r - split2;
       value = PolyValue ( 8, e, r ) / PolyValue ( 8, f, r );
    }

    if ( q < 0.0 )
    {
      value = -value;
    }

  }

  return value;
}

// Numerical Computation of Multivariate Normal Probabilities, Alan Genz, 1992
template<class T>
T MultivariateNormalIntegral(const std::vector<T>& Mu,
                             const std::vector<std::vector<T> >& Sigma,
                             const std::vector<T>& a,
                             const std::vector<T>& b,
                             const T epsilon,
                             const int NMax = 100000,
                             const T alpha = 2.5)
{
  size_t dim = Sigma.size();
  std::vector<std::vector<T> > C;
  C.resize(dim);
  for (index_t i = 0; i < dim; ++i)
    {
      C[i].resize(dim);
    }
  cholesky(Sigma, C);

  std::vector<T> aShift;
  std::vector<T> bShift;
  aShift.resize(dim);
  bShift.resize(dim);
  for (index_t i = 0; i < dim; ++i)
    {
      aShift[i] = a[i] - Mu[i];
      bShift[i] = b[i] - Mu[i];
    }

  std::vector<T> d;
  std::vector<T> e;
  std::vector<T> f;
  std::vector<T> w;
  std::vector<T> y;
  d.resize(dim);
  e.resize(dim);
  f.resize(dim);
  w.resize(dim-1);
  y.resize(dim-1);

  T Intsum = 0;
  T Varsum = 0;
  T tmp = 0;
  T delta = 0;
  int N = 0;

  if (IsInf(-aShift[0]))
    {
      d[0] = 0;
    }
  else
    {
      d[0] = UnivariateNormalCDF(aShift[0] / C[0][0]);
    }

  if (IsInf(bShift[0]))
    {
      e[0] = 1;
    }
  else
    {
      e[0] = UnivariateNormalCDF(bShift[0] / C[0][0]);
    }

  f[0] = e[0] - d[0];

  T Error = HUGE_VAL;
  srand(time(NULL));
  while(Error >= epsilon && N < NMax)
    {
      for (index_t i = 0; i < (dim-1); ++i)
        {
          w[i] = (double) rand() / RAND_MAX;
        }

      for (index_t i = 1; i < dim; ++i)
        {
          if (!IsInf(-aShift[i]) || !IsInf(bShift[i]))
            {
              y[i-1] = UnivariateNormalCDFInverse(d[i-1] + w[i-1] * (e[i-1] - d[i-1]));
              tmp = 0;
              for (index_t j = 0; j < i; ++j)
                {
                  if (C[i][j] != 0)
                    {
                      tmp += C[i][j] * y[j];
                    }
                }
            }

          if (IsInf(-aShift[i]))
            {
              d[i] = 0;
            }
          else
            {
              d[i] = UnivariateNormalCDF((aShift[i] - tmp) / C[i][i]);
            }

          if (IsInf(bShift[i]))
            {
              e[i] = 1;
            }
          else
            {
              e[i] = UnivariateNormalCDF((bShift[i] - tmp) / C[i][i]);
            }

          f[i] = (e[i] - d[i]) * f[i-1];
        }

      N = N + 1;
      delta = (f[dim-1] - Intsum) / N;
      Intsum = Intsum + delta;
      Varsum = (N - 1) * Varsum / N + delta * delta;
      Error = alpha * sqrt(Varsum);
    }

//  std::cout << "\n   --- Integration ---   " << std::endl;
//  std::cout << "Mean : ";
//  for (index_t i = 0; i < dim; ++i)
//    {
//      std::cout << Mu[i] << " ";
//    }
//  std::cout << std::endl;
//  std::cout << "Cov  : ";
//  for (index_t i = 0; i < dim; ++i)
//    {
//      for (index_t j = 0; j < dim; ++j)
//        {
//          std::cout << Sigma[i][j] << " ";
//        }
//      if (i < (dim-1))
//        {
//          std::cout << std::endl << "       ";
//        }
//      else
//        {
//          std::cout << std::endl;
//        }
//    }
//  std::cout << "upper : ";
//  for (index_t i = 0; i < dim; ++i)
//    {
//      std::cout << b[i] << " ";
//    }
//  std::cout << std::endl;
//  std::cout << "lower : ";
//  for (index_t i = 0; i < dim; ++i)
//    {
//      std::cout << a[i] << " ";
//    }
//  std::cout << std::endl;

//  std::cout << "Intsum = " << Intsum << ", Varsum = " << Varsum
//            << ", delta = " << delta << std::endl;

//  std::cout << "Error = " << Error << ", N = " << N << std::endl;

  return Intsum;
}

#endif // INTEGRATION_H
