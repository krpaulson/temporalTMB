#include <TMB.hpp>

// Reference: https://github.com/taylorokonek/stbench/blob/main/src/TMB/u5mr_intercepts_iidtime_rw2s.hpp

// thetahat ~ N(theta, V)
// theta = intercept + delta_t + epsilon_t
// delta_t ~ RW2(tau_delta)
// epsilon_t ~ IID(tau_epsilon)
// tau_delta ~ gamma(1, 5*10^-4)
// tau_epsilon ~ gamma(1, 5*10^-4)

template<class Type>
Type objective_function<Type>::operator() ()
{
  using namespace density;
  using namespace Eigen;
  
  // data
  DATA_VECTOR(thetahat);
  DATA_MATRIX(V);
  DATA_SPARSE_MATRIX(R); // Structure matrix for random walk
  int n_years = V.rows();
  
  // parameters
  PARAMETER(intercept);
  PARAMETER_VECTOR(delta_t);
  PARAMETER_VECTOR(epsilon_t);
  PARAMETER(log_tau_delta);
  PARAMETER(log_tau_epsilon);
  
  // initialize
  Type nll = 0.0;
  
  // intercept
  nll -= dnorm(intercept, Type(0.0), Type(31.62278), true);
  
  // RW2 (delta)
  Eigen::SparseMatrix<Type> Q(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta);
    }
  }
  nll += GMRF(Q)(delta_t);
  
  // iid effects (epsilon)
  Type sd_epsilon = exp(-0.5 * log_tau_epsilon);
  for (int i = 0; i < n_years; i++) {
    nll -= dnorm(epsilon_t(i), Type(0.0), sd_epsilon, true);
  }
  
  // hyperpriors
  nll -= dlgamma(log_tau_delta, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon, Type(1.0), Type(1/0.00005), true);
  
  // likelihood (thetahat-theta ~ N(0,V)); theta = intercept + delta + epsilon
  vector<Type> x(n_years);
  for (int i = 0; i < n_years; i++) {
    x(i) = thetahat(i) - intercept - delta_t(i) - epsilon_t(i);
  }
  MVNORM_t<Type> neg_log_dmvnorm(V);
  nll += neg_log_dmvnorm(x);

  return nll;
}