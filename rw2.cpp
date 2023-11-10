#include <TMB.hpp>

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
  PARAMETER_VECTOR(theta);
  PARAMETER(intercept);
  PARAMETER_VECTOR(delta);
  PARAMETER_VECTOR(epsilon);
  PARAMETER(log_tau_delta);
  PARAMETER(log_tau_epsilon);
  
  // linear predictor
  for (int i = 0; i < n_years; i++) {
    theta[i] = intercept + delta[i] + epsilon[i];
  }

  // initialize
  Type nll = 0.0;
    
  // likelihood (thetahat-theta ~ N(0,V))
  MVNORM_t<Type> neg_log_dmvnorm(V);
  nll += neg_log_dmvnorm(thetahat - theta);
  
  // intercept
  nll -= dnorm(intercept, Type(0.0), Type(31.62278), true);
  
  // RW2 (delta)
  Eigen::SparseMatrix<Type> Q(n_years, n_years);
  for (int i = 0; i < n_years; i++) {
    for (int j = 0; j < n_years; j++) {
      Q.coeffRef(i, j) = R.coeffRef(i, j) * exp(log_tau_delta);
    }
  }
  nll += GMRF(Q)(delta);
  
  // RW2 constraint
  Type sum_delta = delta.sum();
  nll -= dnorm(sum_delta, Type(0.0), Type(0.00001), true);
  
  // iid effects (epsilon)
  Type sd_epsilon = exp(-0.5 * log_tau_epsilon);
  for (int i = 0; i < n_years; i++) {
    nll -= dnorm(epsilon(i), Type(0.0), sd_epsilon, true);
  }
  
  // hyperpriors
  nll -= dlgamma(log_tau_delta, Type(1.0), Type(1/0.00005), true);
  nll -= dlgamma(log_tau_epsilon, Type(1.0), Type(1/0.00005), true);

  return nll;
}