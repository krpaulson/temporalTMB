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
  
  /////////////////////
  // get things from R
  /////////////////////
  
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
  
  ////////////////////////
  // priors + hyperpriors
  ////////////////////////
  
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
  
  ////////////////
  // constraints
  ////////////////
  
  // a constrained vector x_c for an unconstrained vector x is calculated as:
  // x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  // for constraint Ax=e, and for GMRF x with precision Q
  
  // Invert precision matrices (possible since we added a small value to the diagonal)
  matrix<Type> Q_inv_rw2 = invertSparseMatrix(Q);
  
  // Create constraint matrices A
  matrix<Type> A_rw2(1, n_years);
  for(int i = 0; i < n_years; i++) {
    A_rw2(0, i) = 1; // sum-to-0 constraint
  }

  // Create A^T
  matrix<Type> A_rw2_T = A_rw2.transpose();

  // Create Q^{-1}A^T
  matrix<Type> QinvA_rw2= Q_inv_rw2 * A_rw2_T;

  // Create AQ^{-1}A^T
  matrix<Type> AQinvA_rw2 = A_rw2 * QinvA_rw2;

  // Create (AQ^{-1}A^T)^{-1}
  matrix<Type> AQinvA_rw2_inv = AQinvA_rw2.inverse(); // okay for small matrices

  // Create Ax
  matrix<Type> Ax_rw2 = (A_rw2 * delta_t.matrix());

  // Convert Ax from matrix to vector form - needed for dnorm & MVNORM
  vector<Type> Ax_rw2_vec(1);
  for(int i = 0; i < 1; i++) {
    Ax_rw2_vec(i) = Ax_rw2(i,0);
  }
  
  // Convert Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e) to vector form for conditioning by kriging correction
  matrix<Type> krig_correct = QinvA_rw2 * AQinvA_rw2_inv * Ax_rw2;
  vector<Type> krig_correct_vec(n_years);
  for (int i = 0; i < n_years; i++) {
    krig_correct_vec(i) = krig_correct(i,0);
  }

  // Construct constrained vector x_c = x - Q^{-1}A'(AQ^{-1}A')^{-1}(Ax-e)
  vector<Type> delta_t_c = delta_t - krig_correct_vec;
  
  ////////////////
  // likelihood
  ////////////////
  
  // likelihood (thetahat-theta ~ N(0,V)); theta = intercept + delta + epsilon
  vector<Type> x(n_years);
  for (int i = 0; i < n_years; i++) {
    x(i) = thetahat(i) - intercept - delta_t_c(i) - epsilon_t(i);
  }
  MVNORM_t<Type> neg_log_dmvnorm(V);
  nll += neg_log_dmvnorm(x);

  return nll;
}