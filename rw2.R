
# RW2 temporal smoothing with TMB

library(tidyverse)
library(TMB)
library(INLA)

# example data
# log-logistic survival model fit, for illustration smooth log-shape parameter only
load("2014_loglogistic.RData")
df <- fit_loglogistic$result
V <- fit_loglogistic$variance[1:20, 1:20]
n_years <- 20


# TMB ---------------------------------------------------------------------

# RW2 structure matrix
inla.rw = utils::getFromNamespace("inla.rw", "INLA")
R <- inla.rw(n_years, order = 2, sparse = T, scale.model = T)

# load TMB model
TMB::compile("rw2.cpp")
dyn.load(TMB::dynlib("rw2"))

# define negative posterior, proportional to negative log-likelihood x prior
nll <- TMB::MakeADFun(
  data = list(thetahat = df$log_shape_mean,
              V = V,
              R = R),
  parameters = list(theta = rep(mean(df$log_shape_mean), n_years),
                    intercept = mean(df$log_shape_mean),
                    delta = rep(0, n_years),
                    epsilon = rep(0, n_years),
                    log_tau_delta = log(4),
                    log_tau_epsilon = log(4))
)

# optimize to get MMAP estimate and standard error
opt <- nlminb(nll$par, nll$fn, nll$gr)
ests <- TMB::sdreport(nll)

# plot
df$log_shape_smoothed <- ests$par.fixed[1:20]
df$log_shape_smoothed_var <- diag(ests$cov.fixed)[1:20]
ggplot(data = df, aes(x = period)) +
  geom_point(aes(y = log_shape_mean)) +
  geom_line(aes(y = log_shape_smoothed)) +
  geom_ribbon(aes(ymin = log_shape_smoothed - 1.96*sqrt(log_shape_smoothed_var),
                  ymax = log_shape_smoothed + 1.96*sqrt(log_shape_smoothed_var)),
              alpha = 0.2) +
  theme_bw()


# inla --------------------------------------------------------------------

# for comparison fit a RW2 model in INLA
df$period_copy <- df$period
formula <- log_shape_mean ~ 
  f(period, model = "rw2", constr = T, scale.model = T) +
  f(period_copy, model = "iid")
fit_inla <- inla(
  formula = formula,
  data = df,
  family = "gaussian",
  control.family = list(hyper = list(prec = list(initial = log(1), fixed = TRUE))),
  scale = 1 / df$log_shape_var, # fixed known variance (note: only uses diagonals, not whole V matrix)
  control.predictor = list(compute = TRUE),
  control.compute = list(return.marginals = TRUE, config = TRUE,
                         return.marginals.predictor = TRUE)
)

# plot
df$log_shape_smoothed_inla <- fit_inla$summary.fitted.values$`0.5quant`
df$log_shape_smoothed_lower_inla <- fit_inla$summary.fitted.values$`0.025quant`
df$log_shape_smoothed_upper_inla <- fit_inla$summary.fitted.values$`0.975quant`
ggplot(data = df, aes(x = period)) +
  geom_point(aes(y = log_shape_mean)) +
  geom_line(aes(y = log_shape_smoothed_inla)) +
  geom_ribbon(aes(ymin = log_shape_smoothed_lower_inla,
                  ymax = log_shape_smoothed_upper_inla),
              alpha = 0.2) +
  theme_bw()