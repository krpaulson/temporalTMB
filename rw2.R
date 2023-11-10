
# RW2 temporal smoothing with TMB
# Reference (Taylor Okonek): 
# https://github.com/taylorokonek/stbench/blob/main/R/tmb_u5mr_intercepts_iidtime_rw2s.R

library(tidyverse)
library(TMB)
library(INLA)

# example data
# log-logistic survival model fit, for illustration smooth log-shape parameter only
load("2014_loglogistic.RData")
df <- fit_loglogistic$result
V <- fit_loglogistic$variance[1:20, 1:20]
n_years <- 20

# just take diagonals to facilitate comparison to INLA
diags <- diag(V)
V <- matrix(0, nrow = 20, ncol = 20)
diag(V) <- diags


# TMB ---------------------------------------------------------------------

# RW2 structure matrix
inla.rw = utils::getFromNamespace("inla.rw", "INLA")
R <- inla.rw(n_years, order = 2, sparse = T, scale.model = T)
R <- R + diag(n_years) * 1e-6

# load TMB model
TMB::compile("rw2.cpp")
dyn.load(TMB::dynlib("rw2"))

# define negative posterior, proportional to negative log-likelihood x prior
obj <- TMB::MakeADFun(
  data = list(thetahat = df$log_shape_mean,
              V = V,
              R = R),
  parameters = list(intercept = 0,
                    delta_t = rep(0, n_years),
                    epsilon_t = rep(0, n_years),
                    log_tau_delta = 0,
                    log_tau_epsilon = 0),
  random = c("epsilon_t", "delta_t"),
  map = list(),
  hessian = TRUE,
  DLL = "rw2"
)

# optimize to get MMAP estimate and standard error
opt <- nlminb(obj$par, obj$fn, obj$gr)
SD0 <- TMB::sdreport(obj,
                     getJointPrecision = TRUE,
                     getReportCovariance = TRUE,
                     bias.correct = TRUE,
                     bias.correct.control = list(sd = TRUE))


# TMB post-processing ------------------------------------------------------

# obtain point estimates (means) for fixed and random effects
mu <- c(SD0$par.fixed, SD0$par.random)

# get ids for different parameters
t.parnames <- names(mu)
t.intercept.idx <- grep("intercept", t.parnames)
t.time.unstruct.idx <- grep("epsilon_t", t.parnames)
t.time.struct.idx <- grep("delta_t", t.parnames)

# create list of constraint matrices for these terms
A.mat.list <- list()
A.mat.list[[1]] <- matrix(1, nrow = 1, ncol = n_years)

# sample
# Reference: https://github.com/taylorokonek/stbench/blob/main/R/multiconstr_prec.R
multiconstr_prec = utils::getFromNamespace("multiconstr_prec", "stbench")
t.draws <- multiconstr_prec(mu = mu,
                            prec = SD0$jointPrecision,
                            n.sims = 1000,
                            constrain.idx.list = list(t.time.struct.idx),
                            A.mat.list = A.mat.list)

# take the constrained draws
t.draws <- t.draws$x.c

# combine draws for linear predictor
fitted <-
  matrix(rep(t.draws[t.intercept.idx,], n_years), nrow = n_years, byrow = T) +
  t.draws[t.time.unstruct.idx,] +
  t.draws[t.time.struct.idx,]

# get summaries
df$log_shape_smoothed_med <- apply(fitted, 1, quantile, 0.5)
df$log_shape_smoothed_lower <- apply(fitted, 1, quantile, 0.025)
df$log_shape_smoothed_upper <- apply(fitted, 1, quantile, 0.975)

# plot
ggplot(data = df, aes(x = period)) +
  geom_point(aes(y = log_shape_mean)) +
  geom_line(aes(y = log_shape_smoothed_med)) +
  geom_ribbon(aes(ymin = log_shape_smoothed_lower,
                  ymax = log_shape_smoothed_upper),
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


# compare -----------------------------------------------------------------

ggplot(data = df, aes(x = log_shape_smoothed_med, y = log_shape_smoothed_inla, color = period)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0) +
  theme_bw() +
  labs(x = "tmb", y = "inla")
