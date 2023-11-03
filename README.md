## Direct estimates with temporal smoothing

Suppose we have some estimates for times $t=1,...,20$ with a variance covariance matrix:

$$
\hat{\theta} \sim \text{MVN}(\theta, V).
$$

But, these estimates may not be smooth in time and we would like to perform some smoothing. To do this, we choose some prior for $\theta$ and use TMB to maximize the posterior to get the MMAP estimate.

## Random Walk Order 2

The random-walk 2 model is:

$$
\begin{aligned}
    \theta_t &= \alpha + \delta_t + \epsilon_t \\
    (\delta_t - \delta_{t-1}) - (\delta_{t-1} - \delta_{t-2}) &\sim N(0, \tau_{\delta}^{-1}) \\
    \epsilon_t &\sim N(0, \tau_{\epsilon}^{-1})
\end{aligned}
$$

where $\alpha$ is an intercept, $\delta_t$ is a RW2 effect and $\epsilon_t$ is an IID random effect.

Another way to write $\delta$ is using an improper multivariate normal distribution with precision matrix $Q = R\tau_{\delta}$ where $R$ is the scaled RW2 structure matrix. So: $\delta \sim \text{MVN}(0, Q^{-1})$.

A sum-to-zero constraint on $\delta$ is used for identifiability.

We use $\tau_{\delta} \sim \text{gamma}(1, 5 \times 10^-4)$ and $\tau_{\epsilon} \sim \text{gamma}(1, 5 \times 10^-4)$ to match INLA default hyperpriors.
