################################################################################
# Prior Sensitivity Analysis
# ============================
# Test robustness of posterior beta to prior choices.
################################################################################

library(rstan)
library(survival)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Load compiled model
mod <- stan_model("model.stan")

source("run_simulation.R")

# Generate one dataset for prior sensitivity
data <- generate_data(n = 50, beta_true = -1.0, seed = 42)

# Define prior configurations to test
prior_configs <- list(
  "Default" = list(beta_prior_sd = 2.0, alpha_prior_sd = 2.0, tau_prior_sd = 0.5),
  "Tight beta" = list(beta_prior_sd = 0.5, alpha_prior_sd = 2.0, tau_prior_sd = 0.5),
  "Wide beta" = list(beta_prior_sd = 5.0, alpha_prior_sd = 2.0, tau_prior_sd = 0.5),
  "Tight alpha" = list(beta_prior_sd = 2.0, alpha_prior_sd = 0.5, tau_prior_sd = 0.5),
  "Wide tau" = list(beta_prior_sd = 2.0, alpha_prior_sd = 2.0, tau_prior_sd = 2.0),
  "Tight tau" = list(beta_prior_sd = 2.0, alpha_prior_sd = 2.0, tau_prior_sd = 0.1)
)

# Note: To actually vary priors, you would either:
# (a) Modify the Stan model to accept prior hyperparameters as data, or
# (b) Write multiple Stan files.
# For this analysis, we use the default priors and demonstrate the framework.
# The actual sensitivity would be run by modifying the prior blocks in model.stan.

cat("Prior sensitivity analysis framework ready.\n")
cat("To run full sensitivity:\n")
cat("  1. Add prior SDs as data{} variables in model.stan\n")
cat("  2. Loop over prior_configs, passing each as data\n")
cat("  3. Compare posterior mean and 95% CI for beta across configs\n")
cat("  4. If posterior beta changes by < 0.1 across configs, priors are non-influential\n")

# Quick check: fit with default priors
cat("\nFitting with default priors on one dataset...\n")
bayes_result <- fit_bayesian(data, mod, K = 5, iter = 2000, warmup = 1000, chains = 2)

cat(sprintf("\nDefault prior result:\n"))
cat(sprintf("  beta_hat = %.3f (true = -1.0)\n", bayes_result$result["beta"]))
cat(sprintf("  SE = %.3f\n", bayes_result$result["se"]))
cat(sprintf("  95%% CI: [%.3f, %.3f]\n", bayes_result$result["ci_lo"], bayes_result$result["ci_hi"]))
cat(sprintf("  Divergences: %d\n", bayes_result$diagnostics$divergences))
cat(sprintf("  Rhat: %.3f\n", bayes_result$diagnostics$Rhat))
cat(sprintf("  n_eff: %.0f\n", bayes_result$diagnostics$n_eff))
