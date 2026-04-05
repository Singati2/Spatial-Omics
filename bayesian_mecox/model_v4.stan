// Bayesian Joint ME-Cox v4 — Fixed variance model, only tau estimated
// =====================================================================
// Key change: sigma_i^2 = W_i(1-W_i)*DEFF_i/m_i * exp(tau*z_i)
// The design-effect variance is computed in data (not estimated).
// Only tau (residual heterogeneity) is estimated.
// This eliminates the alpha0/alpha1/alpha2 near-non-identifiability.

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> K;
  vector[N] W;
  matrix[N, P] Z;
  vector<lower=0>[N] T_obs;
  array[N] int<lower=0, upper=1> event;
  vector[N] log_sig2_theory;   // log(W*(1-W)*DEFF/m), precomputed
  vector[K-1] s;
}

transformed data {
  array[N] int<lower=1, upper=K> ki;
  for (i in 1:N) {
    ki[i] = K;
    for (k in 1:(K-1)) {
      if (T_obs[i] <= s[k]) { ki[i] = k; break; }
    }
  }
}

parameters {
  real mu_eta;
  real<lower=0> sigma_eta;
  vector[N] z_eta;
  real beta;
  vector[P] gamma_z;
  vector[K] log_lam0;
  real<lower=0> tau;            // only scale parameter estimated
  vector[N] z_sig;              // non-centered residual
}

transformed parameters {
  vector[N] eta_i = mu_eta + sigma_eta * z_eta;
  vector[N] X = inv_logit(eta_i);

  // Variance: theory-based center + random residual
  vector[N] log_sig2 = log_sig2_theory + tau * z_sig;
  vector<lower=0>[N] sig;
  for (i in 1:N) {
    sig[i] = sqrt(exp(log_sig2[i]) + 1e-6);
  }

  vector<lower=0>[K] lam0 = exp(log_lam0);
}

model {
  mu_eta ~ normal(-0.6, 1.5);
  sigma_eta ~ normal(0, 1);
  z_eta ~ std_normal();

  beta ~ normal(0, 2);
  gamma_z ~ normal(0, 1);
  log_lam0 ~ normal(-2, 1);

  tau ~ normal(0, 0.5);         // tight: theory-based center is trusted
  z_sig ~ std_normal();

  W ~ normal(X, sig);

  {
    vector[N] lin = beta * X + Z * gamma_z;
    vector[N] exp_lin = exp(lin);
    for (i in 1:N) {
      real cum_haz = 0; real prev = 0;
      for (k in 1:(ki[i]-1)) {
        cum_haz += lam0[k] * (s[k] - prev) * exp_lin[i]; prev = s[k];
      }
      cum_haz += lam0[ki[i]] * (T_obs[i] - prev) * exp_lin[i];
      if (event[i] == 1) target += log(lam0[ki[i]]) + lin[i] - cum_haz;
      else target += -cum_haz;
    }
  }
}

generated quantities {
  real HR = exp(beta);
}
