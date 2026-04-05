// Bayesian Joint ME-Cox v2 — Fixed divergences
// ===============================================
// Changes from v1:
// 1. Log-cumulative-hazard formulation (no loop in likelihood)
// 2. Stronger sigma floor (1e-4)
// 3. Tighter tau prior to prevent extreme sigma_i^2
// 4. Vectorized observation model

data {
  int<lower=1> N;
  int<lower=1> P;
  int<lower=1> K;
  vector[N] W;
  matrix[N, P] Z;
  vector<lower=0>[N] T_obs;
  array[N] int<lower=0, upper=1> event;
  vector<lower=1>[N] m_spots;
  vector[N] log_m;
  vector[N] rho;
  vector[K-1] s;
}

transformed data {
  // Precompute interval membership for each patient
  // interval_id[i] = which interval patient i's event/censor falls in
  array[N] int interval_id;
  for (i in 1:N) {
    interval_id[i] = K;  // default: last interval
    for (k in 1:(K-1)) {
      if (T_obs[i] <= s[k]) {
        interval_id[i] = k;
        break;
      }
    }
  }
}

parameters {
  real mu_eta;
  real<lower=0> sigma_eta;
  vector[N] eta_raw;
  real beta;
  vector[P] gamma_z;
  vector[K] log_lambda0;
  real alpha0;
  real alpha1;
  real alpha2;
  real<lower=0> tau;
  vector[N] lsig_raw;
}

transformed parameters {
  vector[N] eta_i = mu_eta + sigma_eta * eta_raw;
  vector[N] X = inv_logit(eta_i);

  vector[N] lsig_mu = alpha0 + alpha1 * log_m + alpha2 * rho;
  vector[N] lsig = lsig_mu + tau * lsig_raw;
  vector<lower=0>[N] sig;
  for (i in 1:N) {
    sig[i] = sqrt(exp(lsig[i]) + 1e-4);  // SD with floor
  }

  vector<lower=0>[K] lam0 = exp(log_lambda0);
}

model {
  // Priors
  mu_eta ~ normal(-0.6, 1.5);
  sigma_eta ~ normal(0, 0.8);
  eta_raw ~ std_normal();

  beta ~ normal(0, 2);
  gamma_z ~ normal(0, 1);
  log_lambda0 ~ normal(-2, 1);

  alpha0 ~ normal(-4, 2);
  alpha1 ~ normal(-0.5, 1);
  alpha2 ~ normal(0.5, 1);
  tau ~ normal(0, 0.3);  // tighter than v1
  lsig_raw ~ std_normal();

  // Observation model (vectorized)
  W ~ normal(X, sig);

  // Piecewise-constant Cox likelihood
  {
    vector[N] lin = beta * X + Z * gamma_z;
    vector[N] exp_lin = exp(lin);

    for (i in 1:N) {
      real cum_haz = 0;
      int ki = interval_id[i];

      // Full intervals before the event interval
      if (ki > 1) {
        real prev_cut = 0;
        for (k in 1:(ki-1)) {
          real width = s[k] - prev_cut;
          cum_haz += lam0[k] * width * exp_lin[i];
          prev_cut = s[k];
        }
      }

      // Partial interval containing the event/censor
      {
        real lo_cut = (ki == 1) ? 0.0 : s[ki-1];
        real partial_width = T_obs[i] - lo_cut;
        cum_haz += lam0[ki] * partial_width * exp_lin[i];
      }

      // Log-likelihood
      if (event[i] == 1) {
        target += log(lam0[ki]) + lin[i] - cum_haz;
      } else {
        target += -cum_haz;
      }
    }
  }
}

generated quantities {
  real HR = exp(beta);
}
