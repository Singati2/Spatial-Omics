// Bayesian Joint ME-Cox for Spatial Biomarkers
// ==============================================
// Latent logit-normal biomarker X_i
// Heteroscedastic observation W_i | X_i ~ N(X_i, sigma_i^2)
// Random sigma_i^2 via log-linear design-effect model
// Piecewise-constant baseline Cox survival
//
// Paper: "Bayesian Measurement Error Correction for Survival Analysis
//         with Spatially-Derived Biomarkers"

functions {
  // Log-likelihood contribution for one subject under piecewise-constant Cox
  // with K intervals defined by cutpoints s[1], ..., s[K-1], s[K]=inf
  // lambda0[k] = baseline hazard in interval k
  // eta_i = beta * X_i + gamma' * Z_i (linear predictor)
  real piecewise_cox_lpdf(real t, int event, vector lambda0, vector s, real eta) {
    int K = num_elements(lambda0);
    real log_lik = 0;
    real cum_haz = 0;
    real exp_eta = exp(eta);

    for (k in 1:K) {
      real lo = (k == 1) ? 0.0 : s[k-1];
      real hi = (k == K) ? t : fmin(t, s[k]);

      if (t > lo) {
        real width = hi - lo;
        cum_haz += lambda0[k] * width * exp_eta;

        if (event == 1 && t <= hi + 1e-10 && t > lo) {
          // Event occurs in this interval
          log_lik += log(lambda0[k]) + eta;
        }
      }
    }
    log_lik -= cum_haz;
    return log_lik;
  }
}

data {
  int<lower=1> N;                    // number of patients
  int<lower=1> P;                    // number of confounders
  int<lower=1> K;                    // number of piecewise intervals

  vector[N] W;                       // observed biomarker (proportion in [0,1])
  matrix[N, P] Z;                    // confounder matrix
  vector<lower=0>[N] T_obs;          // observed survival time
  array[N] int<lower=0, upper=1> event;    // event indicator (1=died, 0=censored)

  // Spatial tissue characteristics (observed)
  vector<lower=1>[N] m;              // number of spots per patient
  vector[N] log_m;                   // log(m_i), precomputed
  vector[N] rho;                     // Moran's I per patient

  // Piecewise interval cutpoints (K-1 interior points)
  vector[K-1] s;                     // e.g., quantiles of event times
}

parameters {
  // Latent biomarker (logit scale)
  real mu_eta;                       // population mean on logit scale
  real<lower=0> sigma_eta;           // population SD on logit scale
  vector[N] eta_raw;                 // non-centered parameterization

  // Biomarker effect on survival
  real beta;                         // log-HR per unit X_i

  // Confounder effects
  vector[P] gamma_z;

  // Baseline hazard (piecewise constant, log scale)
  vector[K] log_lambda0;

  // Heteroscedastic error variance model
  // log(sigma_i^2) ~ N(alpha0 + alpha1*log(m_i) + alpha2*rho_i, tau^2)
  real alpha0;
  real alpha1;                       // expect negative (more spots -> less error)
  real alpha2;                       // expect positive (more autocorr -> more error)
  real<lower=0> tau;                 // residual SD of log-variance model
  vector[N] log_sigma2_raw;         // non-centered residual
}

transformed parameters {
  // Latent biomarker on original scale
  vector[N] eta_i = mu_eta + sigma_eta * eta_raw;
  vector[N] X;
  for (i in 1:N) {
    X[i] = inv_logit(eta_i[i]);
  }

  // Patient-specific error variance (random, not fixed)
  vector[N] log_sigma2_mean;
  vector[N] log_sigma2;
  vector<lower=0>[N] sigma2;

  for (i in 1:N) {
    log_sigma2_mean[i] = alpha0 + alpha1 * log_m[i] + alpha2 * rho[i];
    log_sigma2[i] = log_sigma2_mean[i] + tau * log_sigma2_raw[i];
    sigma2[i] = exp(log_sigma2[i]) + 1e-6;  // floor to prevent zero
  }

  // Baseline hazard on natural scale
  vector<lower=0>[K] lambda0 = exp(log_lambda0);
}

model {
  // ---- PRIORS ----

  // Latent biomarker population
  mu_eta ~ normal(-0.6, 1.0);       // logit(0.35) ≈ -0.62
  sigma_eta ~ normal(0, 1.0);       // half-normal (constrained positive)
  eta_raw ~ std_normal();            // non-centered

  // Survival effect
  beta ~ normal(0, 2.0);            // weakly informative
  gamma_z ~ normal(0, 1.0);

  // Baseline hazard
  log_lambda0 ~ normal(-2, 1.5);    // exp(-2) ≈ 0.14, reasonable baseline

  // Variance model parameters
  alpha0 ~ normal(-4, 2.0);         // exp(-4) ≈ 0.018, typical sigma^2
  alpha1 ~ normal(-1, 1.0);         // negative: more spots -> less variance
  alpha2 ~ normal(1, 1.0);          // positive: more autocorr -> more variance
  tau ~ normal(0, 0.5);             // half-normal, moderate residual
  log_sigma2_raw ~ std_normal();    // non-centered

  // ---- LIKELIHOOD ----

  // Observation model: W_i | X_i ~ N(X_i, sigma_i^2)
  for (i in 1:N) {
    W[i] ~ normal(X[i], sqrt(sigma2[i]));
  }

  // Survival model: piecewise-constant Cox
  for (i in 1:N) {
    real lin_pred = beta * X[i] + dot_product(Z[i], gamma_z);
    target += piecewise_cox_lpdf(T_obs[i] | event[i], lambda0, s, lin_pred);
  }
}

generated quantities {
  // Hazard ratio
  real HR = exp(beta);

  // Patient-specific reliability ratio (for comparison with GRC)
  vector[N] lambda_i;
  {
    real sigma_X2 = square(sigma_eta) * 0.25;  // approx Var(X) from logit-normal
    for (i in 1:N) {
      lambda_i[i] = sigma_X2 / (sigma_X2 + sigma2[i]);
    }
  }

  // Posterior predictive checks: replicate W
  vector[N] W_rep;
  for (i in 1:N) {
    W_rep[i] = normal_rng(X[i], sqrt(sigma2[i]));
  }

  // Log-likelihood for LOO-CV
  vector[N] log_lik;
  for (i in 1:N) {
    real lin_pred = beta * X[i] + dot_product(Z[i], gamma_z);
    log_lik[i] = normal_lpdf(W[i] | X[i], sqrt(sigma2[i]))
                 + piecewise_cox_lpdf(T_obs[i] | event[i], lambda0, s, lin_pred);
  }
}
