################################################################################
# Bayesian ME-Cox: Simulation Wrapper
# =====================================
# Generates data from the DGM, fits Oracle/Naive/RC/GRC/Bayesian,
# compares all five methods.
################################################################################

library(rstan)
library(survival)
library(dplyr)
library(tidyr)

rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# Compile Stan model (do this once)
stan_model_path <- "model.stan"
mod <- stan_model(stan_model_path)

################################################################################
# 1. DATA-GENERATING MECHANISM (matches Python DGM exactly)
################################################################################

generate_data <- function(n = 50, beta_true = -1.0, gamma_true = 0.5,
                          rho_range = c(0.02, 0.10), seed = 42) {
  set.seed(seed)

  # Latent biomarker (logit-normal)
  mu_eta <- qlogis(0.35)  # logit(0.35)
  sigma_eta <- 0.8
  eta <- rnorm(n, mu_eta, sigma_eta)
  X <- plogis(eta)

  # Tissue characteristics
  m <- round(exp(rnorm(n, log(2200), 0.45)))
  m <- pmin(pmax(m, 200), 4000)
  rho <- runif(n, rho_range[1], rho_range[2])

  # Design effect and error variance
  DEFF <- 1 + (m - 1) * rho
  sigma2 <- X * (1 - X) * DEFF / m

  # Observed biomarker (Beta-Binomial approximation)
  W <- numeric(n)
  for (i in 1:n) {
    if (rho[i] < 0.001) {
      S <- rbinom(1, m[i], X[i])
    } else {
      phi_bb <- (1 - rho[i]) / rho[i]
      a <- max(X[i] * phi_bb, 0.01)
      b <- max((1 - X[i]) * phi_bb, 0.01)
      p_draw <- rbeta(1, a, b)
      p_draw <- pmin(pmax(p_draw, 1e-6), 1 - 1e-6)
      S <- rbinom(1, m[i], p_draw)
    }
    W[i] <- S / m[i]
  }

  # Confounder
  Z <- rnorm(n)

  # Survival (exponential baseline for DGM; model uses piecewise)
  lambda0 <- 0.1
  rate <- lambda0 * exp(beta_true * X + gamma_true * Z)
  T_event <- rexp(n, rate)

  # Censoring (~40%)
  mean_rate <- mean(rate)
  lambda_C <- 0.4 * mean_rate / (1 - 0.4)
  C <- rexp(n, lambda_C)

  T_obs <- pmin(T_event, C)
  event <- as.integer(T_event <= C)

  list(
    n = n, X = X, W = W, Z = matrix(Z, ncol = 1), T_obs = T_obs, event = event,
    m = m, rho = rho, DEFF = DEFF, sigma2 = sigma2,
    beta_true = beta_true, gamma_true = gamma_true
  )
}

################################################################################
# 2. FIT METHODS
################################################################################

# --- Oracle ---
fit_oracle <- function(data) {
  df <- data.frame(T = data$T_obs, event = data$event, X = data$X, Z = data$Z[,1])
  fit <- coxph(Surv(T, event) ~ X + Z, data = df)
  s <- summary(fit)
  c(beta = coef(fit)["X"], se = s$coefficients["X", "se(coef)"],
    ci_lo = confint(fit)["X", 1], ci_hi = confint(fit)["X", 2],
    p = s$coefficients["X", "Pr(>|z|)"])
}

# --- Naive ---
fit_naive <- function(data) {
  df <- data.frame(T = data$T_obs, event = data$event, X = data$W, Z = data$Z[,1])
  fit <- coxph(Surv(T, event) ~ X + Z, data = df)
  s <- summary(fit)
  c(beta = coef(fit)["X"], se = s$coefficients["X", "se(coef)"],
    ci_lo = confint(fit)["X", 1], ci_hi = confint(fit)["X", 2],
    p = s$coefficients["X", "Pr(>|z|)"])
}

# --- Standard RC ---
fit_rc <- function(data) {
  W <- data$W
  sigma2_i <- pmax(W * (1 - W) * data$DEFF / data$m, 1e-8)
  var_W <- var(W)
  sigma_X2 <- max(var_W - mean(sigma2_i), 0)
  if (sigma_X2 == 0) return(c(beta=NA, se=NA, ci_lo=NA, ci_hi=NA, p=NA))

  lambda_global <- sigma_X2 / (sigma_X2 + mean(sigma2_i))
  mu_W <- mean(W)
  X_rc <- mu_W + lambda_global * (W - mu_W)

  df <- data.frame(T = data$T_obs, event = data$event, X = X_rc, Z = data$Z[,1])
  fit <- coxph(Surv(T, event) ~ X + Z, data = df)
  s <- summary(fit)
  c(beta = coef(fit)["X"], se = s$coefficients["X", "se(coef)"],
    ci_lo = confint(fit)["X", 1], ci_hi = confint(fit)["X", 2],
    p = s$coefficients["X", "Pr(>|z|)"])
}

# --- GRC ---
fit_grc <- function(data) {
  W <- data$W
  sigma2_i <- pmax(W * (1 - W) * data$DEFF / data$m, 1e-8)
  var_W <- var(W)
  sigma_X2 <- max(var_W - mean(sigma2_i), 0)
  if (sigma_X2 == 0) return(c(beta=NA, se=NA, ci_lo=NA, ci_hi=NA, p=NA))

  lambda_i <- sigma_X2 / (sigma_X2 + sigma2_i)
  mu_W <- mean(W)
  X_grc <- mu_W + lambda_i * (W - mu_W)

  df <- data.frame(T = data$T_obs, event = data$event, X = X_grc, Z = data$Z[,1])
  fit <- coxph(Surv(T, event) ~ X + Z, data = df)
  s <- summary(fit)
  c(beta = coef(fit)["X"], se = s$coefficients["X", "se(coef)"],
    ci_lo = confint(fit)["X", 1], ci_hi = confint(fit)["X", 2],
    p = s$coefficients["X", "Pr(>|z|)"])
}

# --- Bayesian ME-Cox ---
fit_bayesian <- function(data, stan_mod, K = 5, iter = 2000, warmup = 1000, chains = 2) {

  # Piecewise interval cutpoints from event time quantiles
  event_times <- data$T_obs[data$event == 1]
  if (length(event_times) < K) {
    s_cuts <- quantile(data$T_obs, probs = seq(0, 1, length.out = K + 1)[-c(1, K + 1)])
  } else {
    s_cuts <- quantile(event_times, probs = seq(0, 1, length.out = K + 1)[-c(1, K + 1)])
  }
  s_cuts <- as.numeric(s_cuts)

  stan_data <- list(
    N = data$n,
    P = ncol(data$Z),
    K = K,
    W = data$W,
    Z = data$Z,
    T_obs = data$T_obs,
    event = data$event,
    m = as.numeric(data$m),
    log_m = log(as.numeric(data$m)),
    rho = data$rho,
    s = s_cuts
  )

  fit <- sampling(stan_mod, data = stan_data,
                  iter = iter, warmup = warmup, chains = chains,
                  refresh = 0, show_messages = FALSE,
                  control = list(adapt_delta = 0.95, max_treedepth = 12))

  # Extract beta posterior
  beta_draws <- extract(fit, "beta")$beta
  beta_hat <- mean(beta_draws)
  beta_se <- sd(beta_draws)
  beta_ci <- quantile(beta_draws, c(0.025, 0.975))
  p_val <- 2 * min(mean(beta_draws > 0), mean(beta_draws < 0))

  # Diagnostics
  diag <- list(
    n_eff = summary(fit, pars = "beta")$summary[, "n_eff"],
    Rhat = summary(fit, pars = "beta")$summary[, "Rhat"],
    divergences = sum(get_divergent_iterations(fit)),
    HR = mean(extract(fit, "HR")$HR)
  )

  list(
    result = c(beta = beta_hat, se = beta_se,
               ci_lo = as.numeric(beta_ci[1]),
               ci_hi = as.numeric(beta_ci[2]),
               p = p_val),
    diagnostics = diag
  )
}

################################################################################
# 3. RUN ONE REPLICATE
################################################################################

run_one_rep <- function(rep_id, beta_true, stan_mod, rho_range = c(0.02, 0.10)) {
  data <- generate_data(n = 50, beta_true = beta_true,
                        rho_range = rho_range, seed = rep_id * 1000 + 42)

  res <- list()
  res$Oracle <- tryCatch(fit_oracle(data), error = function(e) rep(NA, 5))
  res$Naive  <- tryCatch(fit_naive(data),  error = function(e) rep(NA, 5))
  res$StdRC  <- tryCatch(fit_rc(data),     error = function(e) rep(NA, 5))
  res$GRC    <- tryCatch(fit_grc(data),    error = function(e) rep(NA, 5))

  bayes_fit <- tryCatch(
    fit_bayesian(data, stan_mod, K = 5, iter = 2000, warmup = 1000, chains = 2),
    error = function(e) list(result = rep(NA, 5),
                             diagnostics = list(n_eff=NA, Rhat=NA, divergences=NA, HR=NA))
  )
  res$Bayesian <- bayes_fit$result
  res$bayes_diag <- bayes_fit$diagnostics

  res$rep_id <- rep_id
  res
}

################################################################################
# 4. RUN FULL SIMULATION
################################################################################

run_simulation <- function(n_reps, beta_true, stan_mod,
                           rho_range = c(0.02, 0.10), label = "S1") {
  cat(sprintf("\n=== %s: %d reps, beta=%.1f ===\n", label, n_reps, beta_true))

  all_results <- vector("list", n_reps)
  t0 <- Sys.time()

  for (rep in 1:n_reps) {
    all_results[[rep]] <- run_one_rep(rep, beta_true, stan_mod, rho_range)

    if (rep %% 10 == 0) {
      elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
      eta <- (n_reps - rep) / rep * elapsed
      cat(sprintf("  [%s] %d/%d | %.0fs | ETA %.0fs\n", label, rep, n_reps, elapsed, eta))
    }
  }

  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cat(sprintf("  [%s] Done: %.0fs (%.1fs/rep)\n", label, elapsed, elapsed/n_reps))

  # Aggregate results
  methods <- c("Oracle", "Naive", "StdRC", "GRC", "Bayesian")
  summary_rows <- list()

  for (method in methods) {
    betas <- sapply(all_results, function(r) r[[method]]["beta"])
    ci_lo <- sapply(all_results, function(r) r[[method]]["ci_lo"])
    ci_hi <- sapply(all_results, function(r) r[[method]]["ci_hi"])
    pvals <- sapply(all_results, function(r) r[[method]]["p"])

    valid <- !is.na(betas)
    nv <- sum(valid)

    if (nv < 20) {
      summary_rows[[method]] <- data.frame(
        Method = method, n_valid = nv, Mean_beta = NA, Bias = NA,
        RelBias = NA, EmpSE = NA, RMSE = NA, Coverage = NA, Power = NA
      )
      next
    }

    bv <- betas[valid]
    bias <- mean(bv) - beta_true
    summary_rows[[method]] <- data.frame(
      Method = method, n_valid = nv,
      Mean_beta = mean(bv),
      Bias = bias,
      RelBias = bias / abs(beta_true) * 100,
      EmpSE = sd(bv),
      RMSE = sqrt(mean((bv - beta_true)^2)),
      Coverage = mean(ci_lo[valid] <= beta_true & beta_true <= ci_hi[valid], na.rm=T) * 100,
      Power = mean(pvals[valid] < 0.05, na.rm = TRUE) * 100
    )
  }

  result_df <- do.call(rbind, summary_rows)

  # Bayesian diagnostics
  div <- sapply(all_results, function(r) r$bayes_diag$divergences)
  rhat <- sapply(all_results, function(r) r$bayes_diag$Rhat)
  neff <- sapply(all_results, function(r) r$bayes_diag$n_eff)

  cat(sprintf("\n  --- %s Results ---\n", label))
  print(result_df, digits = 3)
  cat(sprintf("\n  Bayesian diagnostics:\n"))
  cat(sprintf("    Divergences: mean=%.1f, max=%d\n", mean(div, na.rm=T), max(div, na.rm=T)))
  cat(sprintf("    Rhat: mean=%.3f, max=%.3f\n", mean(rhat, na.rm=T), max(rhat, na.rm=T)))
  cat(sprintf("    n_eff: mean=%.0f, min=%.0f\n", mean(neff, na.rm=T), min(neff, na.rm=T)))

  # Save
  write.csv(result_df, file.path("results", paste0(label, "_summary.csv")), row.names = FALSE)

  # Save replicate-level
  rep_data <- do.call(rbind, lapply(1:n_reps, function(i) {
    r <- all_results[[i]]
    do.call(rbind, lapply(methods, function(m) {
      data.frame(rep = i, method = m,
                 beta_hat = r[[m]]["beta"], se = r[[m]]["se"],
                 ci_lo = r[[m]]["ci_lo"], ci_hi = r[[m]]["ci_hi"],
                 p_val = r[[m]]["p"])
    }))
  }))
  write.csv(rep_data, file.path("results", paste0(label, "_replicates.csv")), row.names = FALSE)

  result_df
}

################################################################################
# 5. MAIN
################################################################################

dir.create("results", showWarnings = FALSE)

cat("Compiling Stan model...\n")
# Model already compiled above

n_reps <- 100  # Start with 100 for validation; scale to 500 for paper

cat("\n========================================\n")
cat("BAYESIAN ME-COX SIMULATION\n")
cat(sprintf("Reps: %d | Methods: Oracle, Naive, RC, GRC, Bayesian\n", n_reps))
cat("========================================\n")

s1 <- run_simulation(n_reps, beta_true = -1.0, stan_mod = mod, label = "S1")

cat("\n\n========================================\n")
cat("THE KEY COMPARISON: BAYESIAN vs GRC vs RC\n")
cat("========================================\n")

for (method in c("Naive", "StdRC", "GRC", "Bayesian")) {
  row <- s1[s1$Method == method, ]
  cat(sprintf("  %10s: bias=%+.3f (%+.1f%%) | cov=%.1f%% | RMSE=%.3f\n",
              method, row$Bias, row$RelBias, row$Coverage, row$RMSE))
}

bayes_row <- s1[s1$Method == "Bayesian", ]
grc_row <- s1[s1$Method == "GRC", ]
rc_row <- s1[s1$Method == "StdRC", ]

cat(sprintf("\n  Bayesian vs GRC: bias improvement = %.1fpp\n",
            abs(grc_row$RelBias) - abs(bayes_row$RelBias)))
cat(sprintf("  Bayesian vs RC:  bias improvement = %.1fpp\n",
            abs(rc_row$RelBias) - abs(bayes_row$RelBias)))
cat(sprintf("  Bayesian coverage: %.1f%%\n", bayes_row$Coverage))

if (abs(bayes_row$RelBias) < abs(grc_row$RelBias) &&
    abs(bayes_row$RelBias) < abs(rc_row$RelBias) &&
    bayes_row$Coverage >= 90) {
  cat("\n  >>> BAYESIAN WINS: Lower bias than both RC and GRC, valid coverage <<<\n")
  cat("  >>> PAPER IDENTITY: Bayesian ME-Cox for spatial biomarkers <<<\n")
} else {
  cat("\n  >>> BAYESIAN DOES NOT CLEARLY WIN <<<\n")
  cat("  >>> Investigate further before committing <<<\n")
}
