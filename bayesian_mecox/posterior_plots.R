################################################################################
# Posterior Summarization and Comparison Plots
# ==============================================
# Produces publication-ready figures comparing
# Oracle vs Naive vs RC vs GRC vs Bayesian
################################################################################

library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)

theme_paper <- theme_minimal(base_size = 12) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )

################################################################################
# Load results (after running run_simulation.R)
################################################################################

load_results <- function(scenario = "S1") {
  path <- file.path("results", paste0(scenario, "_summary.csv"))
  if (!file.exists(path)) stop(paste("Results not found:", path))
  read.csv(path) %>%
    mutate(Method = factor(Method, levels = c("Oracle", "Naive", "StdRC", "GRC", "Bayesian")))
}

load_replicates <- function(scenario = "S1") {
  path <- file.path("results", paste0(scenario, "_replicates.csv"))
  if (!file.exists(path)) stop(paste("Replicates not found:", path))
  read.csv(path) %>%
    mutate(method = factor(method, levels = c("Oracle", "Naive", "StdRC", "GRC", "Bayesian")))
}

################################################################################
# Figure 1: Bias Comparison (bar chart)
################################################################################

plot_bias <- function(scenario = "S1", beta_true = -1.0) {
  df <- load_results(scenario)

  ggplot(df, aes(x = Method, y = RelBias, fill = Method)) +
    geom_col(alpha = 0.85, width = 0.6) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
    geom_hline(yintercept = c(-5, 5), linetype = "dotted", color = "gray60") +
    scale_fill_manual(values = c(Oracle="#2ecc71", Naive="#e74c3c", StdRC="#f39c12",
                                  GRC="#3498db", Bayesian="#9b59b6")) +
    labs(title = paste0(scenario, ": Relative Bias (%) by Method"),
         subtitle = paste("True beta =", beta_true),
         y = "Relative Bias (%)", x = "") +
    theme_paper +
    theme(legend.position = "none") +
    coord_cartesian(ylim = c(-35, 35))
}

################################################################################
# Figure 2: Coverage Comparison
################################################################################

plot_coverage <- function(scenario = "S1") {
  df <- load_results(scenario)

  ggplot(df, aes(x = Method, y = Coverage, fill = Method)) +
    geom_col(alpha = 0.85, width = 0.6) +
    geom_hline(yintercept = 95, linetype = "dashed", color = "gray40") +
    geom_hline(yintercept = 90, linetype = "dotted", color = "red", alpha = 0.5) +
    scale_fill_manual(values = c(Oracle="#2ecc71", Naive="#e74c3c", StdRC="#f39c12",
                                  GRC="#3498db", Bayesian="#9b59b6")) +
    labs(title = paste0(scenario, ": 95% CI Coverage by Method"),
         y = "Coverage (%)", x = "") +
    theme_paper +
    theme(legend.position = "none") +
    coord_cartesian(ylim = c(70, 100))
}

################################################################################
# Figure 3: Beta-hat Distribution (violin/boxplot)
################################################################################

plot_beta_distribution <- function(scenario = "S1", beta_true = -1.0) {
  df <- load_replicates(scenario) %>%
    filter(!is.na(beta_hat))

  ggplot(df, aes(x = method, y = beta_hat, fill = method)) +
    geom_violin(alpha = 0.4, scale = "width") +
    geom_boxplot(width = 0.15, outlier.size = 0.5, alpha = 0.8) +
    geom_hline(yintercept = beta_true, linetype = "dashed", color = "red", linewidth = 0.8) +
    scale_fill_manual(values = c(Oracle="#2ecc71", Naive="#e74c3c", StdRC="#f39c12",
                                  GRC="#3498db", Bayesian="#9b59b6")) +
    labs(title = paste0(scenario, ": Distribution of beta-hat across replicates"),
         subtitle = paste("Red line = true beta =", beta_true),
         y = expression(hat(beta)), x = "") +
    theme_paper +
    theme(legend.position = "none")
}

################################################################################
# Figure 4: Combined Panel (for paper)
################################################################################

plot_combined <- function(scenario = "S1", beta_true = -1.0) {
  p1 <- plot_bias(scenario, beta_true)
  p2 <- plot_coverage(scenario)
  p3 <- plot_beta_distribution(scenario, beta_true)

  (p1 | p2) / p3 +
    plot_annotation(
      title = paste("Bayesian ME-Cox Simulation Results:", scenario),
      theme = theme(plot.title = element_text(face = "bold", size = 14))
    )
}

################################################################################
# Generate all plots (after results are available)
################################################################################

generate_all_plots <- function() {
  dir.create("figures", showWarnings = FALSE)

  for (sc in c("S1")) {
    cat(sprintf("Generating plots for %s...\n", sc))

    p <- plot_combined(sc)
    ggsave(file.path("figures", paste0(sc, "_combined.pdf")), p, width = 12, height = 10)
    ggsave(file.path("figures", paste0(sc, "_combined.png")), p, width = 12, height = 10, dpi = 300)

    cat(sprintf("  Saved to figures/%s_combined.pdf\n", sc))
  }

  cat("All plots generated.\n")
}

# Run if executed directly
if (interactive()) {
  cat("Call generate_all_plots() after running the simulation.\n")
} else {
  generate_all_plots()
}
