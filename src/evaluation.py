"""Generate the 5 key figures for the pilot study."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict


class PilotEvaluation:
    """Generate the 5 required figures for the pilot study."""

    def __init__(
        self,
        cci_full: pd.DataFrame,
        cci_imputed: Dict[str, pd.DataFrame],
        cci_corrected: Dict[str, pd.DataFrame],
        stability: Dict[str, pd.DataFrame],
    ):
        self.cci_full = cci_full
        self.cci_imputed = cci_imputed
        self.cci_corrected = cci_corrected
        self.stability = stability
        self.methods = list(cci_imputed.keys())

    def figure1_significant_counts(self, save_path: str = None):
        """
        Figure 1: Bar chart of significant CCI pair counts.
        Full data vs. each imputed method vs. each corrected method.
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        categories = ["Full Data\n(Ground Truth)"]
        counts = [self.cci_full["significant"].sum()]
        colors = ["#2ecc71"]

        for method in self.methods:
            categories.append(f"{method}\n(Imputed)")
            counts.append(self.cci_imputed[method]["significant"].sum())
            colors.append("#e74c3c")

            categories.append(f"{method}\n(Corrected)")
            counts.append(self.cci_corrected[method]["significant"].sum())
            colors.append("#3498db")

        bars = ax.bar(
            categories, counts, color=colors, edgecolor="black", linewidth=0.5
        )
        ax.axhline(
            y=counts[0], color="#2ecc71", linestyle="--", alpha=0.7,
            label="Ground truth count",
        )
        ax.set_ylabel("Number of Significant CCI Pairs (FDR < 0.05)", fontsize=12)
        ax.set_title("CCI Inflation by Imputation Method", fontsize=14)
        ax.legend()

        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                str(count),
                ha="center",
                va="bottom",
                fontweight="bold",
            )

        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def figure2_false_positive_analysis(self, save_path: str = None):
        """
        Figure 2: False positive analysis per method.
        True positives, false positives (imputed), false positives (corrected), missed.
        """
        fig, axes = plt.subplots(1, len(self.methods), figsize=(5 * len(self.methods), 5))
        if len(self.methods) == 1:
            axes = [axes]

        true_sig = set(
            self.cci_full[self.cci_full["significant"]].apply(
                lambda r: f"{r['ligand']}_{r['receptor']}", axis=1
            )
        )

        for ax, method in zip(axes, self.methods):
            imputed_sig = set(
                self.cci_imputed[method][self.cci_imputed[method]["significant"]].apply(
                    lambda r: f"{r['ligand']}_{r['receptor']}", axis=1
                )
            )
            corrected_sig = set(
                self.cci_corrected[method][
                    self.cci_corrected[method]["significant"]
                ].apply(lambda r: f"{r['ligand']}_{r['receptor']}", axis=1)
            )

            true_positives = imputed_sig & true_sig
            false_positives = imputed_sig - true_sig
            fp_after_correction = corrected_sig - true_sig
            missed = true_sig - imputed_sig

            categories = [
                "True\nPositives",
                "False\nPositives\n(Imputed)",
                "False\nPositives\n(Corrected)",
                "Missed",
            ]
            values = [
                len(true_positives),
                len(false_positives),
                len(fp_after_correction),
                len(missed),
            ]
            colors_bar = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]

            ax.bar(categories, values, color=colors_bar, edgecolor="black", linewidth=0.5)
            ax.set_title(f"{method}", fontsize=12, fontweight="bold")
            ax.set_ylabel("Number of LR Pairs")

            if len(false_positives) > 0:
                reduction = (
                    1 - len(fp_after_correction) / len(false_positives)
                ) * 100
                ax.text(
                    0.5, 0.95, f"FP reduction: {reduction:.0f}%",
                    transform=ax.transAxes, ha="center", va="top", fontsize=11,
                    color="#2c3e50",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                )

        plt.suptitle(
            "False Positive Analysis: Imputed vs Corrected CCI",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def figure3_bias_scatter(self, save_path: str = None):
        """
        Figure 3: Score-vs-score scatter plot (ground truth vs imputed).
        Deviation from diagonal = bias.
        """
        fig, axes = plt.subplots(1, len(self.methods), figsize=(5 * len(self.methods), 5))
        if len(self.methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, self.methods):
            merged = self.cci_full.merge(
                self.cci_imputed[method],
                on=["ligand", "receptor"],
                suffixes=("_true", "_imputed"),
            )

            x = merged["score_true"]
            y = merged["score_imputed"]

            ax.scatter(x, y, alpha=0.4, s=15, c="#3498db")

            lims = [min(x.min(), y.min()), max(x.max(), y.max())]
            ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)

            slope, intercept = np.polyfit(x, y, 1)
            ax.plot(
                lims, [slope * l + intercept for l in lims],
                "r-", alpha=0.7, linewidth=2, label=f"slope={slope:.2f}",
            )

            bias = (y - x).mean()
            ax.set_xlabel("Ground Truth CCI Score", fontsize=11)
            ax.set_ylabel("Imputed CCI Score", fontsize=11)
            ax.set_title(f"{method}\nMean bias = {bias:.4f}", fontsize=12)
            ax.legend()

        plt.suptitle(
            "CCI Score Bias: Imputed vs Ground Truth", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def figure4_reliability_vs_error(self, save_path: str = None):
        """
        Figure 4: Reliability score vs actual error.
        Target: Spearman correlation >= 0.7.
        """
        fig, axes = plt.subplots(1, len(self.methods), figsize=(5 * len(self.methods), 5))
        if len(self.methods) == 1:
            axes = [axes]

        for ax, method in zip(axes, self.methods):
            merged = self.cci_full.merge(
                self.cci_imputed[method],
                on=["ligand", "receptor"],
                suffixes=("_true", "_imputed"),
            )
            stab = self.stability[method]
            merged = merged.merge(stab, on=["ligand", "receptor"])

            actual_error = np.abs(merged["score_imputed"] - merged["score_true"])
            reliability = merged["stability"]

            ax.scatter(reliability, actual_error, alpha=0.4, s=15, c="#8e44ad")

            spearman_r, spearman_p = stats.spearmanr(reliability, actual_error)
            pearson_r, pearson_p = stats.pearsonr(reliability, actual_error)

            ax.set_xlabel("Predicted Reliability (Stability)", fontsize=11)
            ax.set_ylabel("Actual Error |T_imputed - T_true|", fontsize=11)
            ax.set_title(
                f"{method}\n"
                f"Spearman r = {spearman_r:.3f} (p={spearman_p:.1e})\n"
                f"Pearson r = {pearson_r:.3f}",
                fontsize=11,
            )

        plt.suptitle(
            "Reliability Predicts Actual Error", fontsize=14, fontweight="bold"
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def figure5_calibration_plot(self, save_path: str = None):
        """
        Figure 5: Calibration plot.
        Bin by predicted reliability, compute observed FDP in each bin.
        Perfect calibration: FDP decreases monotonically with reliability.
        """
        fig, axes = plt.subplots(1, len(self.methods), figsize=(5 * len(self.methods), 5))
        if len(self.methods) == 1:
            axes = [axes]

        true_sig = set(
            self.cci_full[self.cci_full["significant"]].apply(
                lambda r: f"{r['ligand']}_{r['receptor']}", axis=1
            )
        )

        for ax, method in zip(axes, self.methods):
            imputed_df = self.cci_imputed[method].copy()
            stab = self.stability[method]
            merged = imputed_df.merge(stab, on=["ligand", "receptor"])
            merged["pair_key"] = merged.apply(
                lambda r: f"{r['ligand']}_{r['receptor']}", axis=1
            )

            sig_merged = merged[merged["significant"]].copy()

            if len(sig_merged) < 10:
                ax.text(
                    0.5, 0.5, "Too few significant\npairs",
                    transform=ax.transAxes, ha="center",
                )
                continue

            sig_merged["is_true_positive"] = sig_merged["pair_key"].isin(true_sig)

            n_bins = 5
            sig_merged["reliability_bin"] = pd.qcut(
                sig_merged["stability"], q=n_bins, duplicates="drop"
            )

            cal_data = (
                sig_merged.groupby("reliability_bin")
                .agg(
                    n_pairs=("pair_key", "count"),
                    n_true_positive=("is_true_positive", "sum"),
                    mean_reliability=("stability", "mean"),
                )
                .reset_index()
            )
            cal_data["fdp"] = 1 - cal_data["n_true_positive"] / cal_data["n_pairs"]

            ax.bar(
                range(len(cal_data)), cal_data["fdp"],
                color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=0.5,
            )
            ax.set_xticks(range(len(cal_data)))
            ax.set_xticklabels(
                [f"{r:.2f}" for r in cal_data["mean_reliability"]], rotation=45
            )
            ax.set_xlabel("Mean Reliability Score (binned)", fontsize=11)
            ax.set_ylabel("False Discovery Proportion", fontsize=11)
            ax.set_title(f"{method}", fontsize=12, fontweight="bold")
            ax.set_ylim(0, 1)

        plt.suptitle(
            "Calibration: FDP Should Decrease with Higher Reliability",
            fontsize=14, fontweight="bold",
        )
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def run_all(self, output_dir: str = "results/pilot"):
        """Generate all 5 figures."""
        import os

        os.makedirs(output_dir, exist_ok=True)

        print("Generating Figure 1: Significant CCI counts...")
        self.figure1_significant_counts(f"{output_dir}/fig1_significant_counts.png")

        print("Generating Figure 2: False positive analysis...")
        self.figure2_false_positive_analysis(f"{output_dir}/fig2_false_positives.png")

        print("Generating Figure 3: Bias scatter...")
        self.figure3_bias_scatter(f"{output_dir}/fig3_bias_scatter.png")

        print("Generating Figure 4: Reliability vs error...")
        self.figure4_reliability_vs_error(f"{output_dir}/fig4_reliability_vs_error.png")

        print("Generating Figure 5: Calibration plot...")
        self.figure5_calibration_plot(f"{output_dir}/fig5_calibration.png")

        print(f"All figures saved to {output_dir}/")
