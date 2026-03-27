#!/usr/bin/env python
"""
Main entry point for the pilot experiment.

Usage:
    python run_pilot.py --spatial path/to/xenium.h5ad --panel-size 300
    python run_pilot.py --spatial path/to/xenium.h5ad --reference path/to/scrna.h5ad --panel-size 300
"""

import argparse
import os
import sys

import numpy as np

from src.data_loader import load_xenium_breast_cancer, load_reference, get_lr_pairs
from src.panel_simulation import simulate_sparse_panel
from src.cci_inference import compute_cci_scores
from src.regression_calibration import estimate_gene_mse, regression_calibrated_cci
from src.stability_analysis import panel_perturbation_stability
from src.evaluation import PilotEvaluation


def run_pilot(
    spatial_path: str,
    reference_path: str = None,
    panel_size: int = 300,
    n_perms: int = 1000,
    n_perturbations: int = 50,
    output_dir: str = "results/pilot",
):
    """Run the complete pilot experiment."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Load data ──
    print("=" * 60)
    print("STEP 0: Loading data")
    print("=" * 60)
    adata = load_xenium_breast_cancer(spatial_path)

    adata_ref = None
    if reference_path:
        adata_ref = load_reference(reference_path)

    # ── LR pairs ──
    lr_pairs = get_lr_pairs(adata)

    # ── Ground truth CCI ──
    print("\n" + "=" * 60)
    print("STEP 1: Ground truth CCI (full panel)")
    print("=" * 60)
    cci_full = compute_cci_scores(adata, lr_pairs, layer="counts", n_perms=n_perms)
    n_sig_full = cci_full["significant"].sum()
    print(f"Significant CCI pairs (ground truth): {n_sig_full}")

    # ── Simulate panel ──
    print("\n" + "=" * 60)
    print("STEP 2: Simulating sparse panel")
    print("=" * 60)
    panel_mask = simulate_sparse_panel(
        adata, m=panel_size, lr_pairs=lr_pairs, strategy="mixed"
    )

    # ── Imputation ──
    print("\n" + "=" * 60)
    print("STEP 3: Running imputation methods")
    print("=" * 60)

    methods = {}

    # GNN-based (always available — no reference needed)
    print("\n  [1] Spatial GNN...")
    from src.imputation.gnn_runner import run_spatial_gnn

    methods["SpatialGNN"] = run_spatial_gnn(adata, panel_mask)

    # Diffusion-based (always available)
    print("\n  [2] Diffusion...")
    from src.imputation.diffusion_runner import run_diffusion

    methods["Diffusion"] = run_diffusion(adata, panel_mask)

    # Reference-based methods (require scRNA-seq reference)
    if adata_ref is not None:
        try:
            print("\n  [3] Tangram...")
            from src.imputation.tangram_runner import run_tangram

            methods["Tangram"] = run_tangram(adata, adata_ref, panel_mask)
        except Exception as e:
            print(f"    Tangram failed: {e}")

        try:
            print("\n  [4] gimVI...")
            from src.imputation.gimvi_runner import run_gimvi

            methods["gimVI"] = run_gimvi(adata, adata_ref, panel_mask)
        except Exception as e:
            print(f"    gimVI failed: {e}")

    # ── CCI on imputed data ──
    print("\n" + "=" * 60)
    print("STEP 4: CCI inference on imputed data")
    print("=" * 60)

    cci_imputed = {}
    cci_corrected = {}
    stability_results = {}

    for method_name, Lambda_hat in methods.items():
        print(f"\n  Processing {method_name}...")

        # Create imputed AnnData
        adata_imp = adata.copy()
        adata_imp.layers["imputed"] = Lambda_hat

        # Naive CCI
        print(f"    Naive CCI...")
        cci_imputed[method_name] = compute_cci_scores(
            adata_imp, lr_pairs, layer="imputed", n_perms=n_perms
        )
        n_sig_imp = cci_imputed[method_name]["significant"].sum()
        print(f"    Significant: {n_sig_imp} (+{n_sig_imp - n_sig_full} vs ground truth)")

        # Gene MSE estimation
        print(f"    Estimating gene MSE...")
        mse = estimate_gene_mse(adata, Lambda_hat, panel_mask)

        # Regression-calibrated CCI
        print(f"    Calibrated CCI...")
        cci_corrected[method_name] = regression_calibrated_cci(
            adata, Lambda_hat, mse, lr_pairs, n_perms=n_perms
        )
        n_sig_corr = cci_corrected[method_name]["significant"].sum()
        print(f"    Significant after correction: {n_sig_corr}")

        # Panel perturbation stability (use GNN for speed)
        print(f"    Panel perturbation stability ({n_perturbations} perturbations)...")

        def quick_impute(ad, mask):
            return run_spatial_gnn(ad, mask, epochs=100)

        stability_results[method_name] = panel_perturbation_stability(
            adata,
            panel_mask,
            quick_impute,
            lr_pairs,
            n_perturbations=n_perturbations,
            drop_fraction=0.2,
        )

    # ── Evaluation ──
    print("\n" + "=" * 60)
    print("STEP 5: Generating figures")
    print("=" * 60)

    evaluator = PilotEvaluation(
        cci_full, cci_imputed, cci_corrected, stability_results
    )
    evaluator.run_all(output_dir)

    # ── Go/No-Go Summary ──
    print("\n" + "=" * 60)
    print("GO / NO-GO SUMMARY")
    print("=" * 60)

    inflation_methods = 0
    for method_name in methods:
        n_imp = cci_imputed[method_name]["significant"].sum()
        n_corr = cci_corrected[method_name]["significant"].sum()
        inflation = n_imp - n_sig_full
        correction = n_imp - n_corr
        inflation_rate = inflation / n_sig_full if n_sig_full > 0 else 0

        print(f"\n  {method_name}:")
        print(f"    Ground truth significant: {n_sig_full}")
        print(f"    Imputed significant:      {n_imp} (+{inflation})")
        print(f"    Corrected significant:    {n_corr}")
        print(f"    False positives removed:  {correction}")
        print(f"    Inflation rate:           {inflation_rate * 100:.1f}%")

        if inflation_rate > 0.20:
            inflation_methods += 1

    print("\n" + "-" * 60)
    print("DECISION CRITERIA:")
    print(f"  Methods with >20% inflation: {inflation_methods} / {len(methods)}")
    print("  GO if: inflation > 20% across >= 2 methods")
    print("         AND correction removes >50% of false positives")
    print("         AND reliability-error correlation > 0.5")
    print("-" * 60)

    if inflation_methods >= 2:
        print("\n  >>> PRELIMINARY: GO signal detected <<<")
    else:
        print("\n  >>> PRELIMINARY: Weak signal — investigate before scaling <<<")

    return {
        "cci_full": cci_full,
        "cci_imputed": cci_imputed,
        "cci_corrected": cci_corrected,
        "stability": stability_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pilot experiment: CCI bias under gene-panel sparsity"
    )
    parser.add_argument("--spatial", required=True, help="Path to spatial .h5ad")
    parser.add_argument("--reference", default=None, help="Path to scRNA-seq .h5ad")
    parser.add_argument("--panel-size", type=int, default=300)
    parser.add_argument("--n-perms", type=int, default=1000)
    parser.add_argument("--n-perturbations", type=int, default=50)
    parser.add_argument("--output-dir", default="results/pilot")

    args = parser.parse_args()

    run_pilot(
        spatial_path=args.spatial,
        reference_path=args.reference,
        panel_size=args.panel_size,
        n_perms=args.n_perms,
        n_perturbations=args.n_perturbations,
        output_dir=args.output_dir,
    )
