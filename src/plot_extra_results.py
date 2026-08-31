from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

TABLE_DIR = Path("results/tables")
FIGURE_DIR = Path("results/figures")

CONTRADICTION_CSV = TABLE_DIR / "contradiction_breakdown.csv"
PMG_CSV = TABLE_DIR / "pmg.csv"


# ============================================================
# CONTRADICTION BEHAVIOR FIGURE
# ============================================================

def plot_contradiction_behavior():
    """
    Generate a 100% stacked bar chart directly from
    results/tables/contradiction_breakdown.csv.

    No result values are hard-coded.
    """

    df = pd.read_csv(CONTRADICTION_CSV)

    expected_modes = ["natural", "anonymized"]

    df = (
        df.set_index("label_mode")
        .loc[expected_modes]
        .reset_index()
    )

    categories = [
        ("gold_only_rate", "Gold only"),
        ("both_rate", "Gold + injected"),
        ("UNKNOWN_rate", "UNKNOWN"),
        ("other_rate", "Other"),
    ]

    x = np.arange(len(df))
    bottom = np.zeros(len(df))

    fig, ax = plt.subplots(
        figsize=(7.0, 4.5)
    )

    for column, label in categories:
        values = df[column].to_numpy(dtype=float) * 100

        bars = ax.bar(
            x,
            values,
            bottom=bottom,
            label=label,
        )

        # Put percentages inside sufficiently large segments.
        for i, value in enumerate(values):
            if value >= 6:
                ax.text(
                    x[i],
                    bottom[i] + value / 2,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                )

        bottom += values

    ax.set_xticks(
        x,
        ["Natural", "Anonymized"],
    )

    ax.set_ylabel(
        "Share of contradiction outputs (%)"
    )

    ax.set_ylim(0, 100)

    ax.legend(
        frameon=False,
        ncol=2,
        loc="lower left",
    )

    fig.tight_layout()

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png_path = (
        FIGURE_DIR
        / "contradiction_behavior.png"
    )

    pdf_path = (
        FIGURE_DIR
        / "contradiction_behavior.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


# ============================================================
# PMG FIGURE
# ============================================================

def plot_pmg():
    """
    Generate PMG with paired-bootstrap 95% confidence intervals
    directly from results/tables/pmg.csv.
    """

    df = pd.read_csv(PMG_CSV)

    condition_order = [
        "entity_substitution",
        "relation_substitution",
        "contradiction",
        "rerouting",
    ]

    df = (
        df.set_index("condition")
        .loc[condition_order]
        .reset_index()
    )

    values = (
        df["pmg"]
        .to_numpy(dtype=float)
        * 100
    )

    lower_error = (
        (
            df["pmg"]
            - df["ci95_low"]
        )
        .to_numpy(dtype=float)
        * 100
    )

    upper_error = (
        (
            df["ci95_high"]
            - df["pmg"]
        )
        .to_numpy(dtype=float)
        * 100
    )

    labels = [
        "Entity\nsubstitution",
        "Relation\nsubstitution",
        "Contradiction",
        "Rerouting",
    ]

    x = np.arange(len(df))

    fig, ax = plt.subplots(
        figsize=(7.0, 4.3)
    )

    ax.bar(
        x,
        values,
        yerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        capsize=4,
    )

    ax.axhline(
        0,
        linewidth=0.9,
    )

    ax.set_xticks(
        x,
        labels,
    )

    ax.set_ylabel(
        "Parametric Masking Gap (percentage points)"
    )

    fig.tight_layout()

    png_path = (
        FIGURE_DIR
        / "pmg_with_ci.png"
    )

    pdf_path = (
        FIGURE_DIR
        / "pmg_with_ci.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {png_path}"
    )

    print(
        f"Saved: {pdf_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    if not CONTRADICTION_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {CONTRADICTION_CSV}"
        )

    if not PMG_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {PMG_CSV}"
        )

    plot_contradiction_behavior()
    plot_pmg()

    print()
    print("RESULT FIGURES COMPLETE")


if __name__ == "__main__":
    main()
