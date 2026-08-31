from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TABLE_DIR = Path("results/tables")
FIGURE_DIR = Path("results/figures")

CONTRADICTION_CSV = (
    TABLE_DIR / "contradiction_breakdown.csv"
)


def main():
    if not CONTRADICTION_CSV.exists():
        raise FileNotFoundError(
            f"Missing: {CONTRADICTION_CSV}"
        )

    df = pd.read_csv(CONTRADICTION_CSV)

    # Keep a deterministic order.
    df = (
        df.set_index("label_mode")
        .loc[["natural", "anonymized"]]
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
        figsize=(6.6, 4.2)
    )

    for column, label in categories:
        values = (
            df[column]
            .to_numpy(dtype=float)
            * 100
        )

        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.62,
            label=label,
        )

        # Label only segments large enough to remain readable.
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

    # Clean legend: above the plot, no surrounding rectangle.
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        columnspacing=1.4,
        handlelength=1.6,
    )

    # Lighten visual clutter.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="both",
        labelsize=9,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.94]
    )

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

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
