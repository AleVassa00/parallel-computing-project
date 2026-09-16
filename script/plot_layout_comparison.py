from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ROW_CSV = (
    ROOT
    / "results"
    / "fase_1_tuning"
    / "campagna_VAS_2_row_major"
    / "campagna_VAS_2_row_major_cuda_warp.csv"
)
COLUMN_CSV = (
    ROOT
    / "results"
    / "fase_1_tuning"
    / "campagna_VAS_block_k_tuning"
    / "campagna_VAS_block_k_tuning_cuda_warp_xcolumn.csv"
)
OUTPUT_DIR = (
    ROOT
    / "plots"
    / "fase_1_tuning"
    / "campagna_VAS_2_row_major"
)

K_VALUES = [3, 6, 8, 20, 32]
BLOCK = 128

COLOR_ROW = "#4C78A8"
COLOR_COLUMN = "#D55E00"
COLOR_GRID = "#B8B8B8"
COLOR_REFERENCE = "#444444"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 17,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

def extract_block(kernel_names: pd.Series) -> pd.Series:
    """Ricava BLOCK dal nome; l'assenza del suffisso indica il default 256."""

    extracted = kernel_names.astype(str).str.extract(r"blk(\d+)", expand=False)
    return extracted.fillna(256).astype(int)

def require_single_value(df: pd.DataFrame, column: str, csv_path: Path):
    values = df[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{csv_path.name}: {column!r} deve avere un solo valore, "
            f"trovati {values.tolist()}."
        )
    return values[0]

def load_layout(csv_path: Path, expected_layout: str) -> tuple[pd.DataFrame, dict]:
    if not csv_path.exists():
        raise FileNotFoundError(f"File non trovato: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {
        "kernel",
        "scalar",
        "M",
        "N",
        "k",
        "P",
        "reps",
        "t_kernel_mean_s",
        "t_kernel_std_s",
        "t_kernel_cv_pct",
        "gflops_kernel",
        "x_layout",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name}: colonne mancanti: {sorted(missing)}")

    df = df.copy()
    numeric_columns = [
        "M",
        "N",
        "k",
        "P",
        "reps",
        "t_kernel_mean_s",
        "t_kernel_std_s",
        "t_kernel_cv_pct",
        "gflops_kernel",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid = df[numeric_columns].isna().any(axis=1)
    if invalid.any():
        rows = (df.index[invalid] + 2).tolist()
        raise ValueError(f"{csv_path.name}: valori numerici non validi alle righe {rows}.")

    df["block"] = extract_block(df["kernel"])
    df = df[df["block"] == BLOCK].copy()
    if df.empty:
        raise ValueError(f"{csv_path.name}: nessun risultato per BLOCK={BLOCK}.")

    layouts = set(df["x_layout"].astype(str))
    if layouts != {expected_layout}:
        raise ValueError(
            f"{csv_path.name}: layout atteso {expected_layout!r}, trovati {sorted(layouts)}."
        )

    observed_k = set(df["k"].astype(int))
    if observed_k != set(K_VALUES):
        raise ValueError(
            f"{csv_path.name}: k attesi {K_VALUES}, trovati {sorted(observed_k)}."
        )

    duplicates = df.duplicated(subset=["k"], keep=False)
    if duplicates.any():
        values = sorted(df.loc[duplicates, "k"].astype(int).unique())
        raise ValueError(f"{csv_path.name}: risultati duplicati per k={values}.")

    if (df["t_kernel_mean_s"] <= 0).any() or (df["gflops_kernel"] <= 0).any():
        raise ValueError(f"{csv_path.name}: tempi e throughput devono essere positivi.")

    flop = 2.0 * df["M"] * df["N"] * df["k"]
    expected_gflops = flop / df["t_kernel_mean_s"] / 1.0e9
    if not np.allclose(
        df["gflops_kernel"], expected_gflops, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError(
            f"{csv_path.name}: gflops_kernel non è coerente con "
            "2*M*N*k/t_kernel_mean_s."
        )

    metadata = {
        "scalar": str(require_single_value(df, "scalar", csv_path)),
        "M": int(require_single_value(df, "M", csv_path)),
        "N": int(require_single_value(df, "N", csv_path)),
        "P": int(require_single_value(df, "P", csv_path)),
        "reps": int(require_single_value(df, "reps", csv_path)),
    }

    keep = ["k", "gflops_kernel", "t_kernel_cv_pct"]
    return df[keep].sort_values("k").reset_index(drop=True), metadata

def verify_same_setup(row_metadata: dict, column_metadata: dict) -> None:
    mismatches = {
        key: (row_metadata[key], column_metadata[key])
        for key in row_metadata
        if row_metadata[key] != column_metadata[key]
    }
    if mismatches:
        raise ValueError(f"Le due campagne non hanno lo stesso setup: {mismatches}")

def build_comparison(row: pd.DataFrame, column: pd.DataFrame) -> pd.DataFrame:
    comparison = row.merge(
        column,
        on="k",
        how="inner",
        validate="one_to_one",
        suffixes=("_row", "_column"),
    )
    comparison["speedup_column_vs_row"] = (
        comparison["gflops_kernel_column"]
        / comparison["gflops_kernel_row"]
    )
    comparison["gain_column_pct"] = 100.0 * (
        comparison["speedup_column_vs_row"] - 1.0
    )
    return comparison

def annotate_bars(ax, bars, values, offset: float) -> None:
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, offset),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            color=bar.get_facecolor(),
        )

def plot_comparison(comparison: pd.DataFrame, metadata: dict) -> tuple[Path, Path]:
    x = np.arange(len(comparison), dtype=float)
    width = 0.34

    fig, (ax_throughput, ax_speedup) = plt.subplots(
        1,
        2,
        figsize=(15.0, 6.7),
        gridspec_kw={"width_ratios": [1.28, 1.0]},
    )
    fig.subplots_adjust(left=0.065, right=0.975, bottom=0.17, top=0.78, wspace=0.30)

    fig.suptitle(
        "CUDA warp: confronto tra layout row-major e column-major",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        (
            f"M={metadata['M']}, N={metadata['N']}, P={metadata['P']}, "
            f"BLOCK={BLOCK}, precisione={metadata['scalar']} · "
            f"{metadata['reps']} ripetizioni · throughput basato sul tempo medio"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    row_values = comparison["gflops_kernel_row"].to_numpy()
    column_values = comparison["gflops_kernel_column"].to_numpy()
    row_bars = ax_throughput.bar(
        x - width / 2.0,
        row_values,
        width,
        color=COLOR_ROW,
        label="Row-major",
    )
    column_bars = ax_throughput.bar(
        x + width / 2.0,
        column_values,
        width,
        color=COLOR_COLUMN,
        label="Column-major",
    )
    annotate_bars(ax_throughput, row_bars, row_values, 4)
    annotate_bars(ax_throughput, column_bars, column_values, 4)

    max_throughput = max(float(row_values.max()), float(column_values.max()))
    ax_throughput.set_ylim(0.0, max_throughput * 1.17)
    ax_throughput.set_xticks(x)
    ax_throughput.set_xticklabels(comparison["k"].astype(int).astype(str))
    ax_throughput.set_xlabel("Ampiezza del multivettore k")
    ax_throughput.set_ylabel("Throughput del kernel [GFLOP/s]")
    ax_throughput.set_title("Prestazioni assolute")
    ax_throughput.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        color=COLOR_GRID,
    )
    ax_throughput.legend(frameon=False, loc="upper left", ncol=2)

    speedup = comparison["speedup_column_vs_row"].to_numpy()
    colors = [COLOR_COLUMN if value >= 1.0 else COLOR_ROW for value in speedup]
    speedup_bars = ax_speedup.bar(x, speedup, width=0.58, color=colors)
    ax_speedup.axhline(
        1.0,
        color=COLOR_REFERENCE,
        linewidth=1.2,
        linestyle="--",
        zorder=0,
    )
    ax_speedup.text(
        len(x) - 0.55,
        1.03,
        "parità",
        ha="right",
        va="bottom",
        fontsize=9,
        color=COLOR_REFERENCE,
    )
    for bar, value in zip(speedup_bars, speedup):
        ax_speedup.annotate(
            f"{value:.2f}×",
            (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            color=bar.get_facecolor(),
            fontweight="bold" if value >= 1.05 else "normal",
        )

    ax_speedup.set_ylim(0.0, max(1.25, float(speedup.max()) * 1.15))
    ax_speedup.set_xticks(x)
    ax_speedup.set_xticklabels(comparison["k"].astype(int).astype(str))
    ax_speedup.set_xlabel("Ampiezza del multivettore k")
    ax_speedup.set_ylabel("Speedup column-major / row-major [×]")
    ax_speedup.set_title("Vantaggio del layout column-major")
    ax_speedup.grid(
        True,
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.55,
        color=COLOR_GRID,
    )

    max_row_cv = float(comparison["t_kernel_cv_pct_row"].max())
    max_column_cv = float(comparison["t_kernel_cv_pct_column"].max())
    fig.text(
        0.5,
        0.045,
        (
            "Speedup = GFLOP/s column-major / GFLOP/s row-major. "
            f"CV massimo: row-major={max_row_cv:.2f}%, "
            f"column-major={max_column_cv:.2f}%."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / "campagna_VAS_2_layout_comparison"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path

def main() -> None:
    row, row_metadata = load_layout(ROW_CSV, expected_layout="row")
    column, column_metadata = load_layout(COLUMN_CSV, expected_layout="column")
    verify_same_setup(row_metadata, column_metadata)
    comparison = build_comparison(row, column)

    print("Confronto CUDA warp: row-major vs column-major")
    print(
        comparison[
            [
                "k",
                "gflops_kernel_row",
                "gflops_kernel_column",
                "speedup_column_vs_row",
                "gain_column_pct",
                "t_kernel_cv_pct_row",
                "t_kernel_cv_pct_column",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )

    png_path, pdf_path = plot_comparison(comparison, row_metadata)
    print(f"Grafico PNG: {png_path}")
    print(f"Grafico PDF: {pdf_path}")

if __name__ == "__main__":
    main()
