from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = (
    ROOT
    / "results"
    / "fase_1_tuning"
    / "campagna_VAS_4_smem_pad_tuning"
)
INPUT_CSV = RESULTS_DIR / "campagna_VAS_4_smem_pad_tuning_cuda_warp_smem.csv"
RAW_CSV = RESULTS_DIR / "campagna_VAS_4_smem_pad_tuning_cuda_warp_smem_raw.csv"
OUTPUT_DIR = (
    ROOT
    / "plots"
    / "fase_1_tuning"
    / "campagna_VAS_4_smem_pad_tuning"
)

K_VALUES = [3, 6, 8, 20, 32]
EXPECTED_REPS = 20
SELECTED_PAD = 1
BLOCK = 256
TILE_GRANULARITY = 32

COLOR_PAD0 = "#4C78A8"
COLOR_PAD1 = "#D55E00"
COLOR_NEGATIVE = "#6B7280"
COLOR_GRID = "#B8B8B8"

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


def extract_pad(kernel_names: pd.Series) -> pd.Series:
    """Ricava SMEM_PAD: il nome senza suffisso corrisponde a pad=1."""

    extracted = kernel_names.astype(str).str.extract(r"pad(\d+)", expand=False)
    return extracted.fillna(1).astype(int)


def require_single_value(df: pd.DataFrame, column: str):
    values = df[column].dropna().unique()
    if len(values) != 1:
        raise ValueError(
            f"{INPUT_CSV.name}: {column!r} deve avere un solo valore, "
            f"trovati {values.tolist()}."
        )
    return values[0]


def load_results() -> tuple[pd.DataFrame, dict]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_CSV}")
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {RAW_CSV}")

    df = pd.read_csv(INPUT_CSV)
    raw = pd.read_csv(RAW_CSV)
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
        "blocks_per_sm",
        "x_rows_per_tile",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{INPUT_CSV.name}: colonne mancanti: {sorted(missing)}")

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
        "blocks_per_sm",
        "x_rows_per_tile",
    ]
    df = df.copy()
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    invalid = df[numeric_columns].isna().any(axis=1)
    if invalid.any():
        rows = (df.index[invalid] + 2).tolist()
        raise ValueError(f"{INPUT_CSV.name}: valori numerici non validi alle righe {rows}.")

    df["smem_pad"] = extract_pad(df["kernel"])
    for column in ["M", "N", "k", "P", "reps", "smem_pad"]:
        df[column] = df[column].astype(int)

    if set(df["smem_pad"]) != {0, 1}:
        raise ValueError(f"Valori SMEM_PAD inattesi: {sorted(set(df['smem_pad']))}")
    if set(df["k"]) != set(K_VALUES):
        raise ValueError(f"Valori di k inattesi: {sorted(set(df['k']))}")
    if len(df) != 2 * len(K_VALUES):
        raise ValueError(f"Attese 10 righe aggregate, trovate {len(df)}.")
    if df.duplicated(subset=["smem_pad", "k"]).any():
        raise ValueError("Sono presenti configurazioni duplicate (SMEM_PAD, k).")
    if (df["t_kernel_mean_s"] <= 0).any() or (df["gflops_kernel"] <= 0).any():
        raise ValueError("Tempi e throughput devono essere positivi.")

    flop = 2.0 * df["M"] * df["N"] * df["k"]
    expected_gflops = flop / df["t_kernel_mean_s"] / 1.0e9
    if not np.allclose(
        df["gflops_kernel"], expected_gflops, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError(
            "gflops_kernel non è coerente con 2*M*N*k/t_kernel_mean_s."
        )

    reps = int(require_single_value(df, "reps"))
    if reps != EXPECTED_REPS:
        raise ValueError(f"Attese {EXPECTED_REPS} ripetizioni, trovate {reps}.")
    if len(raw) != len(df) * reps:
        raise ValueError(
            f"Attese {len(df) * reps} righe raw, trovate {len(raw)}."
        )

    metadata = {
        "scalar": str(require_single_value(df, "scalar")),
        "M": int(require_single_value(df, "M")),
        "N": int(require_single_value(df, "N")),
        "P": int(require_single_value(df, "P")),
        "reps": reps,
    }
    return df, metadata


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    comparison = df.pivot(index="k", columns="smem_pad", values="gflops_kernel")
    comparison = comparison.reindex(index=K_VALUES, columns=[0, 1])
    if comparison.isna().any().any():
        raise ValueError("Dati incompleti per il confronto tra SMEM_PAD=0 e 1.")

    comparison.columns = ["pad0_gflops", "pad1_gflops"]
    comparison["speedup"] = comparison["pad1_gflops"] / comparison["pad0_gflops"]
    comparison["change_pct"] = 100.0 * (comparison["speedup"] - 1.0)
    comparison["best_gflops"] = comparison[["pad0_gflops", "pad1_gflops"]].max(
        axis=1
    )
    comparison["pad0_relative_pct"] = (
        100.0 * comparison["pad0_gflops"] / comparison["best_gflops"]
    )
    comparison["pad1_relative_pct"] = (
        100.0 * comparison["pad1_gflops"] / comparison["best_gflops"]
    )
    return comparison


def plot_padding_comparison(
    comparison: pd.DataFrame, metadata: dict
) -> tuple[Path, Path]:
    x = np.arange(len(K_VALUES), dtype=float)
    fig, (ax_perf, ax_change) = plt.subplots(1, 2, figsize=(15.0, 6.6))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.19, top=0.75, wspace=0.28)
    fig.suptitle(
        "CUDA warp shared-memory: effetto del padding",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.885,
        (
            f"M={metadata['M']}, N={metadata['N']}, P={metadata['P']}, "
            f"precisione={metadata['scalar']}, BLOCK={BLOCK}, "
            f"granularità={TILE_GRANULARITY} · {metadata['reps']} ripetizioni"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    ax_perf.plot(
        x,
        comparison["pad0_gflops"],
        marker="o",
        markersize=7,
        linewidth=2.2,
        color=COLOR_PAD0,
        label="SMEM_PAD=0",
    )
    ax_perf.plot(
        x,
        comparison["pad1_gflops"],
        marker="o",
        markersize=8,
        linewidth=3.0,
        color=COLOR_PAD1,
        label="SMEM_PAD=1 (selezionato)",
    )
    ax_perf.set_ylim(bottom=0.0)
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels([str(k) for k in K_VALUES])
    ax_perf.set_xlabel("Ampiezza del multivettore k")
    ax_perf.set_ylabel("Throughput del kernel [GFLOP/s]")
    ax_perf.set_title("Throughput con e senza padding")
    ax_perf.grid(True, linestyle="--", linewidth=0.7, alpha=0.55, color=COLOR_GRID)
    ax_perf.legend(frameon=False, loc="lower left")

    changes = comparison["change_pct"].to_numpy(dtype=float)
    bar_colors = [COLOR_PAD1 if value >= 0 else COLOR_NEGATIVE for value in changes]
    bars = ax_change.bar(x, changes, width=0.62, color=bar_colors)
    ax_change.axhline(0.0, color="#333333", linewidth=1.0)
    padding = max(2.5, 0.04 * (changes.max() - changes.min()))
    for bar, value in zip(bars, changes):
        vertical = padding if value >= 0 else -padding
        alignment = "bottom" if value >= 0 else "top"
        ax_change.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + vertical,
            f"{value:+.2f}%",
            ha="center",
            va=alignment,
            fontsize=9.5,
            color="#222222",
        )
    lower = min(-5.0, float(changes.min()) - 2.0 * padding)
    upper = float(changes.max()) + 3.0 * padding
    ax_change.set_ylim(lower, upper)
    ax_change.set_xticks(x)
    ax_change.set_xticklabels([str(k) for k in K_VALUES])
    ax_change.set_xlabel("Ampiezza del multivettore k")
    ax_change.set_ylabel("Variazione di throughput [%]")
    ax_change.set_title("Variazione di SMEM_PAD=1 rispetto a SMEM_PAD=0")
    ax_change.grid(
        True, axis="y", linestyle="--", linewidth=0.7, alpha=0.55, color=COLOR_GRID
    )

    pad1_mean = comparison["pad1_relative_pct"].mean()
    pad1_worst = comparison["pad1_relative_pct"].min()
    fig.text(
        0.5,
        0.055,
        (
            f"SMEM_PAD=1 selezionato · prestazione relativa media={pad1_mean:.2f}% · "
            f"caso peggiore={pad1_worst:.2f}% · "
            "il padding evita il forte calo osservato a k=32"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / "campagna_VAS_4_smem_padding_comparison"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    df, metadata = load_results()
    comparison = build_comparison(df)
    png_path, pdf_path = plot_padding_comparison(comparison, metadata)

    print("Campagna VAS 4 - confronto padding shared memory")
    print(f"Configurazioni aggregate: {len(df)}")
    print(f"Ripetizioni raw: {len(df) * metadata['reps']}")
    for k, row in comparison.iterrows():
        print(
            f"k={k}: pad0={row['pad0_gflops']:.3f} GFLOP/s, "
            f"pad1={row['pad1_gflops']:.3f} GFLOP/s, "
            f"variazione={row['change_pct']:+.3f}%"
        )
    print(
        "SMEM_PAD=1: "
        f"media relativa={comparison['pad1_relative_pct'].mean():.3f}%, "
        f"peggior caso={comparison['pad1_relative_pct'].min():.3f}%"
    )
    print(f"Grafico PNG: {png_path}")
    print(f"Grafico PDF: {pdf_path}")


if __name__ == "__main__":
    main()
