from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "results"
    / "fase_1_tuning"
    / "campagna_VAS_block_k_tuning"
)

OUTPUT_DIR = (
    ROOT
    / "plots"
    / "fase_1_tuning"
    / "campagna_VAS_block_k_tuning"
)

EXPERIMENTS = {
    "cuda_naive": {
        "csv": "campagna_VAS_block_k_tuning_cuda_naive.csv",
        "failure_csv": None,
        "title": "CUDA naive",
    },
    "cuda_warp_xcolumn": {
        "csv": "campagna_VAS_block_k_tuning_cuda_warp_xcolumn.csv",
        "failure_csv": "campagna_VAS_block_k_tuning_cuda_warp_xcolumn_failures.csv",
        "title": "CUDA warp, X column-major",
    },
    "cuda_warp_smem": {
        "csv": "campagna_VAS_block_k_tuning_cuda_warp_smem.csv",
        "failure_csv": "campagna_VAS_block_k_tuning_cuda_warp_smem_failures.csv",
        "title": "CUDA warp con shared memory",
    },
}

K_VALUES = [3, 6, 8, 20, 32]
BLOCK_VALUES = [64, 128, 192, 256, 384, 512, 1024]

MEAN_R_TIE_TOLERANCE_PCT = 0.5

COLOR_STANDARD = "#4C78A8"
COLOR_CHOSEN = "#D55E00"
COLOR_MAX = "#222222"
COLOR_MISSING = "#E5E7EB"
COLOR_GRID = "#B8B8B8"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
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
            f"{csv_path.name}: la colonna {column!r} deve avere un solo "
            f"valore, trovati {values.tolist()}."
        )
    return values[0]

def load_results(csv_path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(csv_path)

    required_columns = {
        "kernel",
        "scalar",
        "M",
        "N",
        "k",
        "P",
        "t_kernel_mean_s",
        "t_kernel_cv_pct",
        "gflops_kernel",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path.name}: colonne mancanti: {sorted(missing)}"
        )

    df = df.copy()
    numeric_columns = [
        "M",
        "N",
        "k",
        "P",
        "t_kernel_mean_s",
        "t_kernel_cv_pct",
        "gflops_kernel",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_numeric = df[numeric_columns].isna().any(axis=1)
    if invalid_numeric.any():
        bad_rows = (df.index[invalid_numeric] + 2).tolist()
        raise ValueError(
            f"{csv_path.name}: valori numerici non validi alle righe {bad_rows}."
        )

    if (df["t_kernel_mean_s"] <= 0).any():
        raise ValueError(
            f"{csv_path.name}: t_kernel_mean_s deve essere positivo."
        )
    if (df["gflops_kernel"] <= 0).any():
        raise ValueError(
            f"{csv_path.name}: gflops_kernel deve essere positivo."
        )

    df[["M", "N", "k", "P"]] = df[["M", "N", "k", "P"]].astype(int)
    df["block"] = extract_block(df["kernel"])

    metadata = {
        "scalar": str(require_single_value(df, "scalar", csv_path)),
        "M": int(require_single_value(df, "M", csv_path)),
        "N": int(require_single_value(df, "N", csv_path)),
        "P": int(require_single_value(df, "P", csv_path)),
    }

    observed_k = set(df["k"])
    unexpected_k = sorted(observed_k - set(K_VALUES))
    if unexpected_k:
        raise ValueError(
            f"{csv_path.name}: valori di k inattesi: {unexpected_k}."
        )

    observed_blocks = set(df["block"])
    unexpected_blocks = sorted(observed_blocks - set(BLOCK_VALUES))
    if unexpected_blocks:
        raise ValueError(
            f"{csv_path.name}: BLOCK inattesi: {unexpected_blocks}."
        )

    duplicates = df.duplicated(subset=["k", "block"], keep=False)
    if duplicates.any():
        pairs = (
            df.loc[duplicates, ["k", "block"]]
            .drop_duplicates()
            .sort_values(["k", "block"])
            .to_dict("records")
        )
        raise ValueError(
            f"{csv_path.name}: configurazioni duplicate: {pairs}."
        )

    flop = 2.0 * df["M"] * df["N"] * df["k"]
    expected_gflops = flop / df["t_kernel_mean_s"] / 1.0e9
    if not np.allclose(
        df["gflops_kernel"],
        expected_gflops,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            f"{csv_path.name}: gflops_kernel non è coerente con "
            "2*M*N*k/t_kernel_mean_s."
        )

    return df, metadata

def load_failures(failure_path: Path | None) -> pd.DataFrame:
    if failure_path is None or not failure_path.exists():
        return pd.DataFrame(columns=["k", "block"])

    df = pd.read_csv(failure_path)
    required_columns = {"k", "block"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{failure_path.name}: servono le colonne k e block."
        )

    if df.empty:
        return pd.DataFrame(columns=["k", "block"])

    df = df.copy()
    df["k"] = pd.to_numeric(df["k"], errors="raise").astype(int)
    df["block"] = pd.to_numeric(df["block"], errors="raise").astype(int)
    return df[["k", "block"]].drop_duplicates()

def validate_coverage(
    df: pd.DataFrame,
    failure_df: pd.DataFrame,
    csv_path: Path,
) -> None:
    result_pairs = set(zip(df["k"], df["block"]))
    failure_pairs = set(zip(failure_df["k"], failure_df["block"]))

    overlap = sorted(result_pairs & failure_pairs)
    if overlap:
        raise ValueError(
            f"{csv_path.name}: configurazioni presenti sia nei risultati "
            f"sia nei fallimenti: {overlap}."
        )

    expected_pairs = {(k, block) for k in K_VALUES for block in BLOCK_VALUES}
    missing_pairs = sorted(expected_pairs - result_pairs - failure_pairs)
    if missing_pairs:
        raise ValueError(
            f"{csv_path.name}: configurazioni senza risultato né fallimento: "
            f"{missing_pairs}."
        )

def build_pivot(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return (
        df.pivot(index="k", columns="block", values=value_column)
        .reindex(index=K_VALUES, columns=BLOCK_VALUES)
    )

def compute_normalized_summary(
    performance_pivot: pd.DataFrame,
    cv_pivot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcola prestazione relativa e perdite per ciascun BLOCK."""

    if performance_pivot.notna().sum(axis=1).eq(0).any():
        missing_k = performance_pivot.index[
            performance_pivot.notna().sum(axis=1).eq(0)
        ].tolist()
        raise ValueError(f"Nessun risultato valido per k={missing_k}.")

    best_per_k = performance_pivot.max(axis=1)
    normalized = performance_pivot.div(best_per_k, axis=0)
    loss = 1.0 - normalized

    summary = pd.DataFrame(index=performance_pivot.columns)
    summary.index.name = "block"
    summary["valid_all_k"] = performance_pivot.notna().all(axis=0)
    summary["mean_R_pct"] = 100.0 * normalized.mean(axis=0)
    summary["mean_loss_pct"] = 100.0 * loss.mean(axis=0)
    summary["max_loss_pct"] = 100.0 * loss.max(axis=0)
    summary["mean_kernel_cv_pct"] = cv_pivot.mean(axis=0)

    valid = summary["valid_all_k"]
    best_mean = summary.loc[valid, "mean_R_pct"].max()
    summary["near_best_mean"] = (
        valid
        & (
            summary["mean_R_pct"]
            >= best_mean - MEAN_R_TIE_TOLERANCE_PCT
        )
    )

    return normalized, summary

def choose_block(summary: pd.DataFrame) -> int:
    """Sceglie il compromesso robusto tra i BLOCK praticamente equivalenti."""

    candidates = summary[summary["near_best_mean"]].reset_index()
    if candidates.empty:
        raise ValueError("Nessun BLOCK valido per la selezione.")

    candidates = candidates.sort_values(
        by=["max_loss_pct", "mean_kernel_cv_pct", "block"],
        ascending=[True, True, True],
        na_position="last",
    )
    return int(candidates.iloc[0]["block"])

def print_summary(
    experiment_name: str,
    metadata: dict,
    performance_pivot: pd.DataFrame,
    summary: pd.DataFrame,
    chosen_block: int,
) -> None:
    print()
    print("=" * 96)
    print(experiment_name)
    print("=" * 96)
    print(
        f"M={metadata['M']} N={metadata['N']} P={metadata['P']} "
        f"precisione={metadata['scalar']}"
    )
    print("\nThroughput del kernel da t_kernel_mean_s [GFLOP/s]:")
    print(performance_pivot.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nRiepilogo della selezione robusta:")
    print(summary.to_string(float_format=lambda x: f"{x:.3f}"))
    print(
        f"\nBLOCK SCELTO: {chosen_block} "
        f"(quasi-parità: {MEAN_R_TIE_TOLERANCE_PCT:.1f} punti percentuali)"
    )

def heatmap_lower_bound(normalized_pct: np.ndarray) -> float:
    finite = normalized_pct[np.isfinite(normalized_pct)]
    if finite.size == 0:
        raise ValueError("Nessuna prestazione valida da rappresentare.")
    return max(0.0, 5.0 * np.floor(float(finite.min()) / 5.0))

def add_heatmap(
    ax,
    normalized: pd.DataFrame,
    failure_df: pd.DataFrame,
    chosen_block: int,
):
    values = 100.0 * normalized.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(COLOR_MISSING)
    vmin = heatmap_lower_bound(values)

    image = ax.imshow(
        masked,
        cmap=cmap,
        vmin=vmin,
        vmax=100.0,
        aspect="auto",
        interpolation="nearest",
    )

    failure_pairs = set(zip(failure_df["k"], failure_df["block"]))
    threshold = (vmin + 100.0) / 2.0

    for row, k in enumerate(K_VALUES):
        valid_row = normalized.loc[k].dropna()
        best_block = int(valid_row.idxmax())

        for col, block in enumerate(BLOCK_VALUES):
            value = values[row, col]
            if np.isfinite(value):
                text_color = "white" if value < threshold else "#111111"
                ax.text(
                    col,
                    row,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="normal",
                )
            else:
                label = "FAIL" if (k, block) in failure_pairs else "N/D"
                ax.text(
                    col,
                    row,
                    label,
                    ha="center",
                    va="center",
                    color="#555555",
                    fontsize=8,
                    fontweight="normal",
                )

        best_col = BLOCK_VALUES.index(best_block)
        ax.scatter(
            [best_col + 0.34],
            [row - 0.30],
            marker="*",
            s=85,
            color="#F2C14E",
            edgecolor=COLOR_MAX,
            linewidth=0.6,
            zorder=4,
        )

    chosen_col = BLOCK_VALUES.index(chosen_block)
    ax.add_patch(
        Rectangle(
            (chosen_col - 0.5, -0.5),
            1.0,
            len(K_VALUES),
            fill=False,
            edgecolor=COLOR_CHOSEN,
            linewidth=3.0,
            clip_on=False,
        )
    )

    ax.set_xticks(np.arange(len(BLOCK_VALUES)))
    ax.set_xticklabels([str(block) for block in BLOCK_VALUES])
    ax.set_yticks(np.arange(len(K_VALUES)))
    ax.set_yticklabels([str(k) for k in K_VALUES])
    ax.set_xlabel("Thread per blocco (BLOCK)")
    ax.set_ylabel("Ampiezza del multivettore k")
    ax.set_title("Prestazione relativa al migliore BLOCK per ogni k")

    for tick, block in zip(ax.get_xticklabels(), BLOCK_VALUES):
        if block == chosen_block:
            tick.set_color(COLOR_CHOSEN)
            tick.set_fontweight("bold")

    colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.048, pad=0.03)
    colorbar.set_label("Prestazione normalizzata [%]")
    return image

def add_loss_plot(ax, summary: pd.DataFrame, chosen_block: int):
    valid = summary[summary["valid_all_k"]].copy().sort_index()
    blocks = valid.index.to_numpy(dtype=int)
    y = np.arange(len(blocks), dtype=float)

    for pos, block in zip(y, blocks):
        row = valid.loc[block]
        mean_loss = float(row["mean_loss_pct"])
        max_loss = float(row["max_loss_pct"])
        color = COLOR_CHOSEN if block == chosen_block else COLOR_STANDARD
        linewidth = 3.2 if block == chosen_block else 2.0
        zorder = 4 if block == chosen_block else 2

        ax.hlines(pos, mean_loss, max_loss, color=color, linewidth=linewidth, zorder=zorder)
        ax.scatter(
            mean_loss,
            pos,
            marker="o",
            s=65 if block == chosen_block else 48,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=zorder + 1,
        )
        ax.scatter(
            max_loss,
            pos,
            marker="X",
            s=70 if block == chosen_block else 52,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            zorder=zorder + 1,
        )
        ax.annotate(
            f"{max_loss:.2f}%",
            (max_loss, pos),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=color,
            fontweight="bold" if block == chosen_block else "normal",
        )

    ax.set_yticks(y)
    ax.set_yticklabels([str(block) for block in blocks])
    ax.invert_yaxis()
    ax.set_xlabel("Perdita rispetto al migliore BLOCK per lo stesso k [%]")
    ax.set_ylabel("Thread per blocco (BLOCK)")
    ax.set_title("Perdita media (●) e massima (×) sui cinque k")
    max_observed_loss = float(valid["max_loss_pct"].max())
    ax.set_xlim(0.0, max_observed_loss * 1.16 + 0.5)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.7, alpha=0.55, color=COLOR_GRID)

    for tick, block in zip(ax.get_yticklabels(), blocks):
        if block == chosen_block:
            tick.set_color(COLOR_CHOSEN)
            tick.set_fontweight("bold")

def plot_tuning_summary(
    experiment_name: str,
    title: str,
    metadata: dict,
    normalized: pd.DataFrame,
    summary: pd.DataFrame,
    chosen_block: int,
    failure_df: pd.DataFrame,
) -> None:
    fig, (ax_heatmap, ax_loss) = plt.subplots(
        1,
        2,
        figsize=(15.6, 7.0),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )

    fig.subplots_adjust(
        left=0.06,
        right=0.975,
        bottom=0.20,
        top=0.78,
        wspace=0.34,
    )

    fig.suptitle(
        f"{title}: tuning della dimensione del blocco",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.895,
        (
            f"M={metadata['M']}, N={metadata['N']}, P={metadata['P']}, "
            f"precisione={metadata['scalar']} · "
            "throughput del benchmark basato sul tempo medio"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    add_heatmap(
        ax=ax_heatmap,
        normalized=normalized,
        failure_df=failure_df,
        chosen_block=chosen_block,
    )
    add_loss_plot(ax=ax_loss, summary=summary, chosen_block=chosen_block)

    chosen = summary.loc[chosen_block]
    fig.text(
        0.5,
        0.030,
        (
            f"BLOCK selezionato: {chosen_block} · prestazione media={chosen['mean_R_pct']:.2f}% "
            f"· perdita media={chosen['mean_loss_pct']:.2f}% · "
            f"perdita massima={chosen['max_loss_pct']:.2f}%\n"
            f"CV medio={chosen['mean_kernel_cv_pct']:.2f}% · "
            f"quasi-parità entro {MEAN_R_TIE_TOLERANCE_PCT:.1f} punti percentuali"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )

    base_name = f"campagna_VAS_block_k_tuning_summary_{experiment_name}"
    png_path = OUTPUT_DIR / f"{base_name}.png"
    pdf_path = OUTPUT_DIR / f"{base_name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Grafico PNG: {png_path}")
    print(f"Grafico PDF: {pdf_path}")

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Tuning BLOCK x k con selezione robusta")
    print(f"Input : {RESULTS_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    for experiment_name, config in EXPERIMENTS.items():
        csv_path = RESULTS_DIR / config["csv"]
        if not csv_path.exists():
            raise FileNotFoundError(f"File non trovato: {csv_path}")

        failure_path = (
            RESULTS_DIR / config["failure_csv"]
            if config["failure_csv"] is not None
            else None
        )

        df, metadata = load_results(csv_path)
        failure_df = load_failures(failure_path)
        validate_coverage(df, failure_df, csv_path)

        performance_pivot = build_pivot(df, "gflops_kernel")
        cv_pivot = build_pivot(df, "t_kernel_cv_pct")
        normalized, summary = compute_normalized_summary(
            performance_pivot=performance_pivot,
            cv_pivot=cv_pivot,
        )
        chosen_block = choose_block(summary)

        print_summary(
            experiment_name=experiment_name,
            metadata=metadata,
            performance_pivot=performance_pivot,
            summary=summary,
            chosen_block=chosen_block,
        )
        plot_tuning_summary(
            experiment_name=experiment_name,
            title=config["title"],
            metadata=metadata,
            normalized=normalized,
            summary=summary,
            chosen_block=chosen_block,
            failure_df=failure_df,
        )

if __name__ == "__main__":
    main()
