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
    / "campagna_VAS_5_smem_granularity_tuning"
)
INPUT_CSV = RESULTS_DIR / "campagna_VAS_5_smem_granularity_merged.csv"
FAILURES_CSV = RESULTS_DIR / "campagna_VAS_5_smem_granularity_failures_merged.csv"
OUTPUT_DIR = (
    ROOT
    / "plots"
    / "fase_1_tuning"
    / "campagna_VAS_5_smem_granularity_tuning"
)

K_VALUES = [3, 6, 8, 20, 32]
BLOCK_VALUES = [64, 128, 192, 256, 384, 512, 1024]
GRANULARITIES = [1, 8, 16, 32]
SELECTED_BLOCK = 256
SELECTED_GRANULARITY = 32
SMEM_PAD = 1
MEAN_R_TIE_TOLERANCE_PCT = 0.5

COLOR_SELECTED = "#D55E00"
COLOR_MISSING = "#E5E7EB"
COLOR_GRID = "#B8B8B8"
SERIES_COLORS = {
    1: "#6B7280",
    8: "#2A9D8F",
    16: "#7A5195",
    32: COLOR_SELECTED,
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
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

def load_results() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {INPUT_CSV}")
    if not FAILURES_CSV.exists():
        raise FileNotFoundError(f"File non trovato: {FAILURES_CSV}")

    df = pd.read_csv(INPUT_CSV)
    failures = pd.read_csv(FAILURES_CSV)

    required = {
        "kernel",
        "tile_granularity",
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
        "tile_granularity",
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

    df["block"] = extract_block(df["kernel"])
    for column in ["tile_granularity", "M", "N", "k", "P", "reps", "block"]:
        df[column] = df[column].astype(int)

    if (df["t_kernel_mean_s"] <= 0).any() or (df["gflops_kernel"] <= 0).any():
        raise ValueError("Tempi e throughput devono essere positivi.")

    if set(df["k"]) != set(K_VALUES):
        raise ValueError(f"Valori di k inattesi: {sorted(set(df['k']))}")

    if set(df["tile_granularity"]) != set(GRANULARITIES):
        raise ValueError(
            "Granularità inattese: "
            f"{sorted(set(df['tile_granularity']))}"
        )

    duplicates = df.duplicated(subset=["tile_granularity", "block", "k"], keep=False)
    if duplicates.any():
        pairs = (
            df.loc[duplicates, ["tile_granularity", "block", "k"]]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(f"Configurazioni duplicate: {pairs}")

    flop = 2.0 * df["M"] * df["N"] * df["k"]
    expected_gflops = flop / df["t_kernel_mean_s"] / 1.0e9
    if not np.allclose(
        df["gflops_kernel"], expected_gflops, rtol=1.0e-5, atol=1.0e-6
    ):
        raise ValueError(
            "gflops_kernel non è coerente con 2*M*N*k/t_kernel_mean_s."
        )

    failure_required = {"tile_granularity", "k", "block"}
    if not failure_required.issubset(failures.columns):
        raise ValueError(
            f"{FAILURES_CSV.name}: servono le colonne "
            "tile_granularity, k e block."
        )
    failures = failures.copy()
    for column in failure_required:
        failures[column] = pd.to_numeric(failures[column], errors="raise").astype(int)
    failures = failures[["tile_granularity", "block", "k"]].drop_duplicates()

    result_keys = set(zip(df["tile_granularity"], df["block"], df["k"]))
    failure_keys = set(
        zip(failures["tile_granularity"], failures["block"], failures["k"])
    )
    expected_keys = {
        (granularity, block, k)
        for granularity in GRANULARITIES
        for block in BLOCK_VALUES
        for k in K_VALUES
    }
    overlap = result_keys & failure_keys
    missing_keys = expected_keys - result_keys - failure_keys
    if overlap:
        raise ValueError(f"Risultati e fallimenti sovrapposti: {sorted(overlap)}")
    if missing_keys:
        raise ValueError(f"Configurazioni senza esito: {sorted(missing_keys)}")

    metadata = {
        "scalar": str(require_single_value(df, "scalar", INPUT_CSV)),
        "M": int(require_single_value(df, "M", INPUT_CSV)),
        "N": int(require_single_value(df, "N", INPUT_CSV)),
        "P": int(require_single_value(df, "P", INPUT_CSV)),
        "reps": int(require_single_value(df, "reps", INPUT_CSV)),
    }
    return df, failures, metadata

def build_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance = df.pivot(
        index="k",
        columns=["tile_granularity", "block"],
        values="gflops_kernel",
    ).reindex(index=K_VALUES)
    performance = performance.reindex(
        columns=pd.MultiIndex.from_product(
            [GRANULARITIES, BLOCK_VALUES],
            names=["tile_granularity", "block"],
        )
    )

    best_per_k = performance.max(axis=1)
    normalized = 100.0 * performance.div(best_per_k, axis=0)

    cv = df.pivot(
        index="k",
        columns=["tile_granularity", "block"],
        values="t_kernel_cv_pct",
    ).reindex(index=K_VALUES, columns=performance.columns)

    summary = pd.DataFrame(index=performance.columns)
    summary["valid_all_k"] = performance.notna().all(axis=0)
    summary["mean_R_pct"] = normalized.mean(axis=0)
    summary["min_R_pct"] = normalized.min(axis=0)
    summary["max_loss_pct"] = 100.0 - summary["min_R_pct"]
    summary["mean_cv_pct"] = cv.mean(axis=0)
    summary["max_cv_pct"] = cv.max(axis=0)

    valid = summary["valid_all_k"]
    best_mean = summary.loc[valid, "mean_R_pct"].max()
    summary["near_best_mean"] = (
        valid
        & (summary["mean_R_pct"] >= best_mean - MEAN_R_TIE_TOLERANCE_PCT)
    )
    return normalized, summary

def choose_configuration(summary: pd.DataFrame) -> tuple[int, int]:
    candidates = summary[summary["near_best_mean"]].reset_index()
    candidates = candidates.sort_values(
        ["max_loss_pct", "max_cv_pct", "tile_granularity", "block"]
    )
    row = candidates.iloc[0]
    return int(row["block"]), int(row["tile_granularity"])

def summary_matrix(summary: pd.DataFrame, column: str) -> pd.DataFrame:
    matrix = summary[column].unstack("block")
    return matrix.reindex(index=GRANULARITIES, columns=BLOCK_VALUES)

def draw_summary_heatmap(
    ax,
    values: pd.DataFrame,
    valid: pd.DataFrame,
    title: str,
    vmin: float,
):
    array = values.to_numpy(dtype=float)
    valid_array = valid.to_numpy(dtype=bool)
    masked = np.ma.masked_where(~valid_array, array)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(COLOR_MISSING)
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=100.0)

    threshold = vmin + 0.34 * (100.0 - vmin)
    for row_index, granularity in enumerate(GRANULARITIES):
        for column_index, block in enumerate(BLOCK_VALUES):
            if valid_array[row_index, column_index]:
                value = array[row_index, column_index]
                color = "white" if value < threshold else "#111111"
                ax.text(
                    column_index,
                    row_index,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=9.5,
                    color=color,
                )
            else:
                ax.text(
                    column_index,
                    row_index,
                    "FAIL",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="#555555",
                )

    selected_column = BLOCK_VALUES.index(SELECTED_BLOCK)
    selected_row = GRANULARITIES.index(SELECTED_GRANULARITY)
    ax.add_patch(
        Rectangle(
            (selected_column - 0.5, selected_row - 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor=COLOR_SELECTED,
            linewidth=3.0,
            clip_on=False,
        )
    )

    ax.set_xticks(np.arange(len(BLOCK_VALUES)))
    ax.set_xticklabels([str(block) for block in BLOCK_VALUES])
    ax.set_yticks(np.arange(len(GRANULARITIES)))
    ax.set_yticklabels(["1", "8", "16", "32 (baseline C1)"])
    ax.set_xlabel("Thread per blocco (BLOCK)")
    ax.set_ylabel("Granularità di arrotondamento g")
    ax.set_title(title)

    for tick, block in zip(ax.get_xticklabels(), BLOCK_VALUES):
        if block == SELECTED_BLOCK:
            tick.set_color(COLOR_SELECTED)
            tick.set_fontweight("bold")
    for tick, granularity in zip(ax.get_yticklabels(), GRANULARITIES):
        if granularity == SELECTED_GRANULARITY:
            tick.set_color(COLOR_SELECTED)
            tick.set_fontweight("bold")
    return image

def plot_joint_summary(
    summary: pd.DataFrame,
    metadata: dict,
) -> tuple[Path, Path]:
    mean_matrix = summary_matrix(summary, "mean_R_pct")
    min_matrix = summary_matrix(summary, "min_R_pct")
    valid_matrix = summary_matrix(summary, "valid_all_k").astype(bool)

    valid_values = np.concatenate(
        [
            mean_matrix.to_numpy()[valid_matrix.to_numpy()],
            min_matrix.to_numpy()[valid_matrix.to_numpy()],
        ]
    )
    vmin = max(0.0, 10.0 * np.floor(float(valid_values.min()) / 10.0))

    fig, (ax_mean, ax_worst) = plt.subplots(1, 2, figsize=(15.2, 6.4))
    fig.subplots_adjust(left=0.075, right=0.93, bottom=0.19, top=0.76, wspace=0.30)
    fig.suptitle(
        "CUDA warp shared-memory: tuning congiunto di BLOCK e granularità",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.885,
        (
            f"M={metadata['M']}, N={metadata['N']}, P={metadata['P']}, "
            f"precisione={metadata['scalar']}, SMEM_PAD={SMEM_PAD} · "
            f"k={K_VALUES} · throughput basato sul tempo medio"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    image = draw_summary_heatmap(
        ax_mean,
        mean_matrix,
        valid_matrix,
        "Prestazione normalizzata media sui cinque k",
        vmin,
    )
    draw_summary_heatmap(
        ax_worst,
        min_matrix,
        valid_matrix,
        "Prestazione normalizzata nel caso peggiore",
        vmin,
    )
    colorbar = fig.colorbar(image, ax=[ax_mean, ax_worst], fraction=0.035, pad=0.025)
    colorbar.set_label("Prestazione rispetto all'ottimo per lo stesso k [%]")

    selected = summary.loc[(SELECTED_GRANULARITY, SELECTED_BLOCK)]
    fig.text(
        0.5,
        0.055,
        (
            f"Configurazione selezionata: BLOCK={SELECTED_BLOCK}, "
            f"g={SELECTED_GRANULARITY} · media={selected['mean_R_pct']:.2f}% · "
            f"caso peggiore={selected['min_R_pct']:.2f}% · "
            f"perdita massima={selected['max_loss_pct']:.2f}% · "
            f"CV massimo={selected['max_cv_pct']:.2f}%"
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / "campagna_VAS_5_joint_tuning_summary"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path

def plot_selected_block_detail(
    df: pd.DataFrame,
    metadata: dict,
) -> tuple[Path, Path]:
    detail = df[df["block"] == SELECTED_BLOCK].copy()
    detail = detail.sort_values(["tile_granularity", "k"])
    if len(detail) != len(GRANULARITIES) * len(K_VALUES):
        raise ValueError(f"Dati incompleti per BLOCK={SELECTED_BLOCK}.")

    x = np.arange(len(K_VALUES), dtype=float)
    fig, (ax_perf, ax_tile) = plt.subplots(1, 2, figsize=(15.0, 6.6))
    fig.subplots_adjust(left=0.07, right=0.975, bottom=0.18, top=0.75, wspace=0.30)
    fig.suptitle(
        f"CUDA warp shared-memory: dettaglio a BLOCK={SELECTED_BLOCK}",
        y=0.965,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.885,
        (
            f"M={metadata['M']}, N={metadata['N']}, P={metadata['P']}, "
            f"precisione={metadata['scalar']}, SMEM_PAD={SMEM_PAD} · "
            f"{metadata['reps']} ripetizioni"
        ),
        ha="center",
        va="center",
        fontsize=10,
        color="#555555",
    )

    handles = []
    labels = []
    for granularity in GRANULARITIES:
        subset = detail[detail["tile_granularity"] == granularity].set_index("k")
        subset = subset.reindex(K_VALUES)
        color = SERIES_COLORS[granularity]
        linewidth = 3.0 if granularity == SELECTED_GRANULARITY else 1.9
        marker_size = 7.5 if granularity == SELECTED_GRANULARITY else 6.0
        label = f"g={granularity}"

        line = ax_perf.plot(
            x,
            subset["gflops_kernel"],
            marker="o",
            markersize=marker_size,
            linewidth=linewidth,
            color=color,
            label=label,
        )[0]
        ax_tile.plot(
            x,
            subset["x_rows_per_tile"],
            marker="o",
            markersize=marker_size,
            linewidth=linewidth,
            color=color,
            label=label,
        )
        handles.append(line)
        labels.append(label)

    ax_perf.set_ylim(bottom=0.0)
    ax_perf.set_xticks(x)
    ax_perf.set_xticklabels([str(k) for k in K_VALUES])
    ax_perf.set_xlabel("Ampiezza del multivettore k")
    ax_perf.set_ylabel("Throughput del kernel [GFLOP/s]")
    ax_perf.set_title("Effetto della granularità sulle prestazioni")
    ax_perf.grid(True, linestyle="--", linewidth=0.7, alpha=0.55, color=COLOR_GRID)

    ax_tile.set_ylim(bottom=0.0)
    ax_tile.set_xticks(x)
    ax_tile.set_xticklabels([str(k) for k in K_VALUES])
    ax_tile.set_xlabel("Ampiezza del multivettore k")
    ax_tile.set_ylabel("Righe di X per tile")
    ax_tile.set_title("Tile effettivamente scelto dal pianificatore")
    ax_tile.grid(True, linestyle="--", linewidth=0.7, alpha=0.55, color=COLOR_GRID)

    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.825),
        ncol=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.055,
        (
            "g è la granularità di arrotondamento, non la dimensione del tile: "
            "x_rows_per_tile viene scelto a runtime in funzione di k e dell'occupancy."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base = OUTPUT_DIR / "campagna_VAS_5_block256_detail"
    png_path = base.with_suffix(".png")
    pdf_path = base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path

def main() -> None:
    df, failures, metadata = load_results()
    _, summary = build_summary(df)
    chosen_block, chosen_granularity = choose_configuration(summary)
    if (chosen_block, chosen_granularity) != (
        SELECTED_BLOCK,
        SELECTED_GRANULARITY,
    ):
        raise ValueError(
            "La selezione calcolata non coincide con quella attesa: "
            f"BLOCK={chosen_block}, g={chosen_granularity}."
        )

    selected = summary.loc[(SELECTED_GRANULARITY, SELECTED_BLOCK)]
    print("Tuning congiunto CUDA warp shared-memory")
    print(f"Configurazioni riuscite: {len(df)}")
    print(f"Configurazioni fallite: {len(failures)}")
    print(
        f"Selezione: BLOCK={SELECTED_BLOCK}, g={SELECTED_GRANULARITY}, "
        f"media={selected['mean_R_pct']:.3f}%, "
        f"peggior caso={selected['min_R_pct']:.3f}%, "
        f"CV massimo={selected['max_cv_pct']:.3f}%"
    )

    summary_png, summary_pdf = plot_joint_summary(summary, metadata)
    detail_png, detail_pdf = plot_selected_block_detail(df, metadata)
    print(f"Grafico principale PNG: {summary_png}")
    print(f"Grafico principale PDF: {summary_pdf}")
    print(f"Grafico di dettaglio PNG: {detail_png}")
    print(f"Grafico di dettaglio PDF: {detail_pdf}")

if __name__ == "__main__":
    main()
