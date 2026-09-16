from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "experiments" / "fase_5_mpi_cuda"
CAMPAIGN_NAME = "campagna_VAS_10_mpi_cuda_hybrid"
CAMPAIGN_DIR = PHASE_DIR / CAMPAIGN_NAME

SIZES = [4096, 8192, 16384]
PROCESS_COUNTS = [1, 2, 4, 8]
K_VALUES = "3 6 8 20 32"

KERNELS = {
    "cuda_naive": {
        "kernel": "cuda_naive",
        "block": 128,
        "extra": "",
        "description": "CUDA naive",
    },
    "cuda_warp_column": {
        "kernel": "cuda_warp",
        "block": 128,
        "extra": "x_layout=column\n",
        "description": "CUDA warp column-major",
    },
    "cuda_warp_smem": {
        "kernel": "cuda_warp_smem",
        "block": 256,
        "extra": "smem_pad=1\ntile_granularity=32\nx_layout=row\n",
        "description": "CUDA warp shared-memory",
    },
    "cublas": {
        "kernel": "cublas",
        "block": 256,
        "extra": "",
        "description": "cuBLAS",
    },
}


def validate_target() -> None:
    expected_parent = (ROOT / "experiments" / "fase_5_mpi_cuda").resolve()
    target = CAMPAIGN_DIR.resolve()
    if target.parent != expected_parent or target.name != CAMPAIGN_NAME:
        raise RuntimeError(f"Target non sicuro per la rigenerazione: {target}")


def config_text(size: int, process_count: int, key: str, spec: dict) -> str:
    return f"""# Campagna VAS 10 - {spec['description']}, S={size}, tutte le griglie di P={process_count}.
name=campagna_VAS_10_s{size}_{key}_p{process_count}_all_grids
experiment=k-sweep
outdir=results/fase_5_mpi_cuda/{CAMPAIGN_NAME}/s{size}

kernel={spec['kernel']}
block={spec['block']}
{spec['extra']}
M={size}
N={size}
ks={K_VALUES}

prec=double
reps=20
warmup=5
seed=20252026

all_grids={process_count}

bind_to=core
map_by=core
a_mode=global
x_mode=global
check=0
group_timing=1
continue_on_error=1
"""


def size_suite_text(size: int, relative_configs: list[str]) -> str:
    factorisations = (
        "P=1: 1x1; P=2: 1x2, 2x1; "
        "P=4: 1x4, 2x2, 4x1; P=8: 1x8, 2x4, 4x2, 8x1"
    )
    entries = "\n".join(relative_configs)
    return f"""# Campagna VAS 10 - MPI + CUDA, matrici quadrate S={size}.
# M=N={size}, FP64, k={K_VALUES}.
# Tutti i backend CUDA e tutte le fattorizzazioni ordinate fino a P=8.
# {factorisations}.
# Input globali distribuiti con MPI (a_mode=global, x_mode=global).
# Il cronometro di gruppo sincronizza i rank immediatamente prima e dopo
# local_gemm e produce t_local_group_* e gflops_local_group.

{entries}
"""


def master_suite_text(relative_configs: list[str]) -> str:
    entries = "\n".join(relative_configs)
    return f"""# Campagna VAS 10 - parallelismo ibrido MPI + CUDA su una singola GPU.
#
# Tre sottocampagne, una per ciascuna taglia quadrata della Fase 2:
# M=N=4096, 8192, 16384. Per ogni taglia si provano tutti i backend CUDA,
# k={K_VALUES}, P=1,2,4,8 e tutte le fattorizzazioni ordinate Pr x Pc.
#
# Le matrici globali vengono generate sul root e distribuite ai rank.
# Tutti i rank usano il device CUDA 0: le run multi-rank misurano anche la
# contesa fra contesti sulla singola GPU del server.
# group_timing=1 aggiunge due MPI_Barrier attorno a local_gemm per misurare
# la finestra comune dal lancio coordinato al completamento dell'ultimo rank.
#
# Esecuzione completa:
#   ./script/run_cuda_experiments.sh --suite experiments/fase_5_mpi_cuda/{CAMPAIGN_NAME}.txt
#
# Le tre suite per taglia possono essere eseguite separatamente.

{entries}
"""


def main() -> None:
    validate_target()
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    if CAMPAIGN_DIR.exists():
        shutil.rmtree(CAMPAIGN_DIR)
    CAMPAIGN_DIR.mkdir(parents=True)

    master_entries: list[str] = []
    for size in SIZES:
        size_dir = CAMPAIGN_DIR / f"s{size}"
        size_dir.mkdir()
        size_entries: list[str] = []

        for key, spec in KERNELS.items():
            for process_count in PROCESS_COUNTS:
                filename = f"{key}_p{process_count}.conf"
                path = size_dir / filename
                path.write_text(
                    config_text(size, process_count, key, spec),
                    encoding="utf-8",
                    newline="\n",
                )
                relative = f"{CAMPAIGN_NAME}/s{size}/{filename}"
                size_entries.append(relative)
                master_entries.append(relative)

        suite_path = PHASE_DIR / f"{CAMPAIGN_NAME}_s{size}.txt"
        suite_path.write_text(
            size_suite_text(size, size_entries), encoding="utf-8", newline="\n"
        )

    master_path = PHASE_DIR / f"{CAMPAIGN_NAME}.txt"
    master_path.write_text(
        master_suite_text(master_entries), encoding="utf-8", newline="\n"
    )

    print(f"Generate {len(SIZES) * len(KERNELS) * len(PROCESS_COUNTS)} configurazioni")
    print(f"Cartella: {CAMPAIGN_DIR}")
    print(f"Suite completa: {master_path}")


if __name__ == "__main__":
    main()
