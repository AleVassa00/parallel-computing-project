from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
PHASE_DIR = ROOT / "experiments" / "fase_4_mpi_cpu"
CAMPAIGN_NAME = "campagna_VAS_9_mpi_cpu_scaling"
CAMPAIGN_DIR = PHASE_DIR / CAMPAIGN_NAME

SIZES = [4096, 8192, 16384]
PROCESS_COUNTS = [1, 2, 4, 8]
K_VALUES = "3 6 8 20 32"

KERNELS = {
    "scheme_a": "schema A di riferimento",
    "scheme_a_jblock": "schema A con blocking sulla dimensione j",
    "scheme_a_jblock_rb": "schema A con j-blocking e register blocking",
}


def validate_target() -> None:
    expected_parent = (ROOT / "experiments" / "fase_4_mpi_cpu").resolve()
    target = CAMPAIGN_DIR.resolve()
    if target.parent != expected_parent or target.name != CAMPAIGN_NAME:
        raise RuntimeError(f"Target non sicuro per la rigenerazione: {target}")


def config_text(size: int, process_count: int, kernel: str, description: str) -> str:
    return f"""# Campagna VAS 9 - {description}, S={size}, tutte le griglie di P={process_count}.
name=campagna_VAS_9_s{size}_{kernel}_p{process_count}_all_grids
experiment=k-sweep
outdir=results/fase_4_mpi_cpu/{CAMPAIGN_NAME}/s{size}

kernel={kernel}
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
x_layout=row
check=0
group_timing=0
continue_on_error=1
"""


def size_suite_text(size: int, relative_configs: list[str]) -> str:
    entries = "\n".join(relative_configs)
    return f"""# Campagna VAS 9 - strong scaling MPI-CPU, matrici quadrate S={size}.
# M=N={size}, FP64, k={K_VALUES}.
# Tre backend CPU e tutte le fattorizzazioni ordinate fino a P=8:
# P=1: 1x1; P=2: 1x2, 2x1;
# P=4: 1x4, 2x2, 4x1; P=8: 1x8, 2x4, 4x2, 8x1.
# Input globali distribuiti con MPI (a_mode=global, x_mode=global).
# Sono 12 file di configurazione e 150 casi aggregati (kernel x k x griglia).

{entries}
"""


def master_suite_text(relative_configs: list[str]) -> str:
    entries = "\n".join(relative_configs)
    return f"""# Campagna VAS 9 - scaling MPI puro su CPU.
#
# Tre esperimenti di strong scaling, uno per ciascuna matrice quadrata:
# M=N=4096, 8192, 16384. Per ogni taglia si provano i tre backend CPU,
# k={K_VALUES}, P=1,2,4,8 e tutte le fattorizzazioni ordinate Pr x Pc.
#
# Le matrici globali vengono generate sul rank 0 e distribuite ai blocchi
# locali (a_mode=global, x_mode=global). Generazione e distribuzione sono
# riportate come preprocessing separato e non entrano nel tempo ufficiale.
#
# P=1 fornisce la baseline della stessa taglia per calcolare:
#   speedup S(P)=T(1)/T(P) ed efficienza E(P)=S(P)/P.
#
# In totale: 36 file di configurazione e 450 casi aggregati.
# JBLOCK_BYTES=65536 e RB_ROWS automatico restano ai default del codice.
#
# Esecuzione completa:
#   ./script/run_cuda_experiments.sh --suite experiments/fase_4_mpi_cpu/{CAMPAIGN_NAME}.txt
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

        for kernel, description in KERNELS.items():
            for process_count in PROCESS_COUNTS:
                filename = f"{kernel}_p{process_count}.conf"
                path = size_dir / filename
                path.write_text(
                    config_text(size, process_count, kernel, description),
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
