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
    return f"""
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
    return f"""

{entries}
"""

def master_suite_text(relative_configs: list[str]) -> str:
    entries = "\n".join(relative_configs)
    return f"""

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
