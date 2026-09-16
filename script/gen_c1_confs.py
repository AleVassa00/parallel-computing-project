#!/usr/bin/env python3
"""Genera i .conf e le suite della campagna C1.

    python3 script/gen_c1_confs.py

Produce experiments/c1_size_ratio/{cpu,gpu,mpi20}_{1x1,3x1,1x2}_{s1..s4}.conf
e le tre suite experiments/c1_size_ratio_{cpu,gpu,mpi20}.txt. I .conf sono
generati: per cambiare taglie, kernel, k o griglia si modifica QUESTO file e
si rilancia, non i 36 .conf a mano. Cosa misura la campagna e perche' e'
fatta cosi' e' in experiments/CAMPAGNE.md, sezione C1.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "experiments", "c1_size_ratio")

SIZES = {
    "s1": {"1x1": (3200, 3200),   "3x1": (5400, 1800),   "1x2": (2250, 4500)},
    "s2": {"1x1": (6400, 6400),   "3x1": (10800, 3600),  "1x2": (4500, 9000)},
    "s3": {"1x1": (12800, 12800), "3x1": (21600, 7200),  "1x2": (9000, 18000)},
    "s4": {"1x1": (25600, 25600), "3x1": (43200, 14400), "1x2": (18000, 36000)},
}
RATIO_LABEL = {"1x1": "1:1 (M=N)", "3x1": "3:1 (M=3N)", "1x2": "1:2 (N=2M)"}
STEP_ORDER = ["1x1", "3x1", "1x2"]

CPU_KERNELS = "scheme_a scheme_a_jblock scheme_a_jblock_rb"
GPU_KERNELS = "cuda_naive cuda_warp cuda_warp_smem cublas"

HEADER = """\
{variant_notes}
name={name}
experiment=compare
outdir={outdir}

kernels={kernels}
M={M}
N={N}

ks=3 6 8 20 32
{gpu_keys}
reps=10
warmup=3
seed=20252026

{grid_keys}{mpi_keys}
a_mode=local
x_mode=local
prec=double

continue_on_error=1
"""

VARIANTS = {
    "cpu": dict(
        variant_desc="CPU, un solo rank (kernel isolato, senza comunicazione)",
        outdir="results/c1_size_ratio_cpu",
        kernels=CPU_KERNELS,
        grid_keys="np=1\npr=1\npc=1\n",
        gpu_keys="",
        mpi_keys="",
        variant_notes="""\
""",
    ),
    "gpu": dict(
        variant_desc="GPU, un solo rank (una GPU = un rank, vedi sotto)",
        outdir="results/c1_size_ratio_gpu",
        kernels=GPU_KERNELS,
        grid_keys="np=1\npr=1\npc=1\n",
        gpu_keys="""\
block=256
smem_pad=1
""",
        mpi_keys="",
        variant_notes="""\
""",
    ),
    "mpi20": dict(
        variant_desc="CPU, 20 rank MPI, tutte le forme di griglia (configurazione di produzione)",
        outdir="results/c1_size_ratio_mpi20",
        kernels=CPU_KERNELS,
        gpu_keys="",
        mpi_keys="""\

bind_to=core
map_by=core
""",
        grid_keys="""\
all_grids=20
""",
        variant_notes="""\
""",
    ),
}

def mn_of(M, N):
    return M * N

def write_conf(variant, ratio, step):
    v = VARIANTS[variant]
    M, N = SIZES[step][ratio]
    name = f"c1_{variant}_{ratio}_{step}"
    text = HEADER.format(
        variant_desc=v["variant_desc"],
        ratio=RATIO_LABEL[ratio], step=step, mn=mn_of(M, N),
        variant_notes=v["variant_notes"],
        name=name, outdir=v["outdir"], kernels=v["kernels"], M=M, N=N,
        gpu_keys=v["gpu_keys"], nk=len(v["kernels"].split()),
        grid_keys=v["grid_keys"], mpi_keys=v["mpi_keys"],
    )
    path = os.path.join(OUT, f"{variant}_{ratio}_{step}.conf")
    with open(path, "w") as f:
        f.write(text)
    return f"c1_size_ratio/{variant}_{ratio}_{step}.conf"

SUITE_HEADER = {
    "cpu": """\
""",
    "gpu": """\
""",
    "mpi20": """\
""",
}

def main():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        if f.endswith(".conf"):
            os.remove(os.path.join(OUT, f))

    for variant in ("cpu", "gpu", "mpi20"):
        entries = []
        for step in ("s1", "s2", "s3", "s4"):
            for ratio in STEP_ORDER:
                entries.append(write_conf(variant, ratio, step))
        suite = os.path.join(ROOT, "experiments", f"c1_size_ratio_{variant}.txt")
        with open(suite, "w") as f:
            f.write(SUITE_HEADER[variant])
            f.write("\n".join(entries) + "\n")
        print(f"{suite}: {len(entries)} conf")

if __name__ == "__main__":
    main()
