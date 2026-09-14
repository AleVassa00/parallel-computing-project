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

# passo -> rapporto -> (M, N); M*N uguale entro il 5% a parita' di passo
SIZES = {
    "s1": {"1x1": (3200, 3200),   "3x1": (5400, 1800),   "1x2": (2250, 4500)},
    "s2": {"1x1": (6400, 6400),   "3x1": (10800, 3600),  "1x2": (4500, 9000)},
    "s3": {"1x1": (12800, 12800), "3x1": (21600, 7200),  "1x2": (9000, 18000)},
    "s4": {"1x1": (25600, 25600), "3x1": (43200, 14400), "1x2": (18000, 36000)},
}
RATIO_LABEL = {"1x1": "1:1 (M=N)", "3x1": "3:1 (M=3N)", "1x2": "1:2 (N=2M)"}
STEP_ORDER = ["1x1", "3x1", "1x2"]

CPU_KERNELS = "scheme_a scheme_a_jblock scheme_a_jblock_rb"
GPU_KERNELS = "cuda_naive cuda_warp cuda_warp_smem cuda_warp_tiled cublas"

HEADER = """\
# C1 - campagna obbligatoria: taglie x rapporto M:N x k x kernel
#     variante: {variant_desc}
#     rapporto {ratio}, taglia {step}, M*N = {mn:.2e}
#
# La traccia chiede: misure ripetute per M ed N crescenti, su almeno 3
# rapporti fra M ed N (con M = N sempre incluso), e il collaudo su
# k = 3, 6, 8, 20, 32. Questa campagna e' il prodotto completo di quei tre
# assi, per ogni kernel: ogni punto (taglia, rapporto) esiste per tutti e
# cinque i k richiesti e per tutti i backend della stessa famiglia.
#
# I quattro passi s1..s4 crescono di un fattore 4 in M*N, e i tre rapporti
# hanno M*N uguale entro il 5% a parita' di passo: sovrapponendo le curve
# sullo stesso asse si legge l'effetto della FORMA a parita' di lavoro.
#
# experiment=compare esegue tutti i kernel di `kernels=` su tutti i `ks=` e
# scrive un solo CSV (name=.csv) con una riga per (kernel, k). I k NON
# specializzati (1, 7, 17, 40, 64) restano in C2: rispondono a un'altra
# domanda e non hanno bisogno di tutte le taglie.
{variant_notes}
name={name}
experiment=compare
outdir={outdir}

kernels={kernels}
M={M}
N={N}

# I cinque k imposti dalla traccia, tutti su ogni punto della griglia.
ks=3 6 8 20 32
{gpu_keys}
# reps=10: sono 12 punti x 5 k x {nk} kernel per variante (x 6 griglie con
# all_grids). La colonna
# t_official_cv_pct dice riga per riga se la media e' stabile.
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
#
# Tre backend di CPU: scheme_a e' la consegna, scheme_a_jblock e
# scheme_a_jblock_rb sono le due ottimizzazioni di cache e di pipeline (vedi
# README). A np=1 non c'e' Bcast ne' Reduce: la colonna da leggere e' gflops
# e misura il nucleo di calcolo da solo. E' il denominatore di ogni speedup.
#
# A s4 la sola A occupa ~5 GB in double sul rank unico: e' il primo punto
# che puo' fallire per memoria, per questo continue_on_error=1.""",
    ),
    "gpu": dict(
        variant_desc="GPU, un solo rank (una GPU = un rank, vedi sotto)",
        outdir="results/c1_size_ratio_gpu",
        kernels=GPU_KERNELS,
        grid_keys="np=1\npr=1\npc=1\n",
        gpu_keys="""\
block=256
smem_pad=1
warp_col_tile=8
""",
        mpi_keys="",
        variant_notes="""\
#
# Cinque backend CUDA: cuda_naive (baseline), cuda_warp, cuda_warp_smem,
# cuda_warp_tiled (candidati) e cublas (riferimento esterno, misurato con
# la stessa pipeline e lo stesso cronometro). La colonna da leggere e'
# gflops_kernel; t_h2d_X / t_d2h_Y restano a parte come chiede la traccia.
#
# Perche' NON esiste una variante multi-rank per la GPU: il server ha una
# sola Quadro RTX 5000. Con np>1 tutti i rank usano il device 0 e i contesti
# CUDA si alternano sulla scheda senza MPS: t_kernel includerebbe il tempo
# in cui il contesto e' sospeso a favore di un altro rank, e la misura
# sarebbe contesa fra contesti, non prestazione del kernel. Su un nodo con
# una GPU la configurazione ottima e' quindi np=1 (README, "Backend CUDA").
#
# A s4 la sola A occupa ~5 GB in double: entra nella VRAM da 15.5 GiB, ma e'
# il primo punto che puo' fallire, per questo continue_on_error=1.""",
    ),
    "mpi20": dict(
        variant_desc="CPU, 20 rank MPI, tutte le forme di griglia (configurazione di produzione)",
        outdir="results/c1_size_ratio_mpi20",
        kernels=CPU_KERNELS,
        gpu_keys="",
        mpi_keys="""\

# Sul server il binding e' obbligatorio: senza, le misure sono rumore.
bind_to=core
map_by=core
""",
        grid_keys="""\
# Tutte le fattorizzazioni ordinate di 20: 1x20, 2x10, 4x5, 5x4, 10x2, 20x1.
# 4x5 e' la forma di default (grid_default_shape), quindi la vista "griglia
# fissa" e' un sottoinsieme di queste righe. Il CSV riporta pr e pc.
all_grids=20
""",
        variant_notes="""\
#
# Stessa griglia di punti della variante cpu, ma sul codice MPI completo:
# Bcast di X lungo col_comm + local_gemm + Reduce di Y lungo row_comm, con
# la stessa metrica della traccia (gflops = 2*M*N*k / t_official). E' la
# tabella di prestazioni "vera" del nucleo parallelo, quella che la
# variante cpu a np=1 isola ma non rappresenta.
#
# Perche' P = 20 e non un altro valore. E' il numero di core fisici del
# nodo (doppio socket) ed e' il massimo usato da C4/C5/C6: e' cio' che un
# utente farebbe con questa macchina. P viene tenuto FISSO lungo tutta la
# griglia di taglie di proposito: se si scegliesse il P migliore per ogni
# taglia si confonderebbe l'effetto della taglia con quello del
# parallelismo. Alle taglie piccole (s1: blocco locale 800x640, 4 MB) la
# comunicazione pesa piu' del calcolo e l'efficienza parallela e' bassa:
# e' un risultato, non un difetto della campagna, ed e' il motivo per cui
# la stessa griglia esiste anche a np=1. Se P=20 sia anche il P PIU' VELOCE
# a problema fisso lo dice C4 (strong scaling), non questa campagna.
#
# Perche' TUTTE le forme di griglia e non solo 4x5. Il volume comunicato
# dipende da griglia e rapporto insieme: Bcast ~ (N/pc)*k, Reduce ~ (M/pr)*k.
# La forma migliore deve quindi spostarsi col rapporto: verso piu' righe
# (10x2, 5x4) per M=3N, verso piu' colonne (2x10, 4x5) per N=2M. C6 misura
# le forme solo a M=N e non puo' vederlo; qui la predizione e' verificabile
# punto per punto. Per la tabella "obbligatoria" si legge la riga 4x5 (il
# default del programma) oppure la migliore per ogni punto; le altre forme
# servono al grafico che conferma o smentisce il modello. Costa 6x le righe
# della suite (90 per conf), ma nulla va perso: 4x5 e' una delle sei.
#
# Con np=20 e a_mode=local ogni rank genera il proprio blocco: a s4 sono
# ~260 MB di A per rank, ben oltre la cache, quindi si misura il kernel e
# la comunicazione, non la L3.""",
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
# C1 - campagna obbligatoria, variante CPU a np=1
#     taglie s1..s4 x rapporti 1:1, 3:1, 1:2 x k = 3 6 8 20 32
#     kernel: scheme_a, scheme_a_jblock, scheme_a_jblock_rb
#
#   ./script/run_cuda_experiments.sh --suite experiments/c1_size_ratio_cpu.txt
#
# Non richiede nvcc. Le taglie sono in ordine crescente: se il punto piu'
# grande non entra in memoria, i punti gia' raccolti restano validi.
""",
    "gpu": """\
# C1 - campagna obbligatoria, variante GPU a np=1
#     taglie s1..s4 x rapporti 1:1, 3:1, 1:2 x k = 3 6 8 20 32
#     kernel: cuda_naive, cuda_warp, cuda_warp_smem, cuda_warp_tiled, cublas
#
#   ./script/run_cuda_experiments.sh --suite experiments/c1_size_ratio_gpu.txt
#
# Richiede nvcc. Le taglie sono in ordine crescente: se il punto piu' grande
# non entra in VRAM, i punti gia' raccolti restano validi.
""",
    "mpi20": """\
# C1 - campagna obbligatoria, variante MPI a P=20, tutte le forme di griglia
#     taglie s1..s4 x rapporti 1:1, 3:1, 1:2 x k = 3 6 8 20 32
#     x griglie 1x20 2x10 4x5 5x4 10x2 20x1 (all_grids=20)
#     kernel: scheme_a, scheme_a_jblock, scheme_a_jblock_rb
#
#   ./script/run_cuda_experiments.sh --suite experiments/c1_size_ratio_mpi20.txt
#
# Non richiede nvcc. Richiede un nodo con 20 core liberi e il binding
# (bind_to=core, map_by=core sono nei .conf). Se il trasporto MPI fallisce
# con un errore di BTL, vedere CAMPAGNE.md ("Trasporto MPI").
""",
}


def main():
    os.makedirs(OUT, exist_ok=True)
    # rimuove i vecchi .conf di C1 (stessa cartella, stesso schema di nomi)
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
