#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CUDA / MPI experiment runner
# parallel-computing-project
# ============================================================
#
# Tutto si configura tramite flag: non ci sono argomenti posizionali.
#
# Esempi:
#
#   ./run_cuda_experiments.sh --experiment k-sweep
#
#   ./run_cuda_experiments.sh \
#       --experiment k-sweep \
#       --kernel cuda_naive \
#       --ks "3 6 8 20 32" \
#       --block 256
#
#   ./run_cuda_experiments.sh \
#       --experiment grid-sweep \
#       --kernel cuda_naive \
#       --k 32 \
#       --block 256 \
#       --all-grids 8
#
#   # --all-grids 8 genera:
#   #   1x8, 2x4, 4x2, 8x1
#
#   ./run_cuda_experiments.sh \
#       --experiment full \
#       --all-grids 4
#
# ============================================================

ORIGINAL_ARGS=("$@")

# -------------------------
# Default
# -------------------------

EXPERIMENT=""
EXPERIMENT_NAME=""
CONFIG_FILE=""
SUITE_FILE=""
OUTDIR_EXPLICIT=0

M=10000
N=10000
REPS=20
WARMUP=5
SEED=""

A_MODE="local"
X_MODE="local"
X_LAYOUT="row"
CHECK=0

NP=1
PR=1
PC=1
NP_EXPLICIT=0
PR_EXPLICIT=0
PC_EXPLICIT=0
ALL_GRIDS_P=""

KERNEL="cuda_naive"
KERNEL_EXPLICIT=0
KERNELS=(cuda_naive cuda_warp cuda_warp_smem)

K_SWEEP_KS=(3 6 8 20 32)
BLOCK_SWEEP_KS=(3 32)
COMPARE_KS=(3 6 8 20 32)

SINGLE_K=""
KS_EXPLICIT=0

BLOCK=256
BLOCK_EXPLICIT=0
BLOCKS=(64 128 192 256 384 512 1024)

PREC="double"
SMEM_PAD=1
SMEM_PADS=(0 1)
WARP_COL_TILE=8
WARP_COL_TILES=(4 8 16 32)
FORCE_GENERIC_K=0
TEST_A_PADDING=0
NVCC_ARCH="sm_75"

NCU_SET="full"

VERBOSE=0
CONTINUE_ON_ERROR=0

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTDIR="results/cuda_${TIMESTAMP}"

BTL="self,sm"
BIND_TO="core"
MAP_BY="core"
REPORT_BINDINGS=0

MPI_FLAGS=()

usage() {
cat <<'EOF'
Uso:
  ./run_cuda_experiments.sh --experiment <tipo> [flag...]
  ./run_cuda_experiments.sh --config <file.conf> [flag di override...]
  ./run_cuda_experiments.sh --suite <suite.txt>

CONFIGURAZIONE DA FILE
  --config FILE
      Legge un esperimento da un file key=value.
      I flag passati sulla command line hanno precedenza sul file.

  --suite FILE
      Il file contiene, una per riga, le configurazioni .conf da eseguire.
      Le righe vuote e quelle che iniziano con # vengono ignorate.

  --name NAME
      Nome logico dell'esperimento. I risultati vengono salvati in:
          results/NAME/<timestamp>/

TIPI DI ESPERIMENTO
  --experiment k-sweep
      Varia k.

  --experiment block-sweep
      Varia BLOCK. Per default usa k=3 e k=32.

  --experiment grid-sweep
      Varia la forma della griglia MPI.
      Richiede --all-grids P.

  --experiment compare
      Confronta i kernel indicati da --kernels.

  --experiment registers
      Compila con ptxas -v e salva registri, spill, stack e barrier.

  --experiment smem-pad-sweep
      Confronta i valori di padding della shared memory.
      Default kernel: cuda_warp_smem.

  --experiment warp-tile-sweep
      Varia WARP_COL_TILE per cuda_warp_tiled sui k richiesti.

  --experiment ncu
      Esegue Nsight Compute.

  --experiment full
      Campagna completa:
        - k-sweep
        - block-sweep su TUTTI i k
        - confronto kernel su TUTTI i k
        - smem-pad-sweep
        - registri ptxas
        - un tentativo Nsight Compute su cuda_naive k=32
      Tutti gli esperimenti benchmark rispettano anche --all-grids P.

FLAG DEL PROBLEMA
  --M N                       Righe di A                  [default 10000]
  --N N                       Colonne di A               [default 10000]
  --k K                       Singolo k
  --ks "3 6 8 20 32"         Lista di k
  --reps N                    Repetition misurate        [default 20]
  --warmup N                  Warmup                     [default 5]
  --seed N                    Seed esplicito
  --check                     Abilita validazione seriale
  --a-mode local|global       Modalita' A                [default local]
  --x-mode local|global       Modalita' X                [default local]
  --x-layout row|column       Layout di X (column solo cuda_warp) [default row]
                              Conversione una volta nel preprocessing.
                              Column aggiunge _xcolumn ai nomi dei risultati.

FLAG MPI / GRIGLIA
  --np P                      Numero processi MPI        [default 1]
  --pr PR                     Righe process grid         [default 1]
  --pc PC                     Colonne process grid       [default 1]

  --all-grids P
      Usa P processi e prova TUTTE le fattorizzazioni ordinate PR x PC.
      Esempio:
          --all-grids 8
      produce:
          1x8, 2x4, 4x2, 8x1

      Questo flag puo' essere combinato con k-sweep, block-sweep,
      compare, smem-pad-sweep, warp-tile-sweep e full.

  --btl VALUE                  OpenMPI BTL               [default self,sm]
  --bind-to VALUE              OpenMPI --bind-to         [default core]
  --map-by VALUE               OpenMPI --map-by          [default core]
  --report-bindings            Aggiunge --report-bindings
  --no-binding                 Disabilita bind-to/map-by

FLAG CUDA / BUILD
  --kernel NAME               Singolo kernel             [default cuda_naive]
  --kernels "..."             Lista kernel per compare/registers
                              [default cuda_naive cuda_warp cuda_warp_smem]

  --block N                   Singolo BLOCK              [default 256]
  --blocks "..."              Lista BLOCK
                              [default 64 128 192 256 384 512 1024]

  --prec double|float         Precisione                 [default double]
  --smem-pad N                Padding shared             [default 1]
  --smem-pads "0 1"           Lista padding per smem-pad-sweep
  --warp-col-tile N           Colonne per warp (cuda_warp_tiled) [default 8]
  --warp-col-tiles "4 8 16 32" Lista tile per warp-tile-sweep
  --force-generic-k 0|1       FORCE_GENERIC_K            [default 0]
  --test-a-padding N          TEST_A_PADDING             [default 0]
  --nvcc-arch ARCH            Architettura NVCC          [default sm_75]

OUTPUT DEI BENCHMARK
  Per un normale esperimento con:
      name=naive_block_sweep

  vengono prodotti:
      naive_block_sweep.csv
      naive_block_sweep_raw.csv
      metadata.txt

  Il CSV principale contiene una riga aggregata per configurazione.
  Il CSV _raw contiene una riga per ogni repetition misurata.
  Le warmup non vengono salvate nel raw.

FLAG PROFILING / OUTPUT
  --ncu-set SET               Set Nsight Compute         [default full]
  --outdir DIR                Directory risultati (override del path automatico)
  --name NAME                    Nome esperimento / cartella risultati
  --config FILE                  Carica parametri da file
  --suite FILE                   Esegue una lista di file .conf
  --verbose                   Mostra le righe CSV anche a terminale
  --continue-on-error         Continua una campagna se una run fallisce
  --help

ESEMPI

  # Da file di configurazione
  ./run_cuda_experiments.sh --config experiments/naive_block_sweep.conf

  # Esegue piu' file di configurazione in sequenza
  ./run_cuda_experiments.sh --suite experiments/cuda_naive_suite.txt

  # Naive, tutti i k standard
  ./run_cuda_experiments.sh \
      --experiment k-sweep \
      --kernel cuda_naive

  # Naive, solo k=32 e BLOCK=128
  ./run_cuda_experiments.sh \
      --experiment k-sweep \
      --kernel cuda_naive \
      --k 32 \
      --block 128

  # Tutte le forme di griglia con P=8, k=32
  ./run_cuda_experiments.sh \
      --experiment grid-sweep \
      --kernel cuda_naive \
      --k 32 \
      --all-grids 8

  # k-sweep per tutte le griglie di 4 processi
  ./run_cuda_experiments.sh \
      --experiment k-sweep \
      --kernel cuda_naive \
      --all-grids 4

  # BLOCK x k x tutte le griglie di 8 processi
  ./run_cuda_experiments.sh \
      --experiment block-sweep \
      --ks "3 32" \
      --all-grids 8

  # Campagna totale su tutte le griglie di 4 processi
  ./run_cuda_experiments.sh \
      --experiment full \
      --all-grids 4
EOF
}


trim() {
    local x="$1"
    x="${x#"${x%%[![:space:]]*}"}"
    x="${x%"${x##*[![:space:]]}"}"
    printf '%s' "$x"
}

apply_config_kv() {
    local key="$1"
    local value="$2"

    case "$key" in
        name) EXPERIMENT_NAME="$value" ;;
        experiment) EXPERIMENT="$value" ;;

        M) M="$value" ;;
        N) N="$value" ;;
        k)
            SINGLE_K="$value"
            KS_EXPLICIT=1
            ;;
        ks)
            read -r -a K_SWEEP_KS <<< "$value"
            read -r -a BLOCK_SWEEP_KS <<< "$value"
            read -r -a COMPARE_KS <<< "$value"
            KS_EXPLICIT=1
            ;;
        reps) REPS="$value" ;;
        warmup) WARMUP="$value" ;;
        seed) SEED="$value" ;;
        check)
            [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" ]] && CHECK=1 || CHECK=0
            ;;
        a_mode|a-mode) A_MODE="$value" ;;
        x_mode|x-mode) X_MODE="$value" ;;
        x_layout|x-layout) X_LAYOUT="$value" ;;

        np)
            NP="$value"
            NP_EXPLICIT=1
            ;;
        pr)
            PR="$value"
            PR_EXPLICIT=1
            ;;
        pc)
            PC="$value"
            PC_EXPLICIT=1
            ;;
        all_grids|all-grids) ALL_GRIDS_P="$value" ;;

        btl) BTL="$value" ;;
        bind_to|bind-to) BIND_TO="$value" ;;
        map_by|map-by) MAP_BY="$value" ;;
        report_bindings|report-bindings)
            [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" ]] && REPORT_BINDINGS=1 || REPORT_BINDINGS=0
            ;;

        kernel)
            KERNEL="$value"
            KERNEL_EXPLICIT=1
            ;;
        kernels)
            read -r -a KERNELS <<< "$value"
            ;;
        block)
            BLOCK="$value"
            BLOCK_EXPLICIT=1
            ;;
        blocks)
            read -r -a BLOCKS <<< "$value"
            ;;

        prec) PREC="$value" ;;
        smem_pad|smem-pad) SMEM_PAD="$value" ;;
        smem_pads|smem-pads)
            read -r -a SMEM_PADS <<< "$value"
            ;;
        warp_col_tile|warp-col-tile) WARP_COL_TILE="$value" ;;
        warp_col_tiles|warp-col-tiles)
            read -r -a WARP_COL_TILES <<< "$value"
            ;;
        force_generic_k|force-generic-k) FORCE_GENERIC_K="$value" ;;
        test_a_padding|test-a-padding) TEST_A_PADDING="$value" ;;
        nvcc_arch|nvcc-arch) NVCC_ARCH="$value" ;;

        ncu_set|ncu-set) NCU_SET="$value" ;;
        verbose)
            [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" ]] && VERBOSE=1 || VERBOSE=0
            ;;
        continue_on_error|continue-on-error)
            [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" ]] && CONTINUE_ON_ERROR=1 || CONTINUE_ON_ERROR=0
            ;;
        outdir)
            OUTDIR="$value"
            OUTDIR_EXPLICIT=1
            ;;
        "")
            ;;
        *)
            echo "Errore in config: chiave sconosciuta '$key'." >&2
            exit 1
            ;;
    esac
}

load_config() {
    local file="$1"

    if [[ ! -f "$file" ]]; then
        echo "Errore: config '$file' non trovato." >&2
        exit 1
    fi

    CONFIG_FILE="$file"

    while IFS= read -r raw || [[ -n "$raw" ]]; do
        local line key value

        line="$(trim "$raw")"
        [[ -z "$line" ]] && continue
        [[ "$line" == \#* ]] && continue

        if [[ "$line" != *=* ]]; then
            echo "Errore in $file: riga non valida: $raw" >&2
            echo "Formato atteso: chiave=valore" >&2
            exit 1
        fi

        key="$(trim "${line%%=*}")"
        value="$(trim "${line#*=}")"

        # Supporta opzionalmente virgolette semplici/doppie attorno all'intero valore.
        if [[ ${#value} -ge 2 ]]; then
            if [[ "$value" == \"*\" && "$value" == *\" ]]; then
                value="${value:1:${#value}-2}"
            elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
                value="${value:1:${#value}-2}"
            fi
        fi

        apply_config_kv "$key" "$value"
    done < "$file"
}

run_suite() {
    local suite="$1"

    if [[ ! -f "$suite" ]]; then
        echo "Errore: suite '$suite' non trovata." >&2
        exit 1
    fi

    local suite_dir
    suite_dir="$(cd "$(dirname "$suite")" && pwd)"

    # La lista viene letta dal descrittore 3, non da stdin, e il figlio riceve
    # stdin da /dev/null.
    #
    # Serve entrambe le cose. Il figlio arriva a mpirun, che per progetto
    # INOLTRA stdin al rank 0 e quindi lo legge fino a EOF: con il ciclo
    # attaccato a stdin, mpirun si mangiava il resto del file della suite. Al
    # giro dopo `read` non trovava piu' niente e il ciclo terminava senza
    # errore, quindi una suite da N configurazioni ne eseguiva UNA e
    # annunciava "Esperimento completato". E' lo stesso inciampo di ssh dentro
    # un while read, e non fa rumore: si vede solo contando i banner SUITE.
    while IFS= read -r raw <&3 || [[ -n "$raw" ]]; do
        local line config_path
        line="$(trim "$raw")"
        [[ -z "$line" ]] && continue
        [[ "$line" == \#* ]] && continue

        if [[ "$line" = /* ]]; then
            config_path="$line"
        else
            config_path="$suite_dir/$line"
        fi

        echo
        echo "################################################################"
        echo "SUITE -> $config_path"
        echo "################################################################"

        "$0" --config "$config_path" < /dev/null
    done 3< "$suite"
}

need_value() {
    if [[ $# -lt 2 || -z "${2:-}" ]]; then
        echo "Errore: il flag '$1' richiede un valore." >&2
        exit 1
    fi
}

# -------------------------
# Pre-scan: config / suite
# -------------------------

# I file vengono caricati prima del parsing normale, cosi' i flag CLI
# successivi possono sovrascrivere i valori del file.
for ((i=0; i<${#ORIGINAL_ARGS[@]}; i++)); do
    case "${ORIGINAL_ARGS[$i]}" in
        --config)
            if (( i + 1 >= ${#ORIGINAL_ARGS[@]} )); then
                echo "Errore: --config richiede un file." >&2
                exit 1
            fi
            load_config "${ORIGINAL_ARGS[$((i+1))]}"
            ;;
        --suite)
            if (( i + 1 >= ${#ORIGINAL_ARGS[@]} )); then
                echo "Errore: --suite richiede un file." >&2
                exit 1
            fi
            SUITE_FILE="${ORIGINAL_ARGS[$((i+1))]}"
            ;;
    esac
done

if [[ -n "$SUITE_FILE" ]]; then
    run_suite "$SUITE_FILE"
    exit 0
fi

# -------------------------
# Parse: SOLO flag
# -------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment)
            need_value "$@"
            EXPERIMENT="$2"
            shift 2
            ;;
        --name)
            need_value "$@"
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --config)
            need_value "$@"
            # gia' caricato nel pre-scan
            shift 2
            ;;
        --suite)
            need_value "$@"
            # gia' gestito nel pre-scan
            shift 2
            ;;
        --M)
            need_value "$@"; M="$2"; shift 2 ;;
        --N)
            need_value "$@"; N="$2"; shift 2 ;;
        --k)
            need_value "$@"
            SINGLE_K="$2"
            KS_EXPLICIT=1
            shift 2
            ;;
        --ks)
            need_value "$@"
            read -r -a K_SWEEP_KS <<< "$2"
            read -r -a BLOCK_SWEEP_KS <<< "$2"
            read -r -a COMPARE_KS <<< "$2"
            KS_EXPLICIT=1
            shift 2
            ;;
        --reps)
            need_value "$@"; REPS="$2"; shift 2 ;;
        --warmup)
            need_value "$@"; WARMUP="$2"; shift 2 ;;
        --seed)
            need_value "$@"; SEED="$2"; shift 2 ;;
        --check)
            CHECK=1
            shift
            ;;
        --a-mode)
            need_value "$@"; A_MODE="$2"; shift 2 ;;
        --x-mode)
            need_value "$@"; X_MODE="$2"; shift 2 ;;
        --x-layout)
            need_value "$@"; X_LAYOUT="$2"; shift 2 ;;

        --np)
            need_value "$@"
            NP="$2"
            NP_EXPLICIT=1
            shift 2
            ;;
        --pr)
            need_value "$@"
            PR="$2"
            PR_EXPLICIT=1
            shift 2
            ;;
        --pc)
            need_value "$@"
            PC="$2"
            PC_EXPLICIT=1
            shift 2
            ;;
        --all-grids)
            need_value "$@"
            ALL_GRIDS_P="$2"
            shift 2
            ;;
        --btl)
            need_value "$@"
            BTL="$2"
            shift 2
            ;;
        --bind-to)
            need_value "$@"
            BIND_TO="$2"
            shift 2
            ;;
        --map-by)
            need_value "$@"
            MAP_BY="$2"
            shift 2
            ;;
        --report-bindings)
            REPORT_BINDINGS=1
            shift
            ;;
        --no-binding)
            BIND_TO=""
            MAP_BY=""
            shift
            ;;

        --kernel)
            need_value "$@"
            KERNEL="$2"
            KERNEL_EXPLICIT=1
            shift 2
            ;;
        --kernels)
            need_value "$@"
            read -r -a KERNELS <<< "$2"
            shift 2
            ;;

        --block)
            need_value "$@"
            BLOCK="$2"
            BLOCK_EXPLICIT=1
            shift 2
            ;;
        --blocks)
            need_value "$@"
            read -r -a BLOCKS <<< "$2"
            shift 2
            ;;

        --prec)
            need_value "$@"; PREC="$2"; shift 2 ;;
        --smem-pad)
            need_value "$@"; SMEM_PAD="$2"; shift 2 ;;
        --smem-pads)
            need_value "$@"
            read -r -a SMEM_PADS <<< "$2"
            shift 2
            ;;
        --force-generic-k)
            need_value "$@"; FORCE_GENERIC_K="$2"; shift 2 ;;
        --warp-col-tile)
            need_value "$@"; WARP_COL_TILE="$2"; shift 2 ;;
        --warp-col-tiles)
            need_value "$@"
            read -r -a WARP_COL_TILES <<< "$2"
            shift 2
            ;;
        --test-a-padding)
            need_value "$@"; TEST_A_PADDING="$2"; shift 2 ;;
        --nvcc-arch)
            need_value "$@"; NVCC_ARCH="$2"; shift 2 ;;

        --ncu-set)
            need_value "$@"; NCU_SET="$2"; shift 2 ;;
        --outdir)
            need_value "$@"
            OUTDIR="$2"
            OUTDIR_EXPLICIT=1
            shift 2
            ;;
        --verbose)
            VERBOSE=1
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Errore: argomento non valido '$1'. Usa solo flag." >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$EXPERIMENT" ]]; then
    echo "Errore: manca --experiment <tipo>." >&2
    usage
    exit 1
fi

if [[ -n "$SINGLE_K" ]]; then
    K_SWEEP_KS=("$SINGLE_K")
    BLOCK_SWEEP_KS=("$SINGLE_K")
    COMPARE_KS=("$SINGLE_K")
fi

if [[ "$EXPERIMENT" == "warp-tile-sweep" && "$KERNEL_EXPLICIT" -eq 0 ]]; then
    KERNEL="cuda_warp_tiled"
fi

# -------------------------
# Validazione
# -------------------------

is_pos_int() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

is_nonneg_int() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

validate_block() {
    local b="$1"
    if ! is_pos_int "$b" || (( b % 32 != 0 || b > 1024 )); then
        echo "Errore: BLOCK=$b deve essere multiplo di 32 e <= 1024." >&2
        exit 1
    fi
}

validate_common() {
    if ! is_pos_int "$WARP_COL_TILE"; then
        echo "Errore: WARP_COL_TILE deve essere un intero positivo." >&2
        exit 1
    fi
    if [[ "$EXPERIMENT" == "warp-tile-sweep" ]]; then
        if [[ "$KERNEL" != "cuda_warp_tiled" ]]; then
            echo "Errore: warp-tile-sweep richiede kernel=cuda_warp_tiled." >&2
            exit 1
        fi
        if [[ "${#WARP_COL_TILES[@]}" -eq 0 || "${#K_SWEEP_KS[@]}" -eq 0 ]]; then
            echo "Errore: warp_col_tiles e ks non possono essere vuoti." >&2
            exit 1
        fi
        local tile
        for tile in "${WARP_COL_TILES[@]}"; do
            if ! is_pos_int "$tile"; then
                echo "Errore: WARP_COL_TILE non valido '$tile'." >&2
                exit 1
            fi
        done
    fi
    if ! is_pos_int "$M" || ! is_pos_int "$N"; then
        echo "Errore: M e N devono essere positivi." >&2
        exit 1
    fi
    if ! is_pos_int "$REPS" || ! is_nonneg_int "$WARMUP"; then
        echo "Errore: reps > 0 e warmup >= 0." >&2
        exit 1
    fi

    case "$A_MODE" in local|global) ;; *)
        echo "Errore: --a-mode deve essere local o global." >&2; exit 1 ;;
    esac
    case "$X_MODE" in local|global) ;; *)
        echo "Errore: --x-mode deve essere local o global." >&2; exit 1 ;;
    esac
    case "$X_LAYOUT" in row|column) ;; *)
        echo "Errore: --x-layout deve essere row o column." >&2; exit 1 ;;
    esac
    if [[ "$X_LAYOUT" == "column" ]]; then
        local layout_kernels=("$KERNEL")
        case "$EXPERIMENT" in
            compare) layout_kernels=("${KERNELS[@]}") ;;
            registers)
                if [[ "$KERNEL_EXPLICIT" -eq 0 ]]; then
                    layout_kernels=("${KERNELS[@]}")
                fi ;;
            full|smem-pad-sweep|warp-tile-sweep)
                echo "Errore: --x-layout column non supportato da $EXPERIMENT." >&2
                exit 1 ;;
        esac
        local layout_kernel
        for layout_kernel in "${layout_kernels[@]}"; do
            if [[ "$layout_kernel" != "cuda_warp" ]]; then
                echo "Errore: --x-layout column supportato solo da cuda_warp, non da $layout_kernel." >&2
                exit 1
            fi
        done
    fi
    case "$PREC" in double|float) ;; *)
        echo "Errore: --prec deve essere double o float." >&2; exit 1 ;;
    esac

    for k in "${K_SWEEP_KS[@]}"; do
        if ! is_pos_int "$k"; then
            echo "Errore: k non valido '$k'." >&2
            exit 1
        fi
    done

    for b in "${BLOCKS[@]}"; do
        validate_block "$b"
    done
    validate_block "$BLOCK"

    if [[ "$FORCE_GENERIC_K" != "0" && "$FORCE_GENERIC_K" != "1" ]]; then
        echo "Errore: --force-generic-k accetta 0 o 1." >&2
        exit 1
    fi

    if ! is_nonneg_int "$TEST_A_PADDING"; then
        echo "Errore: --test-a-padding deve essere >= 0." >&2
        exit 1
    fi
}

# -------------------------
# Griglie MPI
# GRIDS: "np:pr:pc"
# -------------------------

GRIDS=()

generate_grids() {
    GRIDS=()

    if [[ -n "$ALL_GRIDS_P" ]]; then
        if ! is_pos_int "$ALL_GRIDS_P"; then
            echo "Errore: --all-grids richiede P > 0." >&2
            exit 1
        fi

        local p="$ALL_GRIDS_P"
        local pr pc

        for ((pr=1; pr<=p; pr++)); do
            if (( p % pr == 0 )); then
                pc=$((p / pr))
                GRIDS+=("${p}:${pr}:${pc}")
            fi
        done
        return
    fi

    if [[ "$PR_EXPLICIT" -eq 1 && "$PC_EXPLICIT" -eq 0 ]] ||
       [[ "$PR_EXPLICIT" -eq 0 && "$PC_EXPLICIT" -eq 1 ]]; then
        echo "Errore: per una griglia fissa specifica sia --pr sia --pc." >&2
        exit 1
    fi

    if [[ "$PR_EXPLICIT" -eq 1 && "$PC_EXPLICIT" -eq 1 ]]; then
        if ! is_pos_int "$PR" || ! is_pos_int "$PC"; then
            echo "Errore: pr e pc devono essere positivi." >&2
            exit 1
        fi

        if [[ "$NP_EXPLICIT" -eq 0 ]]; then
            NP=$((PR * PC))
        fi
    elif [[ "$NP_EXPLICIT" -eq 1 && "$NP" -ne 1 ]]; then
        echo "Errore: con --np $NP specifica anche --pr/--pc oppure usa --all-grids $NP." >&2
        exit 1
    fi

    if ! is_pos_int "$NP" || (( PR * PC != NP )); then
        echo "Errore: griglia non valida: np=$NP, pr=$PR, pc=$PC." >&2
        exit 1
    fi

    GRIDS+=("${NP}:${PR}:${PC}")
}

# -------------------------
# Helpers
# -------------------------

build_mpi_flags() {
    MPI_FLAGS=()

    if [[ -n "$BTL" ]]; then
        MPI_FLAGS+=(--mca btl "$BTL")
    fi
    if [[ -n "$BIND_TO" ]]; then
        MPI_FLAGS+=(--bind-to "$BIND_TO")
    fi
    if [[ -n "$MAP_BY" ]]; then
        MPI_FLAGS+=(--map-by "$MAP_BY")
    fi
    if [[ "$REPORT_BINDINGS" -eq 1 ]]; then
        MPI_FLAGS+=(--report-bindings)
    fi
}

warn_multi_rank_gpu_contention() {
    local has_multi=0
    local g np pr pc

    for g in "${GRIDS[@]}"; do
        IFS=: read -r np pr pc <<< "$g"
        if (( np > 1 )); then
            has_multi=1
            break
        fi
    done

    if [[ "$has_multi" -eq 1 ]]; then
        echo
        echo "ATTENZIONE CUDA/MPI:"
        echo "  il server ha una sola GPU e i backend CUDA usano tutti device 0."
        echo "  Con P>1 i rank condividono la stessa GPU; t_kernel include anche"
        echo "  la contesa / alternanza tra contesti. Usa queste run per studiare"
        echo "  il comportamento MPI/globale, non per caratterizzare il kernel puro."
        echo "  Per confrontare cuda_naive/warp/smem o BLOCK, -np 1 resta il riferimento."
        echo
    fi
}

finalize_output_dir() {
    if [[ -z "$EXPERIMENT_NAME" ]]; then
        EXPERIMENT_NAME="$EXPERIMENT"
    fi

    EXPERIMENT_NAME="${EXPERIMENT_NAME// /_}"
    EXPERIMENT_NAME="${EXPERIMENT_NAME//\//_}"
    if [[ "$X_LAYOUT" == "column" ]]; then
        EXPERIMENT_NAME="${EXPERIMENT_NAME}_xcolumn"
    fi

    if [[ "$OUTDIR_EXPLICIT" -eq 0 ]]; then
        OUTDIR="results/${EXPERIMENT_NAME}"
    fi

    mkdir -p "$OUTDIR"
}

result_path() {
    local stem="$1"
    local ext="$2"
    echo "$OUTDIR/${EXPERIMENT_NAME}_${stem}.${ext}"
}

benchmark_csv_path() {
    local stem="$1"

    # Per gli esperimenti singoli il CSV prende esattamente il nome indicato
    # da name= nel .conf. "full" genera piu' dataset nella stessa directory,
    # quindi in quel solo caso viene mantenuto un suffisso descrittivo.
    if [[ "$EXPERIMENT" == "full" ]]; then
        echo "$OUTDIR/${EXPERIMENT_NAME}_${stem}.csv"
    else
        echo "$OUTDIR/${EXPERIMENT_NAME}.csv"
    fi
}

benchmark_raw_csv_path() {
    local stem="$1"

    if [[ "$EXPERIMENT" == "full" ]]; then
        echo "$OUTDIR/${EXPERIMENT_NAME}_${stem}_raw.csv"
    else
        echo "$OUTDIR/${EXPERIMENT_NAME}_raw.csv"
    fi
}

log() {
    echo
    echo "================================================================"
    echo "$*"
    echo "================================================================"
}

config_suffix() {
    local kernel="$1"
    local block="$2"
    local smem_pad="$3"
    local suffix=""

    if [[ "$PREC" != "double" ]]; then
        suffix="${suffix}-${PREC}"
    fi
    if [[ "$kernel" != "scheme_a" ]]; then
        suffix="${suffix}-${kernel}"
    fi
    if [[ "$FORCE_GENERIC_K" != "0" ]]; then
        suffix="${suffix}-generic"
    fi
    if [[ "$TEST_A_PADDING" != "0" ]]; then
        suffix="${suffix}-pad${TEST_A_PADDING}"
    fi
    if [[ "$smem_pad" != "1" ]]; then
        suffix="${suffix}-smempad${smem_pad}"
    fi
    if [[ "$block" != "256" ]]; then
        suffix="${suffix}-blk${block}"
    fi
    if [[ "$kernel" == "cuda_warp_tiled" && "$WARP_COL_TILE" != "8" ]]; then
        suffix="${suffix}-tile${WARP_COL_TILE}"
    fi
    if [[ "$X_LAYOUT" == "column" ]]; then
        suffix="${suffix}-xcol"
    fi

    echo "$suffix"
}

bin_for() {
    local kernel="$1"
    local block="$2"
    local smem_pad="$3"
    local suffix
    suffix="$(config_suffix "$kernel" "$block" "$smem_pad")"
    echo "./bin/matmul_mpi${suffix}"
}

build_kernel() {
    local kernel="$1"
    local block="$2"
    local smem_pad="$3"
    local extra_nvcc="${4:-}"
    local tile_args=()
    if [[ "$kernel" == "cuda_warp_tiled" ]]; then
        tile_args+=(WARP_COL_TILE="$WARP_COL_TILE")
    fi

    validate_block "$block"

    if [[ "$kernel" == "cuda_warp_smem" ]]; then
        log "BUILD kernel=$kernel BLOCK=$block SMEM_PAD=$smem_pad PREC=$PREC"
    elif [[ "$kernel" == "cuda_warp" ]]; then
        log "BUILD kernel=$kernel BLOCK=$block PREC=$PREC X_LAYOUT=$X_LAYOUT"
    else
        log "BUILD kernel=$kernel BLOCK=$block PREC=$PREC"
    fi

    make \
        KERNEL="$kernel" \
        X_LAYOUT="$X_LAYOUT" \
        BLOCK="$block" \
        PREC="$PREC" \
        SMEM_PAD="$smem_pad" \
        FORCE_GENERIC_K="$FORCE_GENERIC_K" \
        TEST_A_PADDING="$TEST_A_PADDING" \
        NVCC_ARCH="$NVCC_ARCH" \
        "${tile_args[@]}" \
        EXTRA_NVCCFLAGS="$extra_nvcc"
}

csv_header() {
    local bin="$1"
    local file="$2"

    mpirun -np 1 "${MPI_FLAGS[@]}" "$bin" --csv-header > "$file"
}

failure_file() {
    echo "$OUTDIR/${EXPERIMENT_NAME}_failures.csv"
}

init_failure_file() {
    local f
    f="$(failure_file)"
    if [[ ! -f "$f" ]]; then
        echo "experiment,kernel,M,N,k,block,smem_pad,np,pr,pc,exit_code" > "$f"
    fi
}

handle_failure() {
    local experiment="$1"
    local kernel="$2"
    local k="$3"
    local block="$4"
    local smem_pad="$5"
    local np="$6"
    local pr="$7"
    local pc="$8"
    local rc="$9"

    if [[ "$kernel" == "cuda_warp_tiled" ]]; then
        kernel="${kernel}(tile${WARP_COL_TILE})"
    fi
    if [[ "$X_LAYOUT" == "column" ]]; then
        kernel="${kernel}(xcol)"
    fi

    init_failure_file
    echo "$experiment,$kernel,$M,$N,$k,$block,$smem_pad,$np,$pr,$pc,$rc" \
        >> "$(failure_file)"

    if [[ "$CONTINUE_ON_ERROR" -eq 1 ]]; then
        echo "ATTENZIONE: run fallita, continuo (--continue-on-error)." >&2
        return 0
    fi

    return "$rc"
}

run_csv_row() {
    local experiment="$1"
    local bin="$2"
    local kernel="$3"
    local k="$4"
    local block="$5"
    local smem_pad="$6"
    local np="$7"
    local pr="$8"
    local pc="$9"
    local file="${10}"
    local raw_file="${11}"

    local args=(
        -M "$M" -N "$N" -k "$k"
        --pr "$pr" --pc "$pc"
        --a-mode "$A_MODE"
        --x-mode "$X_MODE"
        --warmup "$WARMUP"
        --reps "$REPS"
        --csv
        --csv-raw-file "$raw_file"
    )

    if [[ -n "$SEED" ]]; then
        args+=(--seed "$SEED")
    fi
    if [[ "$CHECK" -eq 1 ]]; then
        args+=(--check)
    fi

    set +e
    if [[ "$VERBOSE" -eq 1 ]]; then
        mpirun -np "$np" "${MPI_FLAGS[@]}" \
            "$bin" "${args[@]}" | tee -a "$file"
        local rc=${PIPESTATUS[0]}
    else
        mpirun -np "$np" "${MPI_FLAGS[@]}" \
            "$bin" "${args[@]}" >> "$file"
        local rc=$?
    fi
    set -e

    if [[ "$rc" -ne 0 ]]; then
        handle_failure "$experiment" "$kernel" "$k" "$block" "$smem_pad" \
            "$np" "$pr" "$pc" "$rc"
    fi
}
write_metadata() {
    local file="$OUTDIR/${EXPERIMENT_NAME}_metadata.txt"
    {
        printf "command="
        printf "%q " "$0" "${ORIGINAL_ARGS[@]}"
        echo

        echo "timestamp=$TIMESTAMP"
        echo "name=$EXPERIMENT_NAME"
        echo "experiment=$EXPERIMENT"
        echo "config_file=${CONFIG_FILE:-none}"
        echo "M=$M"
        echo "N=$N"
        echo "reps=$REPS"
        echo "warmup=$WARMUP"
        echo "seed=${SEED:-default}"
        echo "check=$CHECK"
        echo "a_mode=$A_MODE"
        echo "x_mode=$X_MODE"
        echo "x_layout=$X_LAYOUT"
        echo "kernel=$KERNEL"
        echo "kernels=${KERNELS[*]}"
        echo "k_sweep=${K_SWEEP_KS[*]}"
        echo "block_sweep_k=${BLOCK_SWEEP_KS[*]}"
        echo "compare_k=${COMPARE_KS[*]}"
        echo "block=$BLOCK"
        echo "blocks=${BLOCKS[*]}"
        echo "prec=$PREC"
        echo "smem_pad=$SMEM_PAD"
        echo "smem_pads=${SMEM_PADS[*]}"
        echo "warp_col_tile=$WARP_COL_TILE"
        echo "warp_col_tiles=${WARP_COL_TILES[*]}"
        echo "force_generic_k=$FORCE_GENERIC_K"
        echo "test_a_padding=$TEST_A_PADDING"
        echo "nvcc_arch=$NVCC_ARCH"
        echo "all_grids=${ALL_GRIDS_P:-no}"
        echo "btl=$BTL"
        echo "bind_to=${BIND_TO:-disabled}"
        echo "map_by=${MAP_BY:-disabled}"
        echo "report_bindings=$REPORT_BINDINGS"
        echo "grids:"
        for g in "${GRIDS[@]}"; do
            IFS=: read -r np pr pc <<< "$g"
            echo "  np=$np pr=$pr pc=$pc"
        done

        echo
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
        echo "git_status:"
        git status --short 2>/dev/null || true

        echo
        echo "nvcc:"
        nvcc --version 2>/dev/null || true

        echo
        echo "mpirun:"
        mpirun --version 2>/dev/null | head -n 2 || true

        echo
        echo "gpu:"
        nvidia-smi --query-gpu=name,driver_version,memory.total \
            --format=csv,noheader 2>/dev/null || true
    } > "$file"
}

# -------------------------
# Esperimenti
# -------------------------

experiment_k_sweep() {
    local kernel="$KERNEL"
    local block="$BLOCK"
    local smem_pad="$SMEM_PAD"

    build_kernel "$kernel" "$block" "$smem_pad"
    local bin
    bin="$(bin_for "$kernel" "$block" "$smem_pad")"

    local stem="k_sweep_${kernel}_block${block}"
    local csv="$(benchmark_csv_path "$stem")"
    local raw_csv="$(benchmark_raw_csv_path "$stem")"

    csv_header "$bin" "$csv"
    : > "$raw_csv"

    log "K-SWEEP -> $csv"
    echo "RAW repetitions -> $raw_csv"

    for g in "${GRIDS[@]}"; do
        IFS=: read -r np pr pc <<< "$g"
        for k in "${K_SWEEP_KS[@]}"; do
            echo "kernel=$kernel k=$k BLOCK=$block grid=${pr}x${pc} P=$np"
            run_csv_row "k-sweep" "$bin" "$kernel" "$k" "$block" "$smem_pad" \
                "$np" "$pr" "$pc" "$csv" "$raw_csv"
        done
    done
}

experiment_block_sweep() {
    local kernel="$KERNEL"
    local smem_pad="$SMEM_PAD"
    local stem="block_sweep_${kernel}"
    local csv="$(benchmark_csv_path "$stem")"
    local raw_csv="$(benchmark_raw_csv_path "$stem")"
    local header_written=0

    : > "$raw_csv"

    log "BLOCK-SWEEP -> $csv"
    echo "RAW repetitions -> $raw_csv"

    for block in "${BLOCKS[@]}"; do
        build_kernel "$kernel" "$block" "$smem_pad"

        local bin
        bin="$(bin_for "$kernel" "$block" "$smem_pad")"

        if [[ "$header_written" -eq 0 ]]; then
            csv_header "$bin" "$csv"
            header_written=1
        fi

        for g in "${GRIDS[@]}"; do
            IFS=: read -r np pr pc <<< "$g"
            for k in "${BLOCK_SWEEP_KS[@]}"; do
                echo "kernel=$kernel k=$k BLOCK=$block grid=${pr}x${pc} P=$np"
                run_csv_row "block-sweep" "$bin" "$kernel" "$k" "$block" "$smem_pad" \
                    "$np" "$pr" "$pc" "$csv" "$raw_csv"
            done
        done
    done
}

experiment_grid_sweep() {
    if [[ -z "$ALL_GRIDS_P" ]]; then
        echo "Errore: --experiment grid-sweep richiede --all-grids P." >&2
        exit 1
    fi

    local kernel="$KERNEL"
    local block="$BLOCK"
    local smem_pad="$SMEM_PAD"

    build_kernel "$kernel" "$block" "$smem_pad"

    local bin
    bin="$(bin_for "$kernel" "$block" "$smem_pad")"

    local stem="grid_sweep_${kernel}_block${block}_P${ALL_GRIDS_P}"
    local csv="$(benchmark_csv_path "$stem")"
    local raw_csv="$(benchmark_raw_csv_path "$stem")"

    csv_header "$bin" "$csv"
    : > "$raw_csv"

    log "GRID-SWEEP P=$ALL_GRIDS_P -> $csv"
    echo "RAW repetitions -> $raw_csv"

    for g in "${GRIDS[@]}"; do
        IFS=: read -r np pr pc <<< "$g"
        for k in "${K_SWEEP_KS[@]}"; do
            echo "kernel=$kernel k=$k BLOCK=$block grid=${pr}x${pc} P=$np"
            run_csv_row "grid-sweep" "$bin" "$kernel" "$k" "$block" "$smem_pad" \
                "$np" "$pr" "$pc" "$csv" "$raw_csv"
        done
    done
}

experiment_compare() {
    local block="$BLOCK"
    local smem_pad="$SMEM_PAD"
    local stem="kernel_compare_block${block}"
    local csv="$(benchmark_csv_path "$stem")"
    local raw_csv="$(benchmark_raw_csv_path "$stem")"
    local header_written=0

    : > "$raw_csv"

    log "KERNEL-COMPARE -> $csv"
    echo "RAW repetitions -> $raw_csv"

    for kernel in "${KERNELS[@]}"; do
        build_kernel "$kernel" "$block" "$smem_pad"

        local bin
        bin="$(bin_for "$kernel" "$block" "$smem_pad")"

        if [[ "$header_written" -eq 0 ]]; then
            csv_header "$bin" "$csv"
            header_written=1
        fi

        for g in "${GRIDS[@]}"; do
            IFS=: read -r np pr pc <<< "$g"
            for k in "${COMPARE_KS[@]}"; do
                echo "kernel=$kernel k=$k BLOCK=$block grid=${pr}x${pc} P=$np"
                run_csv_row "compare" "$bin" "$kernel" "$k" "$block" "$smem_pad" \
                    "$np" "$pr" "$pc" "$csv" "$raw_csv"
            done
        done
    done
}

experiment_registers() {
    local kernels=("${KERNELS[@]}")

    if [[ "$KERNEL_EXPLICIT" -eq 1 ]]; then
        kernels=("$KERNEL")
    fi

    local summary="$(result_path "registers_summary" "txt")"
    : > "$summary"

    log "PTXAS REGISTERS -> $OUTDIR"

    for kernel in "${kernels[@]}"; do
        local tile_args=()
        local label="$kernel"
        if [[ "$kernel" == "cuda_warp_tiled" ]]; then
            tile_args+=(WARP_COL_TILE="$WARP_COL_TILE")
            label="${kernel}_tile${WARP_COL_TILE}"
        fi
        local txt="$(result_path "ptxas_${label}" "txt")"
        local regs="$(result_path "ptxas_${label}_registers" "txt")"

        echo "kernel=$kernel"

        # -B e' necessario: EXTRA_NVCCFLAGS non entra nel nome CONFIG del Makefile.
        set +e
        make -B \
            KERNEL="$kernel" \
            X_LAYOUT="$X_LAYOUT" \
            BLOCK="$BLOCK" \
            PREC="$PREC" \
            SMEM_PAD="$SMEM_PAD" \
            FORCE_GENERIC_K="$FORCE_GENERIC_K" \
            TEST_A_PADDING="$TEST_A_PADDING" \
            NVCC_ARCH="$NVCC_ARCH" \
            "${tile_args[@]}" \
            EXTRA_NVCCFLAGS="-Xptxas -v" \
            2>&1 | tee "$txt"
        local rc=${PIPESTATUS[0]}
        set -e

        if [[ "$rc" -ne 0 ]]; then
            echo "Compilazione ptxas fallita per $kernel." >&2
            if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
                return "$rc"
            fi
            continue
        fi

        grep -iE "Used [0-9]+ registers|spill stores|spill loads|stack frame|barriers" \
            "$txt" > "$regs" || true

        {
            echo "===== $label ====="
            cat "$regs"
            echo
        } >> "$summary"
    done
}

experiment_smem_pad_sweep() {
    local kernel="$KERNEL"
    if [[ "$KERNEL_EXPLICIT" -eq 0 ]]; then
        kernel="cuda_warp_smem"
    fi

    local block="$BLOCK"
    local stem="smem_pad_sweep_${kernel}_block${block}"
    local csv="$(benchmark_csv_path "$stem")"
    local raw_csv="$(benchmark_raw_csv_path "$stem")"
    local header_written=0

    : > "$raw_csv"

    log "SMEM-PAD-SWEEP -> $csv"
    echo "RAW repetitions -> $raw_csv"

    for pad in "${SMEM_PADS[@]}"; do
        if ! is_nonneg_int "$pad"; then
            echo "Errore: SMEM_PAD non valido '$pad'." >&2
            exit 1
        fi

        build_kernel "$kernel" "$block" "$pad"

        local bin
        bin="$(bin_for "$kernel" "$block" "$pad")"

        if [[ "$header_written" -eq 0 ]]; then
            csv_header "$bin" "$csv"
            header_written=1
        fi

        for g in "${GRIDS[@]}"; do
            IFS=: read -r np pr pc <<< "$g"
            for k in "${K_SWEEP_KS[@]}"; do
                echo "kernel=$kernel k=$k BLOCK=$block SMEM_PAD=$pad grid=${pr}x${pc} P=$np"
                run_csv_row "smem-pad-sweep" "$bin" "$kernel" "$k" "$block" "$pad" \
                    "$np" "$pr" "$pc" "$csv" "$raw_csv"
            done
        done
    done
}

experiment_warp_tile_sweep() {
    local kernel="$KERNEL"
    local block="$BLOCK"
    local smem_pad="$SMEM_PAD"
    local stem="warp_tile_sweep_${kernel}_block${block}"
    local csv="$(benchmark_csv_path "$stem")"
    local raw_csv="$(benchmark_raw_csv_path "$stem")"
    local header_written=0
    local bin g np pr pc k
    # La variabile locale e' visibile anche agli helper Bash chiamati qui:
    # build, nome binario e registrazione errori usano sempre lo stesso tile.
    local WARP_COL_TILE

    : > "$raw_csv"
    log "WARP-TILE-SWEEP -> $csv"
    echo "RAW repetitions -> $raw_csv"

    for WARP_COL_TILE in "${WARP_COL_TILES[@]}"; do
        build_kernel "$kernel" "$block" "$smem_pad"
        bin="$(bin_for "$kernel" "$block" "$smem_pad")"
        if [[ "$header_written" -eq 0 ]]; then
            csv_header "$bin" "$csv"
            header_written=1
        fi
        for g in "${GRIDS[@]}"; do
            IFS=: read -r np pr pc <<< "$g"
            for k in "${K_SWEEP_KS[@]}"; do
                echo "kernel=$kernel k=$k BLOCK=$block WARP_COL_TILE=$WARP_COL_TILE grid=${pr}x${pc} P=$np"
                run_csv_row "warp-tile-sweep" "$bin" "$kernel" "$k" "$block" "$smem_pad" \
                    "$np" "$pr" "$pc" "$csv" "$raw_csv"
            done
        done
    done
}

experiment_ncu() {
    if ! command -v ncu >/dev/null 2>&1; then
        echo "ncu non trovato: profiling saltato." >&2
        return 0
    fi

    local kernel="$KERNEL"
    local block="$BLOCK"
    local smem_pad="$SMEM_PAD"
    local k="${K_SWEEP_KS[0]}"

    # Se non e' stato passato --k/--ks, il default per ncu e' k=32.
    if [[ "$KS_EXPLICIT" -eq 0 ]]; then
        k=32
    fi

    build_kernel "$kernel" "$block" "$smem_pad"

    local bin
    bin="$(bin_for "$kernel" "$block" "$smem_pad")"

    log "NSIGHT COMPUTE kernel=$kernel k=$k BLOCK=$block"

    for g in "${GRIDS[@]}"; do
        IFS=: read -r np pr pc <<< "$g"

        local base="$OUTDIR/${EXPERIMENT_NAME}_ncu_${kernel}_k${k}_block${block}_P${np}_${pr}x${pc}"
        if [[ "$kernel" == "cuda_warp_tiled" ]]; then
            base="${base}_tile${WARP_COL_TILE}"
        fi
        local txt="${base}.txt"

        local args=(
            -M "$M" -N "$N" -k "$k"
            --pr "$pr" --pc "$pc"
            --a-mode "$A_MODE"
            --x-mode "$X_MODE"
            --warmup 0
            --reps 1
        )

        if [[ -n "$SEED" ]]; then
            args+=(--seed "$SEED")
        fi

        echo "grid=${pr}x${pc} P=$np"

        set +e
        mpirun -np "$np" "${MPI_FLAGS[@]}" \
            ncu \
            --set "$NCU_SET" \
            -o "$base" \
            "$bin" "${args[@]}" \
            2>&1 | tee "$txt"
        local rc=${PIPESTATUS[0]}
        set -e

        if grep -q "ERR_NVGPUCTRPERM" "$txt"; then
            echo "NVIDIA performance counters non accessibili: salvo il log e interrompo ncu."
            return 0
        fi

        if [[ "$rc" -ne 0 ]]; then
            if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
                return "$rc"
            fi
        fi
    done
}

experiment_full() {
    log "FULL CUDA EXPERIMENT CAMPAIGN"

    local saved_kernel="$KERNEL"
    local saved_kernel_explicit="$KERNEL_EXPLICIT"
    local saved_block_ks=("${BLOCK_SWEEP_KS[@]}")
    local saved_compare_ks=("${COMPARE_KS[@]}")

    # 1) k-sweep naive
    KERNEL="cuda_naive"
    KERNEL_EXPLICIT=1
    experiment_k_sweep

    # 2) block-sweep COMPLETO: tutti i k standard
    BLOCK_SWEEP_KS=("${K_SWEEP_KS[@]}")
    experiment_block_sweep

    # 3) confronto kernel: tutti i k standard
    COMPARE_KS=("${K_SWEEP_KS[@]}")
    experiment_compare

    # 4) padding shared memory
    KERNEL="cuda_warp_smem"
    KERNEL_EXPLICIT=1
    experiment_smem_pad_sweep

    # 5) registri di tutti i kernel
    KERNEL_EXPLICIT=0
    experiment_registers

    # 6) ncu: un solo caso rappresentativo, naive k=32.
    # Se sul server i counter sono vietati, viene registrato e saltato.
    KERNEL="cuda_naive"
    KERNEL_EXPLICIT=1

    local saved_k_sweep=("${K_SWEEP_KS[@]}")
    local saved_ks_explicit="$KS_EXPLICIT"
    K_SWEEP_KS=(32)
    KS_EXPLICIT=1

    # In full profiliamo solo la prima griglia per non moltiplicare replay/costi.
    local saved_grids=("${GRIDS[@]}")
    GRIDS=("${GRIDS[0]}")
    experiment_ncu || true
    GRIDS=("${saved_grids[@]}")

    K_SWEEP_KS=("${saved_k_sweep[@]}")
    KS_EXPLICIT="$saved_ks_explicit"

    KERNEL="$saved_kernel"
    KERNEL_EXPLICIT="$saved_kernel_explicit"
    BLOCK_SWEEP_KS=("${saved_block_ks[@]}")
    COMPARE_KS=("${saved_compare_ks[@]}")
}

# -------------------------
# Main
# -------------------------

validate_common
generate_grids
build_mpi_flags
finalize_output_dir
warn_multi_rank_gpu_contention
write_metadata

case "$EXPERIMENT" in
    k-sweep)
        experiment_k_sweep
        ;;
    block-sweep)
        experiment_block_sweep
        ;;
    grid-sweep)
        experiment_grid_sweep
        ;;
    compare)
        experiment_compare
        ;;
    registers)
        experiment_registers
        ;;
    smem-pad-sweep)
        experiment_smem_pad_sweep
        ;;
    warp-tile-sweep)
        experiment_warp_tile_sweep
        ;;
    ncu)
        experiment_ncu
        ;;
    full)
        experiment_full
        ;;
    *)
        echo "Errore: esperimento sconosciuto '$EXPERIMENT'." >&2
        usage
        exit 1
        ;;
esac

echo
echo "Esperimento completato."
echo "Risultati: $OUTDIR"
