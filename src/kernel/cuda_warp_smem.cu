/* Backend CUDA warp-per-row con la fetta di X messa in SHARED MEMORY.  M4.3.
 *
 * Non e' un algoritmo nuovo: e' cuda_warp con una sola variabile cambiata, ed
 * e' questo che lo rende misurabile. L'ordine dei cicli resta i -> j -> c, la
 * riga di A resta letta una volta sola, i k accumulatori restano in registro
 * distribuiti sulle 32 lane, e la riduzione finale resta __shfl_down_sync.
 * Cambia soltanto DA DOVE arrivano gli elementi di X.
 *
 * ---------------------------------------------------------------------------
 * Il difetto di cuda_warp che questo kernel attacca
 * ---------------------------------------------------------------------------
 * In cuda_warp ogni warp percorre TUTTA X: calcolando la riga i di Y legge
 * X[j][0..k-1] per ogni j = 0..n-1. Con un warp per riga di Y i warp sono m, e
 * il traffico di lettura su X vale
 *
 *     m * n * k elementi    contro    m * n elementi di A
 *
 * cioe' k volte il traffico di A. X e' piccola (n x k) e sta in L2, quindi non
 * si vede in DRAM: quel traffico e' banda L2, e cresce linearmente con k. E'
 * l'ipotesi da verificare - se il kernel e' gia' compute-bound, come il
 * roofline in FP64 suggerisce, questo kernel non puo' guadagnare niente e la
 * misura serve a dimostrarlo, non a nasconderlo.
 *
 * Il rimedio sfrutta una coincidenza della mappatura: gli 8 warp di uno stesso
 * blocco lavorano su righe DIVERSE di A, ma allo stesso passo j vogliono le
 * STESSE righe di X. Quindi il blocco stagia cooperativamente un tile
 * x_rows_per_tile x k di X in shared memory e i suoi 8 warp lo riusano: il traffico verso
 * L2 viene diviso per il numero di warp per blocco.
 *
 *     cuda_warp        ogni warp legge X da L2          m * n * k
 *     cuda_warp_smem   un blocco legge X da L2 una      m * n * k / 8
 *                      volta per i suoi 8 warp
 *
 * ---------------------------------------------------------------------------
 * Il conflitto sui banchi e il padding a k+1
 * ---------------------------------------------------------------------------
 * La shared memory ha 32 banchi da 32 bit. Un double ne occupa DUE. La lane l
 * legge X_tile[l*tile_row_stride + c], quindi fra due lane consecutive
 * ci sono tile_row_stride scalari, cioe' 2*tile_row_stride banchi.
 *
 *   tile_row_stride = k = 32  ->  64 banchi = 0 (mod 32): tutte e 32 le lane
 *                             cadono sullo stesso banco. Conflitto a 32 vie,
 *                             l'accesso viene serializzato 32 volte.
 *   tile_row_stride = k + 1   ->  66 banchi = 2 (mod 32): lo stride torna dispari
 *                             in double e l'accesso scende al minimo che il
 *                             double consente (l'hardware serve i 64 bit in
 *                             due fasi da 16 lane).
 *
 * E' l'analogo esatto del padding LDA = N + 24 gia' studiato sulla CPU contro
 * i conflict miss di L1: stesso fenomeno, altra gerarchia di memoria. Il
 * padding e' un knob di compilazione proprio per poter misurare i due casi:
 *
 *     make KERNEL=cuda_warp_smem              -> bin/matmul_mpi-cuda_warp_smem
 *     make KERNEL=cuda_warp_smem SMEM_PAD=0   -> ...-cuda_warp_smem-smempad0
 *
 * I due binari coesistono, quindi il confronto e' una sola campagna e non c'e'
 * modo di misurare per sbaglio la build sbagliata.
 *
 * ---------------------------------------------------------------------------
 * Perche' x_rows_per_tile si sceglie a runtime, e su cosa agisce davvero
 * ---------------------------------------------------------------------------
 * Il tile in shared memory pesa x_rows_per_tile * (k + SMEM_PAD) * sizeof(scalar_t),
 * e k e' noto solo a runtime: la memoria e' quindi allocata dinamicamente al
 * lancio e le righe si scelgono in local_gemm_create.
 *
 * ATTENZIONE a cosa cambia e cosa no. Il traffico verso L2 NON dipende dalla
 * dimensione del tile: dipende da quanti warp ci sono per blocco. Ogni blocco
 * legge comunque tutta X una volta, che lo faccia in due tile grandi o in venti
 * piccoli. Il fattore di riuso resta WARPS_PER_BLOCK e basta.
 *
 * Quello su cui il tile agisce davvero e' il NUMERO DI BARRIERE e l'ILP: ogni
 * tile costa due __syncthreads(), e un tile che contiene un solo passo di warp
 * (x_rows_per_tile == 32) fa sbattere ogni warp nella barriera successiva dopo
 * una sola iterazione del ciclo j, senza nessuna iterazione con cui coprire la
 * latenza delle letture da shared. Dimezzare il numero di tile dimezza le
 * barriere; non tocca il traffico.
 *
 * Il budget di shared memory per blocco non va quindi scelto in assoluto, ma in
 * modo che NON SIA LUI il vincolo attivo sull'occupancy: vedi plan_tile.
 *
 * ---------------------------------------------------------------------------
 * Perche' non ci sono return anticipati
 * ---------------------------------------------------------------------------
 * cuda_warp fa `if (warp_id >= m) return;` in cima. Qui NON si puo': il ciclo
 * sui tile contiene __syncthreads(), che e' una barriera di BLOCCO. Se i warp
 * dell'ultimo blocco (quello parzialmente pieno) uscissero prima, i warp
 * rimasti aspetterebbero a una barriera che nessuno raggiunge piu'. Tutti i
 * thread del blocco percorrono quindi lo stesso numero di iterazioni e di
 * barriere; e' solo l'accumulo, e la scrittura finale, a essere condizionato
 * da `active`.
 */

#include <cuda_runtime.h>

#include "kernel/kernel.h"
#include "common/util.h"

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess)                                              \
            die("CUDA error at %s:%d: %s: %s", __FILE__, __LINE__,            \
                cudaGetErrorName(err_), cudaGetErrorString(err_));            \
    } while (0)

#define WARP_SIZE 32

/* Thread per blocco.
 *
 * 256 e' il default: 8 warp, e un DIVISORE di 1024, il massimo di thread
 * residenti per SM su Turing (sm_75) - quattro blocchi riempiono l'SM, mentre
 * 192 o 384 si fermerebbero a 960 e 768 thread su 1024.
 *
 * Come in cuda_warp il multiplo di 32 e' un requisito di CORRETTEZZA:
 * WARPS_PER_BLOCK decide quante righe di Y elabora un blocco e la riduzione
 * finale e' interna al warp. Qui BLOCK_THREADS governa anche il caricamento
 * COOPERATIVO del tile di X in shared memory (il passo del ciclo di load), ma
 * non la sua DIMENSIONE, che dipende solo da x_rows_per_tile e k: cambiare BLOCK non altera
 * quindi il budget di shared memory ne' il risultato.
 * Il valore si sostituisce dal Makefile con BLOCK=<n> (SCPA_BLOCK_THREADS). */
#ifndef SCPA_BLOCK_THREADS
#define SCPA_BLOCK_THREADS 256
#endif

#if SCPA_BLOCK_THREADS < 32 || SCPA_BLOCK_THREADS > 1024 || (SCPA_BLOCK_THREADS % 32) != 0
#error "SCPA_BLOCK_THREADS deve essere un multiplo di 32 compreso fra 32 e 1024"
#endif

#define BLOCK_THREADS SCPA_BLOCK_THREADS

#define WARPS_PER_BLOCK (BLOCK_THREADS / WARP_SIZE)

#define RUNTIME_TILE 4

/* Il server di dipartimento su cui la consegna richiede di misurare ha una
 * sola GPU: non c'e' nessun device da scegliere, e una logica di selezione
 * basata sul rank locale MPI sarebbe codice che non puo' mai fare nulla.
 * Tutti i rank sullo stesso nodo usano quindi il device 0.
 *
 * Conseguenza da tenere presente nel leggere le misure: con piu' rank MPI per
 * GPU i contesti si alternano sulla scheda (MPS non attivo), quindi t_kernel
 * comprende anche il tempo in cui il contesto di questo rank e' sospeso a
 * favore di un altro. Per caratterizzare il kernel in se' va usato un rank
 * singolo; con piu' rank il numero resta valido come throughput AGGREGATO
 * della GPU sul problema globale, non come tempo di una GPU dedicata. */
#define CUDA_DEVICE_ID 0

#define BYTES_PER_GIB 1073741824.0

/* Scalari di padding aggiunti a ogni riga del tile di X in shared memory.
 * 1 e' il valore che rompe il conflitto a 32 vie su k=32; 0 e' il termine di
 * paragone da misurare. Si imposta dal Makefile con SMEM_PAD=<n>. */
#ifndef SCPA_SMEM_PAD
#define SCPA_SMEM_PAD 1
#endif
#if SCPA_SMEM_PAD < 0
#error "SCPA_SMEM_PAD deve essere >= 0"
#endif

/* Limite architetturale della shared memory dinamica per blocco senza opt-in
 * via cudaFuncSetAttribute: oltre questo il LANCIO fallisce. E' un tetto vero,
 * espresso in byte, e resta la rete di sicurezza del piano di tiling.
 *
 * Non esiste piu' un massimo espresso in RIGHE: non era un vincolo fisico e,
 * appena il budget cresce, sarebbe diventato silenziosamente il vincolo attivo
 * ai k piccoli, rendendo inefficace qualunque sweep sul budget. Il minimo di un
 * passo di warp invece resta, in plan_tile, perche' un significato ce l'ha. */
#define SMEM_MAX_BYTES    49152

/* Granularita' di arrotondamento delle righe per tile.
 *
 * NON e' un requisito di correttezza: rows_in_tile gestisce gia' il tile
 * parziale e il ciclo `for (j = lane; j < rows_in_tile; j += WARP_SIZE)` gestisce
 * gia' un tile che non sia multiplo di 32. E' solo un'ottimizzazione - "nessuna
 * lane inattiva nell'ultimo passo di warp" - e come tale ha un costo: a k=32 in
 * double le righe che entrano nel budget sono 62, e arrotondare a 32 ne butta
 * via la meta', raddoppiando le barriere per guadagnare zero lane.
 *
 * Costo e beneficio vanno quindi MISURATI, non assunti: da qui il parametro.
 * Il default resta 32, cosi' il comportamento non cambia finche' non si chiede
 * esplicitamente altro. Si imposta dal Makefile con TILE_GRANULARITY=<n>. */
#ifndef SCPA_TILE_GRANULARITY
#define SCPA_TILE_GRANULARITY 32
#endif
#if SCPA_TILE_GRANULARITY < 1
#error "SCPA_TILE_GRANULARITY deve essere >= 1"
#endif

/* ---------------------------------------------------------------------------
 * Kernel specializzato: K e' una costante di compilazione
 * ------------------------------------------------------------------------ */
template<int K>
static __global__ void smem_kernel_fixed(int m_loc, int n_loc, int x_rows_per_tile, const scalar_t *__restrict__ A_loc, int lda, const scalar_t *__restrict__ X_loc, int ldx, scalar_t *__restrict__ Y_loc_part, int ldy) {

    /* Shared memory dinamica: la dimensione in byte la fissa il lancio (vedi
     * launch_smem_kernel), il tipo lo fissa questa dichiarazione. */
    extern __shared__ scalar_t X_tile[];

    const int tile_row_stride = K + SCPA_SMEM_PAD;

    const int lane = threadIdx.x & (WARP_SIZE - 1); // posizione nel warp -> 0...31 - equivale a prendere il resto della divisione per 32
    const long long row = (long long)blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5); // posizione del warp all'interno del blocco

    /* Uniforme sul warp: row non dipende da lane. Serve piu' sotto per la
     * maschera delle shuffle. */
    const bool active = (row < m_loc);

    const scalar_t *const arow = A_loc + (size_t)(active ? row : 0) * (size_t)lda; // ripiegamento a riga 0 se il warp è inattivo, per non scatenare undefined behavior

    scalar_t acc[K];

    int c, j, tile_first_row, load_index, offset;

#pragma unroll
    for (c = 0; c < K; ++c)
        acc[c] = (scalar_t)0;

    for (tile_first_row = 0; tile_first_row < n_loc; tile_first_row += x_rows_per_tile) {

        const int rows_in_tile = (n_loc - tile_first_row < x_rows_per_tile) ? (n_loc - tile_first_row) : x_rows_per_tile;

        /* Prima barriera: nessuno sovrascrive il tile finche' tutti i warp
         * del blocco hanno finito di consumare quello precedente.
         *
         * Il costo è che tutti aspettano il più lento.
         * Il beneficio è che dopo la barriera il tile è completo e visibile a tutto il blocco.
         *
        */
        __syncthreads();

        /* Staging cooperativo: i 256 thread del blocco si spartiscono le
         * rows_in_tile x K letture. Thread consecutivi hanno tile_col consecutivo,
         * leggono elementi contigui della stessa riga di X: coalescente. */
        for (load_index = threadIdx.x; load_index < rows_in_tile * K; load_index += BLOCK_THREADS) {

            const int tile_row = load_index / K;
            const int tile_col = load_index - tile_row * K;

            X_tile[tile_row * tile_row_stride + tile_col] =
                X_loc[(size_t)(tile_first_row + tile_row) * (size_t)ldx + tile_col];
        }

        /* Seconda barriera: il tile e' completo e visibile a tutto il blocco. */
        __syncthreads();

        if (active) {
            /* Identico al cuore di cuda_warp, con xrow che ora punta in
             * shared invece che in globale: la lane l prende j = l, l+32, ...
             * quindi le letture di A restano perfettamente coalescenti e ogni
             * valore di A letto viene riusato per tutti e K gli accumulatori. */
            for (j = lane; j < rows_in_tile; j += WARP_SIZE) {
                const scalar_t a = arow[tile_first_row + j];
                const scalar_t *const xrow = X_tile + j * tile_row_stride;
#pragma unroll
                for (c = 0; c < K; ++c)
                    acc[c] += a * xrow[c];
            }
        }
    }

    if (!active)
        return;

    /* Maschera piena e non __activemask(): da Volta in poi le lane possono
     * divergere e riconvergere in modo indipendente, e __activemask() non
     * implica convergenza. Qui `active` e' uniforme sul warp, quindi o le 32
     * lane sono tutte vive o nessuna lo e', e la maschera e' nota a priori. */
#pragma unroll
    for (offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (c = 0; c < K; ++c)
            acc[c] += __shfl_down_sync(0xffffffffu, acc[c], offset);
    }

    if (lane == 0) {
        scalar_t *const yrow = Y_loc_part + (size_t)row * (size_t)ldy;
#pragma unroll
        for (c = 0; c < K; ++c)
            yrow[c] = acc[c];
    }
}

/* ---------------------------------------------------------------------------
 * Fallback per k arbitrario
 * ---------------------------------------------------------------------------
 * Stessa politica di specializzazione di scheme_a e cuda_warp: i cinque k
 * richiesti sono istanze template, ogni altro k passa di qui. Le colonne si
 * lavorano quattro alla volta - come nel fallback di cuda_warp - perche' un
 * acc[] indicizzato da una variabile a runtime finirebbe in local memory
 * invece che nei registri. Il tile in shared contiene quindi solo le quattro
 * colonne del passo corrente, ed e' minuscolo: x_rows_per_tile x (4 + SMEM_PAD).
 * Non c'e' nessun limite superiore su k. */
static __global__ void smem_kernel_runtime(int m_loc, int n_loc, int k, int x_rows_per_tile, const scalar_t *__restrict__ A_loc, int lda, const scalar_t *__restrict__ X_loc, int ldx, scalar_t *__restrict__ Y_loc_part, int ldy) {

    /* Shared memory dinamica: la dimensione in byte la fissa il lancio (vedi
     * launch_smem_kernel), il tipo lo fissa questa dichiarazione. */
    extern __shared__ scalar_t X_tile[];

    const int tile_row_stride = RUNTIME_TILE + SCPA_SMEM_PAD;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const long long row = (long long)blockIdx.x * WARPS_PER_BLOCK
                          + (threadIdx.x >> 5);
    const bool active = (row < m_loc);
    const scalar_t *const arow =
        A_loc + (size_t)(active ? row : 0) * (size_t)lda;
    scalar_t acc[RUNTIME_TILE];
    int col_tile_start, col_in_tile, j, tile_first_row, load_index, offset;

    /* k, n e x_rows_per_tile sono uniformi sul blocco: tutti i thread fanno lo stesso
     * numero di giri, quindi le __syncthreads() qui sotto sono raggiunte da
     * tutti anche nell'ultimo blocco parzialmente pieno. */
    for (col_tile_start = 0; col_tile_start < k; col_tile_start += RUNTIME_TILE) {
#pragma unroll
        for (col_in_tile = 0; col_in_tile < RUNTIME_TILE; ++col_in_tile)
            acc[col_in_tile] = (scalar_t)0;

        for (tile_first_row = 0; tile_first_row < n_loc; tile_first_row += x_rows_per_tile) {
            const int rows_in_tile = (n_loc - tile_first_row < x_rows_per_tile) ? (n_loc - tile_first_row) : x_rows_per_tile;

            __syncthreads();

            for (load_index = threadIdx.x; load_index < rows_in_tile * RUNTIME_TILE;
                 load_index += BLOCK_THREADS) {
                const int tile_row = load_index / RUNTIME_TILE;
                const int tile_col = load_index - tile_row * RUNTIME_TILE;
                /* Le colonne oltre k si azzerano: cosi' il cuore del calcolo
                 * resta senza rami e il resto di k costa solo la guardia
                 * sulla scrittura finale. */
                X_tile[tile_row * tile_row_stride + tile_col] =
                    (col_tile_start + tile_col < k)
                        ? X_loc[(size_t)(tile_first_row + tile_row) * (size_t)ldx + col_tile_start + tile_col]
                        : (scalar_t)0;
            }

            __syncthreads();

            if (active) {
                for (j = lane; j < rows_in_tile; j += WARP_SIZE) {
                    const scalar_t a = arow[tile_first_row + j];
                    const scalar_t *const xrow = X_tile + j * tile_row_stride;
#pragma unroll
                    for (col_in_tile = 0; col_in_tile < RUNTIME_TILE; ++col_in_tile)
                        acc[col_in_tile] += a * xrow[col_in_tile];
                }
            }
        }

        if (active) {
#pragma unroll
            for (offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
                for (col_in_tile = 0; col_in_tile < RUNTIME_TILE; ++col_in_tile)
                    acc[col_in_tile] += __shfl_down_sync(0xffffffffu, acc[col_in_tile], offset);
            }

            if (lane == 0) {
                scalar_t *const yrow =
                    Y_loc_part + (size_t)row * (size_t)ldy + col_tile_start;
#pragma unroll
                for (col_in_tile = 0; col_in_tile < RUNTIME_TILE; ++col_in_tile)
                    if (col_tile_start + col_in_tile < k)
                        yrow[col_in_tile] = acc[col_in_tile];
            }
        }
    }
}

/* ---------------------------------------------------------------------------
 * Pianificazione del tile
 * ---------------------------------------------------------------------------
 * Il piano dipende solo da k e da n_loc, che sono fissi per tutta l'esecuzione:
 * si calcola UNA VOLTA in local_gemm_create e non ha nulla da fare nel cammino
 * cronometrato. Vedi il commento in local_gemm_create sul perche' quella
 * distinzione qui sia critica e non stilistica. */
typedef struct {
    int    x_rows_per_tile;
    size_t smem_bytes;
    int    blocks_per_sm;   /* esposto nel CSV: senza, le curve non si spiegano */
} tile_plan_t;

/* Serve solo al budget derivato: con SCPA_SMEM_BUDGET_BYTES definito la
 * funzione non verrebbe chiamata da nessuno e -Wunused-function la segnalerebbe
 * a ogni build dello sweep. */
#ifndef SCPA_SMEM_BUDGET_BYTES
static int shared_per_sm(void) {
    static int cached = -1;
    if (cached < 0) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, CUDA_DEVICE_ID));
        cached = (int)prop.sharedMemPerMultiprocessor;
    }
    return cached;
}
#endif

static int round_down_g(int v, int g) { return v - (v % g); }
static int round_up_g(int v, int g)   { return ((v + g - 1) / g) * g; }

/* Sceglie le righe per tile in modo che la shared memory non sia il vincolo
 * attivo sull'occupancy.
 *
 * Il problema dell'uovo e della gallina - l'API dell'occupancy vuole i byte di
 * shared che stiamo calcolando - si rompe interrogandola con smem = 0: cosi' si
 * ottiene quanti blocchi consentono REGISTRI e thread per SM da soli, cioe'
 * l'obiettivo da non peggiorare. A k=32 acc[32] costa gia' 64 registri a 32 bit
 * per thread, quindi il vincolo attivo sono quasi certamente quelli: un budget
 * di shared scelto a mano piu' stretto starebbe rimpicciolendo il tile per
 * proteggere un'occupancy gia' persa altrove. */
static tile_plan_t plan_tile(const void *kernel, int tile_row_stride, int n_loc) {

    const int g         = SCPA_TILE_GRANULARITY;
    const int row_bytes = tile_row_stride * (int)sizeof(scalar_t);
    int target, budget, rows, achieved = -1;
    tile_plan_t plan;

    /* 1. Blocchi che entrerebbero con shared memory gratis. */
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &target, kernel, BLOCK_THREADS, 0));
    if (target < 1)
        target = 1;

    /* 2. Fetta di shared memory che spetta a un blocco. */
#ifdef SCPA_SMEM_BUDGET_BYTES
    budget = SCPA_SMEM_BUDGET_BYTES;   /* forzato dal Makefile, per lo sweep */
#else
    budget = shared_per_sm() / target; /* derivato dall'obiettivo di occupancy */
#endif
    if (budget > SMEM_MAX_BYTES)
        budget = SMEM_MAX_BYTES;       /* limite di lancio, non di occupancy */

    /* 3. Righe che ci stanno. */
    rows = round_down_g(budget / row_bytes, g);
    if (rows < WARP_SIZE)
        rows = WARP_SIZE;                 /* almeno un passo di warp completo */
    if (n_loc > 0 && n_loc < rows)
        rows = round_up_g(n_loc, g);      /* inutile allocare piu' righe di
                                           * quante X ne abbia davvero */

    /* 4. Verifica invece di fidarsi. Il driver riserva shared memory per blocco
     *    OLTRE a quella richiesta, quindi il punto 2 puo' essere ottimista.
     *    Quella riserva non va modellata con una costante inventata: la si
     *    scopre interrogando l'API e scendendo di un passo finche' l'obiettivo
     *    e' raggiunto. Il ciclo termina sempre, perche' rows non scende sotto
     *    WARP_SIZE e li' si esce comunque. */
    for (;;) {
        plan.smem_bytes = (size_t)rows * (size_t)row_bytes;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &achieved, kernel, BLOCK_THREADS, plan.smem_bytes));
        if (achieved >= target || rows <= WARP_SIZE)
            break;
        rows -= g;
        if (rows < WARP_SIZE)
            rows = WARP_SIZE;
    }

    if (plan.smem_bytes > (size_t)SMEM_MAX_BYTES)
        die("cuda_warp_smem: il tile richiede %zu byte di shared memory per "
            "blocco, oltre il limite di %d (righe per tile=%d, stride=%d, pad=%d): "
            "abbassare SMEM_BUDGET_BYTES",
            plan.smem_bytes, SMEM_MAX_BYTES, rows, tile_row_stride, SCPA_SMEM_PAD);

    plan.x_rows_per_tile = rows;
    plan.blocks_per_sm   = achieved;
    return plan;
}

/* plan_tile vuole il puntatore all'ISTANZA template concreta: i registri di
 * smem_kernel_fixed<3> e smem_kernel_fixed<32> non sono gli stessi, e chiedere
 * l'occupancy della funzione sbagliata darebbe un piano sbagliato. */
template<int K>
static tile_plan_t plan_fixed(int n_loc) {
    return plan_tile((const void *)smem_kernel_fixed<K>,
                     K + SCPA_SMEM_PAD, n_loc);
}

static tile_plan_t plan_for_k(int k, int n_loc) {
    switch (k) {
    case 3:  return plan_fixed<3>(n_loc);
    case 6:  return plan_fixed<6>(n_loc);
    case 8:  return plan_fixed<8>(n_loc);
    case 20: return plan_fixed<20>(n_loc);
    case 32: return plan_fixed<32>(n_loc);
    default: return plan_tile((const void *)smem_kernel_runtime,
                              RUNTIME_TILE + SCPA_SMEM_PAD, n_loc);
    }
}

/* Nel cammino cronometrato resta soltanto il lancio: il piano arriva gia'
 * pronto dal contesto. */
static void launch_smem_kernel(const tile_plan_t *plan, int m_loc, int n_loc, int k, const scalar_t *A, int lda, const scalar_t *X, int ldx, scalar_t *Y, int ldy) {

    const int blocks = (int)(((long long)m_loc + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    const int rows = plan->x_rows_per_tile;
    const size_t smem_bytes = plan->smem_bytes;

    switch (k) {
    case 3:
        smem_kernel_fixed<3><<<blocks, BLOCK_THREADS, smem_bytes>>>(m_loc, n_loc, rows, A, lda, X, ldx, Y, ldy);
        break;
    case 6:
        smem_kernel_fixed<6><<<blocks, BLOCK_THREADS, smem_bytes>>>(m_loc, n_loc, rows, A, lda, X, ldx, Y, ldy);
        break;
    case 8:
        smem_kernel_fixed<8><<<blocks, BLOCK_THREADS, smem_bytes>>>(m_loc, n_loc, rows, A, lda, X, ldx, Y, ldy);
        break;
    case 20:
        smem_kernel_fixed<20><<<blocks, BLOCK_THREADS, smem_bytes>>>(m_loc, n_loc, rows, A, lda, X, ldx, Y, ldy);
        break;
    case 32:
        smem_kernel_fixed<32><<<blocks, BLOCK_THREADS, smem_bytes>>>(m_loc, n_loc, rows, A, lda, X, ldx, Y, ldy);
        break;
    default:
        smem_kernel_runtime<<<blocks, BLOCK_THREADS, smem_bytes>>>(m_loc, n_loc, k, rows, A, lda, X, ldx, Y, ldy);
        break;
    }
}

/* ---------------------------------------------------------------------------
 * Contesto e ciclo di vita: identici a cuda_warp
 * ------------------------------------------------------------------------ */
struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda, ldx, ldy;
    scalar_t *dA_loc, *dX_loc, *dY_loc_part;
    cudaEvent_t ev_start, ev_stop;
    tile_plan_t plan;   /* calcolato una volta sola: k e n_loc non cambiano piu' */
    double t_setup;
    double t_last;
};

static size_t nonzero(size_t bytes) {
    return (bytes != 0) ? bytes : 1;
}

local_gemm_t *local_gemm_create(int m_loc, int n_loc, int k, const scalar_t *A_loc, int lda, int ldx, int ldy) {

    local_gemm_t *local_gemm_context;
    size_t bytes_A, bytes_X, bytes_Y, need, free_b = 0, total_b = 0;
    cudaError_t err;
    const double t0 = now_seconds();

    if (m_loc < 0 || n_loc < 0 || k < 0)
        die("local_gemm_create: invalid local block %dx%d with k=%d", m_loc, n_loc, k);
    if (lda < n_loc)
        die("local_gemm_create: lda %d is smaller than n %d", lda, n_loc);
    if (ldx < k || ldy < k)
        die("local_gemm_create: ldx %d and ldy %d must both be at least k=%d",
            ldx, ldy, k);
    if (n_loc > 0 && m_loc > 0 && A_loc == NULL)
        die("local_gemm_create: A is NULL for a non-empty %dx%d block", m_loc, n_loc);

    local_gemm_context = (local_gemm_t *)xmalloc(sizeof *local_gemm_context);
    local_gemm_context->m_loc = m_loc;
    local_gemm_context->n_loc = n_loc;
    local_gemm_context->k = k;
    local_gemm_context->lda = lda;
    local_gemm_context->ldx = ldx;
    local_gemm_context->ldy = ldy;
    local_gemm_context->dA_loc = NULL;
    local_gemm_context->dX_loc = NULL;
    local_gemm_context->dY_loc_part = NULL;
    local_gemm_context->t_last = -1.0;

    CUDA_CHECK(cudaSetDevice(CUDA_DEVICE_ID));
    CUDA_CHECK(cudaFree(0));

    bytes_A = (size_t)m_loc * (size_t)lda * sizeof(scalar_t);
    bytes_X = (size_t)n_loc * (size_t)ldx * sizeof(scalar_t);
    bytes_Y = (size_t)m_loc * (size_t)ldy * sizeof(scalar_t);
    need = bytes_A + bytes_X + bytes_Y;

    /* Capienza in VRAM: si VERIFICA TENTANDO, non stimando prima.
     *
     * Il driver e' l'unico a sapere quanto costano davvero allineamento e
     * frammentazione, e - su questo server, dove piu' rank MPI condividono
     * l'unica GPU - quanto hanno allocato gli altri processi un istante fa.
     * Un controllo preventivo con cudaMemGetInfo confronterebbe una fotografia
     * gia' obsoleta e andrebbe corretto con un margine arbitrario; qui la
     * risposta la da' cudaMalloc, che non puo' sbagliarsi.
     *
     * La catena si ferma alla prima allocazione che non entra: le successive
     * non vengono nemmeno tentate, ed err arriva al controllo qui sotto. */
    err = cudaMalloc((void **)&local_gemm_context->dA_loc, nonzero(bytes_A));
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&local_gemm_context->dX_loc, nonzero(bytes_X));
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&local_gemm_context->dY_loc_part, nonzero(bytes_Y));

    if (err == cudaErrorMemoryAllocation) {
        /* Serve solo a comporre il messaggio: se anche questa query fallisse,
         * i due valori resterebbero a zero e la diagnosi degraderebbe senza
         * mascherare l'errore vero. Meglio questo del laconico "out of
         * memory" del runtime, che non dice ne' quanto serviva ne' che fare. */
        cudaMemGetInfo(&free_b, &total_b);
        die("%s: out of memory on device %d: the local block needs about "
            "%.2f GiB (A %dx%d, X %dx%d, Y %dx%d in %s), but only %.2f GiB "
            "of %.2f GiB were free: use more MPI processes or a smaller M/N",
            kernel_name(), CUDA_DEVICE_ID, (double)need / BYTES_PER_GIB,
            m_loc, lda, n_loc, ldx, m_loc, ldy, SCALAR_NAME,
            (double)free_b / BYTES_PER_GIB, (double)total_b / BYTES_PER_GIB);
    }
    CUDA_CHECK(err);

    /* A non cambia mai fra un'invocazione e l'altra: la H2D avviene una volta
     * sola, qui nel preprocessing. X e Y sono gia' dimensionate sopra, quindi
     * nessuna cudaMalloc puo' cadere nella prima repetition, neppure con
     * --warmup 0. */
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dA_loc, A_loc, bytes_A, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemset(local_gemm_context->dY_loc_part, 0, nonzero(bytes_Y)));

    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_stop));

    /* Il piano di tiling si calcola QUI, e non in launch_smem_kernel, per una
     * ragione di misura e non di stile.
     *
     * launch_smem_kernel viene invocata fra le due cudaEventRecord. Se ci si
     * infilassero cudaGetDeviceProperties e le query all'occupancy API, la GPU
     * resterebbe ferma con lo stream vuoto mentre l'host calcola, e quel tempo
     * finirebbe dentro t_kernel: si misurerebbe il pianificatore invece del
     * kernel. Qui invece k e n_loc sono gia' noti e fissi per tutta
     * l'esecuzione, il piano vale per tutte le repetition, e il suo costo
     * compare in t_setup, che e' il posto onesto dove farlo stare.
     *
     * Va dopo cudaSetDevice / cudaFree(0): prima di quelli il contesto non e'
     * inizializzato e le query cadrebbero nel vuoto. */
    local_gemm_context->plan = plan_for_k(k, n_loc);

    CUDA_CHECK(cudaDeviceSynchronize());

    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t *RESTRICT X_loc, int ldx, scalar_t *RESTRICT Y, int ldy) {

    const int m_loc = local_gemm_context->m_loc, n_loc = local_gemm_context->n_loc, k = local_gemm_context->k;
    float ms = 0.0f;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("cuda_warp_smem: leading dimensions changed between calls " "(ldx %d -> %d, ldy %d -> %d)", local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX_loc, X_loc, (size_t)n_loc * (size_t)ldx * sizeof(scalar_t), cudaMemcpyHostToDevice));

    /* Fra i due cudaEventRecord non deve esserci NIENTE che faccia lavorare
     * l'host: solo il lancio, che e' asincrono, e la cudaGetLastError che ne
     * legge l'esito. Il piano di tiling e' gia' pronto nel contesto. */
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_start, 0));
    if (m_loc > 0 && k > 0) {
        launch_smem_kernel(&local_gemm_context->plan, m_loc, n_loc, k, local_gemm_context->dA_loc, local_gemm_context->lda, local_gemm_context->dX_loc, ldx, local_gemm_context->dY_loc_part, ldy);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_stop, 0));

    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y, local_gemm_context->dY_loc_part, (size_t)m_loc * (size_t)ldy * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventSynchronize(local_gemm_context->ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_start, local_gemm_context->ev_stop));
    local_gemm_context->t_last = (double)ms * 1.0e-3;
}

void local_gemm_destroy(local_gemm_t *local_gemm_context) {
    if (local_gemm_context == NULL)
        return;
    if (local_gemm_context->dA_loc != NULL) cudaFree(local_gemm_context->dA_loc);
    if (local_gemm_context->dX_loc != NULL) cudaFree(local_gemm_context->dX_loc);
    if (local_gemm_context->dY_loc_part != NULL) cudaFree(local_gemm_context->dY_loc_part);

    cudaEventDestroy(local_gemm_context->ev_start);
    cudaEventDestroy(local_gemm_context->ev_stop);

    xfree(local_gemm_context);
}

double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context) {
    return (local_gemm_context != NULL) ? local_gemm_context->t_last : -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context) {
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup : 0.0;
}

int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context) {
    return (local_gemm_context != NULL) ? local_gemm_context->plan.blocks_per_sm : -1;
}

int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context) {
    return (local_gemm_context != NULL) ? local_gemm_context->plan.x_rows_per_tile : -1;
}

/* I thread di CPU non sono un concetto di questo backend: il parallelismo qui
 * e' quello della GPU, gia' descritto dalle due funzioni sopra. Sentinella
 * negativa, come per tutto cio' che il backend non ha. */
int local_gemm_threads(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

/* Il nome porta padding, dimensione del blocco, granularita' e budget quando non
 * sono quelli di default: nella tabella dei risultati le varianti non possono
 * essere confuse fra loro, e uno sweep resta leggibile nel CSV.
 *
 * Il budget DERIVATO non aggiunge suffisso, perche' e' il default: le righe
 * cosi' etichettate restano pero' distinguibili da quelle raccolte prima di
 * questa modifica grazie alle colonne x_rows_per_tile e blocks_per_sm, che
 * prima non esistevano. */
#define SCPA_STR_(x) #x
#define SCPA_STR(x)  SCPA_STR_(x)

#if SCPA_SMEM_PAD == 1
#define SCPA_PAD_SUFFIX ""
#else
#define SCPA_PAD_SUFFIX "(pad" SCPA_STR(SCPA_SMEM_PAD) ")"
#endif

#if SCPA_BLOCK_THREADS == 256
#define SCPA_BLK_SUFFIX ""
#else
#define SCPA_BLK_SUFFIX "(blk" SCPA_STR(SCPA_BLOCK_THREADS) ")"
#endif

#if SCPA_TILE_GRANULARITY == 32
#define SCPA_GRAN_SUFFIX ""
#else
#define SCPA_GRAN_SUFFIX "(g" SCPA_STR(SCPA_TILE_GRANULARITY) ")"
#endif

#ifdef SCPA_SMEM_BUDGET_BYTES
#define SCPA_BUD_SUFFIX "(bud" SCPA_STR(SCPA_SMEM_BUDGET_BYTES) ")"
#else
#define SCPA_BUD_SUFFIX ""
#endif

const char *kernel_name(void)
{
    return "cuda_warp_smem" SCPA_PAD_SUFFIX SCPA_BLK_SUFFIX
           SCPA_GRAN_SUFFIX SCPA_BUD_SUFFIX;
}
