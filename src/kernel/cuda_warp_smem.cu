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
 * TJ x k di X in shared memory e i suoi 8 warp lo riusano: il traffico verso
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
 * legge xs[(j0+l)*row_stride + c], quindi fra due lane consecutive ci sono
 * row_stride scalari, cioe' 2*row_stride banchi.
 *
 *   row_stride = k = 32   ->  64 banchi = 0 (mod 32): tutte e 32 le lane
 *                             cadono sullo stesso banco. Conflitto a 32 vie,
 *                             l'accesso viene serializzato 32 volte.
 *   row_stride = k + 1    ->  66 banchi = 2 (mod 32): lo stride torna dispari
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
 * Perche' TJ si sceglie a runtime
 * ---------------------------------------------------------------------------
 * Il tile in shared memory pesa TJ * (k + SMEM_PAD) * sizeof(scalar_t), e k e'
 * noto solo a runtime. TJ viene quindi scelto dal budget di shared memory per
 * blocco (SMEM_BUDGET_BYTES) e la memoria e' allocata dinamicamente al lancio.
 * Il budget e' 16 KiB e non i 48 KiB massimi per un motivo di occupancy: la
 * Turing ha 64 KiB di shared per SM, quindi 16 KiB per blocco lasciano
 * risiedere 4 blocchi (1024 thread, 32 warp) per SM. Con 48 KiB ne resterebbe
 * uno solo, cioe' 8 warp: si guadagnerebbe traffico L2 e si perderebbe molto
 * piu' parallelismo di quanto si guadagna.
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
 * non la sua DIMENSIONE, che dipende solo da tj e k: cambiare BLOCK non altera
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

/* Budget di shared memory per blocco (vedi la nota sull'occupancy sopra) e
 * limite architetturale oltre il quale il lancio fallirebbe. */
#define SMEM_BUDGET_BYTES 16384
#define SMEM_MAX_BYTES    49152
#define TJ_MAX            512

/* ---------------------------------------------------------------------------
 * Kernel specializzato: K e' una costante di compilazione
 * ------------------------------------------------------------------------ */
template<int K>
static __global__ void smem_kernel_fixed(int m_loc, int n_loc, int tj,
                                         const scalar_t *__restrict__ A,
                                         int lda,
                                         const scalar_t *__restrict__ X,
                                         int ldx,
                                         scalar_t *__restrict__ Y,
                                         int ldy)
{
    /* Shared memory dinamica: la dimensione in byte la fissa il lancio (vedi
     * launch_smem_kernel), il tipo lo fissa questa dichiarazione. */
    extern __shared__ scalar_t xs[];
    const int row_stride = K + SCPA_SMEM_PAD;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const long long row = (long long)blockIdx.x * WARPS_PER_BLOCK
                          + (threadIdx.x >> 5);
    /* Uniforme sul warp: row non dipende da lane. Serve piu' sotto per la
     * maschera delle shuffle. */
    const bool active = (row < m_loc);
    const scalar_t *const arow =
        A + (size_t)(active ? row : 0) * (size_t)lda;
    scalar_t acc[K];
    int c, j, j0, e, offset;

#pragma unroll
    for (c = 0; c < K; ++c)
        acc[c] = (scalar_t)0;

    for (j0 = 0; j0 < n_loc; j0 += tj) {
        const int jcount = (n_loc - j0 < tj) ? (n_loc - j0) : tj;

        /* Prima barriera: nessuno sovrascrive il tile finche' tutti i warp
         * del blocco hanno finito di consumare quello precedente. */
        __syncthreads();

        /* Staging cooperativo: i 256 thread del blocco si spartiscono le
         * jcount x K letture. Thread consecutivi hanno cc consecutivo, quindi
         * leggono elementi contigui della stessa riga di X: coalescente. */
        for (e = threadIdx.x; e < jcount * K; e += BLOCK_THREADS) {
            const int jj = e / K;
            const int cc = e - jj * K;
            xs[jj * row_stride + cc] =
                X[(size_t)(j0 + jj) * (size_t)ldx + cc];
        }

        /* Seconda barriera: il tile e' completo e visibile a tutto il blocco. */
        __syncthreads();

        if (active) {
            /* Identico al cuore di cuda_warp, con xrow che ora punta in
             * shared invece che in globale: la lane l prende j = l, l+32, ...
             * quindi le letture di A restano perfettamente coalescenti e ogni
             * valore di A letto viene riusato per tutti e K gli accumulatori. */
            for (j = lane; j < jcount; j += WARP_SIZE) {
                const scalar_t a = arow[j0 + j];
                const scalar_t *const xrow = xs + j * row_stride;
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
        scalar_t *const yrow = Y + (size_t)row * (size_t)ldy;
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
 * colonne del passo corrente, ed e' minuscolo: TJ x (4 + SMEM_PAD).
 * Non c'e' nessun limite superiore su k. */
static __global__ void smem_kernel_runtime(int m_loc, int n_loc, int k, int tj,
                                           const scalar_t *__restrict__ A,
                                           int lda,
                                           const scalar_t *__restrict__ X,
                                           int ldx,
                                           scalar_t *__restrict__ Y,
                                           int ldy)
{
    /* Shared memory dinamica: la dimensione in byte la fissa il lancio (vedi
     * launch_smem_kernel), il tipo lo fissa questa dichiarazione. */
    extern __shared__ scalar_t xs[];
    const int row_stride = RUNTIME_TILE + SCPA_SMEM_PAD;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const long long row = (long long)blockIdx.x * WARPS_PER_BLOCK
                          + (threadIdx.x >> 5);
    const bool active = (row < m_loc);
    const scalar_t *const arow =
        A + (size_t)(active ? row : 0) * (size_t)lda;
    scalar_t acc[RUNTIME_TILE];
    int c0, q, j, j0, e, offset;

    /* k, n e tj sono uniformi sul blocco: tutti i thread fanno lo stesso
     * numero di giri, quindi le __syncthreads() qui sotto sono raggiunte da
     * tutti anche nell'ultimo blocco parzialmente pieno. */
    for (c0 = 0; c0 < k; c0 += RUNTIME_TILE) {
#pragma unroll
        for (q = 0; q < RUNTIME_TILE; ++q)
            acc[q] = (scalar_t)0;

        for (j0 = 0; j0 < n_loc; j0 += tj) {
            const int jcount = (n_loc - j0 < tj) ? (n_loc - j0) : tj;

            __syncthreads();

            for (e = threadIdx.x; e < jcount * RUNTIME_TILE;
                 e += BLOCK_THREADS) {
                const int jj = e / RUNTIME_TILE;
                const int cc = e - jj * RUNTIME_TILE;
                /* Le colonne oltre k si azzerano: cosi' il cuore del calcolo
                 * resta senza rami e il resto di k costa solo la guardia
                 * sulla scrittura finale. */
                xs[jj * row_stride + cc] =
                    (c0 + cc < k)
                        ? X[(size_t)(j0 + jj) * (size_t)ldx + c0 + cc]
                        : (scalar_t)0;
            }

            __syncthreads();

            if (active) {
                for (j = lane; j < jcount; j += WARP_SIZE) {
                    const scalar_t a = arow[j0 + j];
                    const scalar_t *const xrow = xs + j * row_stride;
#pragma unroll
                    for (q = 0; q < RUNTIME_TILE; ++q)
                        acc[q] += a * xrow[q];
                }
            }
        }

        if (active) {
#pragma unroll
            for (offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
                for (q = 0; q < RUNTIME_TILE; ++q)
                    acc[q] += __shfl_down_sync(0xffffffffu, acc[q], offset);
            }

            if (lane == 0) {
                scalar_t *const yrow =
                    Y + (size_t)row * (size_t)ldy + c0;
#pragma unroll
                for (q = 0; q < RUNTIME_TILE; ++q)
                    if (c0 + q < k)
                        yrow[q] = acc[q];
            }
        }
    }
}

/* Numero di righe di X per tile: il piu' grande multiplo di 32 che sta nel
 * budget di shared memory, mai piu' delle righe che X ha davvero. */
static int choose_tj(int row_stride, int n_loc)
{
    const long long fit = (long long)SMEM_BUDGET_BYTES
                          / ((long long)row_stride * (long long)sizeof(scalar_t));
    int tj = (int)(fit - (fit % WARP_SIZE));

    if (tj < WARP_SIZE)
        tj = WARP_SIZE;          /* k grandissimo: almeno un passo di warp */
    if (tj > TJ_MAX)
        tj = TJ_MAX;
    if (n_loc > 0 && n_loc < tj)
        tj = ((n_loc + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return tj;
}

static void launch_smem_kernel(int m_loc, int n_loc, int k,
                               const scalar_t *A, int lda,
                               const scalar_t *X, int ldx,
                               scalar_t *Y, int ldy)
{
    const int blocks = (int)(((long long)m_loc + WARPS_PER_BLOCK - 1)
                             / WARPS_PER_BLOCK);
    const int specialized = (k == 3 || k == 6 || k == 8 || k == 20 || k == 32);
    const int row_stride = (specialized ? k : RUNTIME_TILE) + SCPA_SMEM_PAD;
    const int tj = choose_tj(row_stride, n_loc);
    const size_t smem = (size_t)tj * (size_t)row_stride * sizeof(scalar_t);

    if (smem > (size_t)SMEM_MAX_BYTES)
        die("cuda_warp_smem: il tile richiede %zu byte di shared memory per "
            "blocco, oltre il limite di %d (k=%d, TJ=%d, pad=%d): abbassare "
            "SMEM_BUDGET_BYTES", smem, SMEM_MAX_BYTES, k, tj, SCPA_SMEM_PAD);

    switch (k) {
    case 3:
        smem_kernel_fixed<3><<<blocks, BLOCK_THREADS, smem>>>(m_loc, n_loc, tj, A, lda, X, ldx, Y, ldy);
        break;
    case 6:
        smem_kernel_fixed<6><<<blocks, BLOCK_THREADS, smem>>>(m_loc, n_loc, tj, A, lda, X, ldx, Y, ldy);
        break;
    case 8:
        smem_kernel_fixed<8><<<blocks, BLOCK_THREADS, smem>>>(m_loc, n_loc, tj, A, lda, X, ldx, Y, ldy);
        break;
    case 20:
        smem_kernel_fixed<20><<<blocks, BLOCK_THREADS, smem>>>(m_loc, n_loc, tj, A, lda, X, ldx, Y, ldy);
        break;
    case 32:
        smem_kernel_fixed<32><<<blocks, BLOCK_THREADS, smem>>>(m_loc, n_loc, tj, A, lda, X, ldx, Y, ldy);
        break;
    default:
        smem_kernel_runtime<<<blocks, BLOCK_THREADS, smem>>>(m_loc, n_loc, k, tj, A, lda, X, ldx, Y, ldy);
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
    double t_setup;
    double t_last;
};

static size_t nonzero(size_t bytes)
{
    return (bytes != 0) ? bytes : 1;
}

local_gemm_t *local_gemm_create(int m_loc, int n_loc, int k,
                                const scalar_t *A, int lda,
                                int ldx, int ldy)
{
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
    if (n_loc > 0 && m_loc > 0 && A == NULL)
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
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dA_loc, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(local_gemm_context->dY_loc_part, 0, nonzero(bytes_Y)));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context,
                const scalar_t *RESTRICT X, int ldx,
                scalar_t *RESTRICT Y, int ldy)
{
    const int m_loc = local_gemm_context->m_loc, n_loc = local_gemm_context->n_loc, k = local_gemm_context->k;
    float ms = 0.0f;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("cuda_warp_smem: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX_loc, X,
                              (size_t)n_loc * (size_t)ldx * sizeof(scalar_t),
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_start, 0));
    if (m_loc > 0 && k > 0) {
        launch_smem_kernel(m_loc, n_loc, k, local_gemm_context->dA_loc, local_gemm_context->lda,
                           local_gemm_context->dX_loc, ldx, local_gemm_context->dY_loc_part, ldy);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_stop, 0));

    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y, local_gemm_context->dY_loc_part,
                              (size_t)m_loc * (size_t)ldy * sizeof(scalar_t),
                              cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventSynchronize(local_gemm_context->ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_start, local_gemm_context->ev_stop));
    local_gemm_context->t_last = (double)ms * 1.0e-3;
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{
    if (local_gemm_context == NULL)
        return;
    if (local_gemm_context->dA_loc != NULL) cudaFree(local_gemm_context->dA_loc);
    if (local_gemm_context->dX_loc != NULL) cudaFree(local_gemm_context->dX_loc);
    if (local_gemm_context->dY_loc_part != NULL) cudaFree(local_gemm_context->dY_loc_part);
    cudaEventDestroy(local_gemm_context->ev_start);
    cudaEventDestroy(local_gemm_context->ev_stop);
    xfree(local_gemm_context);
}

double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_last : -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup : 0.0;
}

/* Il nome porta il padding e la dimensione del blocco quando non sono quelli di
 * default: nella tabella dei risultati le varianti non possono essere confuse
 * fra loro, e uno sweep su BLOCK resta leggibile nel CSV. */
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

const char *kernel_name(void)
{
    return "cuda_warp_smem" SCPA_PAD_SUFFIX SCPA_BLK_SUFFIX;
}
