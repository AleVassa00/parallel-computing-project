/* Backend CUDA warp-per-row con suddivisione delle colonne di Y fra warp.
 * Ogni warp calcola un tile di WARP_COL_TILE colonne di una sola riga.
 * Le lane continuano a visitare j=lane,lane+32,..., come in cuda_warp.
 * Tile piu' piccoli riducono gli accumulatori ma ripetono le letture di A.
 */

#include <cuda_runtime.h>
#include <limits.h>

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

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256
#endif
#if BLOCK_THREADS < 32 || BLOCK_THREADS > 1024 || (BLOCK_THREADS % 32) != 0
#error "BLOCK_THREADS deve essere un multiplo di 32 compreso fra 32 e 1024"
#endif
#define WARPS_PER_BLOCK (BLOCK_THREADS / WARP_SIZE)

#ifndef WARP_COL_TILE
#define WARP_COL_TILE 8
#endif
#if WARP_COL_TILE <= 0
#error "WARP_COL_TILE deve essere positivo"
#endif

/* Stesso device e lifecycle di cuda_warp; i rank sul nodo condividono GPU 0. */
#define CUDA_DEVICE_ID 0
#define BYTES_PER_GIB 1073741824.0

/* K>0: specializzazione con numero di colonne noto al compilatore.
 * K=0: fallback con k runtime, senza un limite prefissato alle colonne.
 * Entrambi i percorsi mantengono solo WARP_COL_TILE accumulatori per lane.
 */
template<int K>
static __global__ void warp_tiled_kernel(int m_loc, int n_loc, int k,
                                         const scalar_t *__restrict__ A, int lda,
                                         const scalar_t *__restrict__ X, int ldx,
                                         scalar_t *__restrict__ Y, int ldy)
{
    const int columns = K > 0 ? K : k;
    const long long warps_per_row = 1 + (columns - 1LL) / WARP_COL_TILE;
    const long long global_thread =
        (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long warp_id = global_thread / WARP_SIZE;
    const long long row = warp_id / warps_per_row;
    const long long tile_id = warp_id % warps_per_row;
    const long long c0 = tile_id * WARP_COL_TILE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    scalar_t acc[WARP_COL_TILE];

    /* Uscita uniforme: tutte le 32 lane dei warp validi partecipano alle shuffle,
     * anche nell'ultimo blocco o quando n_loc e' minore di WARP_SIZE. */
    if (row >= m_loc)
        return;

    const scalar_t *arow = A + (size_t)row * (size_t)lda;
#pragma unroll
    for (int q = 0; q < WARP_COL_TILE; ++q)
        acc[q] = (scalar_t)0;

    for (long long j = lane; j < n_loc; j += WARP_SIZE) {
        const scalar_t a = arow[j];
        const scalar_t *xrow = X + (size_t)j * (size_t)ldx + c0;
#pragma unroll
        for (int q = 0; q < WARP_COL_TILE; ++q)
            if (q < columns - c0)
                acc[q] += a * xrow[q];
    }

#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (int q = 0; q < WARP_COL_TILE; ++q)
            acc[q] += __shfl_down_sync(0xffffffffu, acc[q], offset);
    }

    if (lane == 0) {
        scalar_t *yrow = Y + (size_t)row * (size_t)ldy + c0;
#pragma unroll
        for (int q = 0; q < WARP_COL_TILE; ++q)
            if (q < columns - c0)
                yrow[q] = acc[q];
    }
}

static void launch_warp_tiled_kernel(int m_loc, int n_loc, int k,
                                     const scalar_t *A, int lda,
                                     const scalar_t *X, int ldx,
                                     scalar_t *Y, int ldy)
{
    const long long warps_per_row = 1 + (k - 1LL) / WARP_COL_TILE;
    const long long total_warps = (long long)m_loc * warps_per_row;
    const long long blocks = (total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    /* Il grid.x CUDA su sm_75 ammette al massimo INT_MAX blocchi. Non troncare
     * il conteggio in un cast: altrimenti si lascerebbero righe non calcolate. */
    if (blocks > INT_MAX)
        die("cuda_warp_tiled: grid needs %lld blocks, exceeds CUDA grid.x limit", blocks);

    switch (k) {
    case 3:
        warp_tiled_kernel<3><<<(unsigned)blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, A, lda, X, ldx, Y, ldy);
        break;
    case 6:
        warp_tiled_kernel<6><<<(unsigned)blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, A, lda, X, ldx, Y, ldy);
        break;
    case 8:
        warp_tiled_kernel<8><<<(unsigned)blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, A, lda, X, ldx, Y, ldy);
        break;
    case 20:
        warp_tiled_kernel<20><<<(unsigned)blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, A, lda, X, ldx, Y, ldy);
        break;
    case 32:
        warp_tiled_kernel<32><<<(unsigned)blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, A, lda, X, ldx, Y, ldy);
        break;
    default:
        warp_tiled_kernel<0><<<(unsigned)blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, A, lda, X, ldx, Y, ldy);
        break;
    }
}

struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda, ldx, ldy;
    scalar_t *dA_loc, *dX_loc, *dY_loc_part;
    /* Tre coppie di event, non una: la consegna esclude dal tempo ufficiale i
     * trasferimenti da e verso la scheda ma consente di misurarli a parte, e
     * per farlo servono confini propri. Gli event sono l'unico strumento
     * corretto: la D2H parte quando il kernel ha finito, quindi un cronometro
     * sull'host le attribuirebbe anche l'attesa del calcolo. */
    cudaEvent_t ev_h2d_X_start, ev_h2d_X_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_d2h_Y_start, ev_d2h_Y_stop;
    double t_setup;
    double t_last;
    double t_last_h2d_X;   /* H2D di X, ultima invocazione (< 0 = mai) */
    double t_last_d2h_Y;   /* D2H di Y, ultima invocazione (< 0 = mai) */
    /* Scomposizione di t_setup: contesto, VRAM e la sola copia H2D di A.
     * Sommate valgono meno del totale, che comprende anche i controlli sugli
     * argomenti e la creazione degli event. */
    double t_setup_device_init;
    double t_setup_device_alloc;
    double t_setup_h2d_A;
    /* Byte che attraversano davvero il PCIe, per convertire i tempi in banda. */
    size_t bytes_h2d_A;
    size_t bytes_h2d_X_per_call;
    size_t bytes_d2h_Y_per_call;
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
    local_gemm_context->t_last_h2d_X = -1.0;
    local_gemm_context->t_last_d2h_Y = -1.0;

    /* Voce 1 del preprocessing: creazione del contesto CUDA. E' un costo
     * fisso, indipendente dalla taglia, e va tenuto separato dalle altre due
     * per non farlo passare per un costo di trasferimento. */
    const double t_device_init_start = now_seconds();
    CUDA_CHECK(cudaSetDevice(CUDA_DEVICE_ID));
    CUDA_CHECK(cudaFree(0));
    local_gemm_context->t_setup_device_init = now_seconds() - t_device_init_start;

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
    /* Voce 2 del preprocessing: le allocazioni in VRAM. Crescono con la
     * taglia del blocco locale ma non sono un trasferimento. */
    const double t_device_alloc_start = now_seconds();
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
    local_gemm_context->t_setup_device_alloc = now_seconds() - t_device_alloc_start;

    /* A non cambia mai fra un'invocazione e l'altra: la H2D avviene una volta
     * sola, qui nel preprocessing. X e Y sono gia' dimensionate sopra, quindi
     * nessuna cudaMalloc puo' cadere nella prima repetition, neppure con
     * --warmup 0. */
    /* Voce 3 del preprocessing, e l'unica delle tre che e' PCIe: e' il
     * trasferimento che la separazione fra create e local_gemm ha tolto dal
     * cammino misurato, ed e' quindi quello che va discusso a parte. */
    const double t_h2d_A_start = now_seconds();
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dA_loc, A_loc, bytes_A, cudaMemcpyHostToDevice));
    /* Una copia da memoria paginabile puo' ritornare prima che il DMA sia
     * concluso: senza questa attesa il tempo misurato sarebbe quello di
     * accodamento, non quello del trasferimento. */
    CUDA_CHECK(cudaDeviceSynchronize());
    local_gemm_context->t_setup_h2d_A = now_seconds() - t_h2d_A_start;

    /* Byte che attraversano davvero il bus, nelle stesse condizioni in cui
     * sono stati cronometrati: e' cio' che rende i tempi convertibili in
     * banda e confrontabili con il picco del PCIe. */
    local_gemm_context->bytes_h2d_A = (bytes_A > 0) ? bytes_A : 0;
    local_gemm_context->bytes_h2d_X_per_call =
        (n_loc > 0 && k > 0) ? (size_t)n_loc * (size_t)ldx * sizeof(scalar_t) : 0;
    local_gemm_context->bytes_d2h_Y_per_call =
        (m_loc > 0 && k > 0) ? (size_t)m_loc * (size_t)ldy * sizeof(scalar_t) : 0;

    CUDA_CHECK(cudaMemset(local_gemm_context->dY_loc_part, 0, nonzero(bytes_Y)));

    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_kernel_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_kernel_stop));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_h2d_X_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_h2d_X_stop));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_d2h_Y_start));
    CUDA_CHECK(cudaEventCreate(&local_gemm_context->ev_d2h_Y_stop));

    CUDA_CHECK(cudaDeviceSynchronize());

    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t *RESTRICT X_loc, int ldx, scalar_t *RESTRICT Y_loc_part, int ldy) {

    const int m_loc = local_gemm_context->m_loc, n_loc = local_gemm_context->n_loc, k = local_gemm_context->k;

    float ms = 0.0f;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("cuda_warp_tiled: leading dimensions changed between calls " "(ldx %d -> %d, ldy %d -> %d)", local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    /* H2D di X: dentro l'invocazione perche' X e' il risultato del Bcast
     * appena concluso, ma delimitata dai suoi event per poterla sottrarre
     * dal tempo ufficiale e discuterla a parte. I record ci sono anche
     * quando non c'e' niente da copiare: cosi' il tempo e' definito
     * (circa zero) invece che indefinito. */
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_start, 0));
    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX_loc, X_loc, (size_t)n_loc * (size_t)ldx * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_stop, 0));

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_start, 0));
    if (m_loc > 0 && k > 0) {
        launch_warp_tiled_kernel(m_loc, n_loc, k, local_gemm_context->dA_loc, local_gemm_context->lda, local_gemm_context->dX_loc, ldx, local_gemm_context->dY_loc_part, ldy);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_stop, 0));

    /* D2H di Y, l'altra copia esclusa dal tempo ufficiale. L'event di
     * partenza viene accodato dopo il kernel, quindi fra i due record
     * resta la sola copia e non l'attesa del calcolo. */
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_start, 0));
    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y_loc_part, local_gemm_context->dY_loc_part, (size_t)m_loc * (size_t)ldy * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_stop, 0));

    /* L'ultimo event accodato e' quello della D2H: aspettare lui vuol dire
     * aspettare tutto il lavoro di questa invocazione. */
    CUDA_CHECK(cudaEventSynchronize(local_gemm_context->ev_d2h_Y_stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_kernel_start, local_gemm_context->ev_kernel_stop));
    local_gemm_context->t_last = (double)ms * 1.0e-3;

    /* Le due voci da discutere a parte. Insieme a t_last e a cio' che
     * avanza (overhead del runtime) ricostruiscono il tempo dell'intera
     * invocazione misurato dal chiamante. */
    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_h2d_X_start, local_gemm_context->ev_h2d_X_stop));
    local_gemm_context->t_last_h2d_X = (double)ms * 1.0e-3;

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_d2h_Y_start, local_gemm_context->ev_d2h_Y_stop));
    local_gemm_context->t_last_d2h_Y = (double)ms * 1.0e-3;
}

void local_gemm_destroy(local_gemm_t *local_gemm_context) {
    if (local_gemm_context == NULL)
        return;
    if (local_gemm_context->dA_loc != NULL) cudaFree(local_gemm_context->dA_loc);
    if (local_gemm_context->dX_loc != NULL) cudaFree(local_gemm_context->dX_loc);
    if (local_gemm_context->dY_loc_part != NULL) cudaFree(local_gemm_context->dY_loc_part);
    cudaEventDestroy(local_gemm_context->ev_kernel_start);
    cudaEventDestroy(local_gemm_context->ev_kernel_stop);
    cudaEventDestroy(local_gemm_context->ev_h2d_X_start);
    cudaEventDestroy(local_gemm_context->ev_h2d_X_stop);
    cudaEventDestroy(local_gemm_context->ev_d2h_Y_start);
    cudaEventDestroy(local_gemm_context->ev_d2h_Y_stop);
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

/* ---------------------------------------------------------------------------
 * Tempi esclusi dalla misura ufficiale, misurati per essere discussi a parte
 * ------------------------------------------------------------------------- */
double local_gemm_last_h2d_X_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_last_h2d_X : -1.0;
}

double local_gemm_last_d2h_Y_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_last_d2h_Y : -1.0;
}

size_t local_gemm_bytes_h2d_A(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->bytes_h2d_A : 0;
}

size_t local_gemm_bytes_h2d_X_per_call(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->bytes_h2d_X_per_call : 0;
}

size_t local_gemm_bytes_d2h_Y_per_call(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->bytes_d2h_Y_per_call : 0;
}

double local_gemm_setup_device_init_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup_device_init : 0.0;
}

double local_gemm_setup_device_alloc_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup_device_alloc : 0.0;
}

double local_gemm_setup_h2d_A_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup_h2d_A : 0.0;
}

/* Questo backend non pianifica nessun tiling in shared memory: non ha il
 * concetto, e il sentinella negativo lo dice invece di inventare un numero. */
int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

/* Il nome porta la dimensione del blocco quando non e' quella di default: nel
 * CSV le righe di uno sweep su BLOCK devono restare distinguibili fra loro. */
#define STR_(x) #x
#define STR(x)  STR_(x)

#if BLOCK_THREADS == 256
#define BLK_SUFFIX ""
#else
#define BLK_SUFFIX "(blk" STR(BLOCK_THREADS) ")"
#endif

const char *kernel_name(void)
{
    return "cuda_warp_tiled(tile" STR(WARP_COL_TILE) ")" BLK_SUFFIX;
}
