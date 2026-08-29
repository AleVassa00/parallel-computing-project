/* Backend CUDA warp-per-row.
 *
 * Un warp calcola una riga di Y. La lane l visita j=l,l+32,...: le letture
 * della riga di A sono coalescenti e ogni valore di A viene riusato per tutte
 * le colonne del multivettore. I cinque k richiesti sono template distinti;
 * ogni altro k usa un fallback runtime a tile, senza limiti sul valore di k.
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
 * Qui, a differenza del backend naive, il multiplo di 32 e' un requisito di
 * CORRETTEZZA e non solo di efficienza: WARPS_PER_BLOCK determina quante righe
 * di Y elabora un blocco, e la riduzione finale e' un __shfl_down_sync interno
 * al warp. Un blocco non multiplo di 32 spezzerebbe un warp fra due righe.
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

template<int K>
static __global__ void warp_kernel_fixed(int m_loc, int n_loc,
                                         const scalar_t *__restrict__ A,
                                         int lda,
                                         const scalar_t *__restrict__ X,
                                         int ldx,
                                         scalar_t *__restrict__ Y,
                                         int ldy)
{
    const long long global_thread =
        (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long warp_id = global_thread / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    scalar_t acc[K];
    const scalar_t *arow;
    unsigned mask;
    int c, j, offset;

    /* Il test e' uniforme per tutto il warp: nessuna lane valida resta fuori
     * dalle shuffle. Questo copre anche l'ultimo blocco parzialmente usato. */
    if (warp_id >= m_loc)
        return;

    arow = A + (size_t)warp_id * (size_t)lda;

#pragma unroll
    for (c = 0; c < K; ++c)
        acc[c] = (scalar_t)0;

    for (j = lane; j < n_loc; j += WARP_SIZE) {
        const scalar_t a = arow[j];
        const scalar_t *xrow = X + (size_t)j * (size_t)ldx;
#pragma unroll
        for (c = 0; c < K; ++c)
            acc[c] += a * xrow[c];
    }

    /* Maschera piena, non __activemask(). Da Volta in poi le lane di un warp
     * possono divergere e riconvergere in modo indipendente, e la guida CUDA
     * dice esplicitamente che il valore di __activemask() non implica
     * convergenza: usarlo come maschera di una __shfl_*_sync e' un
     * comportamento non definito se due lane ne ottengono una diversa. Qui il
     * return sopra e' UNIFORME sul warp (warp_id non dipende da lane), quindi
     * le 32 lane sono tutte vive e la maschera corretta e' nota a priori. */
    mask = 0xffffffffu;
#pragma unroll
    for (offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
        for (c = 0; c < K; ++c)
            acc[c] += __shfl_down_sync(mask, acc[c], offset);
    }

    if (lane == 0) {
        scalar_t *yrow = Y + (size_t)warp_id * (size_t)ldy;
#pragma unroll
        for (c = 0; c < K; ++c)
            yrow[c] = acc[c];
    }
}

/* Fallback per k arbitrario. Quattro colonne alla volta limitano la pressione
 * sui registri e riusano ogni lettura di A; l'ultimo tile gestisce qualsiasi
 * resto. Non e' richiesta alcuna dimensione massima di k. */
static __global__ void warp_kernel_runtime(int m_loc, int n_loc, int k,
                                           const scalar_t *__restrict__ A,
                                           int lda,
                                           const scalar_t *__restrict__ X,
                                           int ldx,
                                           scalar_t *__restrict__ Y,
                                           int ldy)
{
    const long long global_thread =
        (long long)blockIdx.x * (long long)blockDim.x + threadIdx.x;
    const long long warp_id = global_thread / WARP_SIZE;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    scalar_t acc[RUNTIME_TILE];
    const scalar_t *arow;
    unsigned mask;
    int c0, q, j, offset;

    if (warp_id >= m_loc)
        return;

    arow = A + (size_t)warp_id * (size_t)lda;
    mask = 0xffffffffu;   /* stesso motivo del kernel template qui sopra */

    for (c0 = 0; c0 < k; c0 += RUNTIME_TILE) {
#pragma unroll
        for (q = 0; q < RUNTIME_TILE; ++q)
            acc[q] = (scalar_t)0;

        for (j = lane; j < n_loc; j += WARP_SIZE) {
            const scalar_t a = arow[j];
            const scalar_t *xrow = X + (size_t)j * (size_t)ldx + c0;
#pragma unroll
            for (q = 0; q < RUNTIME_TILE; ++q)
                if (c0 + q < k)
                    acc[q] += a * xrow[q];
        }

#pragma unroll
        for (offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
#pragma unroll
            for (q = 0; q < RUNTIME_TILE; ++q)
                acc[q] += __shfl_down_sync(mask, acc[q], offset);
        }

        if (lane == 0) {
            scalar_t *yrow = Y + (size_t)warp_id * (size_t)ldy + c0;
#pragma unroll
            for (q = 0; q < RUNTIME_TILE; ++q)
                if (c0 + q < k)
                    yrow[q] = acc[q];
        }
    }
}

static void launch_warp_kernel(int m_loc, int n_loc, int k,
                               const scalar_t *A, int lda,
                               const scalar_t *X, int ldx,
                               scalar_t *Y, int ldy)
{
    const int blocks = (int)(((long long)m_loc + WARPS_PER_BLOCK - 1)
                             / WARPS_PER_BLOCK);

    switch (k) {
    case 3:
        warp_kernel_fixed<3><<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, A, lda, X, ldx, Y, ldy);
        break;
    case 6:
        warp_kernel_fixed<6><<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, A, lda, X, ldx, Y, ldy);
        break;
    case 8:
        warp_kernel_fixed<8><<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, A, lda, X, ldx, Y, ldy);
        break;
    case 20:
        warp_kernel_fixed<20><<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, A, lda, X, ldx, Y, ldy);
        break;
    case 32:
        warp_kernel_fixed<32><<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, A, lda, X, ldx, Y, ldy);
        break;
    default:
        warp_kernel_runtime<<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, k,
                                                       A, lda, X, ldx, Y, ldy);
        break;
    }
}

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

    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (64u << 20) > free_b)
        die("cuda_warp: device %d has %.2f GiB free out of %.2f GiB, "
            "but the local block needs about %.2f GiB "
            "(A %dx%d, X %dx%d, Y %dx%d in %s): use more MPI processes "
            "or a smaller M/N",
            CUDA_DEVICE_ID, (double)free_b / BYTES_PER_GIB,
            (double)total_b / BYTES_PER_GIB, (double)need / BYTES_PER_GIB,
            m_loc, lda, n_loc, ldx, m_loc, ldy, SCALAR_NAME);

    CUDA_CHECK(cudaMalloc((void **)&local_gemm_context->dA_loc, nonzero(bytes_A)));
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dA_loc, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&local_gemm_context->dX_loc, nonzero(bytes_X)));
    CUDA_CHECK(cudaMalloc((void **)&local_gemm_context->dY_loc_part, nonzero(bytes_Y)));
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
        die("cuda_warp: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX_loc, X,
                              (size_t)n_loc * (size_t)ldx * sizeof(scalar_t),
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_start, 0));
    if (m_loc > 0 && k > 0) {
        launch_warp_kernel(m_loc, n_loc, k, local_gemm_context->dA_loc, local_gemm_context->lda,
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

/* Il nome porta la dimensione del blocco quando non e' quella di default: nel
 * CSV le righe di uno sweep su BLOCK devono restare distinguibili fra loro. */
#define SCPA_STR_(x) #x
#define SCPA_STR(x)  SCPA_STR_(x)

#if SCPA_BLOCK_THREADS == 256
#define SCPA_BLK_SUFFIX ""
#else
#define SCPA_BLK_SUFFIX "(blk" SCPA_STR(SCPA_BLOCK_THREADS) ")"
#endif

const char *kernel_name(void)
{
    return "cuda_warp" SCPA_BLK_SUFFIX;
}
