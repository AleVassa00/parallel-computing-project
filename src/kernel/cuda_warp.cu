/* Backend CUDA warp-per-row.
 *
 * Un warp calcola una riga di Y. La lane l visita j=l,l+32,...: le letture
 * della riga di A sono coalescenti e ogni valore di A viene riusato per tutte
 * le colonne del multivettore. I cinque k richiesti sono template distinti;
 * ogni altro k usa un fallback runtime a tile, senza limiti sul valore di k.
 */

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdlib.h>

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
#define BLOCK_THREADS 256
#define WARPS_PER_BLOCK (BLOCK_THREADS / WARP_SIZE)
#define RUNTIME_TILE 4
#define BYTES_PER_GIB 1073741824.0

template<int K>
static __global__ void warp_kernel_fixed(int m, int n,
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
    if (warp_id >= m)
        return;

    arow = A + (size_t)warp_id * (size_t)lda;

#pragma unroll
    for (c = 0; c < K; ++c)
        acc[c] = (scalar_t)0;

    for (j = lane; j < n; j += WARP_SIZE) {
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
static __global__ void warp_kernel_runtime(int m, int n, int k,
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

    if (warp_id >= m)
        return;

    arow = A + (size_t)warp_id * (size_t)lda;
    mask = 0xffffffffu;   /* stesso motivo del kernel template qui sopra */

    for (c0 = 0; c0 < k; c0 += RUNTIME_TILE) {
#pragma unroll
        for (q = 0; q < RUNTIME_TILE; ++q)
            acc[q] = (scalar_t)0;

        for (j = lane; j < n; j += WARP_SIZE) {
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

static void launch_warp_kernel(int m, int n, int k,
                               const scalar_t *A, int lda,
                               const scalar_t *X, int ldx,
                               scalar_t *Y, int ldy)
{
    const int blocks = (int)(((long long)m + WARPS_PER_BLOCK - 1)
                             / WARPS_PER_BLOCK);

    switch (k) {
    case 3:
        warp_kernel_fixed<3><<<blocks, BLOCK_THREADS>>>(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 6:
        warp_kernel_fixed<6><<<blocks, BLOCK_THREADS>>>(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 8:
        warp_kernel_fixed<8><<<blocks, BLOCK_THREADS>>>(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 20:
        warp_kernel_fixed<20><<<blocks, BLOCK_THREADS>>>(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 32:
        warp_kernel_fixed<32><<<blocks, BLOCK_THREADS>>>(m, n, A, lda, X, ldx, Y, ldy);
        break;
    default:
        warp_kernel_runtime<<<blocks, BLOCK_THREADS>>>(m, n, k,
                                                       A, lda, X, ldx, Y, ldy);
        break;
    }
}

struct local_gemm_ctx {
    int m, n, k;
    int lda, ldx, ldy;
    scalar_t *dA, *dX, *dY;
    cudaEvent_t ev_start, ev_stop;
    int device;
    double t_setup;
    double t_last;
};

static int pick_device(void)
{
    const char *env, *local_rank_env, *local_size_env;
    int ndev = 0;
    long id, local_rank, local_size;

    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1)
        die("cuda_warp: no CUDA device available");

    local_rank_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (local_rank_env == NULL)
        local_rank_env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (local_rank_env == NULL)
        local_rank_env = getenv("SLURM_LOCALID");
    local_rank = (local_rank_env != NULL) ? strtol(local_rank_env, NULL, 10) : 0;
    if (local_rank < 0)
        local_rank = 0;

    local_size_env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
    if (local_size_env == NULL)
        local_size_env = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
    if (local_size_env == NULL)
        local_size_env = getenv("SLURM_NTASKS_PER_NODE");
    local_size = (local_size_env != NULL) ? strtol(local_size_env, NULL, 10) : 1;

    if (local_rank == 0 && local_size > ndev)
        fprintf(stderr,
                "warning: %ld local MPI ranks share %d CUDA device(s); "
                "correctness is preserved, but official/kernel GPU timing "
                "does not represent one-rank-per-GPU throughput\n",
                local_size, ndev);

    env = getenv("SCPA_CUDA_DEVICE");
    id = (env != NULL) ? strtol(env, NULL, 10) : local_rank;
    if (id < 0)
        id = 0;
    return (int)(id % ndev);
}

static size_t nonzero(size_t bytes)
{
    return (bytes != 0) ? bytes : 1;
}

local_gemm_t *local_gemm_create(int m, int n, int k,
                                const scalar_t *A, int lda,
                                int ldx, int ldy)
{
    local_gemm_t *ctx;
    size_t bytes_A, bytes_X, bytes_Y, need, free_b = 0, total_b = 0;
    const double t0 = now_seconds();

    if (m < 0 || n < 0 || k < 0)
        die("local_gemm_create: invalid local block %dx%d with k=%d", m, n, k);
    if (lda < n)
        die("local_gemm_create: lda %d is smaller than n %d", lda, n);
    if (ldx < k || ldy < k)
        die("local_gemm_create: ldx %d and ldy %d must both be at least k=%d",
            ldx, ldy, k);
    if (n > 0 && m > 0 && A == NULL)
        die("local_gemm_create: A is NULL for a non-empty %dx%d block", m, n);

    ctx = (local_gemm_t *)xmalloc(sizeof *ctx);
    ctx->m = m;
    ctx->n = n;
    ctx->k = k;
    ctx->lda = lda;
    ctx->ldx = ldx;
    ctx->ldy = ldy;
    ctx->dA = NULL;
    ctx->dX = NULL;
    ctx->dY = NULL;
    ctx->t_last = -1.0;

    ctx->device = pick_device();
    CUDA_CHECK(cudaSetDevice(ctx->device));
    CUDA_CHECK(cudaFree(0));

    bytes_A = (size_t)m * (size_t)lda * sizeof(scalar_t);
    bytes_X = (size_t)n * (size_t)ldx * sizeof(scalar_t);
    bytes_Y = (size_t)m * (size_t)ldy * sizeof(scalar_t);
    need = bytes_A + bytes_X + bytes_Y;

    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (64u << 20) > free_b)
        die("cuda_warp: device %d has %.2f GiB free out of %.2f GiB, "
            "but the local block needs about %.2f GiB "
            "(A %dx%d, X %dx%d, Y %dx%d in %s): use more MPI processes "
            "or a smaller M/N",
            ctx->device, (double)free_b / BYTES_PER_GIB,
            (double)total_b / BYTES_PER_GIB, (double)need / BYTES_PER_GIB,
            m, lda, n, ldx, m, ldy, SCALAR_NAME);

    CUDA_CHECK(cudaMalloc((void **)&ctx->dA, nonzero(bytes_A)));
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(ctx->dA, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&ctx->dX, nonzero(bytes_X)));
    CUDA_CHECK(cudaMalloc((void **)&ctx->dY, nonzero(bytes_Y)));
    CUDA_CHECK(cudaMemset(ctx->dY, 0, nonzero(bytes_Y)));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_start));
    CUDA_CHECK(cudaEventCreate(&ctx->ev_stop));
    CUDA_CHECK(cudaDeviceSynchronize());

    ctx->t_setup = now_seconds() - t0;
    return ctx;
}

void local_gemm(local_gemm_t *ctx,
                const scalar_t *SCPA_RESTRICT X, int ldx,
                scalar_t *SCPA_RESTRICT Y, int ldy)
{
    const int m = ctx->m, n = ctx->n, k = ctx->k;
    float ms = 0.0f;

    if (ldx != ctx->ldx || ldy != ctx->ldy)
        die("cuda_warp: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            ctx->ldx, ldx, ctx->ldy, ldy);

    if (n > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(ctx->dX, X,
                              (size_t)n * (size_t)ldx * sizeof(scalar_t),
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ctx->ev_start, 0));
    if (m > 0 && k > 0) {
        launch_warp_kernel(m, n, k, ctx->dA, ctx->lda,
                           ctx->dX, ldx, ctx->dY, ldy);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop, 0));

    if (m > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y, ctx->dY,
                              (size_t)m * (size_t)ldy * sizeof(scalar_t),
                              cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventSynchronize(ctx->ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ctx->ev_start, ctx->ev_stop));
    ctx->t_last = (double)ms * 1.0e-3;
}

void local_gemm_destroy(local_gemm_t *ctx)
{
    if (ctx == NULL)
        return;
    if (ctx->dA != NULL) cudaFree(ctx->dA);
    if (ctx->dX != NULL) cudaFree(ctx->dX);
    if (ctx->dY != NULL) cudaFree(ctx->dY);
    cudaEventDestroy(ctx->ev_start);
    cudaEventDestroy(ctx->ev_stop);
    xfree(ctx);
}

double local_gemm_last_compute_seconds(const local_gemm_t *ctx)
{
    return (ctx != NULL) ? ctx->t_last : -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *ctx)
{
    return (ctx != NULL) ? ctx->t_setup : 0.0;
}

const char *kernel_name(void)
{
    return "cuda_warp";
}
