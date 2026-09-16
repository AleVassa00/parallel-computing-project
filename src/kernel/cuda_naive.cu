
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

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256
#endif
#if BLOCK_THREADS < 32 || BLOCK_THREADS > 1024 || (BLOCK_THREADS % 32) != 0
#error "BLOCK_THREADS deve essere un multiplo di 32 compreso fra 32 e 1024"
#endif

#define CUDA_DEVICE_ID 0

#define BYTES_PER_GIB 1073741824.0

static __global__ void naive_kernel(int m_loc, int n_loc, int k, const scalar_t *__restrict__ A, int lda, const scalar_t *__restrict__ X, int ldx, scalar_t *__restrict__ Y, int ldy)
{
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)m_loc * (long long)k;

    int i, c, j;

    const scalar_t *arow;

    scalar_t acc = (scalar_t)0;

    if (tid >= total)
        return;

    i = (int)(tid / k);
    c = (int)(tid % k);

    arow = A + (size_t)i * (size_t)lda;

    for (j = 0; j < n_loc; j++)
        acc += arow[j] * X[(size_t)j * (size_t)ldx + c];

    Y[(size_t)i * (size_t)ldy + c] = acc;
}

struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda;
    int ldx, ldy;
    scalar_t *dA_loc;
    scalar_t *dX_loc;
    scalar_t *dY_loc_part;

    cudaEvent_t ev_h2d_X_start, ev_h2d_X_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_d2h_Y_start, ev_d2h_Y_stop;
    double t_setup;
    double t_last;
    double t_last_h2d_X;
    double t_last_d2h_Y;

    double t_setup_device_init;
    double t_setup_device_alloc;
    double t_setup_h2d_A;

    size_t bytes_h2d_A;
    size_t bytes_h2d_X_per_call;
    size_t bytes_d2h_Y_per_call;
};

static size_t nonzero(size_t bytes)
{

    return (bytes != 0) ? bytes : 1;
}

local_gemm_t *local_gemm_create(int m_loc, int n_loc, int k, const scalar_t *A_loc, int lda, int ldx, int ldy)
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

    const double t_device_init_start = now_seconds();
    CUDA_CHECK(cudaSetDevice(CUDA_DEVICE_ID));

    CUDA_CHECK(cudaFree(0));
    local_gemm_context->t_setup_device_init = now_seconds() - t_device_init_start;

    bytes_A = (size_t)m_loc * (size_t)lda * sizeof(scalar_t);
    bytes_X = (size_t)n_loc * (size_t)ldx * sizeof(scalar_t);
    bytes_Y = (size_t)m_loc * (size_t)ldy * sizeof(scalar_t);

    need = bytes_A + bytes_X + bytes_Y;

    const double t_device_alloc_start = now_seconds();
    err = cudaMalloc((void **)&local_gemm_context->dA_loc, nonzero(bytes_A));
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&local_gemm_context->dX_loc, nonzero(bytes_X));
    if (err == cudaSuccess)
        err = cudaMalloc((void **)&local_gemm_context->dY_loc_part, nonzero(bytes_Y));

    if (err == cudaErrorMemoryAllocation) {

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

    const double t_h2d_A_start = now_seconds();
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dA_loc, A_loc, bytes_A, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    local_gemm_context->t_setup_h2d_A = now_seconds() - t_h2d_A_start;

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

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t *RESTRICT X_loc, int ldx, scalar_t *RESTRICT Y_loc_part, int ldy)
{
    const int m_loc = local_gemm_context->m_loc, n_loc = local_gemm_context->n_loc, k = local_gemm_context->k;
    const long long threads = (long long)m_loc * (long long)k;
    float ms = 0.0f;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("cuda_naive: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_start, 0));
    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX_loc, X_loc, (size_t)n_loc * (size_t)ldx * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_stop, 0));

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_start, 0));
    if (threads > 0) {
        const int blocks = (int)((threads + BLOCK_THREADS - 1) / BLOCK_THREADS);
        naive_kernel<<<blocks, BLOCK_THREADS>>>(m_loc, n_loc, k, local_gemm_context->dA_loc, local_gemm_context->lda, local_gemm_context->dX_loc, ldx, local_gemm_context->dY_loc_part, ldy);

        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_stop, 0));

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_start, 0));
    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y_loc_part, local_gemm_context->dY_loc_part, (size_t)m_loc * (size_t)ldy * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_stop, 0));

    CUDA_CHECK(cudaEventSynchronize(local_gemm_context->ev_d2h_Y_stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_kernel_start, local_gemm_context->ev_kernel_stop));
    local_gemm_context->t_last = (double)ms * 1.0e-3;

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_h2d_X_start, local_gemm_context->ev_h2d_X_stop));
    local_gemm_context->t_last_h2d_X = (double)ms * 1.0e-3;

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_d2h_Y_start, local_gemm_context->ev_d2h_Y_stop));
    local_gemm_context->t_last_d2h_Y = (double)ms * 1.0e-3;
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{
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

#define STR_(x) #x
#define STR(x)  STR_(x)

#if BLOCK_THREADS == 256
#define BLK_SUFFIX ""
#else
#define BLK_SUFFIX "(blk" STR(BLOCK_THREADS) ")"
#endif

const char *kernel_name(void)
{
    return "cuda_naive" BLK_SUFFIX;
}
