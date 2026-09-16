
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

#ifndef BLOCK_THREADS
#define BLOCK_THREADS 256
#endif

#if BLOCK_THREADS < 32 || BLOCK_THREADS > 1024 || (BLOCK_THREADS % 32) != 0
#error "BLOCK_THREADS deve essere un multiplo di 32 compreso fra 32 e 1024"
#endif

#define WARPS_PER_BLOCK (BLOCK_THREADS / WARP_SIZE)

#define RUNTIME_TILE 4

#define CUDA_DEVICE_ID 0

#define BYTES_PER_GIB 1073741824.0

#ifndef SMEM_PAD
#define SMEM_PAD 1
#endif
#if SMEM_PAD < 0
#error "SMEM_PAD deve essere >= 0"
#endif

#define SMEM_MAX_BYTES    49152

#ifndef TILE_GRANULARITY
#define TILE_GRANULARITY 32
#endif
#if TILE_GRANULARITY < 1
#error "TILE_GRANULARITY deve essere >= 1"
#endif

template<int K>
static __global__ void smem_kernel_fixed(int m_loc, int n_loc, int x_rows_per_tile, const scalar_t *__restrict__ A_loc, int lda, const scalar_t *__restrict__ X_loc, int ldx, scalar_t *__restrict__ Y_loc_part, int ldy) {

    extern __shared__ scalar_t X_tile[];

    const int tile_row_stride = K + SMEM_PAD;

    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const long long row = (long long)blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x >> 5);

    const bool active = (row < m_loc);

    const scalar_t *const arow = A_loc + (size_t)(active ? row : 0) * (size_t)lda;

    scalar_t acc[K];

    int c, j, tile_first_row, load_index, offset;

#pragma unroll
    for (c = 0; c < K; ++c)
        acc[c] = (scalar_t)0;

    for (tile_first_row = 0; tile_first_row < n_loc; tile_first_row += x_rows_per_tile) {

        const int rows_in_tile = (n_loc - tile_first_row < x_rows_per_tile) ? (n_loc - tile_first_row) : x_rows_per_tile;

        __syncthreads();

        for (load_index = threadIdx.x; load_index < rows_in_tile * K; load_index += BLOCK_THREADS) {

            const int tile_row = load_index / K;
            const int tile_col = load_index - tile_row * K;

            X_tile[tile_row * tile_row_stride + tile_col] =
                X_loc[(size_t)(tile_first_row + tile_row) * (size_t)ldx + tile_col];
        }

        __syncthreads();

        if (active) {

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

static __global__ void smem_kernel_runtime(int m_loc, int n_loc, int k, int x_rows_per_tile, const scalar_t *__restrict__ A_loc, int lda, const scalar_t *__restrict__ X_loc, int ldx, scalar_t *__restrict__ Y_loc_part, int ldy) {

    extern __shared__ scalar_t X_tile[];

    const int tile_row_stride = RUNTIME_TILE + SMEM_PAD;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    const long long row = (long long)blockIdx.x * WARPS_PER_BLOCK
                          + (threadIdx.x >> 5);
    const bool active = (row < m_loc);
    const scalar_t *const arow =
        A_loc + (size_t)(active ? row : 0) * (size_t)lda;
    scalar_t acc[RUNTIME_TILE];
    int col_tile_start, col_in_tile, j, tile_first_row, load_index, offset;

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

typedef struct {
    int    x_rows_per_tile;
    size_t smem_bytes;
    int    blocks_per_sm;
} tile_plan_t;

#ifndef SMEM_BUDGET_BYTES
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

static tile_plan_t plan_tile(const void *kernel, int tile_row_stride, int n_loc) {

    const int g         = TILE_GRANULARITY;
    const int row_bytes = tile_row_stride * (int)sizeof(scalar_t);
    int target, budget, rows, achieved = -1;
    tile_plan_t plan;

    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &target, kernel, BLOCK_THREADS, 0));
    if (target < 1)
        target = 1;

#ifdef SMEM_BUDGET_BYTES
    budget = SMEM_BUDGET_BYTES;
#else
    budget = shared_per_sm() / target;
#endif
    if (budget > SMEM_MAX_BYTES)
        budget = SMEM_MAX_BYTES;

    rows = round_down_g(budget / row_bytes, g);
    if (rows < WARP_SIZE)
        rows = WARP_SIZE;
    if (n_loc > 0 && n_loc < rows)
        rows = round_up_g(n_loc, g);

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
            plan.smem_bytes, SMEM_MAX_BYTES, rows, tile_row_stride, SMEM_PAD);

    plan.x_rows_per_tile = rows;
    plan.blocks_per_sm   = achieved;
    return plan;
}

template<int K>
static tile_plan_t plan_fixed(int n_loc) {
    return plan_tile((const void *)smem_kernel_fixed<K>,
                     K + SMEM_PAD, n_loc);
}

static tile_plan_t plan_for_k(int k, int n_loc) {
    switch (k) {
    case 3:  return plan_fixed<3>(n_loc);
    case 6:  return plan_fixed<6>(n_loc);
    case 8:  return plan_fixed<8>(n_loc);
    case 20: return plan_fixed<20>(n_loc);
    case 32: return plan_fixed<32>(n_loc);
    default: return plan_tile((const void *)smem_kernel_runtime,
                              RUNTIME_TILE + SMEM_PAD, n_loc);
    }
}

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

struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda, ldx, ldy;
    scalar_t *dA_loc, *dX_loc, *dY_loc_part;

    cudaEvent_t ev_h2d_X_start, ev_h2d_X_stop;
    cudaEvent_t ev_kernel_start, ev_kernel_stop;
    cudaEvent_t ev_d2h_Y_start, ev_d2h_Y_stop;
    tile_plan_t plan;
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

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_start, 0));
    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(local_gemm_context->dX_loc, X_loc, (size_t)n_loc * (size_t)ldx * sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_h2d_X_stop, 0));

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_start, 0));
    if (m_loc > 0 && k > 0) {
        launch_smem_kernel(&local_gemm_context->plan, m_loc, n_loc, k, local_gemm_context->dA_loc, local_gemm_context->lda, local_gemm_context->dX_loc, ldx, local_gemm_context->dY_loc_part, ldy);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_kernel_stop, 0));

    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_start, 0));
    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y, local_gemm_context->dY_loc_part, (size_t)m_loc * (size_t)ldy * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(local_gemm_context->ev_d2h_Y_stop, 0));

    CUDA_CHECK(cudaEventSynchronize(local_gemm_context->ev_d2h_Y_stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, local_gemm_context->ev_kernel_start, local_gemm_context->ev_kernel_stop));
    local_gemm_context->t_last = (double)ms * 1.0e-3;

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

double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context) {
    return (local_gemm_context != NULL) ? local_gemm_context->t_last : -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context) {
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

int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context) {
    return (local_gemm_context != NULL) ? local_gemm_context->plan.blocks_per_sm : -1;
}

int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context) {
    return (local_gemm_context != NULL) ? local_gemm_context->plan.x_rows_per_tile : -1;
}

#define STR_(x) #x
#define STR(x)  STR_(x)

#if SMEM_PAD == 1
#define PAD_SUFFIX ""
#else
#define PAD_SUFFIX "(pad" STR(SMEM_PAD) ")"
#endif

#if BLOCK_THREADS == 256
#define BLK_SUFFIX ""
#else
#define BLK_SUFFIX "(blk" STR(BLOCK_THREADS) ")"
#endif

#if TILE_GRANULARITY == 32
#define GRAN_SUFFIX ""
#else
#define GRAN_SUFFIX "(g" STR(TILE_GRANULARITY) ")"
#endif

#ifdef SMEM_BUDGET_BYTES
#define BUD_SUFFIX "(bud" STR(SMEM_BUDGET_BYTES) ")"
#else
#define BUD_SUFFIX ""
#endif

const char *kernel_name(void)
{
    return "cuda_warp_smem" PAD_SUFFIX BLK_SUFFIX
           GRAN_SUFFIX BUD_SUFFIX;
}
