/* Backend cuBLAS: RIFERIMENTO ESTERNO, non una proposta.  M4.4.
 *
 * Serve a rispondere alla domanda che all'orale arriva sempre: "perche' avete
 * scritto un kernel invece di chiamare una libreria?". La risposta ha valore
 * solo se il numero della libreria e' stato misurato sulla stessa macchina,
 * con la stessa pipeline, gli stessi trasferimenti e lo stesso cronometro
 * degli altri backend. Per questo cuBLAS entra nel progetto come un backend
 * come gli altri, dietro la stessa local_gemm_create / local_gemm /
 * local_gemm_destroy, e non come uno script a parte.
 *
 * L'attesa e' che vada MALE, ed e' un risultato: cuBLAS e' un GEMM
 * general-purpose, ottimizzato per matrici in cui tutte e tre le dimensioni
 * sono grandi. Qui k <= 32, cioe' il multivettore e' strettissimo: la libreria
 * non riesce ad ammortizzare il proprio percorso di selezione dell'algoritmo,
 * il tiling che sceglie ha molte piu' righe che colonne utili e gran parte del
 * lavoro va sprecato. Un numero basso qui e' l'argomento piu' forte per
 * giustificare l'intero progetto.
 *
 * ---------------------------------------------------------------------------
 * Il trucco riga/colonna: nessuna trasposizione, nessuna copia
 * ---------------------------------------------------------------------------
 * cuBLAS e' COLUMN-MAJOR, tutto il progetto e' ROW-MAJOR. La soluzione non e'
 * trasporre (costerebbe piu' del prodotto) ma reinterpretare: una matrice
 * P x Q row-major con leading dimension ld E' GIA', bit per bit, la matrice
 * Q x P column-major con la stessa ld. Leggere gli stessi byte con l'altra
 * convenzione equivale a trasporre gratis.
 *
 * Quindi, con  _rm = come la vede il progetto  e  _cm = come la vede cuBLAS:
 *
 *     A_rm (m x n, lda)  ==  A_cm (n x m, lda)
 *     X_rm (n x k, ldx)  ==  X_cm (k x n, ldx)
 *     Y_rm (m x k, ldy)  ==  Y_cm (k x m, ldy)
 *
 * e la trasposta del prodotto ribalta l'ordine dei fattori:
 *
 *     Y_cm = (Y_rm)^T = (A_rm * X_rm)^T = (X_rm)^T * (A_rm)^T = X_cm * A_cm
 *            (k x m)                                            (k x n)(n x m)
 *
 * Le dimensioni tornano. Nella convenzione di cublas<t>gemm(op, op, M, N, K,
 * alpha, A, lda, B, ldb, beta, C, ldc) - che calcola C(MxN) = A(MxK)*B(KxN) -
 * il nostro prodotto si scrive percio':
 *
 *     M = k        N = m        K = n
 *     A = dX (ldx) B = dA (lda) C = dY (ldy)
 *
 * con entrambe le op a CUBLAS_OP_N. I vincoli di cuBLAS sulle leading
 * dimension (lda >= righe della matrice column-major) diventano ldx >= k,
 * lda >= n, ldy >= k: esattamente le tre condizioni che local_gemm_create gia'
 * verifica.
 *
 * ---------------------------------------------------------------------------
 * Cosa entra nel cronometro
 * ---------------------------------------------------------------------------
 * Identico agli altri backend CUDA, altrimenti il confronto non varrebbe:
 * l'handle cuBLAS, i cudaMalloc e la H2D di A stanno in create (t_setup); i
 * cudaEvent circondano la sola chiamata gemm, non le copie di X e Y. La
 * creazione dell'handle e' costosa (carica i kernel della libreria) ed e'
 * proprio per questo che sta in create: metterla nel cammino misurato
 * misurerebbe l'inizializzazione della libreria, non il suo GEMM.
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdio.h>
#include <stdlib.h>

#include "kernel/kernel.h"
#include "common/util.h"

#define BYTES_PER_GIB 1073741824.0

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess)                                              \
            die("CUDA error at %s:%d: %s: %s", __FILE__, __LINE__,            \
                cudaGetErrorName(err_), cudaGetErrorString(err_));            \
    } while (0)

/* cublasGetStatusName esiste solo dalle versioni recenti del toolkit: qui la
 * tabella e' esplicita, cosi' il backend compila anche sui moduli piu' vecchi
 * che potrebbero esserci sul server. */
static const char *cublas_status_name(cublasStatus_t s)
{
    switch (s) {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:    return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:    return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:    return "CUBLAS_STATUS_LICENSE_ERROR";
    default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t st_ = (call);                                          \
        if (st_ != CUBLAS_STATUS_SUCCESS)                                     \
            die("cuBLAS error at %s:%d: %s (%d)", __FILE__, __LINE__,         \
                cublas_status_name(st_), (int)st_);                           \
    } while (0)

/* La precisione e' gia' un flag di compilazione per tutto il progetto
 * (scalar_t): qui si limita a scegliere quale delle due gemm chiamare. */
#ifdef USE_FLOAT
#define CUBLAS_GEMM cublasSgemm
#else
#define CUBLAS_GEMM cublasDgemm
#endif

struct local_gemm_ctx {
    int m, n, k;
    int lda, ldx, ldy;
    scalar_t *dA, *dX, *dY;
    cublasHandle_t handle;
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
        die("cublas: no CUDA device available");

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
        die("cublas: device %d has %.2f GiB free out of %.2f GiB, "
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

    /* Handle e stream: la creazione carica i kernel della libreria ed e'
     * costosa, quindi sta qui, fuori dal cammino misurato. Lo stream esplicito
     * e' quello di default, lo stesso su cui vengono registrati gli eventi: e'
     * cosi' che i cudaEvent misurano davvero il gemm. */
    CUBLAS_CHECK(cublasCreate(&ctx->handle));
    CUBLAS_CHECK(cublasSetStream(ctx->handle, 0));
    CUBLAS_CHECK(cublasSetPointerMode(ctx->handle, CUBLAS_POINTER_MODE_HOST));

    CUDA_CHECK(cudaDeviceSynchronize());

    ctx->t_setup = now_seconds() - t0;
    return ctx;
}

void local_gemm(local_gemm_t *ctx,
                const scalar_t *SCPA_RESTRICT X, int ldx,
                scalar_t *SCPA_RESTRICT Y, int ldy)
{
    const int m = ctx->m, n = ctx->n, k = ctx->k;
    const scalar_t alpha = (scalar_t)1;
    const scalar_t beta  = (scalar_t)0;
    float ms = 0.0f;

    if (ldx != ctx->ldx || ldy != ctx->ldy)
        die("cublas: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            ctx->ldx, ldx, ctx->ldy, ldy);

    if (n > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(ctx->dX, X,
                              (size_t)n * (size_t)ldx * sizeof(scalar_t),
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ctx->ev_start, 0));
    if (m > 0 && k > 0) {
        if (n > 0) {
            /* Y_cm(k x m) = X_cm(k x n) * A_cm(n x m): vedi il trucco
             * riga/colonna in testa al file. */
            CUBLAS_CHECK(CUBLAS_GEMM(ctx->handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     k, m, n,
                                     &alpha,
                                     ctx->dX, ldx,
                                     ctx->dA, ctx->lda,
                                     &beta,
                                     ctx->dY, ldy));
        } else {
            /* Blocco senza colonne: Y = 0. Un gemm con K=0 sarebbe legale ma
             * qui il caso e' esplicito, e resta dentro gli eventi cosi' che
             * t_kernel misuri la stessa regione in tutti i casi. */
            CUDA_CHECK(cudaMemsetAsync(ctx->dY, 0,
                                       (size_t)m * (size_t)ldy * sizeof(scalar_t),
                                       0));
        }
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
    cublasDestroy(ctx->handle);
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
    return "cublas";
}
