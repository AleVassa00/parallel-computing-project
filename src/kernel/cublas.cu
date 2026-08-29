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
 *     M = k            N = m_loc        K = n_loc
 *     A = dX_loc (ldx) B = dA_loc (lda) C = dY_loc_part (ldy)
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


#include "kernel/kernel.h"
#include "common/util.h"

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

struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda, ldx, ldy;
    scalar_t *dA_loc, *dX_loc, *dY_loc_part;
    cublasHandle_t handle;
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
    local_gemm_t *ctx;
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

    ctx = (local_gemm_t *)xmalloc(sizeof *ctx);
    ctx->m_loc = m_loc;
    ctx->n_loc = n_loc;
    ctx->k = k;
    ctx->lda = lda;
    ctx->ldx = ldx;
    ctx->ldy = ldy;
    ctx->dA_loc = NULL;
    ctx->dX_loc = NULL;
    ctx->dY_loc_part = NULL;
    ctx->t_last = -1.0;

    CUDA_CHECK(cudaSetDevice(CUDA_DEVICE_ID));
    CUDA_CHECK(cudaFree(0));

    bytes_A = (size_t)m_loc * (size_t)lda * sizeof(scalar_t);
    bytes_X = (size_t)n_loc * (size_t)ldx * sizeof(scalar_t);
    bytes_Y = (size_t)m_loc * (size_t)ldy * sizeof(scalar_t);
    need = bytes_A + bytes_X + bytes_Y;

    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    if (need + (64u << 20) > free_b)
        die("cublas: device %d has %.2f GiB free out of %.2f GiB, "
            "but the local block needs about %.2f GiB "
            "(A %dx%d, X %dx%d, Y %dx%d in %s): use more MPI processes "
            "or a smaller M/N",
            CUDA_DEVICE_ID, (double)free_b / BYTES_PER_GIB,
            (double)total_b / BYTES_PER_GIB, (double)need / BYTES_PER_GIB,
            m_loc, lda, n_loc, ldx, m_loc, ldy, SCALAR_NAME);

    CUDA_CHECK(cudaMalloc((void **)&ctx->dA_loc, nonzero(bytes_A)));
    if (bytes_A > 0)
        CUDA_CHECK(cudaMemcpy(ctx->dA_loc, A, bytes_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void **)&ctx->dX_loc, nonzero(bytes_X)));
    CUDA_CHECK(cudaMalloc((void **)&ctx->dY_loc_part, nonzero(bytes_Y)));
    CUDA_CHECK(cudaMemset(ctx->dY_loc_part, 0, nonzero(bytes_Y)));
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
                const scalar_t *RESTRICT X, int ldx,
                scalar_t *RESTRICT Y, int ldy)
{
    const int m_loc = ctx->m_loc, n_loc = ctx->n_loc, k = ctx->k;
    const scalar_t alpha = (scalar_t)1;
    const scalar_t beta  = (scalar_t)0;
    float ms = 0.0f;

    if (ldx != ctx->ldx || ldy != ctx->ldy)
        die("cublas: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            ctx->ldx, ldx, ctx->ldy, ldy);

    if (n_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(ctx->dX_loc, X,
                              (size_t)n_loc * (size_t)ldx * sizeof(scalar_t),
                              cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ctx->ev_start, 0));
    if (m_loc > 0 && k > 0) {
        if (n_loc > 0) {
            /* Y_cm(k x m) = X_cm(k x n) * A_cm(n x m): vedi il trucco
             * riga/colonna in testa al file. */
            CUBLAS_CHECK(CUBLAS_GEMM(ctx->handle,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     k, m_loc, n_loc,
                                     &alpha,
                                     ctx->dX_loc, ldx,
                                     ctx->dA_loc, ctx->lda,
                                     &beta,
                                     ctx->dY_loc_part, ldy));
        } else {
            /* Blocco senza colonne: Y = 0. Un gemm con K=0 sarebbe legale ma
             * qui il caso e' esplicito, e resta dentro gli eventi cosi' che
             * t_kernel misuri la stessa regione in tutti i casi. */
            CUDA_CHECK(cudaMemsetAsync(ctx->dY_loc_part, 0,
                                       (size_t)m_loc * (size_t)ldy * sizeof(scalar_t),
                                       0));
        }
    }
    CUDA_CHECK(cudaEventRecord(ctx->ev_stop, 0));

    if (m_loc > 0 && k > 0)
        CUDA_CHECK(cudaMemcpy(Y, ctx->dY_loc_part,
                              (size_t)m_loc * (size_t)ldy * sizeof(scalar_t),
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
    if (ctx->dA_loc != NULL) cudaFree(ctx->dA_loc);
    if (ctx->dX_loc != NULL) cudaFree(ctx->dX_loc);
    if (ctx->dY_loc_part != NULL) cudaFree(ctx->dY_loc_part);
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
