/* Schema A: ordine dei cicli i -> j -> c, tutto row-major.
 *
 * E' lo schema che NON rilegge A k volte, ed e' il motivo per cui il prodotto
 * matrice-multivettore vale piu' di k prodotti matrice-vettore.
 *
 *   for i:
 *     acc[0..k) = 0
 *     for j:
 *       a = A[i][j]              <- letto UNA sola volta
 *       for c: acc[c] += a * X[j][c]   <- riusato k volte, dai registri
 *     Y[i][0..k) = acc[0..k)
 *
 * Proprieta':
 *  - entrambi gli stream in memoria sono a stride 1: la riga di A e la riga
 *    di X (che e' contigua perche' X e' row-major con k contiguo);
 *  - il riuso di A vale esattamente k, per qualunque k, senza casi degeneri.
 *    L'intensita' aritmetica passa da k/4 * (1/k) = 1/4 dello schema B a
 *    k/4 FLOP/byte, che e' il massimo ottenibile leggendo A una volta.
 *
 * Confronto con lo schema B (i -> c -> j, X column-major): li' la riga di A
 * viene riletta una volta per ogni colonna di X, e il riuso si recupera solo
 * con l'unrolling su c, che pero' satura a min(k, ampiezza del gruppo). Qui
 * il riuso e' k per costruzione.
 *
 * I k obbligatori hanno funzioni con aggiornamenti espliciti, cosi' il
 * compilatore vede a compile-time il numero di accumulatori. Gli altri k
 * usano il fallback bloccato generico; per k > KB A viene riletta
 * ceil(k/KB) volte. */

#include "kernel/kernel.h"

#include <stddef.h>

#include "common/util.h"

/* Ampiezza del blocco di colonne tenuto negli accumulatori.
 * 32 copre tutti i k del collaudo (3, 6, 8, 20, 32): in quei casi il ciclo
 * su c0 e' degenere e A viene letta una volta sola. */
#define KB 32

static void kernel_generic(int m, int n, int k,
                           const scalar_t *restrict A, int lda,
                           const scalar_t *restrict X, int ldx,
                           scalar_t *restrict Y, int ldy)
{
    int c0;

    for (c0 = 0; c0 < k; c0 += KB) {
        const int cw = (k - c0 < KB) ? (k - c0) : KB;
        int i;

        for (i = 0; i < m; i++) {
            const scalar_t *restrict arow = A + (size_t)i * (size_t)lda;
            scalar_t *restrict yrow = Y + (size_t)i * (size_t)ldy + c0;
            scalar_t acc[KB];
            int j, c;

            for (c = 0; c < cw; c++)
                acc[c] = (scalar_t)0;

            for (j = 0; j < n; j++) {
                const scalar_t a = arow[j];
                const scalar_t *restrict xrow = X + (size_t)j * (size_t)ldx + c0;
                for (c = 0; c < cw; c++)
                    acc[c] += a * xrow[c];
            }

            for (c = 0; c < cw; c++)
                yrow[c] = acc[c];
        }
    }
}

#ifndef FORCE_GENERIC_K

/* Una sola lista per ogni ampiezza genera dichiarazione, aggiornamento e
 * store. Il percorso caldo risultante contiene istruzioni C esplicite per
 * ogni colonna, senza un limite runtime sul ciclo c. */
#define COLS_3(M)  M(0) M(1) M(2)
#define COLS_6(M)  COLS_3(M) M(3) M(4) M(5)
#define COLS_8(M)  COLS_6(M) M(6) M(7)
#define COLS_20(M) COLS_8(M) M(8) M(9) M(10) M(11) M(12) M(13) \
                   M(14) M(15) M(16) M(17) M(18) M(19)
#define COLS_32(M) COLS_20(M) M(20) M(21) M(22) M(23) M(24) M(25) \
                   M(26) M(27) M(28) M(29) M(30) M(31)

#define DECLARE_ACC(c) scalar_t acc##c = (scalar_t)0;
#define UPDATE_ACC(c)  acc##c += a * xrow[c];
#define STORE_ACC(c)   yrow[c] = acc##c;

#define DEFINE_FIXED_KERNEL(K, COLS)                                         \
    static void kernel_k##K(int m, int n,                                    \
                            const scalar_t *restrict A, int lda,              \
                            const scalar_t *restrict X, int ldx,              \
                            scalar_t *restrict Y, int ldy)                    \
    {                                                                         \
        int i;                                                                \
        for (i = 0; i < m; i++) {                                             \
            const scalar_t *restrict arow =                                   \
                A + (size_t)i * (size_t)lda;                                  \
            scalar_t *restrict yrow = Y + (size_t)i * (size_t)ldy;            \
            int j;                                                            \
            COLS(DECLARE_ACC)                                                 \
            for (j = 0; j < n; j++) {                                         \
                const scalar_t a = arow[j];                                   \
                const scalar_t *restrict xrow =                               \
                    X + (size_t)j * (size_t)ldx;                              \
                COLS(UPDATE_ACC)                                              \
            }                                                                 \
            COLS(STORE_ACC)                                                   \
        }                                                                     \
    }

DEFINE_FIXED_KERNEL(3, COLS_3)
DEFINE_FIXED_KERNEL(6, COLS_6)
DEFINE_FIXED_KERNEL(8, COLS_8)
DEFINE_FIXED_KERNEL(20, COLS_20)
DEFINE_FIXED_KERNEL(32, COLS_32)

#undef DEFINE_FIXED_KERNEL
#undef STORE_ACC
#undef UPDATE_ACC
#undef DECLARE_ACC
#undef COLS_32
#undef COLS_20
#undef COLS_8
#undef COLS_6
#undef COLS_3

#endif /* FORCE_GENERIC_K */

/* Stato del backend.
 *
 * Per lo schema A scalare non c'e' nulla da preparare: A e' gia' nella memoria
 * del processo e il kernel la legge direttamente, quindi il contesto si limita
 * a registrare forma e puntatore. Esiste comunque, e con la stessa interfaccia
 * degli altri backend, perche' e' qui che il backend CUDA terra' il puntatore
 * alla copia di A in VRAM: da quel lato create() e' una cudaMemcpy H2D che
 * deve avvenire UNA volta sola, fuori dalla regione cronometrata. */
struct local_gemm_ctx {
    int m, n, k;
    int lda;
    const scalar_t *A;
    double t_setup;   /* misurato davvero, anche se qui e' ~1 us: il confronto
                       * con il backend CUDA ha senso solo se lo stesso numero
                       * viene dallo stesso punto del codice in entrambi. */
};

local_gemm_t *local_gemm_create(int m, int n, int k,
                                const scalar_t *A, int lda)
{
    local_gemm_t *ctx;
    const double t0 = now_seconds();

    if (m < 0 || n < 0 || k < 0)
        die("local_gemm_create: invalid local block %dx%d with k=%d", m, n, k);
    if (lda < n)
        die("local_gemm_create: lda %d is smaller than n %d", lda, n);
    if (n > 0 && m > 0 && A == NULL)
        die("local_gemm_create: A is NULL for a non-empty %dx%d block", m, n);

    ctx = xmalloc(sizeof *ctx);
    ctx->m = m;
    ctx->n = n;
    ctx->k = k;
    ctx->lda = lda;
    ctx->A = A;
    ctx->t_setup = now_seconds() - t0;
    return ctx;
}

void local_gemm(local_gemm_t *ctx,
                const scalar_t *SCPA_RESTRICT X, int ldx,
                scalar_t *SCPA_RESTRICT Y, int ldy)
{
    /* Copie locali: il puntatore ad A torna a essere restrict all'interno di
     * questa funzione, cosi' i kernel specializzati ricevono la stessa
     * garanzia di non aliasing che avevano quando A era un parametro. */
    const scalar_t *SCPA_RESTRICT A = ctx->A;
    const int m = ctx->m, n = ctx->n, k = ctx->k, lda = ctx->lda;

#ifdef FORCE_GENERIC_K
    kernel_generic(m, n, k, A, lda, X, ldx, Y, ldy);
#else
    switch (k) {
    case 3:
        kernel_k3(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 6:
        kernel_k6(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 8:
        kernel_k8(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 20:
        kernel_k20(m, n, A, lda, X, ldx, Y, ldy);
        break;
    case 32:
        kernel_k32(m, n, A, lda, X, ldx, Y, ldy);
        break;
    default:
        kernel_generic(m, n, k, A, lda, X, ldx, Y, ldy);
        break;
    }
#endif
}

void local_gemm_destroy(local_gemm_t *ctx)
{
    /* Nessuna risorsa esterna da rilasciare: A appartiene al chiamante.
     * Il backend CUDA fara' qui la cudaFree della copia in VRAM. */
    xfree(ctx);
}

/* Su CPU il kernel E' l'invocazione: non esiste un tempo di calcolo distinto
 * da t_local, e restituire t_local qui vorrebbe dire duplicare in una colonna
 * un numero che il chiamante ha gia'. Il valore negativo dice "non applicabile"
 * e il driver lo riporta come tale, cosi' nel CSV si vede a colpo d'occhio
 * quali righe vengono da un backend con trasferimenti e quali no. */
double local_gemm_last_compute_seconds(const local_gemm_t *ctx)
{
    (void)ctx;
    return -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *ctx)
{
    return (ctx != NULL) ? ctx->t_setup : 0.0;
}

const char *kernel_name(void)
{
#ifdef FORCE_GENERIC_K
    return "scheme_a_generic";
#else
    return "scheme_a";
#endif
}
