/* Schema A con j-blocking: X mattonellata in cache.
 *
 * scheme_a legge A una volta sola, ma per OGNI riga di A scorre tutta X
 * (n_loc x k). Il traffico su X vale quindi m_loc*n_loc*k scalari, cioe' k
 * volte quello su A: finche' X sta in L2 lo paga la cache, quando non ci sta
 * X viene ristreamata dalla L3 m_loc volte e il kernel si ferma sulla banda
 * della L3. Nella campagna C3 (M=N=10000, un processo, double) il crollo e'
 * visibile fra k=8 (X=640 KB, 10.9 GFLOPS) e k=20 (X=1.6 MB, 5.2 GFLOPS).
 *
 * Qui l'ordine dei cicli viene invertito fra tile e righe:
 *
 *   for j0 (tile di X di BJ righe, BJ*k*sizeof(scalar_t) ~ JBLOCK_BYTES):
 *     for i (TUTTE le righe di A):
 *       acc = (j0 == 0) ? 0 : Y[i][*]
 *       for j in tile:  a = A[i][j];  acc[c] += a * X[j][c]
 *       Y[i][*] = acc
 *
 * Il tile di X viene caricato dalla memoria una volta e riusato m_loc volte
 * restando in L1/L2. A continua a essere letta una volta sola: ogni riga viene
 * consumata a segmenti di BJ elementi, uno per tile.
 *
 * Il prezzo e' Y: non puo' piu' vivere nei registri per tutta la riga, perche'
 * ci si torna sopra a ogni tile. Il traffico extra vale 2*k scalari per riga e
 * per tile, cioe' 2k/BJ rispetto ad A: con k=32 e BJ=256 e' il 25%, con k=8 e
 * BJ=1024 l'1.5%. Il tile va quindi scelto grande abbastanza da rendere Y
 * trascurabile e piccolo abbastanza da stare in L1/L2: JBLOCK_BYTES e' il
 * parametro da misurare, non una costante magica.
 *
 * E' lo stesso principio di cuda_warp_smem: portare un blocco di X in una
 * memoria vicina (li' la shared memory, qui la cache) e riusarlo per molte
 * righe prima di passare al blocco successivo.
 *
 * Come in scheme_a, i k obbligatori hanno kernel con accumulatori espliciti;
 * gli altri k usano il fallback generico, anch'esso mattonellato. */

#include "kernel/kernel.h"

#include <stddef.h>
#include <string.h>

#include "common/util.h"

#if X_COLUMN_MAJOR
#error "scheme_a_jblock richiede X row-major (X_LAYOUT=row)"
#endif

/* Byte di X per tile. 64 KB e' un valore che sta nella L2 di qualunque core
 * x86 recente e quasi nella L1 dei core con 48 KB o piu'; e' il default dello
 * sweep, non il suo risultato. */
#ifndef JBLOCK_BYTES
#define JBLOCK_BYTES 65536
#endif

/* Ampiezza del blocco di colonne del fallback generico (vedi scheme_a). */
#define KB 32

/* Righe di X per tile: almeno una, mai piu' di n_loc. */
static int tile_rows(int n_loc, int k)
{
    const size_t row_bytes = (size_t)k * sizeof(scalar_t);
    size_t bj = (row_bytes > 0) ? (size_t)JBLOCK_BYTES / row_bytes : (size_t)n_loc;

    if (bj < 1)
        bj = 1;
    if (bj > (size_t)n_loc)
        bj = (size_t)n_loc;
    return (int)bj;
}

/* Fallback generico su [j0, j1): first dice se gli accumulatori partono da
 * zero o dal valore parziale gia' in Y. */
static void kernel_generic(int m_loc, int j0, int j1, int first, int k,
                           const scalar_t *restrict A_loc, int lda,
                           const scalar_t *restrict X_loc, int ldx,
                           scalar_t *restrict Y_loc_part, int ldy)
{
    int c0;

    for (c0 = 0; c0 < k; c0 += KB) {
        const int cw = (k - c0 < KB) ? (k - c0) : KB;
        int i;

        for (i = 0; i < m_loc; i++) {
            const scalar_t *restrict arow = A_loc + (size_t)i * (size_t)lda;
            scalar_t *restrict yrow = Y_loc_part + (size_t)i * (size_t)ldy + c0;
            scalar_t acc[KB];
            int j, c;

            if (first) {
                for (c = 0; c < cw; c++)
                    acc[c] = (scalar_t)0;
            } else {
                for (c = 0; c < cw; c++)
                    acc[c] = yrow[c];
            }

            for (j = j0; j < j1; j++) {
                const scalar_t a = arow[j];
                const scalar_t *restrict xrow = X_loc + (size_t)j * (size_t)ldx + c0;
                for (c = 0; c < cw; c++)
                    acc[c] += a * xrow[c];
            }

            for (c = 0; c < cw; c++)
                yrow[c] = acc[c];
        }
    }
}

#ifndef FORCE_GENERIC_K

#define COLS_3(M)  M(0) M(1) M(2)
#define COLS_6(M)  COLS_3(M) M(3) M(4) M(5)
#define COLS_8(M)  COLS_6(M) M(6) M(7)
#define COLS_20(M) COLS_8(M) M(8) M(9) M(10) M(11) M(12) M(13) \
                   M(14) M(15) M(16) M(17) M(18) M(19)
#define COLS_32(M) COLS_20(M) M(20) M(21) M(22) M(23) M(24) M(25) \
                   M(26) M(27) M(28) M(29) M(30) M(31)

/* Rispetto a scheme_a cambia solo l'inizializzazione: al primo tile gli
 * accumulatori partono da zero, ai successivi dal parziale gia' in Y. */
#define DECLARE_ACC(c) scalar_t acc##c = first ? (scalar_t)0 : yrow[c];
#define UPDATE_ACC(c)  acc##c += a * xrow[c];
#define STORE_ACC(c)   yrow[c] = acc##c;

#define DEFINE_FIXED_KERNEL(K, COLS)                                          \
    static void kernel_k##K(int m_loc, int j0, int j1, int first,             \
                            const scalar_t *restrict A_loc, int lda,          \
                            const scalar_t *restrict X_loc, int ldx,          \
                            scalar_t *restrict Y_loc_part, int ldy)           \
    {                                                                         \
        int i;                                                                \
        for (i = 0; i < m_loc; i++) {                                         \
            const scalar_t *restrict arow =                                   \
                A_loc + (size_t)i * (size_t)lda;                              \
            scalar_t *restrict yrow = Y_loc_part + (size_t)i * (size_t)ldy;   \
            int j;                                                            \
            COLS(DECLARE_ACC)                                                 \
            for (j = j0; j < j1; j++) {                                       \
                const scalar_t a = arow[j];                                   \
                const scalar_t *restrict xrow =                               \
                    X_loc + (size_t)j * (size_t)ldx;                          \
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

/* Un tile di X, [j0, j1), applicato a tutte le righe di A. */
static void run_tile(int m_loc, int j0, int j1, int first, int k,
                     const scalar_t *restrict A_loc, int lda,
                     const scalar_t *restrict X_loc, int ldx,
                     scalar_t *restrict Y_loc_part, int ldy)
{
#ifdef FORCE_GENERIC_K
    kernel_generic(m_loc, j0, j1, first, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
#else
    switch (k) {
    case 3:
        kernel_k3(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 6:
        kernel_k6(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 8:
        kernel_k8(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 20:
        kernel_k20(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 32:
        kernel_k32(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    default:
        kernel_generic(m_loc, j0, j1, first, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    }
#endif
}

/* Stato del backend: come in scheme_a, solo forma e puntatore. In piu' il
 * numero di righe di X per tile, calcolato una volta in create. */
struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda;
    int ldx, ldy;
    int bj;           /* righe di X per tile */
    const scalar_t *A_loc;
    double t_setup;
};

local_gemm_t *local_gemm_create(int m_loc, int n_loc, int k, const scalar_t *A_loc, int lda, int ldx, int ldy)
{
    local_gemm_t *local_gemm_context;
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

    local_gemm_context = xmalloc(sizeof *local_gemm_context);
    local_gemm_context->m_loc = m_loc;
    local_gemm_context->n_loc = n_loc;
    local_gemm_context->k = k;
    local_gemm_context->lda = lda;
    local_gemm_context->ldx = ldx;
    local_gemm_context->ldy = ldy;
    local_gemm_context->bj = tile_rows(n_loc, k);
    local_gemm_context->A_loc = A_loc;
    local_gemm_context->t_setup = now_seconds() - t0;
    return local_gemm_context;
}

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t * RESTRICT X_loc, int ldx, scalar_t * RESTRICT Y_loc_part, int ldy)
{
    const scalar_t *RESTRICT A_loc = local_gemm_context->A_loc;
    const int m_loc = local_gemm_context->m_loc,
              n_loc = local_gemm_context->n_loc,
              k = local_gemm_context->k,
              lda = local_gemm_context->lda,
              bj = local_gemm_context->bj;
    int j0;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("local_gemm: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

    /* Y = A*X e' un'assegnazione: con n_loc = 0 non c'e' nessun tile che la
     * scriva, e la somma vuota vale zero. */
    if (n_loc == 0) {
        int i;
        for (i = 0; i < m_loc; i++)
            memset(Y_loc_part + (size_t)i * (size_t)ldy, 0, (size_t)k * sizeof(scalar_t));
        return;
    }

    for (j0 = 0; j0 < n_loc; j0 += bj) {
        const int j1 = (n_loc - j0 < bj) ? n_loc : j0 + bj;
        run_tile(m_loc, j0, j1, j0 == 0, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
    }
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{
    xfree(local_gemm_context);
}

/* Canali di misura: backend di CPU, stesse sentinelle di scheme_a. */
double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1.0;
}

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->t_setup : 0.0;
}

double local_gemm_last_h2d_X_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1.0;
}

double local_gemm_last_d2h_Y_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1.0;
}

size_t local_gemm_bytes_h2d_A(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return 0;
}

size_t local_gemm_bytes_h2d_X_per_call(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return 0;
}

size_t local_gemm_bytes_d2h_Y_per_call(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return 0;
}

double local_gemm_setup_device_init_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return 0.0;
}

double local_gemm_setup_device_alloc_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return 0.0;
}

double local_gemm_setup_h2d_A_seconds(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return 0.0;
}

int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context)
{
    (void)local_gemm_context;
    return -1;
}

/* Qui il concetto ESISTE: e' il tile di X in cache, l'analogo del tile in
 * shared memory di cuda_warp_smem. Riportarlo nella stessa colonna rende
 * confrontabili le due campagne di tiling. */
int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context)
{
    return (local_gemm_context != NULL) ? local_gemm_context->bj : -1;
}

#define STR_(x) #x
#define STR(x)  STR_(x)

#if JBLOCK_BYTES == 65536
#define JB_SUFFIX ""
#else
#define JB_SUFFIX "(jb" STR(JBLOCK_BYTES) ")"
#endif

const char *kernel_name(void)
{
#ifdef FORCE_GENERIC_K
    return "scheme_a_jblock_generic" JB_SUFFIX;
#else
    return "scheme_a_jblock" JB_SUFFIX;
#endif
}
