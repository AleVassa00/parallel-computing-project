/* Schema A con j-blocking E register blocking sulle righe di A.
 *
 * scheme_a_jblock toglie il limite di banda della L3 (X mattonellata in
 * cache). Resta il limite che si vede ai k piccoli: in scheme_a ogni acc[c] e'
 * una catena seriale di FMA lungo j, e l'unico parallelismo di istruzione e'
 * fra le k colonne. Nella campagna C3 i GFLOPS crescono linearmente con k
 * (4.5, 8.4, 10.9 per k = 3, 6, 8): con latenza FMA 4 cicli e due porte
 * servono almeno 8 catene VETTORIALI indipendenti per saturare la pipeline,
 * e k=8 in double AVX2 ne offre 2.
 *
 * Qui il kernel lavora su RB righe di A per volta:
 *
 *   for j0 (tile di X, come in scheme_a_jblock):
 *     for i a passi di RB:
 *       acc[r][c] = (j0 == 0) ? 0 : Y[i+r][c]
 *       for j in tile:
 *         x[c] = X[j][c]                     <- caricato UNA volta
 *         for r: a = A[i+r][j]; acc[r][c] += a * x[c]   <- riusato RB volte
 *       Y[i+r][c] = acc[r][c]
 *
 * Due effetti:
 *  - le catene indipendenti diventano RB * ceil(k/W), con W scalari per
 *    vettore: le righe di A sono indipendenti per costruzione;
 *  - ogni load di X viene riusato RB volte dai registri, quindi anche il
 *    traffico su X (gia' in cache grazie al tile) cala di un fattore RB.
 *
 * RB e' limitato dal register file: RB*ceil(k/W) accumulatori + RB broadcast
 * di A devono stare nei registri vettoriali, altrimenti gli accumulatori
 * finiscono in stack (spill) e si perde piu' di quanto si guadagna. La scelta
 * e' fatta a compile-time per ogni k dal numero di registri e dall'ampiezza
 * vettoriale dell'architettura (vedi VEC_W / VEC_NREGS); RB_ROWS la forza.
 *
 * I k obbligatori hanno kernel con accumulatori espliciti per RB = 4, 2 e 1
 * righe: per ogni k viene usato il piu' grande che sta nei registri, e il
 * kernel a 1 riga chiude le righe residue (m_loc mod RB). Gli altri k usano il
 * fallback generico, mattonellato ma a una riga. */

#include "kernel/kernel.h"

#include <stddef.h>
#include <string.h>

#include "common/util.h"

#if X_COLUMN_MAJOR
#error "scheme_a_jblock_rb richiede X row-major (X_LAYOUT=row)"
#endif

#ifndef JBLOCK_BYTES
#define JBLOCK_BYTES 65536
#endif

#define KB 32

/* ---------------------------------------------------------------------------
 * Scelta di RB per ogni k
 * ---------------------------------------------------------------------------
 * VEC_W:     scalari per registro vettoriale;  VEC_NREGS: registri vettoriali.
 * Il budget e' VEC_NREGS - 2 (un registro per il puntatore-X caricato, uno di
 * scorta), e RB deve soddisfare RB*ceil(k/VEC_W) + RB <= budget. */
#if defined(__AVX512F__)
#define VEC_BYTES 64
#define VEC_NREGS 32
#elif defined(__AVX__)
#define VEC_BYTES 32
#define VEC_NREGS 16
#elif defined(__ARM_NEON) || defined(__aarch64__)
#define VEC_BYTES 16
#define VEC_NREGS 32
#elif defined(__SSE2__)
#define VEC_BYTES 16
#define VEC_NREGS 16
#else
#define VEC_BYTES ((int)sizeof(scalar_t))
#define VEC_NREGS 8
#endif

#define VEC_W ((int)(VEC_BYTES / sizeof(scalar_t)))
#define NVEC(k) (((k) + VEC_W - 1) / VEC_W)
#define RB_FITS(k, rb) ((rb) * NVEC(k) + (rb) <= VEC_NREGS - 2)

#ifdef RB_ROWS
#if RB_ROWS != 1 && RB_ROWS != 2 && RB_ROWS != 4
#error "RB_ROWS deve valere 1, 2 oppure 4"
#endif
#define RB_FOR(k) (RB_ROWS)
#else
#define RB_FOR(k) (RB_FITS(k, 4) ? 4 : RB_FITS(k, 2) ? 2 : 1)
#endif

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

/* Fallback generico: j-blocked, una riga per volta (vedi scheme_a_jblock). */
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

/* Liste di colonne: M riceve (riga, colonna). */
#define COLS_3(M, r)  M(r, 0) M(r, 1) M(r, 2)
#define COLS_6(M, r)  COLS_3(M, r) M(r, 3) M(r, 4) M(r, 5)
#define COLS_8(M, r)  COLS_6(M, r) M(r, 6) M(r, 7)
#define COLS_20(M, r) COLS_8(M, r) M(r, 8) M(r, 9) M(r, 10) M(r, 11) M(r, 12) \
                      M(r, 13) M(r, 14) M(r, 15) M(r, 16) M(r, 17) M(r, 18) M(r, 19)
#define COLS_32(M, r) COLS_20(M, r) M(r, 20) M(r, 21) M(r, 22) M(r, 23) M(r, 24) \
                      M(r, 25) M(r, 26) M(r, 27) M(r, 28) M(r, 29) M(r, 30) M(r, 31)

/* Liste di righe: applicano una lista di colonne a ogni riga del blocco.
 * ROWS_ONLY applica M (una macro di sola riga) a ogni riga. */
#define ROWS_1(COLS, M) COLS(M, 0)
#define ROWS_2(COLS, M) COLS(M, 0) COLS(M, 1)
#define ROWS_4(COLS, M) COLS(M, 0) COLS(M, 1) COLS(M, 2) COLS(M, 3)
#define ROWS_ONLY_1(M) M(0)
#define ROWS_ONLY_2(M) M(0) M(1)
#define ROWS_ONLY_4(M) M(0) M(1) M(2) M(3)

#define DECLARE_ROW(r)                                                       \
    const scalar_t *restrict arow##r = A_loc + (size_t)(i + r) * (size_t)lda; \
    scalar_t *restrict yrow##r = Y_loc_part + (size_t)(i + r) * (size_t)ldy;
#define LOAD_A(r)        const scalar_t a##r = arow##r[j];
#define DECLARE_ACC(r, c) scalar_t acc##r##_##c = first ? (scalar_t)0 : yrow##r[c];
#define UPDATE_ACC(r, c)  acc##r##_##c += a##r * xrow[c];
#define STORE_ACC(r, c)   yrow##r[c] = acc##r##_##c;

/* Kernel a RB righe: processa le righe [0, m_loc) a blocchi di RB e lascia al
 * chiamante le residue (m_loc mod RB), che passano dal kernel a 1 riga. */
#define DEFINE_FIXED_KERNEL(K, COLS, RB, ROWS, ROWS_ONLY)                     \
    static void kernel_k##K##_r##RB(int m_loc, int j0, int j1, int first,    \
                            const scalar_t *restrict A_loc, int lda,          \
                            const scalar_t *restrict X_loc, int ldx,          \
                            scalar_t *restrict Y_loc_part, int ldy)           \
    {                                                                         \
        int i;                                                                \
        for (i = 0; i + RB <= m_loc; i += RB) {                               \
            int j;                                                            \
            ROWS_ONLY(DECLARE_ROW)                                            \
            ROWS(COLS, DECLARE_ACC)                                           \
            for (j = j0; j < j1; j++) {                                       \
                const scalar_t *restrict xrow =                               \
                    X_loc + (size_t)j * (size_t)ldx;                          \
                ROWS_ONLY(LOAD_A)                                             \
                ROWS(COLS, UPDATE_ACC)                                        \
            }                                                                 \
            ROWS(COLS, STORE_ACC)                                             \
        }                                                                     \
    }

#define DEFINE_FIXED_KERNELS(K, COLS)                                         \
    DEFINE_FIXED_KERNEL(K, COLS, 4, ROWS_4, ROWS_ONLY_4)                      \
    DEFINE_FIXED_KERNEL(K, COLS, 2, ROWS_2, ROWS_ONLY_2)                      \
    DEFINE_FIXED_KERNEL(K, COLS, 1, ROWS_1, ROWS_ONLY_1)                      \
    /* Dispatch per questo k: RB_FOR(K) e' una costante, il compilatore     \
     * elimina i rami morti e le varianti non usate. */                       \
    static void run_k##K(int m_loc, int j0, int j1, int first,                \
                         const scalar_t *restrict A_loc, int lda,             \
                         const scalar_t *restrict X_loc, int ldx,             \
                         scalar_t *restrict Y_loc_part, int ldy)              \
    {                                                                         \
        const int rb = RB_FOR(K);                                             \
        /* righe coperte dal kernel a RB righe; con rb == 1 sono tutte       \
         * del kernel a 1 riga */                                             \
        const int m_blk = (rb == 1) ? 0 : m_loc - m_loc % rb;                 \
        if (rb == 4)                                                          \
            kernel_k##K##_r4(m_blk, j0, j1, first, A_loc, lda, X_loc, ldx,    \
                             Y_loc_part, ldy);                                \
        else if (rb == 2)                                                     \
            kernel_k##K##_r2(m_blk, j0, j1, first, A_loc, lda, X_loc, ldx,    \
                             Y_loc_part, ldy);                                \
        kernel_k##K##_r1(m_loc - m_blk, j0, j1, first,                        \
                         A_loc + (size_t)m_blk * (size_t)lda, lda,            \
                         X_loc, ldx,                                          \
                         Y_loc_part + (size_t)m_blk * (size_t)ldy, ldy);      \
    }

DEFINE_FIXED_KERNELS(3, COLS_3)
DEFINE_FIXED_KERNELS(6, COLS_6)
DEFINE_FIXED_KERNELS(8, COLS_8)
DEFINE_FIXED_KERNELS(20, COLS_20)
DEFINE_FIXED_KERNELS(32, COLS_32)

#undef DEFINE_FIXED_KERNELS
#undef DEFINE_FIXED_KERNEL
#undef STORE_ACC
#undef UPDATE_ACC
#undef DECLARE_ACC
#undef LOAD_A
#undef DECLARE_ROW
#undef ROWS_ONLY_4
#undef ROWS_ONLY_2
#undef ROWS_ONLY_1
#undef ROWS_4
#undef ROWS_2
#undef ROWS_1
#undef COLS_32
#undef COLS_20
#undef COLS_8
#undef COLS_6
#undef COLS_3

#endif /* FORCE_GENERIC_K */

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
        run_k3(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 6:
        run_k6(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 8:
        run_k8(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 20:
        run_k20(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 32:
        run_k32(m_loc, j0, j1, first, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    default:
        kernel_generic(m_loc, j0, j1, first, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    }
#endif
}

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

    /* Y = A*X e' un'assegnazione: con n_loc = 0 nessun tile la scrive. */
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

#ifdef RB_ROWS
#define RB_SUFFIX "(rb" STR(RB_ROWS) ")"
#else
#define RB_SUFFIX ""
#endif

const char *kernel_name(void)
{
#ifdef FORCE_GENERIC_K
    return "scheme_a_jblock_rb_generic" JB_SUFFIX RB_SUFFIX;
#else
    return "scheme_a_jblock_rb" JB_SUFFIX RB_SUFFIX;
#endif
}
