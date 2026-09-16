
#include "kernel/kernel.h"

#include <stddef.h>

#include "common/util.h"

#define KB 32

static void kernel_generic(int m_loc, int n_loc, int k,
                           const scalar_t *restrict A_loc, int lda,
                           const scalar_t *restrict X_loc, int ldx,
                           scalar_t *restrict Y_part_loc, int ldy)
{
    int c0;

    for (c0 = 0; c0 < k; c0 += KB) {
        const int cw = (k - c0 < KB) ? (k - c0) : KB;
        int i;

        for (i = 0; i < m_loc; i++) {
            const scalar_t *restrict arow = A_loc + (size_t)i * (size_t)lda;
            scalar_t *restrict yrow = Y_part_loc + (size_t)i * (size_t)ldy + c0;
            scalar_t acc[KB];
            int j, c;

            for (c = 0; c < cw; c++)
                acc[c] = (scalar_t)0;

            for (j = 0; j < n_loc; j++) {
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

#define DECLARE_ACC(c) scalar_t acc##c = (scalar_t)0;
#define UPDATE_ACC(c)  acc##c += a * xrow[c];
#define STORE_ACC(c)   yrow[c] = acc##c;

#define DEFINE_FIXED_KERNEL(K, COLS)                                         \
    static void kernel_k##K(int m_loc, int n_loc,                                    \
                            const scalar_t *restrict A_loc, int lda,              \
                            const scalar_t *restrict X_loc, int ldx,              \
                            scalar_t *restrict Y_loc_part, int ldy)                    \
    {                                                                         \
        int i;                                                                \
        for (i = 0; i < m_loc; i++) {                                             \
            const scalar_t *restrict arow =                                   \
                A_loc + (size_t)i * (size_t)lda;                                  \
            scalar_t *restrict yrow = Y_loc_part + (size_t)i * (size_t)ldy;            \
            int j;                                                            \
            COLS(DECLARE_ACC)                                                 \
            for (j = 0; j < n_loc; j++) {                                         \
                const scalar_t a = arow[j];                                   \
                const scalar_t *restrict xrow =                               \
                    X_loc + (size_t)j * (size_t)ldx;                              \
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

#endif

struct local_gemm_context {
    int m_loc, n_loc, k;
    int lda;
    int ldx, ldy;
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
              lda = local_gemm_context->lda;

    if (ldx != local_gemm_context->ldx || ldy != local_gemm_context->ldy)
        die("local_gemm: leading dimensions changed between calls "
            "(ldx %d -> %d, ldy %d -> %d)",
            local_gemm_context->ldx, ldx, local_gemm_context->ldy, ldy);

#ifdef FORCE_GENERIC_K
    kernel_generic(m_loc, n_loc, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
#else
    switch (k) {
    case 3:
        kernel_k3(m_loc, n_loc, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 6:
        kernel_k6(m_loc, n_loc, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 8:
        kernel_k8(m_loc, n_loc, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 20:
        kernel_k20(m_loc, n_loc, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    case 32:
        kernel_k32(m_loc, n_loc, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    default:
        kernel_generic(m_loc, n_loc, k, A_loc, lda, X_loc, ldx, Y_loc_part, ldy);
        break;
    }
#endif
}

void local_gemm_destroy(local_gemm_t *local_gemm_context)
{

    xfree(local_gemm_context);
}

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
    (void)local_gemm_context;
    return -1;
}

const char *kernel_name(void)
{
#ifdef FORCE_GENERIC_K
    return "scheme_a_generic";
#else
    return "scheme_a";
#endif
}
