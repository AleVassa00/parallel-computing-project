#ifndef KERNEL_H
#define KERNEL_H

#include <stddef.h>

#include "common/scalar.h"

#ifndef X_COLUMN_MAJOR
#define X_COLUMN_MAJOR 0
#endif
#if X_COLUMN_MAJOR
#define X_LAYOUT_NAME "column"
#else
#define X_LAYOUT_NAME "row"
#endif

#if defined(__cplusplus)
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

#if defined(__cplusplus)
extern "C" {
#endif

typedef struct local_gemm_context local_gemm_t;

local_gemm_t *local_gemm_create(int m, int n, int k, const scalar_t *A_loc, int lda, int ldx, int ldy);

void local_gemm(local_gemm_t *local_gemm_context, const scalar_t *RESTRICT X, int ldx, scalar_t *RESTRICT Y, int ldy);

void local_gemm_destroy(local_gemm_t *local_gemm_context);

double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context);

double local_gemm_last_h2d_X_seconds(const local_gemm_t *local_gemm_context);

double local_gemm_last_d2h_Y_seconds(const local_gemm_t *local_gemm_context);

size_t local_gemm_bytes_h2d_A(const local_gemm_t *local_gemm_context);
size_t local_gemm_bytes_h2d_X_per_call(const local_gemm_t *local_gemm_context);
size_t local_gemm_bytes_d2h_Y_per_call(const local_gemm_t *local_gemm_context);

double local_gemm_setup_device_init_seconds(const local_gemm_t *local_gemm_context);
double local_gemm_setup_device_alloc_seconds(const local_gemm_t *local_gemm_context);
double local_gemm_setup_h2d_A_seconds(const local_gemm_t *local_gemm_context);

double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context);

int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context);

int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context);

const char *kernel_name(void);

#if defined(__cplusplus)
}
#endif

#endif
