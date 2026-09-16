#ifndef DISTRIB_H
#define DISTRIB_H

#include "common/scalar.h"
#include "mpi/grid.h"

typedef struct {
    int M, N, k;

    int m_loc;
    int n_loc;
    int row0;
    int col0;

    int lda;

    int ldx;

    int ldy;

} layout_t;

void layout_init(layout_t *l, const grid_t *g, int M, int N, int k);

void distribute_global_A(const grid_t *g, const layout_t *l,
                         const scalar_t *A_global, scalar_t *A_loc);

void distribute_global_X(const grid_t *g, const layout_t *l,
                         const scalar_t *X_global, scalar_t *X_loc);

void layout_y_counts(const layout_t *l, const grid_t *g, int *counts, int *displs);

#endif
