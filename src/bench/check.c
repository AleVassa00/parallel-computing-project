#include "bench/check.h"

#include <math.h>
#include <stddef.h>

#include "common/util.h"
#include "gen/gen.h"
#include "serial/serial.h"

double check_against_serial(const grid_t *g, const layout_t *l,
                            const scalar_t *Y_loc, uint64_t seed)
{
    double err = 0.0;
    scalar_t *Y_all = NULL;
    int *counts = NULL, *displs = NULL;

    if (l->ldy != l->k)
        die("check_against_serial: Y deve essere contigua (ldy=%d, k=%d)",
            l->ldy, l->k);

    if (g->my_col == 0) {
        if (g->my_row == 0) {
            Y_all = xmalloc((size_t)l->M * l->k * sizeof *Y_all);
            counts = xmalloc((size_t)g->pr * sizeof *counts);
            displs = xmalloc((size_t)g->pr * sizeof *displs);
            layout_y_counts(l, g, counts, displs);
        }

        MPI_Gatherv(Y_loc, l->m_loc * l->k, SCALAR_MPI_TYPE,
                    Y_all, counts, displs, SCALAR_MPI_TYPE,
                    0, g->col_comm);
    }

    if (g->rank == 0) {
        scalar_t *A = xmalloc((size_t)l->M * l->N * sizeof *A);
        scalar_t *X = xmalloc((size_t)l->N * l->k * sizeof *X);
        scalar_t *Yref = xmalloc((size_t)l->M * l->k * sizeof *Yref);
        double num = 0.0, den = 0.0;
        size_t idx, n_y = (size_t)l->M * l->k;

        gen_block_A(A, l->N, l->M, l->N, 0, 0, l->N, seed);
        gen_block_X(X, l->k, l->N, l->k, 0, seed);
        serial_gemm(l->M, l->N, l->k, A, l->N, X, l->k, Yref, l->k);

        for (idx = 0; idx < n_y; idx++) {
            double d = (double)Y_all[idx] - (double)Yref[idx];
            num += d * d;
            den += (double)Yref[idx] * (double)Yref[idx];
        }
        err = (den > 0.0) ? sqrt(num / den) : sqrt(num);

        xfree(A);
        xfree(X);
        xfree(Yref);
    }

    xfree(Y_all);
    xfree(counts);
    xfree(displs);

    MPI_Bcast(&err, 1, MPI_DOUBLE, 0, g->grid_comm);
    return err;
}
