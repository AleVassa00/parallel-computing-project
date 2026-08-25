#include "mpi/matmul_mpi.h"

#include <stddef.h>

#include "kernel/kernel.h"

void mpi_matmul(const grid_t *g, const layout_t *l,
                const scalar_t *A_loc,
                scalar_t *X_loc,
                scalar_t *Ypart,
                scalar_t *Y_loc,
                matmul_time_t *t)
{
    const int x_count = l->n_loc * l->k;
    const int y_count = l->m_loc * l->k;
    double t0, t1, t2, t3;

    t0 = MPI_Wtime();

    /* 1. In g->col il rank 0 coincide, per l'invariante verificato da
     *    grid_create(), con la coordinata cartesiana di riga 0. */
    MPI_Bcast(X_loc, x_count, SCALAR_MPI_TYPE, 0, g->col);
    t1 = MPI_Wtime();

    /* 2. Contributo locale: righe [row0, row0+m_loc) di Y, limitatamente alle
     *    colonne [col0, col0+n_loc) di A. E' un risultato PARZIALE. */
    local_gemm(l->m_loc, l->n_loc, l->k,
               A_loc, l->lda,
               X_loc, l->ldx,
               Ypart, l->ldy);

    t2 = MPI_Wtime();

    /* 3. I Pc parziali della stessa banda di righe vivono sui processi della
     *    riga my_row: sommandoli si ottiene Y. count = m_loc*k, non 1: la
     *    reduce agisce lungo la dimensione dei processi e preserva la
     *    posizione, producendo m_loc*k somme indipendenti.
     *    root = rank 0 di row, cioe' il processo di colonna 0, che e' gia' il
     *    proprietario designato di quella porzione di Y: nessuna
     *    ridistribuzione finale. */
    MPI_Reduce(Ypart, Y_loc, y_count, SCALAR_MPI_TYPE, MPI_SUM, 0, g->row);
    t3 = MPI_Wtime();

    if (t != NULL) {
        t->t_bcast = t1 - t0;
        t->t_local = t2 - t1;
        t->t_reduce = t3 - t2;
        t->t_total = t3 - t0;
    }
}
