#include "mpi/matmul_mpi.h"

#include "kernel/kernel.h"

double mpi_distribute_X(const grid_t *g, const layout_t *l, scalar_t *X_loc)
{
    const int x_count = l->n_loc * l->k;
    double t0 = MPI_Wtime();

    /* In g->col il rank 0 coincide, per l'invariante verificato da
     * grid_create(), con la coordinata cartesiana di riga 0. */
    MPI_Bcast(X_loc, x_count, SCALAR_MPI_TYPE, 0, g->col);
    return MPI_Wtime() - t0;
}

void mpi_matmul(const grid_t *g, const layout_t *l,
                const scalar_t *A_loc,
                const scalar_t *X_loc,
                scalar_t *Ypart,
                scalar_t *Y_loc,
                matmul_time_t *t)
{
    const int y_count = l->m_loc * l->k;
    double t0, t1, t2;

    /* 1. Contributo locale: righe [row0, row0+m_loc) di Y, limitatamente alle
     *    colonne [col0, col0+n_loc) di A. E' un risultato PARZIALE. */
    t0 = MPI_Wtime();
    local_gemm(l->m_loc, l->n_loc, l->k,
               A_loc, l->lda,
               X_loc, l->ldx,
               Ypart, l->ldy);

    /* 2. I Pc parziali della stessa banda di righe vivono sui processi della
     *    riga my_row: sommandoli si ottiene Y. count = m_loc*k, non 1: la
     *    reduce agisce lungo la dimensione dei processi e preserva la
     *    posizione, producendo m_loc*k somme indipendenti.
     *    root = rank 0 di row, cioe' il processo di colonna 0, che e' gia' il
     *    proprietario designato di quella porzione di Y: nessuna
     *    ridistribuzione finale. */
    t1 = MPI_Wtime();
    MPI_Reduce(Ypart, Y_loc, y_count, SCALAR_MPI_TYPE, MPI_SUM, 0, g->row);
    t2 = MPI_Wtime();

    if (t != NULL) {
        t->t_local = t1 - t0;
        t->t_reduce = t2 - t1;
        t->t_total = t->t_local + t->t_reduce;
    }
}
