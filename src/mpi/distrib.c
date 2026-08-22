#include "mpi/distrib.h"

#include "index/index.h"

void layout_init(layout_t *l, const grid_t *g, int M, int N, int k)
{
    l->M = M;
    l->N = N;
    l->k = k;

    /* Le righe di A (e quindi di Y) sono divise fra le Pr righe della griglia,
     * le colonne di A (e quindi le righe di X) fra le Pc colonne. */
    l->m_loc = block_size(M, g->pr, g->my_row);
    l->row0 = block_start(M, g->pr, g->my_row);
    l->n_loc = block_size(N, g->pc, g->my_col);
    l->col0 = block_start(N, g->pc, g->my_col);

    /* m_loc dipende solo da my_row e n_loc solo da my_col: tutti i processi di
     * una stessa riga della griglia concordano su m_loc (count della reduce) e
     * tutti quelli di una stessa colonna su n_loc (count del broadcast).
     * E' la condizione di consistenza delle due collettive. */

    l->lda = l->n_loc; /* nessun padding per ora: il gancio e' qui */
    l->ldx = k;
    l->ldy = k;
}

void layout_y_counts(const layout_t *l, const grid_t *g, int *counts, int *displs)
{
    int r;
    for (r = 0; r < g->pr; r++) {
        counts[r] = block_size(l->M, g->pr, r) * l->k;
        displs[r] = block_start(l->M, g->pr, r) * l->k;
    }
}
