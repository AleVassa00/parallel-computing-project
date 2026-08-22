#include "mpi/grid.h"

#include "common/util.h"

/*
 * Calcola la forma della griglia in modo da renderla il più quadrata possibile.
 * Se p non e' un quadrato perfetto, la griglia sara' rettangolare con pr < pc.
 * Se p e' primo, la griglia sara' 1 x p.
*/
void grid_default_shape(int p, int *pr, int *pc)
{
    int r = 1, i;
    for (i = 1; i * i <= p; i++)
        if (p % i == 0)
            r = i;
    *pr = r;
    *pc = p / r;
}


/*
 * Creazione di una griglia cartesiana 2D con reorder.
 *
 * I rank in grid possono differire da quelli di MPI_COMM_WORLD.
 * 
 * La griglia e' divisa in righe e colonne per permettere Broadcast e Reduce lungo le due dimensioni.
 */
void grid_create(MPI_Comm comm, int pr, int pc, grid_t *g)
{
    int dims[2], periods[2] = { 0, 0 }, coords[2];
    int remain_row[2], remain_col[2];
    int size, sub_rank;

    MPI_Comm_size(comm, &size);
    if (pr <= 0 || pc <= 0 || pr * pc != size)
        die("invalid grid shape %dx%d for %d processes (pr*pc must equal P)",
            pr, pc, size);

    dims[0] = pr;
    dims[1] = pc;

    /* reorder = 1 lascia all'implementazione la liberta' di rinumerare i rank
     * per aderire alla topologia fisica.
     * Da qui in poi l'unico rank valido e' quello di g->grid.
    */
    MPI_Cart_create(comm, 2, dims, periods, 1, &g->grid);
    MPI_Comm_rank(g->grid, &g->rank);
    MPI_Cart_coords(g->grid, g->rank, 2, coords);

    g->nprocs = size;
    g->pr = pr;
    g->pc = pc;
    g->my_row = coords[0];
    g->my_col = coords[1];

    /* row: si tiene fissa la dimensione 0 (riga) e si lascia variare la 1 */
    remain_row[0] = 0;
    remain_row[1] = 1;
    MPI_Cart_sub(g->grid, remain_row, &g->row);

    /* col: si lascia variare la dimensione 0 (riga), fissa la colonna */
    remain_col[0] = 1;
    remain_col[1] = 0;
    MPI_Cart_sub(g->grid, remain_col, &g->col);

    /* Il codice conta su questa proprieta': in un sotto-comunicatore con una
     * sola dimensione residua i rank sono ordinati per coordinata crescente,
     * quindi il rank 0 di col e' il processo di riga 0 (root del broadcast di
     * X) e il rank 0 di row e' il processo di colonna 0 (root della reduce di
     * Y). Il assert la rende un'ipotesi verificata anziche' sperata. */
    MPI_Comm_rank(g->row, &sub_rank);
    if (sub_rank != g->my_col)
        die("row_comm rank %d != grid column %d", sub_rank, g->my_col);
    MPI_Comm_rank(g->col, &sub_rank);
    if (sub_rank != g->my_row)
        die("col_comm rank %d != grid row %d", sub_rank, g->my_row);
}

void grid_free(grid_t *g)
{
    if (g->row != MPI_COMM_NULL)
        MPI_Comm_free(&g->row);
    if (g->col != MPI_COMM_NULL)
        MPI_Comm_free(&g->col);
    if (g->grid != MPI_COMM_NULL)
        MPI_Comm_free(&g->grid);
}
