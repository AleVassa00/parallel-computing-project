#include "mpi/grid.h"

#include "common/util.h"

void grid_default_shape(int p, int *pr, int *pc)
{
    int r = 1, i;
    for (i = 1; i * i <= p; i++)
        if (p % i == 0)
            r = i;
    *pr = r;
    *pc = p / r;
}

void grid_create(MPI_Comm comm, int pr, int pc, grid_t *grid)
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

    MPI_Cart_create(comm, 2, dims, periods, 1, &grid->grid_comm);
    MPI_Comm_rank(grid->grid_comm, &grid->rank);
    MPI_Cart_coords(grid->grid_comm, grid->rank, 2, coords);

    grid->nprocs = size;
    grid->pr = pr;
    grid->pc = pc;
    grid->my_row = coords[0];
    grid->my_col = coords[1];

    remain_row[0] = 0;
    remain_row[1] = 1;
    MPI_Cart_sub(grid->grid_comm, remain_row, &grid->row_comm);

    remain_col[0] = 1;
    remain_col[1] = 0;
    MPI_Cart_sub(grid->grid_comm, remain_col, &grid->col_comm);

    MPI_Comm_rank(grid->row_comm, &sub_rank);
    if (sub_rank != grid->my_col)
        die("row_comm rank %d != grid column %d", sub_rank, grid->my_col);
    MPI_Comm_rank(grid->col_comm, &sub_rank);
    if (sub_rank != grid->my_row)
        die("col_comm rank %d != grid row %d", sub_rank, grid->my_row);
}

void grid_free(grid_t *grid)
{
    if (grid->row_comm != MPI_COMM_NULL)
        MPI_Comm_free(&grid->row_comm);
    if (grid->col_comm != MPI_COMM_NULL)
        MPI_Comm_free(&grid->col_comm);
    if (grid->grid_comm != MPI_COMM_NULL)
        MPI_Comm_free(&grid->grid_comm);
}
