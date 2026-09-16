#ifndef GRID_H
#define GRID_H

#include <mpi.h>

typedef struct {
    MPI_Comm grid_comm;
    MPI_Comm row_comm;
    MPI_Comm col_comm;

    int nprocs;
    int rank;

    int pr, pc;
    int my_row;
    int my_col;
} grid_t;

void grid_create(MPI_Comm comm, int pr, int pc, grid_t *g);

void grid_free(grid_t *g);

void grid_default_shape(int p, int *pr, int *pc);

#endif
