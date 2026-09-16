#ifndef MATMUL_MPI_H
#define MATMUL_MPI_H

#include "common/scalar.h"
#include "kernel/kernel.h"
#include "mpi/distrib.h"
#include "mpi/grid.h"

typedef struct {
    double bcast_time;
    double local_phase_time;
    double reduce_time;
    double total_time;

    double official_time;

    double kernel_time;

    double h2d_X_transfer_time;
    double d2h_Y_transfer_time;

    double launch_overhead_time;

    double local_group_time;
} matmul_time_t;

void mpi_matmul(const grid_t *grid, const layout_t *layout,
                local_gemm_t *local_gemm_context,
                scalar_t *X_loc,
                scalar_t *Y_loc_part,
                scalar_t *Y_row_col0,
                matmul_time_t *times_struct_rep,
                int group_timing);

#endif
