#include "mpi/matmul_mpi.h"

#include <stddef.h>

#include "common/util.h"

void mpi_matmul(const grid_t *grid, const layout_t *layout, local_gemm_t *local_gemm_context,  scalar_t *X_loc, scalar_t *Y_loc_part, scalar_t *Y_row_col0, matmul_time_t *times_struct_rep, int group_timing)
{
    const int x_count = layout->n_loc * layout->k;
    const int y_count = layout->m_loc * layout->k;
    double t0, t1, t2, t3;
    double tg0 = 0.0, tg1 = 0.0;

    if (layout->ldx != layout->k || layout->ldy != layout->k)
        die("mpi_matmul: X e Y devono essere contigue (ldx=%d, ldy=%d, k=%d): "
            "MPI_Bcast e MPI_Reduce usano conteggi contigui",
            layout->ldx, layout->ldy, layout->k);

    t0 = MPI_Wtime();

    MPI_Bcast(X_loc, x_count, SCALAR_MPI_TYPE, 0, grid->col_comm);

    t1 = MPI_Wtime();

    if (group_timing) {
        MPI_Barrier(grid->grid_comm);
        tg0 = MPI_Wtime();
    }

    local_gemm(local_gemm_context, X_loc, layout->ldx, Y_loc_part, layout->ldy);

    if (group_timing) {
        MPI_Barrier(grid->grid_comm);
        tg1 = MPI_Wtime();
    }

    t2 = MPI_Wtime();

    MPI_Reduce(Y_loc_part, Y_row_col0, y_count, SCALAR_MPI_TYPE, MPI_SUM, 0, grid->row_comm);

    t3 = MPI_Wtime();

    if (times_struct_rep != NULL) {
        const double kernel_time = local_gemm_last_compute_seconds(local_gemm_context);
        const double h2d_X_transfer_time = local_gemm_last_h2d_X_seconds(local_gemm_context);
        const double d2h_Y_transfer_time = local_gemm_last_d2h_Y_seconds(local_gemm_context);

        times_struct_rep->bcast_time = t1 - t0;
        times_struct_rep->local_phase_time = t2 - t1;
        times_struct_rep->reduce_time = t3 - t2;
        times_struct_rep->total_time = t3 - t0;

        times_struct_rep->kernel_time = kernel_time;
        times_struct_rep->official_time = (kernel_time >= 0.0)
                      ? times_struct_rep->bcast_time + kernel_time + times_struct_rep->reduce_time
                      : times_struct_rep->total_time;

        times_struct_rep->h2d_X_transfer_time = h2d_X_transfer_time;
        times_struct_rep->d2h_Y_transfer_time = d2h_Y_transfer_time;

        times_struct_rep->launch_overhead_time =
            (kernel_time >= 0.0 && h2d_X_transfer_time >= 0.0 && d2h_Y_transfer_time >= 0.0)
              ? times_struct_rep->local_phase_time - kernel_time
                    - h2d_X_transfer_time - d2h_Y_transfer_time
              : -1.0;

        times_struct_rep->local_group_time = group_timing ? (tg1 - tg0) : -1.0;
    }
}
