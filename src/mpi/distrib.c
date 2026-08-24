#include "mpi/distrib.h"

#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "index/index.h"

#define TAG_DISTRIBUTE_A 1701

static void mpi_abort_error(MPI_Comm comm, int error_code, const char *operation)
{
    char message[MPI_MAX_ERROR_STRING];
    int length = 0, rank = -1;

    MPI_Comm_rank(comm, &rank);
    MPI_Error_string(error_code, message, &length);
    fprintf(stderr, "fatal: grid rank %d: %s failed: %.*s\n",
            rank, operation, length, message);
    fflush(stderr);
    MPI_Abort(comm, error_code);
    exit(EXIT_FAILURE);
}

static int block_element_count(MPI_Comm comm, int m, int n)
{
    if (m > 0 && n > INT_MAX / m)
        mpi_abort_error(comm, MPI_ERR_COUNT,
                        "local A block count exceeds the MPI int count range");
    return m * n;
}

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

void distribute_global_A(const grid_t *g, const layout_t *l,
                         const scalar_t *A_global, scalar_t *A_loc)
{
    const int root = 0;
    int error_code;

    if (g->rank == root) {
        int r, c;

        if (A_global == NULL)
            mpi_abort_error(g->grid, MPI_ERR_BUFFER,
                            "A_global is NULL on grid rank 0");

        for (r = 0; r < g->pr; r++) {
            const int row0 = block_start(l->M, g->pr, r);
            const int m_loc = block_size(l->M, g->pr, r);

            for (c = 0; c < g->pc; c++) {
                const int col0 = block_start(l->N, g->pc, c);
                const int n_loc = block_size(l->N, g->pc, c);
                const int count = block_element_count(g->grid, m_loc, n_loc);
                int coords[2] = { r, c };
                int destination;

                error_code = MPI_Cart_rank(g->grid, coords, &destination);
                if (error_code != MPI_SUCCESS)
                    mpi_abort_error(g->grid, error_code, "MPI_Cart_rank");

                /* Nessun messaggio e nessun datatype per un blocco vuoto.
                 * Il processo destinatario segue la stessa condizione. */
                if (count == 0)
                    continue;

                if (destination == root) {
                    int i;
                    for (i = 0; i < m_loc; i++) {
                        const scalar_t *source =
                            A_global + (size_t)(row0 + i) * (size_t)l->N + col0;
                        scalar_t *target = A_loc + (size_t)i * (size_t)l->lda;
                        memcpy(target, source, (size_t)n_loc * sizeof *target);
                    }
                } else {
                    MPI_Datatype block_type = MPI_DATATYPE_NULL;
                    const scalar_t *start =
                        A_global + (size_t)row0 * (size_t)l->N + col0;

                    error_code = MPI_Type_vector(m_loc, n_loc, l->N,
                                                 SCALAR_MPI_TYPE, &block_type);
                    if (error_code != MPI_SUCCESS)
                        mpi_abort_error(g->grid, error_code, "MPI_Type_vector");

                    error_code = MPI_Type_commit(&block_type);
                    if (error_code != MPI_SUCCESS) {
                        MPI_Type_free(&block_type);
                        mpi_abort_error(g->grid, error_code, "MPI_Type_commit");
                    }

                    /* MPI_Send e' bloccante: al ritorno il buffer globale e il
                     * datatype non sono piu' in uso e il tipo puo' essere
                     * liberato immediatamente. */
                    error_code = MPI_Send(start, 1, block_type, destination,
                                          TAG_DISTRIBUTE_A, g->grid);
                    if (error_code != MPI_SUCCESS) {
                        MPI_Type_free(&block_type);
                        mpi_abort_error(g->grid, error_code, "MPI_Send(A block)");
                    }

                    error_code = MPI_Type_free(&block_type);
                    if (error_code != MPI_SUCCESS)
                        mpi_abort_error(g->grid, error_code, "MPI_Type_free");
                }
            }
        }
    } else {
        const int count = block_element_count(g->grid, l->m_loc, l->n_loc);

        if (count > 0) {
            MPI_Status status;
            error_code = MPI_Recv(A_loc, count, SCALAR_MPI_TYPE, root,
                                  TAG_DISTRIBUTE_A, g->grid, &status);
            if (error_code != MPI_SUCCESS)
                mpi_abort_error(g->grid, error_code, "MPI_Recv(A block)");
        }
    }
}

void layout_y_counts(const layout_t *l, const grid_t *g, int *counts, int *displs)
{
    int r;
    for (r = 0; r < g->pr; r++) {
        counts[r] = block_size(l->M, g->pr, r) * l->k;
        displs[r] = block_start(l->M, g->pr, r) * l->k;
    }
}
