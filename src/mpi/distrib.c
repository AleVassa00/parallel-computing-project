#include "mpi/distrib.h"

#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "index/index.h"

#define TAG_DISTRIBUTE_A 1701

#ifndef TEST_A_PADDING
#define TEST_A_PADDING 0
#endif

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

static int element_count(MPI_Comm comm, int rows, int columns,
                         const char *operation)
{
    if (rows > 0 && columns > INT_MAX / rows)
        mpi_abort_error(comm, MPI_ERR_COUNT, operation);
    return rows * columns;
}

void layout_init(layout_t *layout, const grid_t *grid, int M, int N, int k)
{
    layout->M = M;
    layout->N = N;
    layout->k = k;

    /* Le righe di A (e quindi di Y) sono divise fra le Pr righe della griglia,
     * le colonne di A (e quindi le righe di X) fra le Pc colonne. */
    layout->m_loc = block_size(M, grid->pr, grid->my_row);
    layout->row0 = block_start(M, grid->pr, grid->my_row);
    layout->n_loc = block_size(N, grid->pc, grid->my_col);
    layout->col0 = block_start(N, grid->pc, grid->my_col);

    /* m_loc dipende solo da my_row e n_loc solo da my_col: tutti i processi di
     * una stessa riga della griglia concordano su m_loc (count della reduce) e
     * tutti quelli di una stessa colonna su n_loc (count del broadcast).
     * E' la condizione di consistenza delle due collettive. */

    /* La build normale usa lda == n_loc. Il target check-padding definisce
     * TEST_A_PADDING=8 per collaudare il datatype ricevente con stride locale
     * senza esporre il padding come opzione pubblica. */
    layout->lda = layout->n_loc + TEST_A_PADDING;
    layout->ldx = k;
    layout->ldy = k;
}

void distribute_global_A(const grid_t *grid, const layout_t *layout, const scalar_t *A_global, scalar_t *A_loc)
{
    const int root = 0;
    int error_code;

    if (grid->rank == root) {
        int r, c;

        if (A_global == NULL)
            mpi_abort_error(grid->grid_comm, MPI_ERR_BUFFER,
                            "A_global is NULL on grid rank 0");

        for (r = 0; r < grid->pr; r++) {
            const int row0 = block_start(layout->M, grid->pr, r);
            const int m_loc = block_size(layout->M, grid->pr, r);

            for (c = 0; c < grid->pc; c++) {
                const int col0 = block_start(layout->N, grid->pc, c);
                const int n_loc = block_size(layout->N, grid->pc, c);
                const int count = element_count(
                    grid->grid_comm, m_loc, n_loc,
                    "local A block count exceeds the MPI int count range");
                int coords[2] = { r, c };
                int destination;

                error_code = MPI_Cart_rank(grid->grid_comm, coords, &destination);
                if (error_code != MPI_SUCCESS)
                    mpi_abort_error(grid->grid_comm, error_code, "MPI_Cart_rank");

                /* Nessun messaggio e nessun datatype per un blocco vuoto.
                 * Il processo destinatario segue la stessa condizione. */
                if (count == 0)
                    continue;

                if (destination == root) {
                    int i;
                    for (i = 0; i < m_loc; i++) {
                        const scalar_t *source =
                            A_global + (size_t)(row0 + i) * (size_t)layout->N + col0;
                        scalar_t *target = A_loc + (size_t)i * (size_t)layout->lda;
                        memcpy(target, source, (size_t)n_loc * sizeof *target);
                    }
                } else {
                    MPI_Datatype block_type = MPI_DATATYPE_NULL;
                    const scalar_t *start =
                        A_global + (size_t)row0 * (size_t)layout->N + col0;

                    error_code = MPI_Type_vector(m_loc, n_loc, layout->N,
                                                 SCALAR_MPI_TYPE, &block_type);
                    if (error_code != MPI_SUCCESS)
                        mpi_abort_error(grid->grid_comm, error_code, "MPI_Type_vector");

                    error_code = MPI_Type_commit(&block_type);
                    if (error_code != MPI_SUCCESS) {
                        MPI_Type_free(&block_type);
                        mpi_abort_error(grid->grid_comm, error_code, "MPI_Type_commit");
                    }

                    /* MPI_Send e' bloccante: al ritorno il buffer globale e il
                     * datatype non sono piu' in uso e il tipo puo' essere
                     * liberato immediatamente. */
                    error_code = MPI_Send(start, 1, block_type, destination,
                                          TAG_DISTRIBUTE_A, grid->grid_comm);
                    if (error_code != MPI_SUCCESS) {
                        MPI_Type_free(&block_type);
                        mpi_abort_error(grid->grid_comm, error_code, "MPI_Send(A block)");
                    }

                    error_code = MPI_Type_free(&block_type);
                    if (error_code != MPI_SUCCESS)
                        mpi_abort_error(grid->grid_comm, error_code, "MPI_Type_free");
                }
            }
        }
    } else {
        const int count = element_count(
            grid->grid_comm, layout->m_loc, layout->n_loc,
            "local A block count exceeds the MPI int count range");

        if (count > 0) {
            MPI_Datatype recv_type = MPI_DATATYPE_NULL;
            MPI_Status status;

            /* Il sender cammina nel globale con stride N; il receiver scrive
             * direttamente nel layout locale, che puo' avere lda > n_loc. */
            error_code = MPI_Type_vector(layout->m_loc, layout->n_loc, layout->lda,
                                         SCALAR_MPI_TYPE, &recv_type);
            if (error_code != MPI_SUCCESS)
                mpi_abort_error(grid->grid_comm, error_code,
                                "MPI_Type_vector(local A block)");
            error_code = MPI_Type_commit(&recv_type);
            if (error_code != MPI_SUCCESS) {
                MPI_Type_free(&recv_type);
                mpi_abort_error(grid->grid_comm, error_code,
                                "MPI_Type_commit(local A block)");
            }

            error_code = MPI_Recv(A_loc, 1, recv_type, root,
                                  TAG_DISTRIBUTE_A, grid->grid_comm, &status);
            if (error_code != MPI_SUCCESS) {
                MPI_Type_free(&recv_type);
                mpi_abort_error(grid->grid_comm, error_code, "MPI_Recv(A block)");
            }
            error_code = MPI_Type_free(&recv_type);
            if (error_code != MPI_SUCCESS)
                mpi_abort_error(grid->grid_comm, error_code,
                                "MPI_Type_free(local A block)");
        }
    }
}

void distribute_global_X(const grid_t *grid, const layout_t *layout,
                         const scalar_t *X_global, scalar_t *X_loc)
{
    const int root = 0;
    MPI_Datatype recv_type = SCALAR_MPI_TYPE;
    int *sendcounts = NULL, *displs = NULL;
    int recv_count, error_code;
    int recv_type_is_derived = 0;

    /* Solo la riga 0 possiede inizialmente X: gli altri processi riceveranno
     * la stessa fetta dal normale broadcast lungo col_comm in mpi_matmul. */
    if (grid->my_row != 0)
        return;

    if (layout->ldx < layout->k)
        mpi_abort_error(grid->grid_comm, MPI_ERR_ARG,
                        "X_loc leading dimension is smaller than k");

    /* MPI_Scatterv usa count e displacement di tipo int. Il controllo sul
     * prodotto globale implica che anche tutti i conteggi e displacement dei
     * singoli blocchi siano rappresentabili. */
    (void)element_count(
        grid->grid_comm, layout->N, layout->k,
        "global X element count exceeds the MPI int count range");
    recv_count = element_count(
        grid->grid_comm, layout->n_loc, layout->k,
        "local X block count exceeds the MPI int count range");

    if (grid->rank == root) {
        int c;

        if (X_global == NULL)
            mpi_abort_error(grid->grid_comm, MPI_ERR_BUFFER,
                            "X_global is NULL on grid rank 0");

        sendcounts = malloc((size_t)grid->pc * sizeof *sendcounts);
        displs = malloc((size_t)grid->pc * sizeof *displs);
        if (sendcounts == NULL || displs == NULL) {
            free(sendcounts);
            free(displs);
            mpi_abort_error(grid->grid_comm, MPI_ERR_NO_MEM,
                            "allocate X Scatterv metadata");
        }

        for (c = 0; c < grid->pc; c++) {
            sendcounts[c] = element_count(
                grid->grid_comm,
                block_size(layout->N, grid->pc, c), layout->k,
                "X Scatterv send count exceeds the MPI int count range");
            displs[c] = element_count(
                grid->grid_comm,
                block_start(layout->N, grid->pc, c), layout->k,
                "X Scatterv displacement exceeds the MPI int range");
        }
    }

    /* X_global e' compatta (ld = k), quindi il lato sorgente usa scalar_t
     * contigui. Solo un eventuale X_loc padded richiede un tipo vettoriale
     * ricevente per saltare da una riga locale alla successiva. */
    if (recv_count > 0 && layout->ldx != layout->k) {
        error_code = MPI_Type_vector(layout->n_loc, layout->k, layout->ldx,
                                     SCALAR_MPI_TYPE, &recv_type);
        if (error_code != MPI_SUCCESS)
            mpi_abort_error(grid->grid_comm, error_code,
                            "MPI_Type_vector(local X block)");
        error_code = MPI_Type_commit(&recv_type);
        if (error_code != MPI_SUCCESS) {
            MPI_Type_free(&recv_type);
            mpi_abort_error(grid->grid_comm, error_code,
                            "MPI_Type_commit(local X block)");
        }
        recv_count = 1;
        recv_type_is_derived = 1;
    }

    error_code = MPI_Scatterv(X_global, sendcounts, displs, SCALAR_MPI_TYPE,
                              X_loc, recv_count, recv_type,
                              root, grid->row_comm);
    if (error_code != MPI_SUCCESS) {
        if (recv_type_is_derived)
            MPI_Type_free(&recv_type);
        free(sendcounts);
        free(displs);
        mpi_abort_error(grid->grid_comm, error_code, "MPI_Scatterv(X blocks)");
    }

    if (recv_type_is_derived) {
        error_code = MPI_Type_free(&recv_type);
        if (error_code != MPI_SUCCESS) {
            free(sendcounts);
            free(displs);
            mpi_abort_error(grid->grid_comm, error_code,
                            "MPI_Type_free(local X block)");
        }
    }

    free(sendcounts);
    free(displs);
}

void layout_y_counts(const layout_t *layout, const grid_t *grid, int *counts, int *displs)
{
    int r;
    for (r = 0; r < grid->pr; r++) {
        counts[r] = block_size(layout->M, grid->pr, r) * layout->k;
        displs[r] = block_start(layout->M, grid->pr, r) * layout->k;
    }
}
