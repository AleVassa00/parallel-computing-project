#include "mpi/matmul_mpi.h"

#include <stddef.h>

#include "common/util.h"

void mpi_matmul(const grid_t *grid, const layout_t *layout, local_gemm_t *local_gemm_context,  scalar_t *X_loc, scalar_t *Y_loc_part, scalar_t *Y_row_col0, matmul_time_t *times_struct_rep)
{
    const int x_count = layout->n_loc * layout->k; //numero elementi di x locali
    const int y_count = layout->m_loc * layout->k; //numero elemento di y locali
    double t0, t1, t2, t3;

    /* Le due collettive qui sotto trattano X_loc e Y come buffer CONTIGUI di
     * n_loc*k e m_loc*k elementi. E' vero soltanto se ldx == k e ldy == k.
     *
     * L'invariante viene IMPOSTO invece che sperato perche' violarlo non fa
     * fallire niente: produce risultati sbagliati in silenzio, che e' il modo
     * peggiore di sbagliare. E non si puo' rimediare costruendo qui un
     * datatype derivato che descriva lo stride: sarebbe una MPI_Type_vector +
     * commit + free per ogni invocazione, cioe' tre chiamate al runtime MPI
     * DENTRO la regione cronometrata, che e' proprio cio' che la misura non
     * deve contenere. Il padding che il progetto misura e' quello di A (lda,
     * vedi TEST_A_PADDING), e A non passa da nessuna collettiva.
     *
     * Il controllo e' fuori dal cronometro e costa due confronti fra int. */
    if (layout->ldx != layout->k || layout->ldy != layout->k)
        die("mpi_matmul: X e Y devono essere contigue (ldx=%d, ldy=%d, k=%d): "
            "MPI_Bcast e MPI_Reduce usano conteggi contigui",
            layout->ldx, layout->ldy, layout->k);

    t0 = MPI_Wtime();

    /* 1. In g->col il rank 0 coincide, per l'invariante verificato da
     *    grid_create(), con la coordinata cartesiana di riga 0. */
    MPI_Bcast(X_loc, x_count, SCALAR_MPI_TYPE, 0, grid->col_comm);

    t1 = MPI_Wtime();

    /* 2. Contributo locale: righe [row0, row0+m_loc) di Y, limitatamente alle
     *    colonne [col0, col0+n_loc) di A. E' un risultato PARZIALE. */
    local_gemm(local_gemm_context, X_loc, layout->ldx, Y_loc_part, layout->ldy);

    t2 = MPI_Wtime();

    /* 3. I Pc parziali della stessa banda di righe vivono sui processi della
     *    riga my_row: sommandoli si ottiene Y. count = m_loc*k, non 1: la
     *    reduce agisce lungo la dimensione dei processi e preserva la
     *    posizione, producendo m_loc*k somme indipendenti.
     *    root = rank 0 di row, cioe' il processo di colonna 0, che e' gia' il
     *    proprietario designato di quella porzione di Y: nessuna
     *    ridistribuzione finale. */
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
        /* Interrogato DOPO t3, non fra t2 e la reduce: e' una lettura di uno
         * stato che il backend ha gia' congelato alla fine di local_gemm, e
         * metterla dentro una finestra cronometrata ne falserebbe il valore. */
        times_struct_rep->kernel_time = kernel_time;
        times_struct_rep->official_time = (kernel_time >= 0.0)
                      ? times_struct_rep->bcast_time + kernel_time + times_struct_rep->reduce_time
                      : times_struct_rep->total_time;

        /* Voci escluse dal tempo ufficiale e raccolte per discuterle a parte.
         * Vengono dagli stessi event gia' letti sopra, quindi non aggiungono
         * nessuna sincronizzazione: e' pura lettura di stato congelato. */
        times_struct_rep->h2d_X_transfer_time = h2d_X_transfer_time;
        times_struct_rep->d2h_Y_transfer_time = d2h_Y_transfer_time;

        /* L'overhead del runtime e' cio' che AVANZA, non una misura diretta:
         * il tempo della fase locale meno le tre cose che sappiamo nominare.
         * Solo un backend che le misura tutte e tre puo' produrlo, altrove
         * resta il sentinella negativo. */
        times_struct_rep->launch_overhead_time =
            (kernel_time >= 0.0 && h2d_X_transfer_time >= 0.0 && d2h_Y_transfer_time >= 0.0)
              ? times_struct_rep->local_phase_time - kernel_time
                    - h2d_X_transfer_time - d2h_Y_transfer_time
              : -1.0;
    }
}
