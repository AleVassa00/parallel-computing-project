#ifndef SCPA_GRID_H
#define SCPA_GRID_H

#include <mpi.h>

/* Griglia bidimensionale di processi Pr x Pc.
 *
 * Perche' due sotto-comunicatori e non uno:
 *  - cosa serve a un processo in ingresso (la fetta di X) dipende SOLO dalle
 *    colonne di A che possiede -> tutti i processi della stessa COLONNA della
 *    griglia vogliono la stessa fetta -> broadcast lungo col_comm;
 *  - cosa produce un processo in uscita (un parziale di Y) dipende SOLO dalle
 *    righe di A che possiede -> tutti i processi della stessa RIGA della
 *    griglia producono parziali della stessa porzione di Y -> reduce lungo
 *    row_comm.
 * Sono due informazioni indipendenti: e' per questo che la comunicazione si
 * spezza in una verticale e una orizzontale. */
typedef struct {
    MPI_Comm grid_comm; /* cartesiano 2D, con reorder */
    MPI_Comm row_comm;  /* processi con la stessa coordinata di RIGA (varia la colonna) */
    MPI_Comm col_comm;  /* processi con la stessa coordinata di COLONNA (varia la riga) */

    int nprocs;
    int rank;      /* rank in grid: NON usare mai quello di MPI_COMM_WORLD,
                    * con reorder = 1 i due possono differire */
    int pr, pc;    /* forma della griglia */
    int my_row;    /* coordinata cartesiana 0 */
    int my_col;    /* coordinata cartesiana 1 */
} grid_t;

/* Costruisce la griglia su comm. Termina il programma se pr*pc != size. */
void grid_create(MPI_Comm comm, int pr, int pc, grid_t *g);

void grid_free(grid_t *g);

/* Fattorizzazione piu' quadrata di p: pr <= pc, pr massimo con pr*pc == p.
 * Usata come default quando l'utente non impone la forma della griglia. */
void grid_default_shape(int p, int *pr, int *pc);

#endif /* SCPA_GRID_H */
