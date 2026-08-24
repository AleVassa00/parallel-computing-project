#ifndef SCPA_DISTRIB_H
#define SCPA_DISTRIB_H

#include "common/scalar.h"
#include "mpi/grid.h"

/* Forma dei blocchi locali sul processo corrente.
 *
 *   A e' tagliata lungo righe E colonne   -> (M/Pr) x (N/Pc)
 *   X e' tagliata solo lungo le righe     -> (N/Pc) x k     sulla riga 0
 *   Y e' tagliata solo lungo le righe     -> (M/Pr) x k     sulla colonna 0
 *
 * k non si divide mai: e' piccolo (i frammenti sarebbero messaggi dominati
 * dalla latenza) ed e' la dimensione da cui viene tutto il riuso di A.
 *
 * INVARIANTE CRITICO: il numero di righe locali di X deve venire dalla stessa
 * partizione con cui si tagliano le colonne di A, e il numero di righe locali
 * di Y dalla stessa con cui si tagliano le righe di A. Se le due divergono i
 * risultati sono sbagliati in modo silenzioso.
 * Qui l'invariante e' garantito per costruzione: esistono solo n_loc e m_loc,
 * calcolati una volta sola, e non due coppie di valori da tenere allineate. */
typedef struct {
    int M, N, k; /* dimensioni globali */

    int m_loc;   /* righe locali di A  = righe locali di Y */
    int n_loc;   /* colonne locali di A = righe locali di X */
    int row0;    /* prima riga globale di A posseduta */
    int col0;    /* prima colonna globale di A posseduta */

    int lda;     /* leading dimension di A_loc (row-major) */
    int ldx;     /* leading dimension di X_loc: k contiguo */
    int ldy;     /* leading dimension di Y_loc: k contiguo */
} layout_t;

void layout_init(layout_t *l, const grid_t *g, int M, int N, int k);

/* Distribuisce A_global, presente soltanto sul grid rank 0, nei blocchi
 * contigui A_loc descritti dal layout 2D. Sul root ogni blocco remoto e'
 * descritto direttamente dentro la matrice row-major con MPI_Type_vector;
 * il destinatario riceve invece m_loc*n_loc scalar_t contigui.
 *
 * A_global puo' essere NULL sui processi diversi dal grid rank 0. Gli errori
 * MPI sono riportati e causano MPI_Abort sul communicator cartesiano. */
void distribute_global_A(const grid_t *g, const layout_t *l,
                         const scalar_t *A_global, scalar_t *A_loc);

/* Vettori counts/displs (in ELEMENTI, non byte) per le collettive a lunghezza
 * variabile su Y lungo la colonna 0 della griglia: pezzi di righe consecutive
 * di Y, contigui perche' Y e' row-major con k contiguo. */
void layout_y_counts(const layout_t *l, const grid_t *g, int *counts, int *displs);

#endif /* SCPA_DISTRIB_H */
