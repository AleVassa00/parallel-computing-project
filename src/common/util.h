#ifndef SCPA_UTIL_H
#define SCPA_UTIL_H

#include <stddef.h>

/* malloc con allineamento a 64 byte (una cache line) e abort su fallimento.
 * L'allineamento serve al kernel: le righe di A e X partono cosi' a inizio
 * cache line, condizione necessaria (non sufficiente) per non sprecare banda. */
void *xmalloc(size_t bytes);
void  xfree(void *p);

/* Orologio monotono, secondi. Usato fuori dalle regioni MPI;
 * dentro l'algoritmo parallelo si usa MPI_Wtime. */
double now_seconds(void);

/* Errore fatale: usa exit prima di MPI_Init/dopo MPI_Finalize e MPI_Abort
 * durante un job MPI, evitando di lasciare rank bloccati nelle collettive. */
void die(const char *fmt, ...);

#endif /* SCPA_UTIL_H */
