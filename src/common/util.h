#ifndef SCPA_UTIL_H
#define SCPA_UTIL_H

#include <stddef.h>

/* Le stesse guardie di kernel.h: un backend .cu vorra' xmalloc e soprattutto
 * die() per la gestione degli errori CUDA, e nvcc compila i .cu come C++.
 * Senza extern "C" i simboli verrebbero cercati con il nome decorato. */
#if defined(__cplusplus)
extern "C" {
#endif

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

#if defined(__cplusplus)
}
#endif

#endif /* SCPA_UTIL_H */
