/* clock_gettime e CLOCK_MONOTONIC non sono ISO C: con -std=c11 (senza
 * estensioni GNU) vanno richiesti esplicitamente. */
#define _POSIX_C_SOURCE 200809L

#include "common/util.h"

#include <mpi.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SCPA_ALIGN 64

void *xmalloc(size_t bytes)
{
    void *p = NULL;

    /* Un blocco locale puo' essere legittimamente vuoto (griglia con piu'
     * processi che indici globali). Si alloca comunque il minimo: il puntatore
     * resta valido, e nessun percorso deve preoccuparsi di aritmetica su NULL
     * o di buffer nulli passati alle collettive con count == 0. */
    if (bytes == 0) bytes = 1;

    /* aligned_alloc esige una dimensione multipla dell'allineamento */
    bytes = (bytes + SCPA_ALIGN - 1) / SCPA_ALIGN * SCPA_ALIGN;
    p = aligned_alloc(SCPA_ALIGN, bytes);
    if (p == NULL)
        die("out of memory: cannot allocate %zu bytes", bytes);
    return p;
}

void xfree(void *p)
{
    free(p);
}

double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

void die(const char *fmt, ...)
{
    int initialized = 0, finalized = 0;
    va_list ap;

    va_start(ap, fmt);
    fputs("fatal: ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    fflush(stderr);

    /* Prima di MPI_Init (e dopo MPI_Finalize) exit e' l'unica terminazione
     * lecita. Durante un job MPI, invece, la morte di un solo rank potrebbe
     * lasciare gli altri bloccati in una collettiva: termina l'intero job. */
    MPI_Initialized(&initialized);
    if (initialized)
        MPI_Finalized(&finalized);
    if (initialized && !finalized)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    exit(EXIT_FAILURE);
}
