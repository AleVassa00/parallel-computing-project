
#define _POSIX_C_SOURCE 200809L

#include "common/util.h"

#include <mpi.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ALIGN 64

void *xmalloc(size_t bytes)
{
    void *p = NULL;

    if (bytes == 0) bytes = 1;

    bytes = (bytes + ALIGN - 1) / ALIGN * ALIGN;
    p = aligned_alloc(ALIGN, bytes);
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

    MPI_Initialized(&initialized);
    if (initialized)
        MPI_Finalized(&finalized);
    if (initialized && !finalized)
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);

    exit(EXIT_FAILURE);
}
