#ifndef SCALAR_H
#define SCALAR_H

#include <float.h>
#include <math.h>

#ifdef USE_FLOAT
typedef float scalar_t;
#define SCALAR_MPI_TYPE  MPI_FLOAT
#define SCALAR_NAME      "float"
#define SCALAR_EPS       ((double)FLT_EPSILON)
#else
typedef double scalar_t;
#define SCALAR_MPI_TYPE  MPI_DOUBLE
#define SCALAR_NAME      "double"
#define SCALAR_EPS       ((double)DBL_EPSILON)
#endif

#define SCALAR_CHECK_SAFETY 8.0

static inline double scalar_check_tol(int n)
{
    const double terms = (n > 1) ? (double)n : 1.0;
    return SCALAR_CHECK_SAFETY * SCALAR_EPS * sqrt(terms);
}

#endif
