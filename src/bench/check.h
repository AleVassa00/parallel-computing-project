#ifndef CHECK_H
#define CHECK_H

#include <stdint.h>

#include "common/scalar.h"
#include "mpi/distrib.h"
#include "mpi/grid.h"

double check_against_serial(const grid_t *g, const layout_t *l,
                            const scalar_t *Y_loc, uint64_t seed);

#endif
