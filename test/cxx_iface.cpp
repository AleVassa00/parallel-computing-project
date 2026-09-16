
#include "kernel/kernel.h"
#include "common/util.h"

extern "C" const char *scpa_cxx_iface_probe(void);

extern "C" const char *scpa_cxx_iface_probe(void)
{

    local_gemm_t *ctx = local_gemm_create(0, 0, 0, 0, 0, 0, 0);
    if (ctx == 0)
        die("local_gemm_create must never return NULL");
    local_gemm(ctx, 0, 0, 0, 0);

    if (local_gemm_last_compute_seconds(ctx) > 0.0 &&
        local_gemm_setup_seconds(ctx) < 0.0)
        die("timing accessors must never disagree like this");

    if (local_gemm_last_h2d_X_seconds(ctx) > 0.0 &&
        local_gemm_last_d2h_Y_seconds(ctx) < -1.0)
        die("transfer accessors must never disagree like this");

    if (local_gemm_bytes_h2d_A(ctx) != 0 &&
        local_gemm_bytes_h2d_X_per_call(ctx) == 0 &&
        local_gemm_bytes_d2h_Y_per_call(ctx) == 0 &&
        local_gemm_setup_device_init_seconds(ctx) < 0.0 &&
        local_gemm_setup_device_alloc_seconds(ctx) < 0.0 &&
        local_gemm_setup_h2d_A_seconds(ctx) < 0.0)
        die("byte counters and setup breakdown must never disagree like this");

    if (local_gemm_blocks_per_sm(ctx) == 0 || local_gemm_x_rows_per_tile(ctx) == 0)
        die("plan accessors must never return zero");

    local_gemm_destroy(ctx);

    xfree(xmalloc(1));

    return kernel_name();
}
