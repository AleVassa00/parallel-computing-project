#include "kernel/kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

/* cuda_warp.cu viene compilato separatamente rinominando soltanto i simboli
 * dell'interfaccia, per confrontare entrambi i backend nello stesso processo. */
extern "C" {
local_gemm_t *reference_create(int, int, int, const scalar_t *, int, int, int);
void reference_gemm(local_gemm_t *, const scalar_t *, int, scalar_t *, int);
void reference_destroy(local_gemm_t *);
}

static void require(bool condition, const char *message)
{
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(EXIT_FAILURE);
    }
}

static void check_case(int m, int n, int k, int padding,
                       double *max_serial_error, double *max_warp_error)
{
    const int lda = n + padding, ldx = k + padding, ldy = k + padding;
    std::vector<scalar_t> a(std::max((size_t)1, (size_t)m * lda), (scalar_t)777);
    std::vector<scalar_t> x(std::max((size_t)1, (size_t)n * ldx), (scalar_t)999);
    std::vector<scalar_t> y(std::max((size_t)1, (size_t)m * ldy));
    std::vector<scalar_t> reference(y.size());
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            a[(size_t)i * lda + j] = (scalar_t)(((i * 17 + j * 7) % 29 - 14) / 17.0);

    local_gemm_t *ctx = local_gemm_create(m, n, k, a.data(), lda, ldx, ldy);
    local_gemm_t *ref = reference_create(m, n, k, a.data(), lda, ldx, ldy);
    require(local_gemm_last_compute_seconds(ctx) < 0, "kernel time before first invocation");
    require(local_gemm_setup_seconds(ctx) >= 0, "setup time");

    /* Due X diverse: verifica la copia a ogni chiamata e Y=A*X, senza accumulo. */
    for (int rep = 0; rep < 2; ++rep) {
        for (int j = 0; j < n; ++j)
            for (int c = 0; c < k; ++c)
                x[(size_t)j * ldx + c] =
                    (scalar_t)(((j * 11 + c * 3 + rep * 5) % 31 - 15) / 19.0);
        std::fill(y.begin(), y.end(), (scalar_t)-123);
        local_gemm(ctx, x.data(), ldx, y.data(), ldy);
        reference_gemm(ref, x.data(), ldx, reference.data(), ldy);
        require(std::isfinite(local_gemm_last_compute_seconds(ctx)) &&
                local_gemm_last_compute_seconds(ctx) >= 0, "finite nonnegative CUDA event time");

        long double serial_norm = 0, serial_diff = 0, warp_norm = 0, warp_diff = 0;
        for (int i = 0; i < m; ++i) {
            for (int c = 0; c < k; ++c) {
                long double expected = 0;
                for (int j = 0; j < n; ++j)
                    expected += (long double)a[(size_t)i * lda + j] * x[(size_t)j * ldx + c];
                const scalar_t actual = y[(size_t)i * ldy + c];
                const scalar_t old = reference[(size_t)i * ldy + c];
                require(std::isfinite(actual) && std::isfinite(old), "finite output");
                serial_norm += expected * expected;
                serial_diff += ((long double)actual - expected) * ((long double)actual - expected);
                warp_norm += (long double)old * old;
                warp_diff += ((long double)actual - old) * ((long double)actual - old);
            }
        }
        const double serial_error = (double)std::sqrt(serial_norm > 0 ? serial_diff / serial_norm : serial_diff);
        const double warp_error = (double)std::sqrt(warp_norm > 0 ? warp_diff / warp_norm : warp_diff);
        *max_serial_error = std::max(*max_serial_error, serial_error);
        *max_warp_error = std::max(*max_warp_error, warp_error);
        if (serial_error > SCALAR_CHECK_TOL || warp_error > SCALAR_CHECK_TOL) {
            std::fprintf(stderr, "m=%d n=%d k=%d padding=%d serial=%.3e cuda_warp=%.3e\n",
                         m, n, k, padding, serial_error, warp_error);
            require(false, "relative L2 tolerance");
        }
    }
    local_gemm_destroy(ctx);
    reference_destroy(ref);
}

int main()
{
    const int ks[] = {3, 6, 8, 20, 32, 1, 7, 17, 40, 65, 257};
    const int shapes[][2] = {{1, 1}, {13, 19}, {17, 65}, {9, 173}, {0, 29}, {5, 0}};
    double max_serial_error = 0, max_warp_error = 0;
    for (int k : ks)
        for (const auto &shape : shapes)
            for (int padding : {0, 3})
                check_case(shape[0], shape[1], k, padding, &max_serial_error, &max_warp_error);
    local_gemm_destroy(NULL);
    std::printf("PASS %s %s: 264 invocations; max relative L2 serial=%.3e cuda_warp=%.3e (tol %.1e)\n",
                kernel_name(), SCALAR_NAME, max_serial_error, max_warp_error, (double)SCALAR_CHECK_TOL);
}
