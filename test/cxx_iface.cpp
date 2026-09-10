/* Verifica di compilazione: kernel.h deve restare usabile da un .cu.
 *
 * nvcc compila i .cu come C++, quindi il backend CUDA vedra' questo header
 * esattamente come lo vede questo file. Non c'e' bisogno di avere CUDA
 * installato per accorgersi in anticipo di una rottura: bastano due proprieta'
 *   1. l'header deve essere PARSABILE da C++  -> `restrict` va scritto
 *      `__restrict__`, ed e' cio' che fa la macro RESTRICT;
 *   2. i simboli devono conservare il LINKAGE C -> senza extern "C" il name
 *      mangling li rinominerebbe e il link con gli oggetti prodotti da mpicc
 *      fallirebbe.
 *
 * Il file non viene mai eseguito: `make check-cxx` lo compila e poi controlla
 * con nm che i simboli richiesti siano ancora quelli non decorati. */

#include "kernel/kernel.h"
#include "common/util.h"

extern "C" const char *scpa_cxx_iface_probe(void);

extern "C" const char *scpa_cxx_iface_probe(void)
{
    /* Un backend CUDA scriverebbe esattamente questa sequenza: create in
     * preprocessing (H2D di A), local_gemm nella regione cronometrata,
     * destroy alla fine (cudaFree). */
    local_gemm_t *ctx = local_gemm_create(0, 0, 0, 0, 0, 0, 0);
    if (ctx == 0)
        die("local_gemm_create must never return NULL");   /* mai vero: e' il contratto */
    local_gemm(ctx, 0, 0, 0, 0);

    /* I due canali di misura fanno parte del contratto tanto quanto il
     * calcolo: se un giorno qualcuno li togliesse da extern "C", il backend
     * CUDA compilerebbe e poi non linkerebbe. Vanno quindi attraversati anche
     * qui, e il risultato va usato perche' la chiamata non sparisca. */
    if (local_gemm_last_compute_seconds(ctx) > 0.0 &&
        local_gemm_setup_seconds(ctx) < 0.0)
        die("timing accessors must never disagree like this");

    /* Stesso discorso per i canali dei tempi che la consegna esclude dalla
     * misura ufficiale ma chiede di potere discutere a parte: trasferimenti
     * per invocazione, byte trasferiti e scomposizione del preprocessing. Sono
     * parte del contratto quanto il calcolo, quindi vanno attraversati anche
     * qui e il risultato va usato, perche' la chiamata non venga eliminata. */
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

    /* Anche il piano fa parte del contratto: se un giorno qualcuno togliesse
     * questi due da extern "C", il backend CUDA compilerebbe e non linkerebbe. */
    if (local_gemm_blocks_per_sm(ctx) == 0 || local_gemm_x_rows_per_tile(ctx) == 0)
        die("plan accessors must never return zero");

    local_gemm_destroy(ctx);

    /* util.h serve al backend CUDA per xmalloc e per die() sugli errori CUDA */
    xfree(xmalloc(1));

    return kernel_name();
}
