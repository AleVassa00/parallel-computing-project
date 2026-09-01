/* Driver di misura del prodotto matrice x multivettore distribuito.
 *
 *   mpirun -np P ./bin/matmul_mpi -M .. -N .. -k .. --pr .. --pc ..
 *
 * Le modalita' predefinite generano localmente i blocchi di A e X; le modalita'
 * alternative materializzano il rispettivo input sul grid rank 0 e lo
 * distribuiscono. Tutti questi percorsi sono preprocessing. */

#include <mpi.h>

#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "bench/check.h"
#include "common/scalar.h"
#include "common/util.h"
#include "gen/gen.h"
#include "kernel/kernel.h"
#include "mpi/distrib.h"
#include "mpi/grid.h"
#include "mpi/matmul_mpi.h"

typedef enum {
    A_MODE_LOCAL,
    A_MODE_GLOBAL
} a_mode_t;

typedef enum {
    X_MODE_LOCAL,
    X_MODE_GLOBAL
} x_mode_t;

typedef struct {
    int M, N, k;
    int pr, pc;       /* 0 = forma della griglia scelta automaticamente */
    int pr_given, pc_given;
    int reps, warmup_reps;
    uint64_t seed;
    int check;
    int csv;
    a_mode_t a_mode;
    x_mode_t x_mode;
} opts_t;

static const char *CSV_HEADER =
    "kernel,scalar,a_mode,x_mode,M,N,k,P,pr,pc,reps,"
    "t_bcast_mean_s,t_local_mean_s,t_reduce_mean_s,t_total_mean_s,"
    "t_official_mean_s,t_total_median_s,t_total_min_s,t_kernel_mean_s,"
    "t_transfer_runtime_overhead_mean_s,t_setup_s,"
    "gflops,gflops_compute,gflops_kernel,rel_err";

static const char *a_mode_name(a_mode_t mode)
{
    return mode == A_MODE_GLOBAL ? "global" : "local";
}

static const char *x_mode_name(x_mode_t mode)
{
    return mode == X_MODE_GLOBAL ? "global" : "local";
}

static void usage(const char *prog)
{
    printf("usage: mpirun -np P %s [options]\n", prog);
    printf("  -M <int>        rows of A                        (default 1024)\n");
    printf("  -N <int>        columns of A                     (default 1024)\n");
    printf("  -k <int>        columns of the multivector       (default 8)\n");
    printf("  --pr <int>      process grid rows                (default: squarest)\n");
    printf("  --pc <int>      process grid columns             (default: squarest)\n");
    printf("  --reps <int>    timed repetitions                (default 10)\n");
    printf("  --warmup <int>  untimed warm-up repetitions      (default 2)\n");
    printf("  --seed <u64>    generator seed                   (default %llu)\n",
           (unsigned long long)GEN_DEFAULT_SEED);
    printf("  --a-mode <mode> generate A locally or distribute global A\n");
    printf("                    local (default), global\n");
    printf("  --x-mode <mode> generate X slices locally or distribute global X\n");
    printf("                    local (default), global\n");
    printf("  --check         validate against the serial reference\n");
    printf("  --csv           print one CSV row instead of the report\n");
    printf("  --csv-header    print the CSV header and exit\n");
    printf("  -h, --help      this message\n");
}

static int arg_int(int argc, char **argv, int *i, const char *name)
{
    const char *value;
    char *end = NULL;
    long parsed;

    if (*i + 1 >= argc)
        die("option %s requires a value", name);
    value = argv[++(*i)];
    errno = 0;
    parsed = strtol(value, &end, 10);
    if (errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX ||
        end == value || *end != '\0')
        die("invalid integer for %s: '%s'", name, value);
    return (int)parsed;
}

static uint64_t arg_seed(int argc, char **argv, int *i)
{
    const char *value;
    char *end = NULL;
    unsigned long long parsed;

    if (*i + 1 >= argc)
        die("option --seed requires a value");
    value = argv[++(*i)];
    errno = 0;
    if (*value == '-')
        die("invalid unsigned integer for --seed: '%s'", value);
    parsed = strtoull(value, &end, 10);
    if (errno == ERANGE || parsed > UINT64_MAX || end == value || *end != '\0')
        die("invalid unsigned integer for --seed: '%s'", value);
    return (uint64_t)parsed;
}

/* Tutti i processi vedono lo stesso argv e prendono quindi le stesse
 * decisioni: nessun bisogno di distribuire le opzioni. */
static void parse_args(int argc, char **argv, opts_t *options, int rank)
{
    int i;

    options->M = 1024;
    options->N = 1024;
    options->k = 8;
    options->pr = 0;
    options->pc = 0;
    options->pr_given = 0;
    options->pc_given = 0;
    options->reps = 10;
    options->warmup_reps = 2;
    options->seed = GEN_DEFAULT_SEED;
    options->check = 0;
    options->csv = 0;
    options->a_mode = A_MODE_LOCAL;
    options->x_mode = X_MODE_LOCAL;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-M"))
            options->M = arg_int(argc, argv, &i, "-M");
        else if (!strcmp(argv[i], "-N"))
            options->N = arg_int(argc, argv, &i, "-N");
        else if (!strcmp(argv[i], "-k"))
            options->k = arg_int(argc, argv, &i, "-k");
        else if (!strcmp(argv[i], "--pr")) {
            options->pr = arg_int(argc, argv, &i, "--pr");
            options->pr_given = 1;
        } else if (!strcmp(argv[i], "--pc")) {
            options->pc = arg_int(argc, argv, &i, "--pc");
            options->pc_given = 1;
        }
        else if (!strcmp(argv[i], "--reps"))
            options->reps = arg_int(argc, argv, &i, "--reps");
        else if (!strcmp(argv[i], "--warmup"))
            options->warmup_reps = arg_int(argc, argv, &i, "--warmup");
        else if (!strcmp(argv[i], "--seed"))
            options->seed = arg_seed(argc, argv, &i);
        else if (!strcmp(argv[i], "--a-mode")) {
            const char *mode;
            if (i + 1 >= argc)
                die("option --a-mode requires a value");
            mode = argv[++i];
            if (!strcmp(mode, "local"))
                options->a_mode = A_MODE_LOCAL;
            else if (!strcmp(mode, "global"))
                options->a_mode = A_MODE_GLOBAL;
            else
                die("invalid --a-mode '%s' (expected local or global)", mode);
        } else if (!strcmp(argv[i], "--x-mode")) {
            const char *mode;
            if (i + 1 >= argc)
                die("option --x-mode requires a value");
            mode = argv[++i];
            if (!strcmp(mode, "local"))
                options->x_mode = X_MODE_LOCAL;
            else if (!strcmp(mode, "global"))
                options->x_mode = X_MODE_GLOBAL;
            else
                die("invalid --x-mode '%s' (expected local or global)", mode);
        } else if (!strcmp(argv[i], "--check"))
            options->check = 1;
        else if (!strcmp(argv[i], "--csv"))
            options->csv = 1;
        else if (!strcmp(argv[i], "--csv-header")) {
            if (rank == 0)
                printf("%s\n", CSV_HEADER);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            if (rank == 0)
                usage(argv[0]);
            MPI_Finalize();
            exit(EXIT_SUCCESS);
        } else
            die("unknown option '%s' (try --help)", argv[i]);
    }

    if (options->M < 1 || options->N < 1 || options->k < 1)
        die("M, N and k must be positive (got %d, %d, %d)", options->M, options->N, options->k);
    if (options->reps < 1)
        die("--reps must be positive");
    if (options->warmup_reps < 0)
        die("--warmup must be non-negative");
    if (options->pr_given && options->pr < 1)
        die("--pr must be positive");
    if (options->pc_given && options->pc < 1)
        die("--pc must be positive");
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double vec_mean(const double *v, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++)
        s += v[i];
    return s / n;
}

int main(int argc, char **argv)
{
    opts_t options;
    grid_t grid;
    layout_t layout;

    scalar_t *A_loc, *X_loc, *Y_loc_part;
    scalar_t *A_global_root = NULL, *X_global_root = NULL;
    scalar_t *Y_row_col0 = NULL;

    local_gemm_t *local_gemm_context;

    double *bcast_times, *local_phase_times, *reduce_times, *total_times, *official_times, *kernel_times, *non_kernel_local_times, *sorted_total_times;
    double mean_bcast_time, mean_local_phase_time, mean_reduce_time, mean_total_time, mean_official_time;
    double mean_kernel_time, mean_compute_time, mean_non_kernel_local_time  = -1.0;
    double median_total_time, min_total_time, gflops, gflops_compute, rel_err = -1.0;
    double gflops_kernel = -1.0, setup_time;
    int world_rank, world_size, rep;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    parse_args(argc, argv, &options, world_rank);

    /* 
     * Se l'utente non ha specificato la forma della griglia, la calcolo in modo
     * da renderla il più quadrata possibile. Se invece ha specificato solo una
     * dimensione, calcolo l'altra in modo da usare tutti i processi.
    */
    if (!options.pr_given && !options.pc_given) {
        grid_default_shape(world_size, &options.pr, &options.pc);
    } else if (!options.pr_given) {
        if (world_size % options.pc != 0)
            die("--pc %d does not divide P=%d; cannot infer --pr",
                options.pc, world_size);
        options.pr = world_size / options.pc;
    } else if (!options.pc_given) {
        if (world_size % options.pr != 0)
            die("--pr %d does not divide P=%d; cannot infer --pc",
                options.pr, world_size);
        options.pc = world_size / options.pr;
    } else if ((long long)options.pr * (long long)options.pc != world_size) {
        die("invalid grid shape %dx%d for %d processes (pr*pc must equal P)",
            options.pr, options.pc, world_size);
    }

    /*
    * Creazione della griglia e del layout locale. La griglia e' cartesiana 2D
    * con reorder, quindi i rank in grid possono differire da quelli di
    * MPI_COMM_WORLD. Il layout locale e' calcolato a partire dalle dimensioni
    * globali e dalla forma della griglia.
    */
    grid_create(MPI_COMM_WORLD, options.pr, options.pc, &grid);

    /* Stabiliamo le porzioni di ogni matrice per il processo corrente.*/
    layout_init(&layout, &grid, options.M, options.N, options.k);

    /* Avviso se la griglia e' piu' grande della matrice: alcuni processi non
     * possiedono dati, ma partecipano comunque alla computazione.
     */
    if (grid.rank == 0 && (options.pr > options.M || options.pc > options.N))
        fprintf(stderr,
                "warning: grid %dx%d is larger than the matrix %dx%d, "
                "some processes own no data\n", options.pr, options.pc, options.M, options.N);

    /* Allocazione dei blocchi locali. */
    A_loc = xmalloc((size_t)layout.m_loc * layout.lda * sizeof *A_loc);
    X_loc = xmalloc((size_t)layout.n_loc * layout.ldx * sizeof *X_loc);
    Y_loc_part = xmalloc((size_t)layout.m_loc * layout.ldy * sizeof *Y_loc_part);
    if (grid.my_col == 0)
        Y_row_col0 = xmalloc((size_t)layout.m_loc * layout.ldy * sizeof *Y_row_col0);

    if (options.a_mode == A_MODE_LOCAL) {
        /* Modalita' predefinita: ogni processo genera direttamente il proprio
         * blocco e fa anche il first touch delle proprie pagine. */
        gen_block_A(A_loc, layout.lda, layout.m_loc, layout.n_loc,
                    layout.row0, layout.col0, layout.N, options.seed);
    } else {
        /* Modalita' alternativa: soltanto il grid rank 0 materializza la
         * matrice globale. Lo stesso generatore e gli stessi indici globali
         * garantiscono valori identici alla generazione locale. */
        if (grid.rank == 0) {
            A_global_root = xmalloc((size_t)layout.M * (size_t)layout.N * sizeof *A_global_root);
            gen_block_A(A_global_root, layout.N, layout.M, layout.N, 0, 0, layout.N, options.seed);
        }
        distribute_global_A(&grid, &layout, A_global_root, A_loc);
        xfree(A_global_root);
        A_global_root = NULL;
    }

    /* Preparazione del backend: e' PREPROCESSING, quindi sta fuori dalla
     * regione cronometrata. Va dopo che A_loc e' stata popolata, perche' e'
     * qui che un backend CUDA copierebbe A in VRAM una volta sola (la
     * consegna esclude dalla misura i trasferimenti da e verso la scheda).
     * Forma e leading dimension vengono dal layout una volta sola: il kernel
     * non puo' piu' essere invocato con dimensioni diverse da quelle con cui
     * A e' stata preparata. */
    local_gemm_context = local_gemm_create(layout.m_loc, layout.n_loc, layout.k, A_loc, layout.lda, layout.ldx, layout.ldy);

    /* X e' inizialmente collocata sulla sola riga 0 della griglia. In modalita'
     * local ogni processo di quella riga genera la propria fetta; in modalita'
     * global il grid rank 0 genera X compatta e la distribuisce sulla riga 0.
     * In entrambi i casi mpi_matmul esegue poi lo stesso broadcast verticale. */
    if (options.x_mode == X_MODE_LOCAL) {
        if (grid.my_row == 0)
            gen_block_X(X_loc, layout.ldx, layout.n_loc, layout.k,
                        layout.col0, options.seed);
    } else {
        /* MPI_Scatterv usa conteggi e displacement int. Controllare prima
         * dell'allocazione evita di materializzare un buffer non distribuibile. */
        if (layout.N > INT_MAX / layout.k)
            die("global X element count exceeds the MPI int count range");
        if (grid.rank == 0) {
            X_global_root = xmalloc((size_t)layout.N * (size_t)layout.k
                                    * sizeof *X_global_root);
            gen_block_X(X_global_root, layout.k, layout.N, layout.k,
                        0, options.seed);
        }
        distribute_global_X(&grid, &layout, X_global_root, X_loc);
        xfree(X_global_root);
        X_global_root = NULL;
    }

    bcast_times = xmalloc((size_t)options.reps * sizeof *bcast_times);
    total_times = xmalloc((size_t)options.reps * sizeof *total_times);
    local_phase_times = xmalloc((size_t)options.reps * sizeof *local_phase_times);
    reduce_times = xmalloc((size_t)options.reps * sizeof *reduce_times);
    official_times = xmalloc((size_t)options.reps * sizeof *official_times);
    kernel_times = xmalloc((size_t)options.reps * sizeof *kernel_times);
    non_kernel_local_times = xmalloc((size_t)options.reps * sizeof *non_kernel_local_times);
    sorted_total_times = xmalloc((size_t)options.reps * sizeof *sorted_total_times);

    for (rep = 0; rep < options.warmup_reps; rep++)
        mpi_matmul(&grid, &layout, local_gemm_context, X_loc, Y_loc_part, Y_row_col0, NULL);

    for (rep = 0; rep < options.reps; rep++) {

        matmul_time_t times_struct_rep;
        /* barriera prima di ogni ripetizione: senza, un processo in anticipo
         * comincerebbe a cronometrare mentre gli altri sono ancora indietro */
        MPI_Barrier(grid.grid_comm);

        mpi_matmul(&grid, &layout, local_gemm_context, X_loc, Y_loc_part, Y_row_col0, &times_struct_rep);

        bcast_times[rep] = times_struct_rep.bcast_time;
        total_times[rep] = times_struct_rep.total_time;
        local_phase_times[rep] = times_struct_rep.local_phase_time;
        reduce_times[rep] = times_struct_rep.reduce_time;
        official_times[rep] = times_struct_rep.official_time;
        kernel_times[rep] = times_struct_rep.kernel_time;
        non_kernel_local_times[rep] = (times_struct_rep.kernel_time >= 0.0) ? times_struct_rep.local_phase_time - times_struct_rep.kernel_time : -1.0;
    }

    /* Il tempo di una invocazione e' il MASSIMO fra i processi, non quello del
     * rank 0: l'operazione e' finita quando ha finito l'ultimo. Le riduzioni
     * si fanno alla fine, su tutto il vettore, per non disturbare le misure. */
    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : bcast_times, bcast_times, options.reps, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);
    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : local_phase_times, local_phase_times, options.reps, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);
    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : reduce_times, reduce_times, options.reps, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);
    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : total_times, total_times, options.reps, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);
    /* t_official e' costruito LOCALMENTE per ogni repetition e solo dopo si
     * prende il massimo: sommare medie/massimi delle singole fasi non darebbe
     * il cammino critico di una vera invocazione. */
    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : official_times, official_times, options.reps, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);
    /* Stesso criterio per il tempo di kernel e per il preprocessing: conta il
     * processo piu' lento, non il rank 0. Il sentinella negativo dei backend di
     * CPU sopravvive al massimo, perche' li' e' negativo su tutti i rank. */
    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : kernel_times, kernel_times, options.reps, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);
    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : non_kernel_local_times, non_kernel_local_times, options.reps, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);

    setup_time = local_gemm_setup_seconds(local_gemm_context);

    MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : &setup_time, &setup_time, 1, MPI_DOUBLE, MPI_MAX, 0, grid.grid_comm);

    if (options.check)
        rel_err = check_against_serial(&grid, &layout, Y_row_col0, options.seed);

    if (grid.rank == 0) {
        memcpy(sorted_total_times, total_times, (size_t)options.reps * sizeof *sorted_total_times);
        qsort(sorted_total_times, (size_t)options.reps, sizeof *sorted_total_times, cmp_double);
        mean_bcast_time = vec_mean(bcast_times, options.reps);
        mean_local_phase_time = vec_mean(local_phase_times, options.reps);
        mean_reduce_time = vec_mean(reduce_times, options.reps);
        mean_total_time = vec_mean(total_times, options.reps);
        mean_official_time = vec_mean(official_times, options.reps);
        mean_kernel_time = vec_mean(kernel_times, options.reps);
        mean_non_kernel_local_time  = vec_mean(non_kernel_local_times, options.reps);
        mean_compute_time = (mean_kernel_time >= 0.0) ? mean_kernel_time : mean_local_phase_time;
        median_total_time = (options.reps % 2) ? sorted_total_times[options.reps / 2]
                                    : 0.5 * (sorted_total_times[options.reps / 2 - 1] + sorted_total_times[options.reps / 2]);
        min_total_time = sorted_total_times[0];

        /* Metrica ufficiale: prodotto MPI completo, ma senza i trasferimenti
         * CUDA come richiesto dalla traccia. Su CPU mean_official coincide con
         * il totale end-to-end misurato direttamente. */
        gflops = 2.0 * (double)options.M * (double)options.N * (double)options.k
                  / mean_official_time / 1.0e9;
        gflops_compute = 2.0 * (double)options.M * (double)options.N * (double)options.k
                          / mean_compute_time / 1.0e9;
        /* Solo se il backend sa distinguere il kernel dai trasferimenti.
         * Questa e' la colonna che, sul backend CUDA, esclude H2D e D2H come
         * la consegna consente esplicitamente di fare. */
        if (mean_kernel_time > 0.0) {
            /* Alias CUDA esplicito, mantenuto nel CSV per distinguere a colpo
             * d'occhio le righe GPU. Il valore coincide con gflops_compute. */
            gflops_kernel = gflops_compute;
        }

        if (options.csv) {
            printf("%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,"
                   "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,"
                   "%.6f,%.6f,%.6f,%.3e\n",
                   kernel_name(), SCALAR_NAME, a_mode_name(options.a_mode),
                   x_mode_name(options.x_mode),
                   options.M, options.N, options.k, grid.nprocs, grid.pr, grid.pc, options.reps,
                   mean_bcast_time, mean_local_phase_time, mean_reduce_time, mean_total_time,
                   mean_official_time, median_total_time, min_total_time, mean_kernel_time,
                   mean_non_kernel_local_time , setup_time,
                   gflops, gflops_compute, gflops_kernel, rel_err);
        } else {
            double bytes_A = (double)options.M * options.N * sizeof(scalar_t);
            printf("matmul_mpi  M=%d N=%d k=%d  grid=%dx%d (P=%d)  %s  kernel=%s  A=%s X=%s\n",
                   options.M, options.N, options.k, grid.pr, grid.pc, grid.nprocs, SCALAR_NAME,
                   kernel_name(), a_mode_name(options.a_mode),
                   x_mode_name(options.x_mode));
            printf("  local block  A %dx%d   X %dx%d   Y %dx%d      A total %.1f MiB\n",
                   layout.m_loc, layout.n_loc, layout.n_loc, layout.k, layout.m_loc, layout.k,
                   bytes_A / 1048576.0);
            printf("  reps=%d warmup=%d seed=%llu\n",
                   options.reps, options.warmup_reps, (unsigned long long)options.seed);
            printf("  Bcast mean              %.3f ms\n", mean_bcast_time * 1e3);
            printf("  Local mean              %.3f ms\n", mean_local_phase_time * 1e3);
            printf("  Reduce mean             %.3f ms\n", mean_reduce_time * 1e3);
            printf("  Total mean (end-to-end) %.3f ms   median %.3f ms   min %.3f ms\n",
                   mean_total_time * 1e3, median_total_time * 1e3, min_total_time * 1e3);
            if (mean_kernel_time >= 0.0) {
                printf("  Official mean           %.3f ms   (GPU transfers excluded)\n",
                       mean_official_time * 1e3);
                printf("  Kernel mean             %.3f ms\n", mean_kernel_time * 1e3);
                printf("  Transfer/runtime ovh.   %.3f ms\n",
                       mean_non_kernel_local_time  * 1e3);
            } else {
                printf("  Official mean           %.3f ms   (same as end-to-end)\n",
                       mean_official_time * 1e3);
            }
            printf("  Backend setup           %.3f ms   (preprocessing, fuori dalla misura)\n",
                   setup_time * 1e3);
            printf("  GFLOPS MPI              %.3f\n", gflops);
            printf("  GFLOPS compute-only     %.3f\n", gflops_compute);
            if (gflops_kernel > 0.0)
                printf("  GFLOPS kernel-only      %.3f\n", gflops_kernel);
            printf("                          (2*M*N*k = %.3f GFLOP; official T = official mean)\n",
                   2.0 * (double)options.M * (double)options.N * (double)options.k / 1e9);
            if (options.check)
                printf("  validation              relative L2 error %.3e   [%s]\n",
                       rel_err, (rel_err <= SCALAR_CHECK_TOL) ? "PASS" : "FAIL");
        }
        fflush(stdout);
    }

    /* prima il backend, poi la memoria di A: il contesto la referenzia */
    local_gemm_destroy(local_gemm_context);
    xfree(A_loc);
    xfree(A_global_root);
    xfree(X_loc);
    xfree(X_global_root);
    xfree(Y_loc_part);
    xfree(Y_row_col0);
    xfree(bcast_times);
    xfree(total_times);
    xfree(local_phase_times);
    xfree(reduce_times);
    xfree(official_times);
    xfree(kernel_times);
    xfree(non_kernel_local_times);
    xfree(sorted_total_times);
    grid_free(&grid);
    MPI_Finalize();

    /* esito non nullo se la validazione fallisce: utile negli script */
    return (options.check && rel_err > SCALAR_CHECK_TOL) ? EXIT_FAILURE : EXIT_SUCCESS;
}
