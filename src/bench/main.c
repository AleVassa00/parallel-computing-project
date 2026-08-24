/* Driver di misura del prodotto matrice x multivettore distribuito.
 *
 *   mpirun -np P ./bin/matmul_mpi -M .. -N .. -k .. --pr .. --pc ..
 *
 * La modalita' predefinita genera A localmente su ciascun processo; la
 * modalita' alternativa genera A globale sul grid rank 0 e la distribuisce.
 * Entrambi i percorsi sono preprocessing, escluso dalla misura del kernel. */

#include <mpi.h>

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

typedef struct {
    int M, N, k;
    int pr, pc;       /* 0 = forma della griglia scelta automaticamente */
    int reps, warmup;
    uint64_t seed;
    int check;
    int csv;
    a_mode_t a_mode;
} opts_t;

static const char *CSV_HEADER =
    "kernel,scalar,a_mode,M,N,k,P,pr,pc,reps,"
    "t_local_mean_s,t_local_median_s,t_local_min_s,"
    "t_reduce_mean_s,t_total_mean_s,t_bcast_setup_s,"
    "gflops,gflops_total,rel_err";

static const char *a_mode_name(a_mode_t mode)
{
    return mode == A_MODE_GLOBAL ? "global" : "local";
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
    printf("  --check         validate against the serial reference\n");
    printf("  --csv           print one CSV row instead of the report\n");
    printf("  --csv-header    print the CSV header and exit\n");
    printf("  -h, --help      this message\n");
}

static int arg_int(int argc, char **argv, int *i, const char *name)
{
    if (*i + 1 >= argc)
        die("option %s requires a value", name);
    return (int)strtol(argv[++(*i)], NULL, 10);
}

/* Tutti i processi vedono lo stesso argv e prendono quindi le stesse
 * decisioni: nessun bisogno di distribuire le opzioni. */
static void parse_args(int argc, char **argv, opts_t *o, int rank)
{
    int i;

    o->M = 1024;
    o->N = 1024;
    o->k = 8;
    o->pr = 0;
    o->pc = 0;
    o->reps = 10;
    o->warmup = 2;
    o->seed = GEN_DEFAULT_SEED;
    o->check = 0;
    o->csv = 0;
    o->a_mode = A_MODE_LOCAL;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-M"))
            o->M = arg_int(argc, argv, &i, "-M");
        else if (!strcmp(argv[i], "-N"))
            o->N = arg_int(argc, argv, &i, "-N");
        else if (!strcmp(argv[i], "-k"))
            o->k = arg_int(argc, argv, &i, "-k");
        else if (!strcmp(argv[i], "--pr"))
            o->pr = arg_int(argc, argv, &i, "--pr");
        else if (!strcmp(argv[i], "--pc"))
            o->pc = arg_int(argc, argv, &i, "--pc");
        else if (!strcmp(argv[i], "--reps"))
            o->reps = arg_int(argc, argv, &i, "--reps");
        else if (!strcmp(argv[i], "--warmup"))
            o->warmup = arg_int(argc, argv, &i, "--warmup");
        else if (!strcmp(argv[i], "--seed")) {
            if (i + 1 >= argc)
                die("option --seed requires a value");
            o->seed = strtoull(argv[++i], NULL, 10);
        } else if (!strcmp(argv[i], "--a-mode")) {
            const char *mode;
            if (i + 1 >= argc)
                die("option --a-mode requires a value");
            mode = argv[++i];
            if (!strcmp(mode, "local"))
                o->a_mode = A_MODE_LOCAL;
            else if (!strcmp(mode, "global"))
                o->a_mode = A_MODE_GLOBAL;
            else
                die("invalid --a-mode '%s' (expected local or global)", mode);
        } else if (!strcmp(argv[i], "--check"))
            o->check = 1;
        else if (!strcmp(argv[i], "--csv"))
            o->csv = 1;
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

    if (o->M < 1 || o->N < 1 || o->k < 1)
        die("M, N and k must be positive (got %d, %d, %d)", o->M, o->N, o->k);
    if (o->reps < 1)
        die("--reps must be positive");
    if (o->warmup < 0)
        die("--warmup must be non-negative");
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
    opts_t o;
    grid_t g;
    layout_t lay;
    scalar_t *A_loc, *A_global = NULL, *X_loc, *Ypart, *Y_loc = NULL;
    double *t_tot, *t_lo, *t_re, *sorted;
    double mean_local, median_local, min_local;
    double mean_reduce, mean_total, gflops, gflops_total, rel_err = -1.0;
    double t_bcast_setup_local, t_bcast_setup = 0.0;
    int world_rank, world_size, r;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    parse_args(argc, argv, &o, world_rank);

    /* 
     * Se l'utente non ha specificato la forma della griglia, la calcolo in modo
     * da renderla il più quadrata possibile. Se invece ha specificato solo una
     * dimensione, calcolo l'altra in modo da usare tutti i processi.
    */
    if (o.pr == 0 && o.pc == 0)
        grid_default_shape(world_size, &o.pr, &o.pc);
    else if (o.pr == 0)
        o.pr = (o.pc > 0) ? world_size / o.pc : 0;
    else if (o.pc == 0)
        o.pc = world_size / o.pr;

    /*
    * Creazione della griglia e del layout locale. La griglia e' cartesiana 2D
    * con reorder, quindi i rank in grid possono differire da quelli di
    * MPI_COMM_WORLD. Il layout locale e' calcolato a partire dalle dimensioni
    * globali e dalla forma della griglia.
    */
    grid_create(MPI_COMM_WORLD, o.pr, o.pc, &g);

    /* Stabiliamo le porzioni di ogni matrice per il processo corrente.*/
    layout_init(&lay, &g, o.M, o.N, o.k);

    /* Avviso se la griglia e' piu' grande della matrice: alcuni processi non
     * possiedono dati, ma partecipano comunque alla computazione.
     */
    if (g.rank == 0 && (o.pr > o.M || o.pc > o.N))
        fprintf(stderr,
                "warning: grid %dx%d is larger than the matrix %dx%d, "
                "some processes own no data\n", o.pr, o.pc, o.M, o.N);

    /* Allocazione dei blocchi locali. */
    A_loc = xmalloc((size_t)lay.m_loc * lay.lda * sizeof *A_loc);
    X_loc = xmalloc((size_t)lay.n_loc * lay.ldx * sizeof *X_loc);
    Ypart = xmalloc((size_t)lay.m_loc * lay.ldy * sizeof *Ypart);
    if (g.my_col == 0)
        Y_loc = xmalloc((size_t)lay.m_loc * lay.ldy * sizeof *Y_loc);

    if (o.a_mode == A_MODE_LOCAL) {
        /* Modalita' predefinita: ogni processo genera direttamente il proprio
         * blocco e fa anche il first touch delle proprie pagine. */
        gen_block_A(A_loc, lay.lda, lay.m_loc, lay.n_loc,
                    lay.row0, lay.col0, lay.N, o.seed);
    } else {
        /* Modalita' alternativa: soltanto il grid rank 0 materializza la
         * matrice globale. Lo stesso generatore e gli stessi indici globali
         * garantiscono valori identici alla generazione locale. */
        if (g.rank == 0) {
            A_global = xmalloc((size_t)lay.M * (size_t)lay.N * sizeof *A_global);
            gen_block_A(A_global, lay.N, lay.M, lay.N, 0, 0, lay.N, o.seed);
        }
        distribute_global_A(&g, &lay, A_global, A_loc);
        xfree(A_global);
        A_global = NULL;
    }

    /* X e' collocata sulla riga 0 della griglia: solo quei processi la
     * generano; il singolo broadcast di setup la replica lungo le colonne. */
    if (g.my_row == 0)
        gen_block_X(X_loc, lay.ldx, lay.n_loc, lay.k, lay.col0, o.seed);

    t_bcast_setup_local = mpi_distribute_X(&g, &lay, X_loc);
    MPI_Reduce(&t_bcast_setup_local, &t_bcast_setup, 1, MPI_DOUBLE,
               MPI_MAX, 0, g.grid);

    t_tot = xmalloc((size_t)o.reps * sizeof *t_tot);
    t_lo = xmalloc((size_t)o.reps * sizeof *t_lo);
    t_re = xmalloc((size_t)o.reps * sizeof *t_re);
    sorted = xmalloc((size_t)o.reps * sizeof *sorted);

    for (r = 0; r < o.warmup; r++)
        mpi_matmul(&g, &lay, A_loc, X_loc, Ypart, Y_loc, NULL);

    for (r = 0; r < o.reps; r++) {
        matmul_time_t tt;
        /* barriera prima di ogni ripetizione: senza, un processo in anticipo
         * comincerebbe a cronometrare mentre gli altri sono ancora indietro */
        MPI_Barrier(g.grid);
        mpi_matmul(&g, &lay, A_loc, X_loc, Ypart, Y_loc, &tt);
        t_tot[r] = tt.t_total;
        t_lo[r] = tt.t_local;
        t_re[r] = tt.t_reduce;
    }

    /* Il tempo di una invocazione e' il MASSIMO fra i processi, non quello del
     * rank 0: l'operazione e' finita quando ha finito l'ultimo. Le riduzioni
     * si fanno alla fine, su tutto il vettore, per non disturbare le misure. */
    MPI_Reduce(g.rank == 0 ? MPI_IN_PLACE : t_tot, t_tot, o.reps, MPI_DOUBLE, MPI_MAX, 0, g.grid);
    MPI_Reduce(g.rank == 0 ? MPI_IN_PLACE : t_lo, t_lo, o.reps, MPI_DOUBLE, MPI_MAX, 0, g.grid);
    MPI_Reduce(g.rank == 0 ? MPI_IN_PLACE : t_re, t_re, o.reps, MPI_DOUBLE, MPI_MAX, 0, g.grid);

    if (o.check)
        rel_err = check_against_serial(&g, &lay, Y_loc, o.seed);

    if (g.rank == 0) {
        memcpy(sorted, t_lo, (size_t)o.reps * sizeof *sorted);
        qsort(sorted, (size_t)o.reps, sizeof *sorted, cmp_double);
        mean_local = vec_mean(t_lo, o.reps);
        median_local = (o.reps % 2) ? sorted[o.reps / 2]
                                    : 0.5 * (sorted[o.reps / 2 - 1] + sorted[o.reps / 2]);
        min_local = sorted[0];
        mean_reduce = vec_mean(t_re, o.reps);
        mean_total = vec_mean(t_tot, o.reps);

        /* Metrica ufficiale: T e' la MEDIA, sulle repetition, del massimo
         * tempo di SOLO local_gemm fra tutti i processi. */
        gflops = 2.0 * (double)o.M * (double)o.N * (double)o.k
                  / mean_local / 1.0e9;
        gflops_total = 2.0 * (double)o.M * (double)o.N * (double)o.k
                        / mean_total / 1.0e9;

        if (o.csv) {
            printf("%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,"
                   "%.9e,%.9e,%.9e,%.9e,%.9e,%.9e,%.6f,%.6f,%.3e\n",
                   kernel_name(), SCALAR_NAME, a_mode_name(o.a_mode),
                   o.M, o.N, o.k, g.nprocs, g.pr, g.pc, o.reps,
                   mean_local, median_local, min_local,
                   mean_reduce, mean_total, t_bcast_setup,
                   gflops, gflops_total, rel_err);
        } else {
            double bytes_A = (double)o.M * o.N * sizeof(scalar_t);
            printf("matmul_mpi  M=%d N=%d k=%d  grid=%dx%d (P=%d)  %s  kernel=%s  A=%s\n",
                   o.M, o.N, o.k, g.pr, g.pc, g.nprocs, SCALAR_NAME,
                   kernel_name(), a_mode_name(o.a_mode));
            printf("  local block  A %dx%d   X %dx%d   Y %dx%d      A total %.1f MiB\n",
                   lay.m_loc, lay.n_loc, lay.n_loc, lay.k, lay.m_loc, lay.k,
                   bytes_A / 1048576.0);
            printf("  reps=%d warmup=%d seed=%llu\n",
                   o.reps, o.warmup, (unsigned long long)o.seed);
            printf("  local_gemm (max ranks)  mean %.3f ms   median %.3f ms   min %.3f ms\n",
                   mean_local * 1e3, median_local * 1e3, min_local * 1e3);
            printf("  phases (mean of max)    local %.3f ms   reduce %.3f ms   total %.3f ms\n",
                   mean_local * 1e3, mean_reduce * 1e3, mean_total * 1e3);
            printf("  setup                    X bcast %.3f ms (excluded from benchmark)\n",
                   t_bcast_setup * 1e3);
            printf("  performance             compute-only %.3f GFLOPS   total %.3f GFLOPS\n",
                   gflops, gflops_total);
            printf("                          (2*M*N*k = %.3f GFLOP; official T = local mean)\n",
                   2.0 * (double)o.M * (double)o.N * (double)o.k / 1e9);
            if (o.check)
                printf("  validation              relative L2 error %.3e   [%s]\n",
                       rel_err, (rel_err <= SCALAR_CHECK_TOL) ? "PASS" : "FAIL");
        }
        fflush(stdout);
    }

    xfree(A_loc);
    xfree(A_global);
    xfree(X_loc);
    xfree(Ypart);
    xfree(Y_loc);
    xfree(t_tot);
    xfree(t_lo);
    xfree(t_re);
    xfree(sorted);
    grid_free(&g);
    MPI_Finalize();

    /* esito non nullo se la validazione fallisce: utile negli script */
    return (o.check && rel_err > SCALAR_CHECK_TOL) ? EXIT_FAILURE : EXIT_SUCCESS;
}
