#ifndef SCPA_KERNEL_H
#define SCPA_KERNEL_H

#include "common/scalar.h"

/* Interfaccia unica del kernel locale, indipendente dal backend.
 *
 * Il codice MPI non deve sapere quale implementazione gira sotto: schema A
 * scalare, OpenMP o CUDA. La scelta e' del Makefile (variabile KERNEL).
 *
 * ---------------------------------------------------------------------------
 * Compatibilita' con nvcc
 * ---------------------------------------------------------------------------
 * nvcc compila i .cu come C++, e il C++ differisce dal C in due punti che
 * toccano proprio questo header:
 *   - `restrict` non e' una parola chiave (si scrive `__restrict__`);
 *   - i nomi delle funzioni vengono decorati (name mangling), quindi senza
 *     extern "C" un backend .cu compilerebbe ma non linkerebbe con gli
 *     oggetti C prodotti da mpicc.
 * Le due protezioni qui sotto rendono l'header includibile da entrambi i lati.
 * Il target `make check-cxx` verifica che la proprieta' resti vera. */
#if defined(__cplusplus)
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

#if defined(__cplusplus)
extern "C" {
#endif

/* ---------------------------------------------------------------------------
 * Ciclo di vita
 * ---------------------------------------------------------------------------
 * A NON cambia mai fra un'invocazione e l'altra: viene generata (o ricevuta)
 * una volta sola in preprocessing e resta identica per tutte le repetition.
 * X e Y invece cambiano a ogni invocazione, perche' X arriva dal MPI_Bcast e
 * Y deve tornare al MPI_Reduce.
 *
 * Da qui la separazione in tre fasi: tutto cio' che riguarda A sta in
 * local_gemm_create, che e' PREPROCESSING e non va cronometrato; il percorso
 * cronometrato e' il solo local_gemm.
 *
 * Per il backend scalare create/destroy si limitano a registrare i puntatori.
 * Per il backend CUDA create e' il punto in cui A viene copiata in VRAM una
 * volta sola: la consegna esclude esplicitamente dalla misura il tempo di
 * trasferimento da e verso la scheda, e con A dell'ordine dei GB una copia
 * H2D per invocazione misurerebbe il PCIe, non la GPU.
 *
 * Lo stato vive in un contesto OPACO e non in variabili statiche del modulo:
 * cosi' e' esplicito nel tipo chi possiede la copia di A, non esiste
 * inizializzazione globale nascosta, e il backend resta riutilizzabile. */
typedef struct local_gemm_context local_gemm_t;

/* Prepara il backend per una A fissa, m x n con leading dimension lda, e per
 * buffer X/Y con leading dimension ldx/ldy. A deve essere gia' popolata e deve
 * restare valida fino a local_gemm_destroy. Per CUDA tutte le allocazioni e la
 * copia H2D di A terminano qui, fuori dalle repetition. Non ritorna mai NULL:
 * in caso di errore termina il job. */
local_gemm_t *local_gemm_create(int m, int n, int k, const scalar_t *A_loc, int lda, int ldx, int ldy);

/* Y = A * X       (assegnazione, NON accumulo)
 *
 *   A: m x n, riga i a partire da A + i*lda        row-major   (dal contesto)
 *   X: n x k, riga j a partire da X + j*ldx        row-major, k contiguo
 *   Y: m x k, riga i a partire da Y + i*ldy        row-major, k contiguo
 *
 * m, n, k, lda, ldx e ldy sono fissati da local_gemm_create. ldx e ldy
 * restano nella firma dell'invocazione per rendere esplicito il layout dei
 * buffer, ma il backend verifica che non siano cambiati.
 * X e Y non devono sovrapporsi fra loro ne' con A (sono dichiarati restrict).
 * Le leading dimension di X e Y restano parametri e non coincidono
 * necessariamente con k: e' il gancio per il padding anti-conflict-miss
 * senza toccare ne' il kernel ne' il codice chiamante. */
void local_gemm(local_gemm_t *ctx, const scalar_t *RESTRICT X, int ldx, scalar_t *RESTRICT Y, int ldy);

/* Rilascia le risorse del backend (per CUDA: la copia di A in VRAM).
 * Tollera ctx == NULL. */
void local_gemm_destroy(local_gemm_t *ctx);

/* ---------------------------------------------------------------------------
 * Canali di misura del backend
 * ---------------------------------------------------------------------------
 * Su CPU il tempo dell'invocazione misurato dal chiamante (t_local, orologio
 * attorno a local_gemm) E' il tempo del kernel: non c'e' nient'altro in mezzo.
 * Su GPU non e' cosi'. Una invocazione contiene tre cose diverse:
 *
 *     H2D di X   ->   lancio del kernel   ->   D2H di Y
 *
 * e la consegna dice esplicitamente che i trasferimenti da e verso la scheda
 * NON vanno inclusi nel tempo T con cui si calcola 2*M*N*k/T, ma possono
 * essere misurati e discussi a parte. Servono percio' due canali distinti:
 *
 *   t_local   (misurato dal chiamante)  = H2D + kernel + D2H
 *   t_kernel  (misurato dal backend)    = solo kernel
 *
 * La differenza t_local-t_kernel misura congiuntamente trasferimenti e overhead
 * del runtime host (lancio, record/sync e controlli), non il solo PCIe. Il
 * backend e' l'unico che puo' misurare t_kernel, perche' su CUDA va fatto con i
 * cudaEvent sullo stream, non con l'orologio dell'host: un lancio e' asincrono
 * e l'orologio dell'host misurerebbe il tempo di accodamento, non quello di
 * esecuzione. */

/* Tempo di calcolo della SOLA ultima invocazione di local_gemm, in secondi.
 * Restituisce un valore NEGATIVO se il backend non distingue il kernel dal
 * resto dell'invocazione (e' il caso dei backend di CPU: li' la risposta e'
 * gia' t_local) oppure se local_gemm non e' ancora stata chiamata. */
double local_gemm_last_compute_seconds(const local_gemm_t *ctx);

/* Tempo speso in preparazione, in secondi: tutto cio' che e' avvenuto una
 * volta sola fuori dalla regione cronometrata. Per un backend di CPU e' circa
 * zero; per CUDA e' creazione del contesto + tutti i cudaMalloc + H2D di A e
 * sincronizzazione finale, cioe' proprio il costo che la scelta di separare
 * create da local_gemm ha tolto dal cammino misurato. Va riportato, non
 * nascosto. */
double local_gemm_setup_seconds(const local_gemm_t *ctx);

/* Nome del backend attivo, per l'intestazione delle misure. */
const char *kernel_name(void);

#if defined(__cplusplus)
}
#endif

#endif /* SCPA_KERNEL_H */
