#ifndef SCPA_KERNEL_H
#define SCPA_KERNEL_H

#include <stddef.h>

#include "common/scalar.h"

/* Interfaccia unica del kernel locale, indipendente dal backend.
 *
 * Il codice MPI non deve sapere quale implementazione gira sotto: schema A
 * scalare o uno dei backend CUDA. La scelta e' del Makefile (variabile KERNEL).
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
 *
 * Ogni backend deve onorare ldx e ldy anche quando sono maggiori di k, e i
 * test li esercitano padded: e' l'interfaccia del KERNEL, e come tale resta
 * generale. Il DRIVER distribuito, pero', li tiene entrambi uguali a k, e li
 * impone: MPI_Bcast di X e MPI_Reduce di Y usano conteggi contigui, e
 * descrivere uno stride richiederebbe di costruire un datatype derivato a
 * ogni invocazione, dentro la regione cronometrata. L'unico padding che il
 * percorso distribuito puo' avere - e che il progetto misura - e' quindi
 * quello di A, cioe' lda, perche' A non attraversa nessuna collettiva. */
void local_gemm(local_gemm_t *local_gemm_context, const scalar_t *RESTRICT X, int ldx, scalar_t *RESTRICT Y, int ldy);

/* Rilascia le risorse del backend (per CUDA: la copia di A in VRAM).
 * Tollera local_gemm_context == NULL. */
void local_gemm_destroy(local_gemm_t *local_gemm_context);

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
 * esecuzione.
 *
 * Quella differenza, da sola, non e' pero' DISCUTIBILE: mette nello stesso
 * numero il PCIe e il runtime, che hanno cause diverse e scalano in modo
 * diverso (il primo con n_loc*k e m_loc*k, il secondo per niente). Per questo
 * i due trasferimenti hanno un canale ciascuno, e l'overhead del runtime resta
 * cio' che avanza:
 *
 *   t_h2d_X + t_kernel + t_d2h_Y + t_launch_overhead = t_local
 *
 * Sono esattamente i tempi che la consegna esclude dalla misura ufficiale ma
 * consente di misurare e discutere a parte. */

/* Tempo di calcolo della SOLA ultima invocazione di local_gemm, in secondi.
 * Restituisce un valore NEGATIVO se il backend non distingue il kernel dal
 * resto dell'invocazione (e' il caso dei backend di CPU: li' la risposta e'
 * gia' t_local) oppure se local_gemm non e' ancora stata chiamata. */
double local_gemm_last_compute_seconds(const local_gemm_t *local_gemm_context);

/* Tempo del trasferimento H2D del multivettore X nella SOLA ultima
 * invocazione di local_gemm, in secondi.
 *
 * E' la prima delle due copie che la consegna permette di escludere dal tempo
 * T e chiede di discutere a parte. Va misurato con i cudaEvent come t_kernel,
 * non con l'orologio dell'host: la copia viene accodata sullo stream, e i due
 * event la delimitano li' dove avviene davvero.
 *
 * Sentinella NEGATIVA sui backend che non trasferiscono nulla (CPU) o prima
 * della prima invocazione, con la stessa convenzione di
 * local_gemm_last_compute_seconds. */
double local_gemm_last_h2d_X_seconds(const local_gemm_t *local_gemm_context);

/* Tempo del trasferimento D2H del blocco parziale di Y nella SOLA ultima
 * invocazione, in secondi. E' l'altra copia esclusa dal tempo ufficiale.
 *
 * Qui gli event non sono una raffinatezza ma una necessita': la D2H parte solo
 * quando il kernel ha finito, quindi un cronometro sull'host misurerebbe
 * "attesa del kernel + copia" e attribuirebbe al PCIe tempo di calcolo.
 *
 * Sentinella negativa come sopra. */
double local_gemm_last_d2h_Y_seconds(const local_gemm_t *local_gemm_context);

/* Byte effettivamente trasferiti da e verso la scheda, per potere trasformare
 * i tempi qui sopra in banda e confrontarli con il picco del PCIe. Sono i byte
 * di QUESTO rank: il driver li somma su tutti i rank che condividono la GPU.
 * Valgono 0 sui backend che non trasferiscono nulla. */
size_t local_gemm_bytes_h2d_A(const local_gemm_t *local_gemm_context);
size_t local_gemm_bytes_h2d_X_per_call(const local_gemm_t *local_gemm_context);
size_t local_gemm_bytes_d2h_Y_per_call(const local_gemm_t *local_gemm_context);

/* Scomposizione del tempo di preparazione (vedi local_gemm_setup_seconds).
 *
 * Il totale da solo non e' discutibile: dice che il preprocessing costa, non
 * PERCHE'. Su CUDA le tre voci hanno cause e andamenti diversi:
 *
 *   device_init   creazione del contesto CUDA: costo fisso di qualche
 *                 centinaio di ms, indipendente dalla taglia del problema;
 *   device_alloc  i cudaMalloc di A, X e Y in VRAM: cresce con la taglia ma
 *                 non e' un trasferimento;
 *   h2d_A         la copia H2D di A, l'unica delle tre che e' PCIe e l'unica
 *                 che scala con M*N; e' il trasferimento che la scelta di
 *                 separare create da local_gemm ha tolto dal cammino misurato,
 *                 ed e' quindi il numero piu' interessante da riportare.
 *
 * Le tre voci sommate sono <= del totale (restano fuori i controlli sugli
 * argomenti e la creazione degli event). Valgono 0 sui backend di CPU, dove
 * anche il totale e' dell'ordine del microsecondo. */
double local_gemm_setup_device_init_seconds(const local_gemm_t *local_gemm_context);
double local_gemm_setup_device_alloc_seconds(const local_gemm_t *local_gemm_context);
double local_gemm_setup_h2d_A_seconds(const local_gemm_t *local_gemm_context);

/* Tempo speso in preparazione, in secondi: tutto cio' che e' avvenuto una
 * volta sola fuori dalla regione cronometrata. Per un backend di CPU e' circa
 * zero; per CUDA e' creazione del contesto + tutti i cudaMalloc + H2D di A e
 * sincronizzazione finale, cioe' proprio il costo che la scelta di separare
 * create da local_gemm ha tolto dal cammino misurato. Va riportato, non
 * nascosto. */
double local_gemm_setup_seconds(const local_gemm_t *local_gemm_context);

/* ---------------------------------------------------------------------------
 * Piano di esecuzione del backend
 * ---------------------------------------------------------------------------
 * Alcuni backend non si limitano a eseguire: scelgono a runtime come mappare il
 * lavoro sull'hardware. cuda_warp_smem, per esempio, deve decidere quante righe
 * di X stanno in un tile di shared memory, e quella scelta determina quante
 * barriere esegue ogni warp e quanti blocchi restano residenti per SM.
 *
 * Senza queste due colonne nel CSV una campagna sul tiling non e' spiegabile:
 * si vedrebbero curve che salgono e scendono senza sapere se e' cambiato il
 * tile, l'occupancy, o entrambi. Non sono metriche di prestazione, sono la
 * CONFIGURAZIONE effettivamente scelta, e vanno registrate accanto ai tempi.
 *
 * Entrambe restituiscono -1 quando il backend non ha il concetto (CPU, cuBLAS,
 * che decide da se' e non lo dice), con la stessa convenzione di sentinella
 * negativa gia' usata da local_gemm_last_compute_seconds. */

/* Blocchi residenti per SM che il runtime CUDA dichiara raggiungibili con la
 * shared memory effettivamente richiesta dal lancio. */
int local_gemm_blocks_per_sm(const local_gemm_t *local_gemm_context);

/* Righe di X messe in un tile di shared memory. */
int local_gemm_x_rows_per_tile(const local_gemm_t *local_gemm_context);

/* Nome del backend attivo, per l'intestazione delle misure. */
const char *kernel_name(void);

#if defined(__cplusplus)
}
#endif

#endif /* SCPA_KERNEL_H */
