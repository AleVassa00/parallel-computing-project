#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

out="${1:-results/warp_tiled_checks}"
mkdir -p "$out"
nvcc="${NVCC:-nvcc}"
arch="${NVCC_ARCH:-sm_75}"
mpicc="${MPICC:-mpicc}"
mpicxx="${MPICXX:-mpicxx}"

for prec in double float; do
    prec_flags=()
    [[ "$prec" == double ]] || prec_flags+=(-DUSE_FLOAT)
    ref="$out/reference_${prec}.o"
    "$nvcc" -O3 -std=c++14 -arch="$arch" -Isrc "${prec_flags[@]}" \
        -Dlocal_gemm_create=reference_create -Dlocal_gemm=reference_gemm \
        -Dlocal_gemm_destroy=reference_destroy \
        -Dlocal_gemm_last_compute_seconds=reference_last_compute_seconds \
        -Dlocal_gemm_setup_seconds=reference_setup_seconds -Dkernel_name=reference_name \
        -Dlocal_gemm_last_h2d_X_seconds=reference_last_h2d_X_seconds \
        -Dlocal_gemm_last_d2h_Y_seconds=reference_last_d2h_Y_seconds \
        -Dlocal_gemm_bytes_h2d_A=reference_bytes_h2d_A \
        -Dlocal_gemm_bytes_h2d_X_per_call=reference_bytes_h2d_X_per_call \
        -Dlocal_gemm_bytes_d2h_Y_per_call=reference_bytes_d2h_Y_per_call \
        -Dlocal_gemm_setup_device_init_seconds=reference_setup_device_init_seconds \
        -Dlocal_gemm_setup_device_alloc_seconds=reference_setup_device_alloc_seconds \
        -Dlocal_gemm_setup_h2d_A_seconds=reference_setup_h2d_A_seconds \
        -c src/kernel/cuda_warp.cu -o "$ref"
    "$mpicc" -O3 -std=c11 -Isrc -c src/common/util.c -o "$out/util.o"

    for tile in 4 8 16 32; do
        # -B forza un rapporto ptxas anche quando esiste gia' la configurazione.
        make -B KERNEL=cuda_warp_tiled WARP_COL_TILE="$tile" PREC="$prec" \
            NVCC="$nvcc" NVCC_ARCH="$arch" EXTRA_NVCCFLAGS="-Xptxas -v" \
            > "$out/build_${prec}_tile${tile}.log" 2>&1
        suffix=""
        [[ "$prec" == double ]] || suffix="-float"
        suffix="${suffix}-cuda_warp_tiled"
        [[ "$tile" == 8 ]] || suffix="${suffix}-tile${tile}"
        "$mpicxx" -O2 -std=c++14 -Isrc "${prec_flags[@]}" \
            test/test_cuda_warp_tiled.cpp "$ref" "$out/util.o" \
            "obj/matmul_mpi${suffix}/kernel/cuda_warp_tiled.o" \
            -L"$(dirname "$(command -v "$nvcc")")/../lib64" -lcudart \
            -o "$out/check_${prec}_tile${tile}"
        "$out/check_${prec}_tile${tile}" | tee "$out/numeric_${prec}_tile${tile}.log"
        make KERNEL=cuda_warp_tiled WARP_COL_TILE="$tile" PREC="$prec" \
            NVCC="$nvcc" NVCC_ARCH="$arch" check \
            > "$out/mpi_${prec}_tile${tile}.log" 2>&1
        echo "PASS MPI grid/input checks: $prec tile=$tile"
    done
done
