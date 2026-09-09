"""Integration checks for CLI/config/build routing, without launching MPI/CUDA."""
import csv
import json
import os
from pathlib import Path
import subprocess
import tempfile


REPO = Path(__file__).resolve().parents[1]
STUB = r'''#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
tool = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ['RUNNER_TEST_LOG'], 'a') as f:
    f.write(json.dumps([tool, args]) + '\n')
if tool == 'make':
    print('ptxas info : Used 32 registers, 0 bytes spill stores, 0 bytes spill loads')
elif tool == 'mpirun' and '--version' not in args:
    if 'ncu' in args:
        sys.exit(0)
    binary = next(x for x in args if x.startswith('./bin/'))
    if '--csv-header' in args:
        print('kernel,k')
    else:
        k = args[args.index('-k')+1]
        row = binary + ',' + k
        if os.environ.get('RUNNER_TEST_FAIL') and '-tile4' in binary:
            sys.exit(42)
        print(row)
        raw = Path(args[args.index('--csv-raw-file')+1])
        with raw.open('a') as f:
            if raw.stat().st_size == 0:
                f.write('kernel,k,rep\n')
            for rep in range(int(args[args.index('--reps')+1])):
                f.write(row + ',' + str(rep+1) + '\n')
'''


def main():
    with tempfile.TemporaryDirectory(prefix='cuda-runner-test-') as temp:
        root = Path(temp)
        fake = root / 'commands'
        fake.mkdir()
        for tool in ('make', 'mpirun', 'nvcc', 'nvidia-smi', 'ncu'):
            path = fake / tool
            path.write_text(STUB)
            path.chmod(0o755)
        env = dict(os.environ, PATH=str(fake) + os.pathsep + os.environ['PATH'])
        runs = 0

        def run(extra, expected=0, fail=False):
            nonlocal runs
            runs += 1
            out = root / str(runs)
            log = root / (str(runs) + '.jsonl')
            log.touch()
            run_env = dict(env, RUNNER_TEST_LOG=str(log))
            if fail:
                run_env['RUNNER_TEST_FAIL'] = '1'
            command = ['bash', str(REPO / 'script/run_cuda_experiments.sh'),
                       '--name', 'routing_test', '--outdir', str(out),
                       '--M', '13', '--N', '19', '--ks', '3 7',
                       '--blocks', '64 256', '--reps', '2', '--warmup', '1'] + extra
            result = subprocess.run(command, cwd=REPO, env=run_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert result.returncode == expected, (command, result.stdout)
            return out, [json.loads(line) for line in log.read_text().splitlines()]

        def builds(log):
            return [dict(arg.split('=', 1) for arg in args if '=' in arg)
                    for tool, args in log if tool == 'make']

        def rows(path):
            with path.open() as f:
                return list(csv.DictReader(f))

        # Default backend, four isolated builds, all k and ordered grid shapes.
        out, log = run(['--experiment', 'warp-tile-sweep', '--all-grids', '4'])
        assert [b['WARP_COL_TILE'] for b in builds(log)] == ['4', '8', '16', '32']
        data = rows(out / 'routing_test.csv')
        assert len(data) == 24 and len(rows(out / 'routing_test_raw.csv')) == 48
        assert {r['kernel'] for r in data} == {
            './bin/matmul_mpi-cuda_warp_tiled',
            './bin/matmul_mpi-cuda_warp_tiled-tile4',
            './bin/matmul_mpi-cuda_warp_tiled-tile16',
            './bin/matmul_mpi-cuda_warp_tiled-tile32'}

        # Existing modes still dispatch; the tile argument affects only the new backend.
        for experiment in ('k-sweep', 'block-sweep', 'compare', 'registers',
                           'smem-pad-sweep', 'ncu', 'full', 'grid-sweep'):
            _, log = run(['--experiment', experiment, '--all-grids', '4'])
            assert builds(log), experiment
            assert all('WARP_COL_TILE' not in b for b in builds(log)), experiment
        for experiment in ('k-sweep', 'block-sweep', 'registers', 'ncu', 'grid-sweep'):
            _, log = run(['--experiment', experiment, '--kernel', 'cuda_warp_tiled',
                          '--warp-col-tile', '16', '--all-grids', '4'])
            assert all(b['WARP_COL_TILE'] == '16' for b in builds(log))
            for tool, args in log:
                if tool == 'mpirun' and '--version' not in args:
                    assert any('-tile16' in arg for arg in args), args
        _, log = run(['--experiment', 'compare', '--kernels', 'cuda_warp cuda_warp_tiled',
                      '--warp-col-tile', '4'])
        assert ['WARP_COL_TILE' in b for b in builds(log)] == [False, True]

        conf = root / 'sweep.conf'
        conf.write_text('experiment=warp-tile-sweep\nkernel=cuda_warp_tiled\n'
                        'warp_col_tile=4\nwarp_col_tiles=4 8 16 32\nks=3 6 8 20 32\n')
        out, log = run(['--config', str(conf), '--warp-col-tiles', '8 16'])
        assert [b['WARP_COL_TILE'] for b in builds(log)] == ['8', '16']
        assert len(rows(out / 'routing_test.csv')) == 4  # CLI --ks overrides config.
        _, log = run(['--config', str(conf), '--experiment', 'k-sweep', '--warp-col-tile', '32'])
        assert builds(log)[0]['WARP_COL_TILE'] == '32'

        for extra in (['--warp-col-tile', '0'], ['--warp-col-tile', '-1'],
                      ['--warp-col-tiles', '4 0'], ['--warp-col-tiles', ''],
                      ['--warp-col-tiles', 'abc'], ['--ks', ''], ['--kernel', 'cuda_warp']):
            _, log = run(['--experiment', 'warp-tile-sweep'] + extra, expected=1)
            assert not builds(log)
        out, _ = run(['--experiment', 'warp-tile-sweep', '--continue-on-error'], fail=True)
        errors = rows(out / 'routing_test_failures.csv')
        assert len(errors) == 2 and all(e['kernel'] == 'cuda_warp_tiled(tile4)' for e in errors)
        print('PASS runner: {} CLI/config/sweep/regression/error scenarios'.format(runs))


if __name__ == '__main__':
    main()
