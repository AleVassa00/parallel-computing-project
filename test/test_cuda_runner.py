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

        def run(extra, expected=0):
            nonlocal runs
            runs += 1
            out = root / str(runs)
            log = root / (str(runs) + '.jsonl')
            log.touch()
            run_env = dict(env, RUNNER_TEST_LOG=str(log))
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

        for experiment in ('k-sweep', 'block-sweep', 'compare', 'registers',
                           'smem-pad-sweep', 'ncu', 'full', 'grid-sweep'):
            _, log = run(['--experiment', experiment, '--all-grids', '4'])
            assert builds(log), experiment

        out, log = run(['--experiment', 'compare',
                        '--kernels', 'cuda_warp cuda_warp_smem'])
        assert len(builds(log)) == 2
        assert len(rows(out / 'routing_test.csv')) == 4
        assert len(rows(out / 'routing_test_raw.csv')) == 8

        conf = root / 'sweep.conf'
        conf.write_text('experiment=k-sweep\nkernel=cuda_warp\nks=3 6 8 20 32\n')
        out, log = run(['--config', str(conf), '--ks', '8 20'])
        assert len(builds(log)) == 1
        assert len(rows(out / 'routing_test.csv')) == 2

        _, log = run(['--experiment', 'k-sweep', '--ks', ''], expected=1)
        assert not builds(log)
        print('PASS runner: {} CLI/config/regression/error scenarios'.format(runs))

if __name__ == '__main__':
    main()
