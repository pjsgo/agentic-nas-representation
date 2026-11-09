"""Run the historical evaluator and unmodified candidate in a temporary directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/resnet')
    args = parser.parse_args()
    if sys.flags.optimize:
        parser.error('The historical evaluator uses assertions; do not use python -O.')
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; choose a fresh directory to preserve prior results.')
    frozen = ROOT / 'experiments/resnet'
    manifest = json.loads((ROOT / 'experiments/manifest.json').read_text())
    for entry in manifest['files']:
        if entry['file'].startswith('experiments/resnet/'):
            data = (ROOT / entry['file']).read_bytes()
            if hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise SystemExit('Artifact changed: ' + entry['file'])
    env = dict(os.environ)
    env.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                               'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                               'PYTHONDONTWRITEBYTECODE')})
    env['CUDA_VISIBLE_DEVICES'] = ''
    with tempfile.TemporaryDirectory(prefix='nas-resnet-') as tmp:
        run = Path(tmp) / 'run'
        shutil.copytree(frozen, run)
        process = subprocess.run([sys.executable, '-B', str(run / 'evaluate_forward.py')],
                                 env=env, capture_output=True, text=True, timeout=110)
        output.mkdir(parents=True, exist_ok=False)
        (output / 'stdout.txt').write_text(process.stdout)
        (output / 'stderr.txt').write_text(process.stderr)
        if (run / 'validation').exists():
            shutil.copytree(run / 'validation', output / 'validation')
        result_path = output / 'validation/results.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        passed = process.returncode == 0 and result.get('overall_pipeline_pass') is True
        # Capture installed package versions without importing candidate code here.
        from importlib.metadata import version
        summary = {'kind': 'historical_evaluator_rerun', 'passed': passed,
                   'candidate_modified': False, 'api_calls': 0,
                   'cases_passed': sum(c.get('status') == 'passed' for c in result.get('cases', [])),
                   'case_count': len(result.get('cases', [])),
                   'environment': {k: version(k) for k in ('torch', 'torchvision', 'numpy')},
                   'python': sys.version.split()[0],
                   'scope': 'BasicBlock identity shortcut, CPU float32 eval; no training or full model claim'}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps(summary, indent=2))
        if not passed:
            print(process.stderr, file=sys.stderr)
            return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
