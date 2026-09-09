"""Generate Kaggle notebooks (T4 x2) for the GPU-heavy parts of the benchmark and their kernel metadata.

    python kaggle/make_kernel.py --kind tier2 --slug sl-project --title Sl-project
    python kaggle/make_kernel.py --kind tier1 --slug sl-project-t1a --title sl-project-t1a --tasks metstab-high metstab-low

Each generated kernel lives in kaggle/<slug>/ and is pushed with `kaggle kernels push -p kaggle/<slug>`.
Input dataset: dipurao/molbench-bundle (project tree incl. results/tuning + the authors' checkpoint).
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
USER = "dipurao"                       # overridden by --user
DATASET = f"{USER}/molbench-bundle"    # overridden by --dataset


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


SETUP_CELLS = [
    code("import subprocess, sys, os, shutil, glob, time, zipfile\n"
         "open('/kaggle/working/progress.txt', 'w').write(time.strftime('%H:%M:%S ') + 'notebook started\\n')\n"
         "try:\n"
         "    print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)\n"
         "except Exception as e:\n"
         "    print('nvidia-smi unavailable:', e)\n"
         "import torch; N_GPU = torch.cuda.device_count(); print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', N_GPU)\n"
         "print('cpus', os.cpu_count())\n"
         "open('/kaggle/working/progress.txt', 'a').write(time.strftime('%H:%M:%S ') + f'gpus={N_GPU} cpus={os.cpu_count()}\\n')"),
    code("r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'rdkit', 'torch_geometric', 'threadpoolctl'], capture_output=True, text=True)\n"
         "print(r.stdout[-800:], r.stderr[-800:])\n"
         "open('/kaggle/working/progress.txt', 'a').write(time.strftime('%H:%M:%S ') + f'pip exit {r.returncode}\\n')\n"
         "import rdkit, torch_geometric; print('rdkit', rdkit.__version__, 'pyg', torch_geometric.__version__)"),
    md("## 1. Unpack the project bundle into /tmp/molbench (only results and logs go to /kaggle/working)"),
    code("import traceback\n"
         "def fail(msg):\n"
         "    open('/kaggle/working/error.txt', 'a').write(time.strftime('%H:%M:%S ') + msg + '\\n' + traceback.format_exc() + '\\n')\n"
         "    open('/kaggle/working/progress.txt', 'a').write(time.strftime('%H:%M:%S ') + 'ERROR ' + msg + '\\n')\n"
         "def find_input():\n"
         "    # Kaggle mounts datasets either at /kaggle/input/<slug> or at /kaggle/input/datasets/<user>/<slug>\n"
         "    cands = ['/kaggle/input/molbench-bundle'] + glob.glob('/kaggle/input/*/*/molbench-bundle') + glob.glob('/kaggle/input/*/molbench-bundle')\n"
         "    for c in cands:\n"
         "        if os.path.isdir(c) and (os.path.isdir(os.path.join(c, 'project')) or os.path.exists(os.path.join(c, 'project.zip'))):\n"
         "            return c\n"
         "    for root, dirs, files in os.walk('/kaggle/input'):\n"
         "        if root.count(os.sep) > 6: continue\n"
         "        if 'project' in dirs or 'project.zip' in files:\n"
         "            return root\n"
         "    return None\n"
         "INP = None\n"
         "for _ in range(60):   # the dataset version can take a few minutes to become visible after a push\n"
         "    INP = find_input()\n"
         "    if INP: break\n"
         "    print('waiting for dataset input ...', os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'no /kaggle/input', flush=True)\n"
         "    time.sleep(10)\n"
         "try:\n"
         "    print('input dir:', INP, os.listdir(INP))\n"
         "except Exception:\n"
         "    fail('dataset input missing'); raise\n"
         "WORK = '/tmp/molbench'\n"
         "shutil.rmtree(WORK, ignore_errors=True)\n"
         "if os.path.isdir(os.path.join(INP, 'project')):\n"
         "    shutil.copytree(os.path.join(INP, 'project'), WORK)\n"
         "else:\n"
         "    zipfile.ZipFile(os.path.join(INP, 'project.zip')).extractall(WORK)\n"
         "os.makedirs(os.path.join(WORK, 'data/pretrained'), exist_ok=True)\n"
         "w = glob.glob(os.path.join(INP, '**', 'mat_pretrained_weights.pt'), recursive=True)\n"
         "if w:\n"
         "    shutil.copy(w[0], os.path.join(WORK, 'data/pretrained/mat_pretrained_weights.pt'))\n"
         "else:\n"
         "    zipfile.ZipFile(os.path.join(INP, 'weights.zip')).extractall(os.path.join(WORK, 'data/pretrained'))\n"
         "os.chdir(WORK)\n"
         "print(sorted(os.listdir(WORK)))\n"
         "print('weights MB', os.path.getsize('data/pretrained/mat_pretrained_weights.pt') // 1_000_000)\n"
         "print('feature cache MB', os.path.getsize('data/processed/features/mol_cache.pkl') // 1_000_000)\n"
         "print('tuning present:', os.path.exists('results/tuning/best_configs.json'))\n"
         "open('/kaggle/working/progress.txt', 'a').write(time.strftime('%H:%M:%S ') + 'bundle unpacked\\n')"),
    code("SRC = os.path.join(WORK, 'src')\n"
         "sys.path.insert(0, SRC)\n"
         "os.environ['PYTHONPATH'] = SRC + os.pathsep + os.environ.get('PYTHONPATH', '')\n"
         "import molbench; print('molbench', molbench.__version__, 'from', molbench.__file__)\n"
         "def progress(msg):\n"
         "    with open('/kaggle/working/progress.txt', 'a') as fh:\n"
         "        fh.write(time.strftime('%H:%M:%S ') + msg + '\\n')\n"
         "    print(msg, flush=True)\n"
         "progress('package importable')"),
    md("## 2. Fidelity check (re-implementation equals the authors' code; checkpoint loads completely)"),
    code("r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_mat_fidelity.py', '-q', '-p', 'no:cacheprovider'],\n"
         "                   capture_output=True, text=True, env=dict(os.environ))\n"
         "print(r.stdout[-1500:]); print(r.stderr[-800:])\n"
         "progress(f'fidelity tests exit code {r.returncode}')\n"
         "if r.returncode != 0:\n"
         "    open('/kaggle/working/error.txt', 'w').write(r.stdout[-4000:] + r.stderr[-4000:])\n"
         "assert r.returncode == 0, 'fidelity tests failed'"),
]


def run_cells(script, extra_args, out_prefix, workers, threads):
    """Two processes, one per GPU (GPU0 random split, GPU1 scaffold split), separate CSVs, merged at the end."""
    return [
        md(f"## 3. Run `{script}` (GPU0 = random split, GPU1 = scaffold split; {workers} worker(s) per process)"),
        code("os.makedirs('results/raw', exist_ok=True)\n"
             "procs = {}\n"
             "if N_GPU >= 1:      # one process per GPU (GPU index wraps if only one GPU is present)\n"
             "    plan = [(0 % N_GPU, 'random', 'cuda', " + str(workers) + "), (1 % N_GPU, 'scaffold', 'cuda', " + str(workers) + ")]\n"
             "else:               # CPU fallback (account without GPU): both splits, workers = CPU count\n"
             "    plan = [(0, 'random', 'cpu', max(1, os.cpu_count() // 2)), (0, 'scaffold', 'cpu', max(1, os.cpu_count() // 2))]\n"
             "progress(f'execution plan: {plan}')\n"
             "for gpu, split, dev, nw in plan:\n"
             "    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu) if dev == 'cuda' else '', MOLBENCH_DEVICE=dev, PYTHONUNBUFFERED='1')\n"
             f"    log = open(f'results/{out_prefix}_{{split}}.log', 'w')\n"
             f"    procs[split] = subprocess.Popen([sys.executable, '-u', 'scripts/{script}', '--splits', split,\n"
             f"                                     '--workers', str(nw), '--threads', '{threads}', '--out', f'results/raw/{out_prefix}_{{split}}.csv'] + {extra_args!r},\n"
             "                                    env=env, stdout=log, stderr=subprocess.STDOUT)\n"
             "t0 = time.time()\n"
             "def n_rows(f):\n"
             "    return (sum(1 for _ in open(f)) - 1) if os.path.exists(f) else 0\n"
             "while any(p.poll() is None for p in procs.values()):\n"
             "    time.sleep(120)\n"
             f"    done = {{s: n_rows(f'results/raw/{out_prefix}_{{s}}.csv') for s in procs}}\n"
             "    for s_ in procs:\n"
             f"        if os.path.exists(f'results/{out_prefix}_{{s_}}.log'):\n"
             f"            shutil.copy(f'results/{out_prefix}_{{s_}}.log', f'/kaggle/working/{out_prefix}_{{s_}}.log')\n"
             f"        if os.path.exists(f'results/raw/{out_prefix}_{{s_}}.csv'):\n"
             f"            shutil.copy(f'results/raw/{out_prefix}_{{s_}}.csv', f'/kaggle/working/{out_prefix}_{{s_}}.csv')\n"
             "    progress(f'{(time.time()-t0)/60:5.1f} min  completed runs: {done}')\n"
             "for s, p in procs.items():\n"
             "    print(s, 'exit code', p.returncode)\n"
             f"    print(open(f'results/{out_prefix}_{{s}}.log').read()[-1500:])\n"
             "    progress(f'{s} process exit code {p.returncode}')"),
        md("## 4. Merge and summarise"),
        code("import pandas as pd\n"
             f"parts = [pd.read_csv(f) for f in ['results/raw/{out_prefix}_random.csv', 'results/raw/{out_prefix}_scaffold.csv'] if os.path.exists(f)]\n"
             "df = pd.concat(parts, ignore_index=True)\n"
             f"df.to_csv('/kaggle/working/{out_prefix}_runs.csv', index=False)\n"
             "for s_ in ['random', 'scaffold']:\n"
             f"    if os.path.exists(f'results/{out_prefix}_{{s_}}.log'):\n"
             f"        shutil.copy(f'results/{out_prefix}_{{s_}}.log', f'/kaggle/working/{out_prefix}_{{s_}}.log')\n"
             "print(len(df), 'rows;', int((df['error'].fillna('') != '').sum()), 'errors')\n"
             "print(df.groupby(['model', 'split'])[['roc_auc', 'rmse', 'epochs_run', 'wall_time_s']].mean().round(3))\n"
             "print(df[df['error'].fillna('') != ''][['model', 'task', 'split', 'seed', 'error']].head(20))\n"
             "progress('merged output written')"),
        md(f"The merged file `/kaggle/working/{out_prefix}_runs.csv` is the notebook output; copy it into the local project's `results/raw/` and continue with `scripts/05_stats.py`."),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["tier1", "tier2"], required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--user", default=USER, help="Kaggle username that owns the kernel")
    ap.add_argument("--dataset", default=None, help="dataset ref (default <user>/molbench-bundle)")
    args = ap.parse_args()
    user = args.user
    dataset = args.dataset or f"{user}/molbench-bundle"
    out = HERE / args.slug
    out.mkdir(exist_ok=True)
    if args.kind == "tier2":
        head = md("# MAT benchmark, Tier 2 on Kaggle (2 x T4): pretrained MAT vs scratch MAT, 42M parameters\n\n"
                  "Full protocol of PRD section 3.3: 5 seeds x fold 0 x 2 splits x 7 tasks x 2 variants = 140 runs, "
                  "15 epochs max, patience 5, identical budget for both variants. Each GPU takes one split type.")
        cells = [head] + SETUP_CELLS + run_cells("04_run_tier2.py", [], "tier2", args.workers or 1, 2)
    else:
        tasks = args.tasks or []
        head = md(f"# MAT benchmark, Tier 1 on Kaggle (2 x T4): tasks {', '.join(tasks)}\n\n"
                  "All eight Tier-1 models (RF, SVM, GCN, MAT, three ablations, hybrid), 5 seeds x 5 folds x 2 splits, "
                  "with the frozen configurations from results/tuning/best_configs.json (PRD section 3.2/3.3). "
                  "Each GPU takes one split type; several worker processes share each GPU.")
        tune = [
            md("## 2b. Complete the hyperparameter search for these tasks on GPU 0 (resume-safe: rows already in "
               "results/tuning/tuning_runs.csv from the local run are reused; only missing configurations are trained)"),
            code("env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', MOLBENCH_DEVICE='cuda', PYTHONUNBUFFERED='1')\n"
                 f"cmd = [sys.executable, '-u', 'scripts/02_tune.py', '--tasks'] + {tasks!r} + ['--workers', '4', '--threads', '1']\n"
                 "t0 = time.time()\n"
                 "r = subprocess.run(cmd, env=env, capture_output=True, text=True)\n"
                 "print(r.stdout[-3000:]); print(r.stderr[-1500:])\n"
                 "progress(f'tuning exit code {r.returncode} after {(time.time()-t0)/60:.1f} min')\n"
                 "shutil.copy('results/tuning/best_configs.json', '/kaggle/working/best_configs.json')\n"
                 "shutil.copy('results/tuning/tuning_runs.csv', '/kaggle/working/tuning_runs.csv')\n"
                 "assert r.returncode == 0, 'tuning failed'"),
        ]
        cells = [head] + SETUP_CELLS + tune + run_cells("03_run_tier1.py", ["--tasks"] + tasks, "tier1", args.workers or 2, 1)
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{args.slug}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {
        "id": f"{user}/{args.slug}", "title": args.title, "code_file": f"{args.slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "keywords": [], "dataset_sources": [dataset], "kernel_sources": [], "competition_sources": [],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print("wrote", out / f"{args.slug}.ipynb")


if __name__ == "__main__":
    main()
