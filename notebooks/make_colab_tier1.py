"""Generate notebooks/colab_tier1_<name>.ipynb: run a Tier-1 task share on a Colab GPU, pulling the project bundle
from the private Kaggle dataset with the user's own kaggle.json (uploaded in the notebook).

    python notebooks/make_colab_tier1.py --name estrogen-alpha --tasks estrogen-alpha
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--dataset", default="dipurao/molbench-bundle")
    args = ap.parse_args()
    tasks = args.tasks
    cells = [
        md(f"# MAT benchmark, Tier 1 share on Colab GPU: tasks {', '.join(tasks)}\n\n"
           "Runtime → Change runtime type → **T4 GPU**. Upload your `kaggle.json` when asked (it is only used to "
           "download the private project bundle). Run all cells; at the end download `tier1_runs.csv` and copy it "
           f"into the local project's `kaggle/output/colab_{args.name}/` folder, then run "
           f"`python scripts/fetch_kaggle_results.py --no-fetch --kernels colab/colab_{args.name}`."),
        code("!nvidia-smi --query-gpu=name,memory.total --format=csv\nimport torch, os; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'cpus', os.cpu_count())"),
        code("!pip install -q kaggle rdkit torch_geometric threadpoolctl 2>&1 | tail -1"),
        code("from google.colab import files\nimport os, json\nup = files.upload()   # choose kaggle.json\n"
             "os.makedirs('/root/.kaggle', exist_ok=True)\n"
             "open('/root/.kaggle/kaggle.json', 'wb').write(up['kaggle.json'])\n"
             "os.chmod('/root/.kaggle/kaggle.json', 0o600)\nprint('credential installed')"),
        code(f"!rm -rf /content/bundle && mkdir -p /content/bundle && kaggle datasets download -d {args.dataset} -p /content/bundle --unzip\n"
             "!ls /content/bundle && ls /content/bundle/project | head"),
        code("import sys, subprocess, time, shutil\n"
             "WORK = '/content/bundle/project'\n"
             "os.chdir(WORK)\n"
             "SRC = os.path.join(WORK, 'src')\n"
             "sys.path.insert(0, SRC)\n"
             "os.environ['PYTHONPATH'] = SRC\n"
             "os.environ['MOLBENCH_DEVICE'] = 'cuda'\n"
             "import molbench; print('molbench', molbench.__version__)\n"
             "print('tuning present:', os.path.exists('results/tuning/best_configs.json'))"),
        code("r = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_models.py', '-q', '-p', 'no:cacheprovider'], capture_output=True, text=True)\n"
             "print(r.stdout[-600:]); assert r.returncode == 0"),
        md("## Run the Tier-1 share (all 8 models, both splits, 5 seeds × 5 folds) — resume-safe; re-run the cell to continue"),
        code("os.makedirs('results/raw', exist_ok=True)\n"
             f"cmd = [sys.executable, '-u', 'scripts/03_run_tier1.py', '--tasks'] + {tasks!r} + ['--workers', str(max(1, os.cpu_count())), '--threads', '1', '--out', 'results/raw/tier1_colab.csv']\n"
             "print(' '.join(cmd))\n"
             "p = subprocess.Popen(cmd, stdout=open('results/tier1_colab.log', 'w'), stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED='1'))\n"
             "t0 = time.time()\n"
             "while p.poll() is None:\n"
             "    time.sleep(120)\n"
             "    n = (sum(1 for _ in open('results/raw/tier1_colab.csv')) - 1) if os.path.exists('results/raw/tier1_colab.csv') else 0\n"
             "    print(f'{(time.time()-t0)/60:5.1f} min  completed runs: {n}', flush=True)\n"
             "print('exit code', p.returncode); print(open('results/tier1_colab.log').read()[-1500:])"),
        code("import pandas as pd\n"
             "df = pd.read_csv('results/raw/tier1_colab.csv')\n"
             "print(len(df), 'rows;', int((df['error'].fillna('') != '').sum()), 'errors')\n"
             "print(df.groupby(['task', 'model']).size().unstack(fill_value=0))\n"
             "out = '/content/tier1_runs.csv'\n"
             "shutil.copy('results/raw/tier1_colab.csv', out)\n"
             "from google.colab import files\nfiles.download(out)"),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / f"colab_tier1_{args.name}.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
