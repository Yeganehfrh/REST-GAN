"""Run a training job on Modal (one A10 GPU), from the code of this checkout.

    modal volume put yaregan data/LEMON_DATA/EC_ch-8_sf-128.nc5 /data/LEMON_DATA/EC_ch-8_sf-128.nc5   # once, and EO
    modal run --detach jobs/modal_train.py --run-name main_s1 --args "--seed 1 --heldout-split splits/lemon_heldout40.json ..."
    modal volume get yaregan logs/main_s1_epoch_100.pt logs/

The job writes to logs/ on the volume. If the container is interrupted, Modal starts it again and the run
resumes from its latest checkpoint.
"""
import pathlib
import subprocess

import modal

ROOT = pathlib.Path(__file__).resolve().parents[1]

app = modal.App('yaregan-train')
image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('git')
    # the versions of the cluster environment
    .pip_install('torch==2.7.1', 'numpy==2.3.3', 'scipy==1.16.2', 'pandas==2.3.3', 'xarray==2025.10.1',
                 'h5netcdf==1.7.2', 'h5py==3.15.1', 'meegkit==0.1.9', 'mne==1.10.2', 'scikit-learn==1.7.2')
    .add_local_dir(ROOT / 'src', '/root/repo/src')
    .add_local_dir(ROOT / 'splits', '/root/repo/splits')
)
volume = modal.Volume.from_name('yaregan', create_if_missing=True)


@app.function(image=image, gpu='A10', cpu=4, memory=16384, volumes={'/vol': volume}, timeout=5 * 3600,
              retries=modal.Retries(max_retries=3, initial_delay=10.0))
def train(run_name: str, args: str, commit: str = ''):
    import glob
    import os
    import re

    os.makedirs('/vol/logs', exist_ok=True)
    os.chdir('/root/repo')
    for name in ('data', 'logs'):
        if not os.path.lexists(name):
            os.symlink(f'/vol/{name}', name)
    cmd = ['python', '-m', 'src.EEGModalNet.pipeline.train_gan_con', '--run-name', run_name, *args.split()]
    checkpoints = sorted(glob.glob(f'logs/{run_name}_epoch_*.pt'),
                         key=lambda p: int(re.search(r'_epoch_(\d+)\.pt$', p).group(1)))
    if checkpoints and '--resume-from' not in cmd:  # restarted after an interruption: continue the run
        cmd += ['--resume-from', checkpoints[-1]]
    print(f'code commit: {commit}\n{" ".join(cmd)}', flush=True)
    subprocess.run(cmd, check=True)
    volume.commit()


@app.local_entrypoint()
def main(run_name: str, args: str, wait: bool = False):
    """--wait: stay attached and show the output (smoke tests); otherwise start the job and return."""
    git = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], capture_output=True, text=True)
    dirty = subprocess.run(['git', '-C', str(ROOT), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True).stdout.strip()
    commit = git.stdout.strip() + (' (with uncommitted changes)' if dirty else '')
    if wait:
        train.remote(run_name, args, commit)
    else:
        call = train.spawn(run_name, args, commit)
        print('started', run_name, '| call id:', call.object_id)
