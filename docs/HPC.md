# Training on the BITS Hyderabad HPC (Sharanga)

The first training runs and sweeps (4-6 Nov) used the cluster's GPUs. From 13 Nov the GPU queues were weeks long
(`--test-only` gave start dates from 10 Dec), so later work ran elsewhere: PatchCore models and sweeps on the Mac
(MPS, ~30 s per config, `datasets/mvtec_full`), and the YOLO seed study on a Kaggle GPU
(`scripts/kaggle/launch_yolo_recipe.sh`: private datasets + a script kernel; the Kaggle account must be
phone-verified to get a GPU, and the kernel installs Ultralytics from an uploaded wheel dataset, without internet).

| | |
|---|---|
| Login | `ssh <user>@<hpc-login-host>` (key-based) |
| Project dir | `/scratch/<user>/edgeqc` (venv in `.venv`, datasets in `datasets/`) |
| Scheduler | Slurm. GPU partitions: `gpu_a100_8`, `gpu_h100_4`, `gpu_h200_8`. `gpu_v100_1` is not usable: this account gets "Invalid qos specification" (and the V100 nodes are unsupported by the CUDA 13 torch build) |
| Internet | Available on the login node (pip, Kaggle, wget) |

## Workflow
```bash
# On the Mac
scripts/hpc/sync.sh push                       # code → HPC (no venvs, datasets or LLM weights)

# On the HPC (login node)
cd /scratch/<user>/edgeqc
bash scripts/hpc/setup_env.sh                  # once: venv with torch (CUDA 12.1 wheels) + ultralytics
# get datasets into datasets/ (see below), then:
sbatch scripts/hpc/train_classifier.sbatch     # YOLOv8n on NEU-DET, about 10-20 min on one GPU
sbatch scripts/hpc/train_anomaly.sbatch        # PatchCore on 5 MVTec categories
squeue -u $USER                                # status;  logs: runs/slurm/*.log

# Back on the Mac
scripts/hpc/sync.sh pull                       # models/{anomaly,classifier} + run plots/metrics → runs/hpc/
```
Jobs request **1 GPU, 8 CPUs and 2 h** (`gpu_h200_8` allows at most 4 CPUs per GPU). GPU queues can be hours to weeks long, so check first with `--test-only`. The `big_compute` CPU queue requires at least 128 CPUs per job.
The cluster rejects multi-partition requests. Jobs default to `gpu_a100_8`; check which queue starts sooner with
`sbatch --test-only -p gpu_h100_4 --gres=gpu:1 -c 8 -t 02:00:00 --wrap true`, then submit with `sbatch -p <partition> ...`.

## Datasets (download straight onto the HPC)
- **NEU-DET** (Kaggle): put your API token at `~/.kaggle/kaggle.json` or `~/.kaggle/access_token` (`chmod 600`), then
  `kaggle datasets download -d kaustubhdikshit/neu-surface-defect-database -p datasets --unzip`
  and move or rename the result to `datasets/NEU-DET`.
- **MVTec AD**: fill in the form at mvtec.com (Company → Research → Datasets → MVTec AD). The emailed links can be
  fetched with `wget` into `datasets/mvtec/`. Extract only the categories you need, for example
  `tar -xf mvtec_anomaly_detection.tar.xz -C datasets/mvtec metal_nut screw grid bottle tile`.

## Notes for the report
Training hardware (GPU type from the job log), epochs and training time all come from `runs/slurm/*.log`
(Kaggle run: `runs/kaggle_yolo_recipe/`).
Inference numbers must come from the Arm device, not the HPC.
