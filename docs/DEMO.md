# Demo video: recording guide (3-5 min)

## Before recording (on the Pi, once)
```bash
ssh <pi>            # then:
cd ~/edgeqc
scripts/demo.sh steel --prepare     # Llama writes advice for the 6 steel defect types (~4 min), cached on disk
scripts/demo.sh bottle --prepare    # same for the bottle sequence (~3 min)
```
Fit the fan first: without it the Pi throttles at 80 C and FPS / advice times drop.

## Recording setup
- **Screen**: on the laptop open `http://<pi-ip>:8000`, record with QuickTime / Cmd+Shift+5 (mic on).
- **Phone**: 10-20 s of the Pi 4 board (and camera, if used) so viewers see where it runs.
- **Terminal**: `ssh <pi>` with `htop` open (4 Cortex-A72 cores) for one shot.
- Start each run and wait ~40 s for the Llama warm-up before you press record.

## Shot list and narration

| # | Time | On screen | Say (suggested) |
|---|---|---|---|
| 1 | 0:00-0:30 | Title + architecture (`docs/figures/architecture.png`) | "EdgeQC is a quality-control station that runs entirely on an Arm CPU. Two pipelines: vision on ONNX Runtime, and a Llama 3.2 1B advisor on ExecuTorch. No cloud, no accelerator." |
| 2 | 0:30-0:45 | Phone shot of the Pi 4 + `htop` | "This is a Raspberry Pi 4: four Cortex-A72 cores, 4 GB, CPU only." |
| 3 | 0:45-1:50 | `scripts/demo.sh steel`: dashboard with boxes, type, severity, REJECT, advice panel | "Each strip is inspected in about 100 ms. YOLOv8n names the defect, rules set the severity, and the Llama explains the likely cause and what to do." Read one Cause / Action / Check. "It is grounded on our defect knowledge base and never changes the verdict." |
| 4 | 1:50-2:10 | Same run, a defect type comes back | "When a defect type repeats, the advice the Llama wrote is reused instantly, with this frame's own log line." |
| 5 | 2:10-2:50 | `scripts/demo.sh bottle`: good bottles PASS, broken / contaminated bottles REJECT with heatmap | "For products where defects are rare, we use anomaly detection trained only on good parts. It has never seen a broken bottle, yet it flags it and shows where." |
| 6 | 2:50-3:20 | Click Acknowledge, scroll the log, export CSV | "Every inspection is logged with its evidence image. The operator acknowledges alerts and exports the log." |
| 7 | 3:20-3:40 | (optional) `scripts/demo.sh steel --fresh`: advice streaming in | "For a new defect type the Llama writes the advice live on the Pi; here it took N seconds, first words after about 10." Show the real time, sped up if needed, with a caption. |
| 8 | 3:40-4:10 | Results slide (numbers from `docs/REPORT.md` summary table) | "98.3% of steel defects named correctly; 88-100% of MVTec defects caught at up to 10% false alarms; 5 FPS; whole system 1.1 GB. Limits: thresholds tuned on small test sets, no defect-free steel images, Llama takes tens of seconds for a new defect type." |

## Offline proof (optional)
On the Pi, `ping -c 2 8.8.8.8` failing while the dashboard keeps working (unplug the uplink, keep the LAN).
The Pi's network watchdog reboots it after 20 min offline, so keep it short.

## Tips
- Cut the waits, but caption real times ("Llama reply: 42 s on the Pi 4, shown 4x"). Do not hide latency.
- Live camera instead of the folder: `scripts/demo.sh steel --camera 0` (test it first).
- Rehearse once; keep each run under a minute of footage.
