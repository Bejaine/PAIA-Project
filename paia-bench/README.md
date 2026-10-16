# paia-bench

Measurement harness for PAIA library baselines: bandwidth probes, cuBLAS/FP8/INT4 GEMV baselines, ncu
byte accounting and whole-token decode traces. All hardware properties are read at runtime, so the same
code runs on DGX Spark (GB10) and on discrete GPUs.

## Setup (once per machine)

```bash
cd paia-bench
docker build -t paia-bench:25.11 -f docker/Dockerfile docker   # needs driver >= 580 (CUDA 13.0)
scripts/drun.sh python -m bench.env --session <session> --platform-tag <tag> --probe-ncu
scripts/drun.sh python -m bench.selfcheck --session <session> --platform-tag <tag>
```

`scripts/drun.sh` runs commands in the image with the host PID namespace (required by the idle gate and
monitor), your uid, the Hugging Face cache, and shared `/tmp`. Always run measurements inside `tmux`.

Check `selfcheck.json` before measuring: any library path with `correct: false` must not be benchmarked
on that platform. See `findings.md` for what is known per platform and `configs/platform_notes.yaml` for
per-platform measurement settings.

## Running a phase

```bash
tmux new -s paia                       # real sessions refuse to run outside tmux
scripts/drun.sh python -m bench.run --phase e1 --session s1_$(date +%Y%m%d) --platform-tag <tag> --contact "<name/email>"
# safe to Ctrl-C or lose the connection: rerun the same command to resume; clean cells are never redone
scripts/drun.sh python -m bench.run --phase e1 --session <session> --list    # status of every cell
scripts/drun.sh python -m tests.test_resume                                   # GPU-free resume test
```

Before every cell the runner waits for an idle machine (no foreign GPU process, GPU util < 5%, other users
< 2 CPU cores, canary bandwidth >= 95% of the session reference). Cells disturbed while running are marked
contaminated and retried. Sessions named `debug_*` skip the waiting and are never results.

## Layout

```
configs/shapes.yaml          frozen shapes and methodology parameters
configs/platform_notes.yaml  per-platform measurement settings (ncu metric names, FP8 route, ...)
bench/env.py                 E0 fingerprint -> results/<tag>/<session>/env.json
bench/selfcheck.py           functional checks of every library path -> selfcheck.json
bench/monitor.py             idle gate, live monitor (separate process), bandwidth canary, lockfile
bench/timing.py              isolated / eager / graph timing, L2 rotation, statistics
bench/run.py                 resumable cell runner
bench/ops/                   one module per implementation (bw_probe: E1 probes)
tests/test_resume.py         kill-and-resume test
docker/Dockerfile            the single image used for every phase
scripts/drun.sh              docker run wrapper
findings.md                  verified facts, surprises, failures
predictions.md               filled in before E1/E2 run
results/<tag>/<session>/     env.json, selfcheck.json, cells/, logs/, ncu/, nsys/, summary/
```
