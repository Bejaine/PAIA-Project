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

## Layout

```
configs/shapes.yaml          frozen shapes and methodology parameters
configs/platform_notes.yaml  per-platform measurement settings (ncu metric names, FP8 route, ...)
bench/env.py                 E0 fingerprint -> results/<tag>/<session>/env.json
bench/selfcheck.py           functional checks of every library path -> selfcheck.json
docker/Dockerfile            the single image used for every phase
scripts/drun.sh              docker run wrapper
findings.md                  verified facts, surprises, failures
predictions.md               filled in before E1/E2 run
results/<tag>/<session>/     env.json, selfcheck.json, cells/, logs/, ncu/, nsys/, summary/
```
