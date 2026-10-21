Frozen per-shape variant choices, one file per platform tag (e.g. dgxspark_gb10.json), written by
`src/tools/dev_sweep_bf16.py --write-tuned`. Entries marked `dev_only: true` come from an ungated developer sweep:
re-validate them under `bench.run --phase s1` before quoting any number. Do not edit by hand; rerun the sweep.
