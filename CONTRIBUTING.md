# Contributing a result

Results from other machines are welcome by pull request: data center GPUs, other laptops, other NPUs, other operating systems.

## Steps

1. Use a release tag, not `main`: `git checkout v0.1.0`.
2. Run on mains power, with other heavy programs closed.
3. Run `uv run sob reproduce --tag v0.1.0 --machine <label>`. Choose a short label that names the hardware, such as `h100-sxm`, `m3-max` or `rtx4090-desktop`.
4. Open a pull request that adds the folder `results/<label>/<run>/` exactly as the command wrote it. Do not edit the files.

## What a result folder holds

| File | Contents |
|---|---|
| `results.csv` | One row per checkpoint, backend, workload and size. The columns are described in `results/README.md`. |
| `env.json` | The machine fingerprint: processor, accelerator, driver, memory, power source, operating system, and every package version on the measured path. No user name, host name or serial number. |
| `run.json` | The command, the settings, the input digest, and whether the checkpoint pins match laya's reviewed pins. |

## What we check before merging

- `run.json` shows the release tag's input digest and `pins_match_laya: true`.
- Every backend row is present: a backend that failed is a row with its error, which is a useful result too.
- Answer parity holds: `parity_argmax` agrees on every answer, or the pull request explains the row that does not.

## Code changes

Open an issue first for a new backend or a change to the timing method. A change that moves numbers needs a new release tag, so earlier results stay comparable.
