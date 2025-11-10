---
title: Running plaknit on HPC
description: Step-by-step guidance for launching plaknit inside Singularity/Apptainer containers.
---

# Run plaknit on HPC with Singularity/Apptainer

Use this guide when you need to execute the [`plaknit` CLI](usage.md) on a shared
cluster where Orfeo ToolBox (OTB) already lives inside a container image. It
covers both an interactive smoke test and a SLURM batch workflow while keeping
your Python environment on a writable filesystem.

## Overview

- Install `plaknit` into a persistent virtual environment that sits outside the
  read-only container image.
- Bind your project data, scratch space, and virtual environments into the
  container each time you launch it.
- Run `plaknit` commands exactly as you would locally once the venv is active.

!!! tip "Use the same flags everywhere"
    All CLI options described in [Usage](usage.md) work unchanged inside the
    container. The only difference is how paths are mounted.

## Prerequisites

### Singularity/Apptainer modules

```bash
# Example -- adjust to the module names provided by your cluster
module load apptainer   # or: module load singularity
```

### OTB-enabled container image

You need a container that already bundles OTB, GDAL, and their dependencies.
Store it somewhere quota-friendly and fast.

```bash
export IMG_DIR=$HOME/containers
mkdir -p "$IMG_DIR"

# Replace the image reference with the one provided by your lab/cluster
apptainer pull "$IMG_DIR/otb.sif" docker://<ORG>/<OTB_IMAGE>:<TAG>
```

### Project folders, scratch, and caches

```bash
export PROJ=$HOME/projects/plaknit-demo
export IMG_DIR_IN=$PROJ/strips
export UDM_DIR_IN=$PROJ/udms
export OUT_DIR=$PROJ/out
export TMP_DIR=$PROJ/tmp        # Prefer node-local scratch when available
export VENVS=$HOME/.venvs
export PIPCACHE=$HOME/.cache/pip

mkdir -p "$PROJ" "$IMG_DIR_IN" "$UDM_DIR_IN" "$OUT_DIR" "$TMP_DIR" "$VENVS" "$PIPCACHE"
```

## Interactive smoke test

1. Export the paths that will be reused in batch jobs:

    ```bash
    export SIF=$IMG_DIR/otb.sif
    export VENV=$VENVS/plaknit
    ```

2. Launch an interactive shell inside the container with your binds:

    ```bash
    apptainer shell \
      --bind "$PROJ":/work \
      --bind "$VENVS":/venvs \
      --bind "$PIPCACHE":/pipcache \
      "$SIF"
    ```

3. Inside the container:

    ```bash
    python3 -m venv /venvs/plaknit
    source /venvs/plaknit/bin/activate
    python -m pip install --upgrade pip
    pip install --cache-dir /pipcache plaknit

    plaknit --help
    ```

Run a small command (for example `plaknit mosaic ...`) to confirm OTB access
before moving on to batch jobs.

## Batch run with SLURM

Create `run_plaknit.slurm` alongside your project:

```bash
cat <<'EOF' > run_plaknit.slurm
#!/bin/bash
#SBATCH --job-name=plaknit-mosaic
#SBATCH --partition=standard
#SBATCH --account=<ACCOUNT>
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=150G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.log

set -euo pipefail
module load apptainer

export PROJ=$HOME/projects/plaknit-demo
export VENVS=$HOME/.venvs
export PIPCACHE=$HOME/.cache/pip
export SIF=$HOME/containers/otb.sif

apptainer exec \
  --bind "$PROJ":/work \
  --bind "$VENVS":/venvs \
  --bind "$PIPCACHE":/pipcache \
  "$SIF" bash -lc '
    set -euo pipefail
    source /venvs/plaknit/bin/activate

    # Optional sanity checks
    command -v otbcli_Mosaic >/dev/null && echo "OTB present"

    plaknit mosaic \
      --inputs /work/strips \
      --udms   /work/udms \
      --output /work/out/final_mosaic.tif \
      --tmpdir /work/tmp \
      --ram 131072 \
      --jobs ${SLURM_CPUS_PER_TASK:-8} \
      -vv
  '
EOF
```

Submit and monitor:

```bash
sbatch run_plaknit.slurm
squeue -u "$USER"
```

## Bind mounts and filesystem layout

- Keep strip imagery and UDM rasters on read-only storage; mount them into the
  container under `/work/strips` and `/work/udms`.
- Mount a writable directory (project `out/` or node-local scratch) to `/work/out`
  and `/work/tmp` for outputs and temp files.
- Mount `$HOME/.venvs` and `$HOME/.cache/pip` into `/venvs` and `/pipcache`
  for faster installs and reuse across jobs.

## Performance guidance

- Match `plaknit`'s `--ram` flag to the SLURM `--mem` request but leave headroom
  for the OS and GDAL caches.
- Set `--jobs` to `SLURM_CPUS_PER_TASK` and keep an eye on I/O contention.
- Point `--tmpdir` to node-local SSDs when available and ensure that temp space
  is cleaned up after the job.
- Consider writing Cloud-Optimized GeoTIFFs downstream if your workflow needs
  repeated reads.

## Troubleshooting

- **OTB not found**: confirm the container image actually ships OTB. If not,
  rebuild or request an image that does.
- **`pip install` fails**: verify that `/venvs` points to a writable path and
  that you activated the virtual environment before installing.
- **Raster driver errors**: always use Python from inside the container so
  GDAL/Rasterio versions stay consistent.
- **Slow throughput**: move `--tmpdir` to faster storage and limit `--jobs`
  until per-node bandwidth stabilizes.
- **OOM kills**: lower `--jobs`, tile the AOI into smaller chunks, or request
  more memory.

## Minimal quick start

```bash
module load apptainer

export IMG_DIR=$HOME/containers
export PROJ=$HOME/projects/plaknit-demo
export VENVS=$HOME/.venvs
export PIPCACHE=$HOME/.cache/pip
mkdir -p "$IMG_DIR" "$PROJ"/{strips,udms,out,tmp} "$VENVS" "$PIPCACHE"

apptainer pull "$IMG_DIR/otb.sif" docker://<ORG>/<OTB_IMAGE>:<TAG>

apptainer exec \
  --bind "$PROJ":/work --bind "$VENVS":/venvs --bind "$PIPCACHE":/pipcache \
  "$IMG_DIR/otb.sif" bash -lc '
  python3 -m venv /venvs/plaknit && source /venvs/plaknit/bin/activate && \
  python -m pip install --upgrade pip && \
  pip install --cache-dir /pipcache plaknit && \
  plaknit --help
'
```

Follow up by submitting the SLURM script from the previous section.

## Adapting to your cluster

- Swap `apptainer` for `singularity` if that is the binary on your system.
- Replace partitions, accounts, and QoS settings in the SLURM script with the
  values your cluster mandates.
- Some centers provide a prebuilt OTB image -- if so, skip the `pull` command and
  reference their path directly.
- Use different bind locations when your organization exposes shared project
  spaces via environment variables such as `$PROJECT`, `$SCRATCH`, or
  `$LOCAL_SCRATCH`.
