---
title: Running plaknit on HPC
description: Tested workflow for running plaknit in an OTB-enabled Singularity/Apptainer container.
---

# Run plaknit inside Singularity/Apptainer

This walkthrough reflects the latest working recipe for launching the
[`plaknit` CLI](usage.md) inside an **OTB-enabled Singularity/Apptainer**
container on a shared HPC cluster. The emphasis is on keeping everything
writable inside your project space while the container remains read-only.

## Requirements

- Writable project or scratch directory (for example `/blue/$USER/...`,
  `/scratch/$USER`, or `/project/...`).
- Singularity/Apptainer module available on the cluster.
- Container image that already includes Orfeo ToolBox (OTB), GDAL, and Python
  3.8 or newer.

!!! tip "Same CLI everywhere"
    `plaknit` runs the same inside or outside the container. Once the venv is
    active you can reuse the commands from [Usage](usage.md) verbatim.

## 1. Directory layout

Set your PROJECT_DIR and SIF paths and create the file structure:

```bash
export PROJECT_DIR=/path/to/your/project
export STRIPS=$PROJECT_DIR/data/strips
export UDMS=$PROJECT_DIR/data/udms
export OUTDIR=$PROJECT_DIR/output
export VENVBASE=$PROJECT_DIR/venvs
export PIPCACHE=$PROJECT_DIR/cache/pip
export BOOT=$PROJECT_DIR/bootstrap
export SIF=/path/to/your/otb_image.sif

mkdir -p "$STRIPS" "$UDMS" "$OUTDIR" "$VENVBASE" "$PIPCACHE" "$BOOT"
```

!!! note
    Keep these assets under quota-friendly paths instead of `$HOME`, especially
    on clusters that enforce small home-directory limits.

## 2. Download get-pip (Python 3.8 example)

Many OTB images ship Python without `pip`. Grab the version-specific bootstrap
script so you can install `pip` in a writable prefix:

```bash
curl -fsSLo "$BOOT/get-pip.py" https://bootstrap.pypa.io/pip/3.8/get-pip.py
```

Use the matching URL for Python 3.9+ images.

## 3. Install pip inside the container

```bash
singularity exec \
  --bind "$BOOT":/bootstrap \
  --bind "$VENVBASE":/venvs \
  "$SIF" bash -lc '
    set -euo pipefail
    mkdir -p /venvs/piproot
    python3 /bootstrap/get-pip.py --prefix /venvs/piproot

    PYVER=$(python3 -c '\''import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'\'')
    export PYTHONPATH=/venvs/piproot/lib/python${PYVER}/site-packages
    export PATH=/venvs/piproot/bin:$PATH

    pip --version
  '
```

This keeps every Python artifact inside `$PROJECT_DIR/venvs`.

## 4. Create the persistent plaknit virtualenv

```bash
singularity exec \
  --bind "$VENVBASE":/venvs \
  --bind "$PIPCACHE":/pipcache \
  "$SIF" bash -lc '
    set -euo pipefail
    PYVER=$(python3 -c '\''import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'\'')
    export PYTHONPATH=/venvs/piproot/lib/python${PYVER}/site-packages
    export PATH=/venvs/piproot/bin:$PATH

    pip install --cache-dir /pipcache virtualenv
    virtualenv --python=python3 /venvs/plaknit
    source /venvs/plaknit/bin/activate
    pip install --cache-dir /pipcache plaknit

    echo "[verify] python -> $(which python)"
    echo "[verify] plaknit -> $(which plaknit)"
    plaknit --version || plaknit --help
  '
```

The venv now lives at `$PROJECT_DIR/venvs/plaknit` and can be reused across jobs.

## 5. Upgrade plaknit in the persistent venv

When a new plaknit release drops, reuse the same venv and cache to minimize
downloads:

```bash
singularity exec \
  --bind "$VENVBASE":/venvs \
  --bind "$PIPCACHE":/pipcache \
  "$SIF" bash -lc '
    set -euo pipefail
    PYVER=$(python3 -c '\''import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'\'')
    export PYTHONPATH=/venvs/piproot/lib/python${PYVER}/site-packages
    export PATH=/venvs/piproot/bin:$PATH

    source /venvs/plaknit/bin/activate
    pip install --cache-dir /pipcache --upgrade plaknit

    echo "[verify] plaknit -> $(which plaknit)"
    plaknit --version || plaknit --help
  '
```

Pin to a specific version with `pip install plaknit==<version>` if you need
repeatable jobs.

## 6. Example processing script:

```bash
# set these to the paths on the host filesystem
export TILES=/blue/jkblackburn/$USER/data/strips          # GeoTIFF strips/tiles
export UDMS=/blue/jkblackburn/$USER/data/udms            # matching UDM
export OUTDIR=/blue/jkblackburn/$USER/output        # mosaic
export VENVBASE=/blue/jkblackburn/$USER/venvs       # contains the env
export SCRATCH=${SLURM_TMPDIR:-/tmp}       # fast scratch space
export SIF=otb.sif   # OTB-enabled image

singularity exec \
  --bind "$TILES":/data/tiles \
  --bind "$UDMS":/data/udms \
  --bind "$OUTDIR":/data/output \
  --bind "$SCRATCH":/localscratch \
  --bind "$VENVBASE":/venvs \
  "$SIF" bash -lc '
    export OTB_APPLICATION_PATH=/app/otb/lib/otb/applications
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-8}
    export OTB_MAX_RAM_HINT=131072
    export GDAL_CACHEMAX=4096
    mkdir -p /localscratch/tmp

    /venvs/plaknit/bin/plaknit \
      --inputs /data/tiles \
      --udms /data/udms \
      --output /data/output/plaknit_mosaic.tif \
      --tmpdir /localscratch/tmp \
      --jobs ${SLURM_CPUS_PER_TASK:-8} \
      --ram 131072 \
      -vv
  '
```

Submit and monitor:

```bash
sbatch plaknit_mosaic.slurm
squeue -u "$USER"
```

## 7. Validation checklist

- [ ] `pip --version` runs successfully inside the container.
- [ ] `/venvs/plaknit/bin/plaknit --version` prints the expected release.
- [ ] `/data/strips`, `/data/udms`, and `/data/output` are visible inside the job.
- [ ] The output mosaic (for example `final_mosaic.tif`) lands in `$OUTDIR`.

## 8. Summary

You now have a reproducible approach that:

- Stores all Python artifacts under your project directory.
- Uses an OTB-enabled container with a persistent `plaknit` venv.
- Supports both interactive validation and SLURM batch execution.
- Avoids `$HOME` pollution and keeps dependencies isolated.

If your site ships different modules or mount points, adjust the bind paths and
SLURM directives but keep the same persistent-venv pattern.
