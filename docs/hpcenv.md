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

Pick directories under the fast filesystem the cluster recommends:

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

## 5. Interactive validation run

```bash
singularity exec \
  --bind "$STRIPS":/data/strips \
  --bind "$UDMS":/data/udms \
  --bind "$OUTDIR":/data/output \
  --bind "${SLURM_TMPDIR:-/tmp}":/localscratch \
  --bind "$VENVBASE":/venvs \
  "$SIF" bash -lc '
    set -euo pipefail

    export OTB_APPLICATION_PATH=/app/otb/lib/otb/applications
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-8}
    export OTB_MAX_RAM_HINT=131072
    export GDAL_CACHEMAX=4096
    mkdir -p /localscratch/tmp

    /venvs/plaknit/bin/plaknit \
      --inputs /data/strips \
      --udms /data/udms \
      --output /data/output/final_mosaic.tif \
      --tmpdir /localscratch/tmp \
      --ram 131072 \
      --jobs ${SLURM_CPUS_PER_TASK:-8} \
      -vv
  '
```

Adjust RAM, jobs, and scratch paths to match your cluster defaults.

## 6. SLURM batch script

`plaknit_mosaic.slurm`:

```bash
#!/usr/bin/env bash
#SBATCH -J plaknit_mosaic
#SBATCH -A <account>
#SBATCH -p <partition>
#SBATCH -c 8
#SBATCH --mem=140G
#SBATCH -t 12:00:00
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

set -euo pipefail

PROJECT_DIR=/path/to/your/project
STRIPS=$PROJECT_DIR/data/strips
UDMS=$PROJECT_DIR/data/udms
OUTDIR=$PROJECT_DIR/output
VENVBASE=$PROJECT_DIR/venvs
SIF=/path/to/your/otb_image.sif

singularity exec \
  --bind "$STRIPS":/data/strips \
  --bind "$UDMS":/data/udms \
  --bind "$OUTDIR":/data/output \
  --bind "${SLURM_TMPDIR:-/tmp}":/localscratch \
  --bind "$VENVBASE":/venvs \
  "$SIF" bash -lc '
    set -euo pipefail

    export OTB_APPLICATION_PATH=/app/otb/lib/otb/applications
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=${SLURM_CPUS_PER_TASK:-8}
    export OTB_MAX_RAM_HINT=131072
    export GDAL_CACHEMAX=4096
    mkdir -p /localscratch/tmp

    /venvs/plaknit/bin/plaknit \
      --inputs /data/strips \
      --udms /data/udms \
      --output /data/output/final_mosaic.tif \
      --tmpdir /localscratch/tmp \
      --ram 131072 \
      --jobs ${SLURM_CPUS_PER_TASK:-8} \
      -vv
  '
```

Submit and monitor:

```bash
sbatch plaknit_mosaic.slurm
squeue -u "$USER"
```

## 7. Common issues

| Problem | Cause | Fix |
| --- | --- | --- |
| `ensurepip` missing | Minimal Python build in container | Use the versioned `get-pip.py` download in Step 2. |
| `pip` still not found | Python cannot see the custom prefix | Export `PYTHONPATH` and `PATH` that point to `/venvs/piproot`. |
| `/venvs/plaknit/bin/activate` missing | venv not created or not bound | Mount `$VENVBASE` into the container before running commands. |
| Wrong `plaknit` binary executes | Host `~/.local/bin` shadows container | Call `/venvs/plaknit/bin/plaknit` explicitly. |
| `error: unrecognized arguments: mosaic` | CLI has no subcommands | Run `plaknit --inputs ...` directly; omit subcommand names. |

## 8. Validation checklist

- [ ] `pip --version` runs successfully inside the container.
- [ ] `/venvs/plaknit/bin/plaknit --version` prints the expected release.
- [ ] `/data/strips`, `/data/udms`, and `/data/output` are visible inside the job.
- [ ] The output mosaic (for example `final_mosaic.tif`) lands in `$OUTDIR`.

## 9. Summary

You now have a reproducible approach that:

- Stores all Python artifacts under your project directory.
- Uses an OTB-enabled container with a persistent `plaknit` venv.
- Supports both interactive validation and SLURM batch execution.
- Avoids `$HOME` pollution and keeps dependencies isolated.

If your site ships different modules or mount points, adjust the bind paths and
SLURM directives but keep the same persistent-venv pattern.
