# Changelog

## Unreleased

**New Features**

- Added `plaknit order` subcommand to submit Planet Orders from a saved plan/GeoJSON.
- Planner now enforces Planet’s 1,500-vertex ROI limit automatically and ignores scenes missing quality metadata.

**Improvements**

- Orders CLI detects “no access to assets …” responses, drops inaccessible scenes, and resubmits so the rest of the plan can still be fulfilled.
