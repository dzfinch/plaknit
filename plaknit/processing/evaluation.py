"""Post-processing and holdout evaluation utilities shared across model types."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from ..data.raster import _log


def _format_confusion_matrix(matrix: np.ndarray, labels: Sequence[str]) -> str:
    label_strings = [str(label) for label in labels]
    max_label = max((len(label) for label in label_strings), default=0)
    max_value = max((len(str(val)) for val in matrix.flatten()), default=1)
    width = max(max_label, max_value)

    header = " " * (width + 1) + " ".join(label.rjust(width) for label in label_strings)
    rows = [header]
    for label, row in zip(label_strings, matrix):
        row_values = " ".join(str(val).rjust(width) for val in row)
        rows.append(f"{label.rjust(width)} {row_values}")
    return "\n".join(rows)


def _collect_holdout_metrics(
    model: RandomForestClassifier,
) -> Optional[Dict[str, Any]]:
    test_samples = getattr(model, "test_samples_", None)
    test_labels = getattr(model, "test_labels_", None)
    if test_samples is None or test_labels is None or len(test_labels) == 0:
        return None

    predictions = model.predict(test_samples)
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = np.unique(test_labels)
    decoder = getattr(model, "label_decoder", None)
    if decoder:
        label_names = [str(decoder.get(int(code), code)) for code in classes]
    else:
        label_names = [str(code) for code in classes]

    matrix = confusion_matrix(test_labels, predictions, labels=classes)
    accuracy = accuracy_score(test_labels, predictions)
    auc: Optional[float] = None
    if len(np.unique(test_labels)) > 1 and 1 in classes:
        positive_class_index = int(np.flatnonzero(classes == 1)[0])
        probabilities = model.predict_proba(test_samples)
        auc = float(roc_auc_score(test_labels, probabilities[:, positive_class_index]))

    importances = getattr(model, "feature_importances_", None)
    bands: Optional[List[Tuple[int, float]]]
    if importances is None:
        bands = None
    else:
        band_indices = getattr(model, "band_indices", None)
        if band_indices is None:
            band_indices = list(range(1, len(importances) + 1))
        bands = list(zip(band_indices, importances))
        bands.sort(key=lambda pair: pair[1], reverse=True)

    return {
        "sample_count": len(test_labels),
        "accuracy": float(accuracy),
        "auc": auc,
        "labels": label_names,
        "classes": classes,
        "matrix": matrix,
        "band_importances": bands,
        "test_labels": test_labels,
        "predictions": predictions,
        "test_ids": getattr(model, "test_ids_", None),
        "test_rows": getattr(model, "test_rows_", None),
        "test_cols": getattr(model, "test_cols_", None),
        "train_shape": getattr(model, "train_shape_", None),
        "train_transform": getattr(model, "train_transform_", None),
        "train_crs": getattr(model, "train_crs_", None),
        "train_grid_size": getattr(model, "train_grid_size_", None),
        "class_band_stats": getattr(model, "class_band_stats_", None),
    }


def _log_holdout_metrics(model: RandomForestClassifier) -> Optional[Dict[str, Any]]:
    metrics = _collect_holdout_metrics(model)
    if metrics is None:
        _log("[yellow]Model has no holdout samples; skipping evaluation.")
        return None

    _log(
        f"[bold cyan]Holdout evaluation: {metrics['sample_count']:,} samples, "
        f"accuracy {metrics['accuracy']:.3f}"
        + (
            f", ROC AUC {metrics['auc']:.3f}"
            if metrics["auc"] is not None
            else ", ROC AUC unavailable"
        )
    )
    _log("[bold cyan]Confusion matrix (rows=true, cols=pred):")
    _log(_format_confusion_matrix(metrics["matrix"], metrics["labels"]))

    bands = metrics["band_importances"]
    if bands is None:
        _log("[yellow]Model lacks feature_importances_; skipping band importance.")
    else:
        lines = ["[bold cyan]Band importance (sorted):"]
        for band_idx, importance in bands:
            lines.append(f"band {band_idx}: {importance:.6f}")
        _log("\n".join(lines))

    return metrics


def _metrics_path(out_path: Path) -> Path:
    base = out_path.with_suffix("")
    metrics_txt = base.with_name(f"{base.name}_metrics.txt")
    return metrics_txt


def _write_point_gpkg(
    out_path: Path,
    rows: np.ndarray,
    cols: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    sample_ids: Optional[np.ndarray],
    *,
    transform: Any,
    crs: Optional[Any],
) -> None:
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
    data: Dict[str, Any] = {
        "true": np.asarray(true_labels),
        "pred": np.asarray(pred_labels),
    }
    if sample_ids is not None:
        data["source_id"] = np.asarray(sample_ids)

    geometry = gpd.points_from_xy(xs, ys, crs=crs)
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", index=False)
    _log(f"[green]Wrote sample points to {out_path}.")


def _misclassified_ids(
    labels: np.ndarray,
    predictions: np.ndarray,
    sample_ids: Optional[np.ndarray],
) -> Optional[List[str]]:
    if sample_ids is None:
        return None
    mismatched = labels != predictions
    if not np.any(mismatched):
        return []
    unique_ids = np.unique(sample_ids[mismatched])
    return [str(value) for value in unique_ids]


def _collect_smoothed_holdout_metrics(
    metrics: Dict[str, Any],
    out_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    test_rows = metrics.get("test_rows")
    test_cols = metrics.get("test_cols")
    if test_rows is None or test_cols is None:
        return None, "Holdout sample locations missing from model."

    test_rows_arr = np.asarray(test_rows)
    test_cols_arr = np.asarray(test_cols)
    test_labels = metrics["test_labels"]
    test_ids = metrics.get("test_ids")

    with rasterio.open(out_path) as src:
        train_shape = metrics.get("train_shape")
        if train_shape and (src.height, src.width) != tuple(train_shape):
            return None, "Prediction grid shape differs from training grid."

        train_transform = metrics.get("train_transform")
        if train_transform is not None:
            if not np.allclose(tuple(src.transform), tuple(train_transform)):
                return None, "Prediction transform differs from training transform."

        train_crs = metrics.get("train_crs")
        if train_crs is not None and src.crs is not None and train_crs != src.crs:
            return None, "Prediction CRS differs from training CRS."

        in_bounds = (
            (test_rows_arr >= 0)
            & (test_rows_arr < src.height)
            & (test_cols_arr >= 0)
            & (test_cols_arr < src.width)
        )
        if not np.any(in_bounds):
            return None, "No holdout samples fall within prediction bounds."

        if not np.all(in_bounds):
            dropped_bounds = int(np.count_nonzero(~in_bounds))
        else:
            dropped_bounds = 0

        test_rows_arr = test_rows_arr[in_bounds]
        test_cols_arr = test_cols_arr[in_bounds]
        test_labels = test_labels[in_bounds]
        if test_ids is not None:
            test_ids = test_ids[in_bounds]

        coords = [
            rasterio.transform.xy(src.transform, int(row), int(col), offset="center")
            for row, col in zip(test_rows_arr, test_cols_arr)
        ]
        sampled = np.array([val[0] for val in src.sample(coords)])

        nodata = src.nodata
        if nodata is None:
            valid = np.ones(sampled.shape[0], dtype=bool)
        elif np.isnan(nodata):
            valid = ~np.isnan(sampled)
        else:
            valid = sampled != nodata

        if not np.any(valid):
            return None, "All holdout samples landed on nodata in predictions."

        dropped_nodata = int(np.count_nonzero(~valid))
        if dropped_nodata:
            test_labels = test_labels[valid]
            sampled = sampled[valid]
            if test_ids is not None:
                test_ids = test_ids[valid]

        classes = metrics["classes"]
        matrix = confusion_matrix(test_labels, sampled, labels=classes)
        accuracy = accuracy_score(test_labels, sampled)

        note = None
        total_dropped = dropped_bounds + dropped_nodata
        if total_dropped:
            note = (
                "Dropped "
                f"{total_dropped} holdout samples outside raster or on nodata."
            )

        return (
            {
                "sample_count": len(test_labels),
                "accuracy": float(accuracy),
                "matrix": matrix,
                "predictions": sampled,
                "test_labels": test_labels,
                "test_ids": test_ids,
                "note": note,
            },
            None,
        )


def _write_holdout_outputs(
    metrics: Dict[str, Any],
    out_path: Path,
    *,
    smoothed: Optional[Dict[str, Any]] = None,
    smoothed_note: Optional[str] = None,
    smooth: str = "none",
) -> None:
    metrics_txt = _metrics_path(out_path)
    labels = metrics["labels"]
    matrix = metrics["matrix"]

    lines = [
        f"Holdout samples: {metrics['sample_count']}",
        f"Raw accuracy: {metrics['accuracy']:.3f}",
        (
            f"Raw ROC AUC: {metrics['auc']:.3f}"
            if metrics.get("auc") is not None
            else "Raw ROC AUC: unavailable."
        ),
        (
            f"Training grid sampling: {metrics['train_grid_size']} px per cell."
            if metrics.get("train_grid_size")
            else "Training grid sampling: none."
        ),
        "",
        "Raw confusion matrix (rows=true, cols=pred):",
        _format_confusion_matrix(matrix, labels),
    ]

    raw_ids = _misclassified_ids(
        metrics["test_labels"], metrics["predictions"], metrics.get("test_ids")
    )
    if raw_ids is None:
        lines.append("")
        lines.append("Misclassified validation IDs (raw): unavailable.")
    else:
        lines.append("")
        lines.append(
            "Misclassified validation IDs (raw): "
            + (", ".join(raw_ids) if raw_ids else "none")
        )

    lines.append("")
    lines.append(f"Smoothing mode: {smooth}")
    if smoothed is None:
        lines.append("Smoothed accuracy: unavailable.")
        if smoothed_note:
            lines.append(f"Smoothed metrics note: {smoothed_note}")
        lines.append("")
        lines.append("Smoothed confusion matrix: unavailable.")
    else:
        lines.append(f"Smoothed accuracy: {smoothed['accuracy']:.3f}")
        lines.append("")
        lines.append("Smoothed confusion matrix (rows=true, cols=pred):")
        lines.append(_format_confusion_matrix(smoothed["matrix"], labels))
        if smoothed.get("note"):
            lines.append(smoothed["note"])

        smoothed_ids = _misclassified_ids(
            smoothed["test_labels"],
            smoothed["predictions"],
            smoothed.get("test_ids"),
        )
        lines.append("")
        if smoothed_ids is None:
            lines.append("Misclassified validation IDs (smoothed): unavailable.")
        else:
            lines.append(
                "Misclassified validation IDs (smoothed): "
                + (", ".join(smoothed_ids) if smoothed_ids else "none")
            )

    bands = metrics["band_importances"]
    if bands is None:
        lines.append("")
        lines.append("Band importance unavailable (model has no feature_importances_).")
    else:
        lines.append("")
        lines.append("Band importance (sorted):")
        for band_idx, importance in bands:
            lines.append(f"band {band_idx}: {importance:.6f}")

    class_stats = metrics.get("class_band_stats")
    if class_stats is None:
        lines.append("")
        lines.append("Per-class band stats (train): unavailable.")
    else:
        stats_classes = class_stats.get("classes")
        counts = class_stats.get("counts")
        means = class_stats.get("mean")
        stds = class_stats.get("std")
        band_indices = class_stats.get("band_indices")
        if (
            stats_classes is None
            or counts is None
            or means is None
            or stds is None
            or band_indices is None
        ):
            lines.append("")
            lines.append("Per-class band stats (train): unavailable.")
        else:
            label_map = {cls: name for cls, name in zip(stats_classes, labels)}
            lines.append("")
            lines.append("Per-class band stats (train):")
            for idx, class_value in enumerate(stats_classes):
                label = label_map.get(class_value, str(class_value))
                lines.append(f"class {label} (n={int(counts[idx])}):")
                for band_idx, mean_val, std_val in zip(
                    band_indices, means[idx], stds[idx]
                ):
                    lines.append(
                        f"band {int(band_idx)}: mean={mean_val:.6f} "
                        f"std={std_val:.6f}"
                    )

    metrics_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log(f"[green]Wrote holdout metrics to {metrics_txt}.")
