# Diagram Mask Clustering Integration

This backend adds a new **diagram-only alternate clustering path** that uses:

`pre-clean -> SAM2 candidate masks -> DINOv2 features -> merge/filter -> ClusterMaps/ClusterRenders`

It preserves the existing downstream artifact boundary, so `timeline.py` and the labeler still consume:

- `ClusterMaps/processed_<n>/clusters.json`
- `ClusterRenders/processed_<n>/*.png`

## Backend switch

The active backend is controlled by:

```powershell
$env:DIAGRAM_CLUSTER_BACKEND="stroke"
```

or

```powershell
$env:DIAGRAM_CLUSTER_BACKEND="sam2_dinov2"
```

Default is `stroke`.

## Where it plugs in

### 1. ImagePipeline stage

`ImagePipeline.py` now branches at the diagram clustering stage:

- `stroke` -> existing `ImageClusters.cluster_in_memory(...)`
- `sam2_dinov2` -> new `DiagramMaskClusters.cluster_diagrams_in_memory(...)`

This keeps clustering generation upstream of the diagram matching and labeling logic.

### 2. Timeline safety net

`timeline.py` also has an on-demand cluster generation hook for the new backend.

When `DIAGRAM_CLUSTER_BACKEND=sam2_dinov2`, the diagram matching path checks for:

- `ClusterMaps/processed_<n>/clusters.json`
- `ClusterRenders/processed_<n>/`

If they are missing, it calls:

```python
DiagramMaskClusters.ensure_processed_clusters(processed_id, save_outputs=True)
```

That means the new backend can be used both:

- during the normal image pipeline run
- and as a lazy recovery step before unified diagram matching

## Output contract

The new backend writes one cluster row per merged component. Each row preserves the old keys:

- `bbox_xyxy`
- `crop_file_mask`
- `group_index_in_color`
- `color_id`
- `color_name`

It also adds optional extra metadata that downstream readers can ignore:

- `pipeline`
- `mask_area_px`
- `sam_stability_score`
- `sam_pred_iou`
- `merged_from_mask_ids`
- `feature_cluster_id`

## Debug bundle

The backend writes a debug bundle under:

```text
PipelineOutputs/diagram_mask_clusters/processed_<n>/
```

This includes:

- `cleaned.png`
- `cleanup_overlay.png`
- `cleanup_removed_mask.png`
- `manifest.json`

## Visual tester

Use the clustering-only tester to inspect raw split quality without labeling:

```powershell
python DrawnOut\DrawnOutWhiteboard\whiteboard_backend\test_diagram_mask_cluster_report.py --processed-id processed_1 --open
```

Optional tuning example:

```powershell
python DrawnOut\DrawnOutWhiteboard\whiteboard_backend\test_diagram_mask_cluster_report.py --processed-id processed_1 --point-grid-step 128 --min-region-area-px 2400 --merge-min-dino-cosine 0.94 --open
```

If the new backend is over-splitting because contrast shifts are making neighboring regions look too different, try the softer merge knobs:

```powershell
python DrawnOut\DrawnOutWhiteboard\whiteboard_backend\test_diagram_mask_cluster_report.py --processed-id processed_1 --merge-min-contrast-color-similarity 0.78 --merge-combined-feature-score 0.82 --merge-soft-min-dino-cosine 0.88 --merge-soft-min-shape-similarity 0.64 --open
```

## Recommended rollout

1. Keep `DIAGRAM_CLUSTER_BACKEND=stroke` as the default.
2. Use the tester to tune SAM2/DINOv2 clustering on diagram-heavy samples.
3. Enable `sam2_dinov2` only for diagram runs you want to compare.
4. Once the new artifact quality is stable, flip the backend for the diagram path only.
