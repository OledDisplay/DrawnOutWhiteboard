from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import List

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import DiagramMaskClusters as dmc  # noqa: E402


class PrecleanTests(unittest.TestCase):
    def test_preclean_removes_black_annotation_line_but_keeps_colored_blob(self) -> None:
        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        img[40:85, 45:90] = np.array([220, 80, 80], dtype=np.uint8)
        img[10:110, 15:18] = 0

        cfg = dmc.build_config(
            dark_threshold=80,
            dark_rgb_ceiling=80,
            thin_component_max_width=12,
            thin_component_min_aspect=3.0,
            dot_component_max_area=40,
            line_component_max_area=600,
            removal_mask_expand_px=2,
            inpaint_radius=2,
        )
        out = dmc._preclean_diagram_copy(img, cfg)
        cleaned = out["cleaned_rgb"]
        removed = out["removed_mask"]

        self.assertGreater(int(np.count_nonzero(removed[:, 15:18])), 0)
        self.assertGreater(float(cleaned[20:100, 15:18].mean()), 180.0)
        self.assertGreater(int(cleaned[55:70, 55:70, 0].mean()), 150)

    def test_preclean_removes_many_thin_annotation_lines(self) -> None:
        img = np.full((220, 220, 3), 255, dtype=np.uint8)
        img[70:155, 75:150] = np.array([60, 150, 220], dtype=np.uint8)

        for x in range(10, 210, 20):
            img[10:210, x:x + 2] = 0
        for y in range(25, 205, 30):
            img[y:y + 2, 15:205] = 0

        cfg = dmc.build_config(
            dark_threshold=80,
            dark_rgb_ceiling=80,
            thin_component_max_width=12,
            thin_component_min_aspect=3.0,
            dot_component_max_area=40,
            line_component_max_area=5000,
            line_hough_threshold=12,
            line_min_length_px=50,
            line_max_gap_px=8,
            line_hough_thickness_px=5,
            removal_mask_expand_px=3,
            inpaint_radius=2,
        )
        out = dmc._preclean_diagram_copy(img, cfg)
        cleaned = out["cleaned_rgb"]
        removed = out["removed_mask"]

        self.assertGreater(int(np.count_nonzero(removed[:, 10:12])), 150)
        self.assertGreater(int(np.count_nonzero(removed[25:27, :])), 150)
        self.assertGreater(float(cleaned[20:200, 10:12].mean()), 180.0)
        self.assertGreater(float(cleaned[25:27, 20:200].mean()), 180.0)
        self.assertGreater(int(cleaned[90:130, 90:130, 2].mean()), 150)


class ProposalSuppressionTests(unittest.TestCase):
    def test_redundant_contained_proposal_is_dropped(self) -> None:
        mask_a = np.zeros((80, 80), dtype=bool)
        mask_a[10:50, 10:50] = True
        mask_b = np.zeros((80, 80), dtype=bool)
        mask_b[12:48, 12:48] = True

        removed_mask = np.zeros((80, 80), dtype=np.uint8)
        cfg = dmc.build_config(
            min_region_area_px=20,
            proposal_nms_iou_thresh=0.5,
            proposal_containment_thresh=0.9,
        )
        kept = dmc._suppress_redundant_proposals(
            [
                {"proposal_id": "a", "mask": mask_a, "bbox_xyxy": [10, 10, 50, 50], "mask_area_px": int(mask_a.sum()), "sam_pred_iou": 0.95, "sam_stability_score": 0.94},
                {"proposal_id": "b", "mask": mask_b, "bbox_xyxy": [12, 12, 48, 48], "mask_area_px": int(mask_b.sum()), "sam_pred_iou": 0.91, "sam_stability_score": 0.90},
            ],
            removed_mask,
            cfg,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["proposal_id"], "a")


class PromptConsolidationTests(unittest.TestCase):
    def test_same_point_same_scale_masks_collapse_to_best_candidate(self) -> None:
        mask_a = np.zeros((90, 90), dtype=bool)
        mask_a[20:42, 20:42] = True
        mask_b = np.zeros((90, 90), dtype=bool)
        mask_b[21:43, 21:43] = True
        mask_parent = np.zeros((90, 90), dtype=bool)
        mask_parent[16:50, 16:50] = True

        cfg = dmc.build_config(
            prompt_keep_max_per_point=3,
            prompt_same_scale_area_ratio_max=1.6,
            prompt_same_scale_iou_thresh=0.5,
            prompt_parent_containment_thresh=0.8,
            prompt_parent_max_area_ratio=5.0,
        )
        kept, prompt_debug = dmc._consolidate_prompt_level_proposals(
            [
                {"proposal_id": "a", "prompt_key": "10_10", "prompt_point": [10.0, 10.0], "mask": mask_a, "bbox_xyxy": [20, 20, 42, 42], "mask_area_px": int(mask_a.sum()), "sam_pred_iou": 0.90, "sam_stability_score": 0.90},
                {"proposal_id": "b", "prompt_key": "10_10", "prompt_point": [10.0, 10.0], "mask": mask_b, "bbox_xyxy": [21, 21, 43, 43], "mask_area_px": int(mask_b.sum()), "sam_pred_iou": 0.93, "sam_stability_score": 0.92},
                {"proposal_id": "p", "prompt_key": "10_10", "prompt_point": [10.0, 10.0], "mask": mask_parent, "bbox_xyxy": [16, 16, 50, 50], "mask_area_px": int(mask_parent.sum()), "sam_pred_iou": 0.91, "sam_stability_score": 0.91},
            ],
            cfg,
        )
        kept_ids = {row["proposal_id"] for row in kept}
        self.assertNotIn("a", kept_ids)
        self.assertIn("b", kept_ids)
        self.assertIn("p", kept_ids)
        parent_row = next(row for row in kept if row["proposal_id"] == "p")
        self.assertIn("b", parent_row.get("prompt_child_mask_ids", []))
        self.assertEqual(len(prompt_debug), 1)


class MergeTests(unittest.TestCase):
    def test_merge_component_proposals_unions_similar_neighbors(self) -> None:
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        mask_a = np.zeros((100, 100), dtype=bool)
        mask_a[20:50, 20:40] = True
        mask_b = np.zeros((100, 100), dtype=bool)
        mask_b[22:52, 38:58] = True
        img[mask_a | mask_b] = np.array([210, 90, 90], dtype=np.uint8)

        base_shape = {"aspect_ratio": 0.7, "solidity": 0.92, "fill_ratio": 0.85, "eccentricity": 0.4}
        proposals = [
            {
                "proposal_id": "p1",
                "mask": mask_a,
                "bbox_xyxy": [20, 20, 40, 50],
                "mask_area_px": int(mask_a.sum()),
                "sam_pred_iou": 0.93,
                "sam_stability_score": 0.91,
                "dino_embedding": [1.0, 0.0, 0.0],
                "color_histogram": [0.5, 0.5, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            {
                "proposal_id": "p2",
                "mask": mask_b,
                "bbox_xyxy": [38, 22, 58, 52],
                "mask_area_px": int(mask_b.sum()),
                "sam_pred_iou": 0.92,
                "sam_stability_score": 0.90,
                "dino_embedding": [0.99, 0.01, 0.0],
                "color_histogram": [0.49, 0.51, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
        ]
        cfg = dmc.build_config(
            merge_min_dino_cosine=0.9,
            merge_min_color_similarity=0.9,
            merge_min_shape_similarity=0.8,
            merge_iou_thresh=0.01,
            merge_containment_thresh=0.95,
            merge_bbox_gap_px=30.0,
            merge_bbox_growth_max=1.0,
        )
        merged, pair_debug = dmc._merge_component_proposals(img, proposals, cfg)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["merge_count"], 2)
        self.assertTrue(pair_debug)

    def test_should_merge_pair_allows_contrast_shift_when_other_features_match(self) -> None:
        mask_a = np.zeros((80, 80), dtype=bool)
        mask_a[12:42, 10:30] = True
        mask_b = np.zeros((80, 80), dtype=bool)
        mask_b[14:44, 28:48] = True

        base_shape = {"aspect_ratio": 0.68, "solidity": 0.93, "fill_ratio": 0.83, "eccentricity": 0.39}
        cfg = dmc.build_config(
            merge_min_dino_cosine=0.94,
            merge_min_color_similarity=0.90,
            merge_min_shape_similarity=0.85,
            merge_min_contrast_color_similarity=0.86,
            merge_combined_feature_score=0.86,
            merge_soft_min_dino_cosine=0.90,
            merge_soft_min_shape_similarity=0.80,
            merge_iou_thresh=0.01,
            merge_containment_thresh=0.95,
            merge_bbox_gap_px=24.0,
            merge_bbox_growth_max=1.0,
        )
        ok, metrics = dmc._should_merge_pair(
            {
                "proposal_id": "p1",
                "mask": mask_a,
                "bbox_xyxy": [10, 12, 30, 42],
                "dino_embedding": [1.0, 0.0, 0.0],
                "color_histogram": [1.0, 0.0, 0.0, 0.0],
                "contrast_color_histogram": [0.6, 0.4, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            {
                "proposal_id": "p2",
                "mask": mask_b,
                "bbox_xyxy": [28, 14, 48, 44],
                "dino_embedding": [0.98, 0.02, 0.0],
                "color_histogram": [0.0, 1.0, 0.0, 0.0],
                "contrast_color_histogram": [0.59, 0.41, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            cfg,
        )
        self.assertTrue(ok)
        self.assertLess(metrics["rgb_color_similarity"], cfg.merge_min_color_similarity)
        self.assertGreaterEqual(metrics["contrast_color_similarity"], cfg.merge_min_contrast_color_similarity)

    def test_should_merge_pair_rejects_soft_merge_when_dino_is_too_weak(self) -> None:
        mask_a = np.zeros((80, 80), dtype=bool)
        mask_a[12:42, 10:30] = True
        mask_b = np.zeros((80, 80), dtype=bool)
        mask_b[14:44, 28:48] = True

        base_shape = {"aspect_ratio": 0.68, "solidity": 0.93, "fill_ratio": 0.83, "eccentricity": 0.39}
        cfg = dmc.build_config(
            merge_min_dino_cosine=0.94,
            merge_min_color_similarity=0.90,
            merge_min_shape_similarity=0.85,
            merge_min_contrast_color_similarity=0.86,
            merge_combined_feature_score=0.86,
            merge_soft_min_dino_cosine=0.90,
            merge_soft_min_shape_similarity=0.80,
            merge_iou_thresh=0.01,
            merge_containment_thresh=0.95,
            merge_bbox_gap_px=24.0,
            merge_bbox_growth_max=1.0,
        )
        ok, metrics = dmc._should_merge_pair(
            {
                "proposal_id": "p1",
                "mask": mask_a,
                "bbox_xyxy": [10, 12, 30, 42],
                "dino_embedding": [1.0, 0.0, 0.0],
                "color_histogram": [1.0, 0.0, 0.0, 0.0],
                "contrast_color_histogram": [0.6, 0.4, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            {
                "proposal_id": "p2",
                "mask": mask_b,
                "bbox_xyxy": [28, 14, 48, 44],
                "dino_embedding": [0.65, 0.35, 0.0],
                "color_histogram": [0.0, 1.0, 0.0, 0.0],
                "contrast_color_histogram": [0.59, 0.41, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            cfg,
        )
        self.assertFalse(ok)
        self.assertLess(metrics["dino_cosine"], cfg.merge_soft_min_dino_cosine)


class LocalIslandGroupingTests(unittest.TestCase):
    def test_local_island_grouping_rejects_simple_similarity_chain(self) -> None:
        cfg = dmc.build_config(
            local_group_max_neighbors=6,
            local_group_mutual_top_k=4,
            local_group_base_gap_px=8.0,
            local_group_gap_diag_scale=0.25,
            local_group_max_center_factor=2.8,
            local_group_area_ratio_max=2.0,
            local_group_min_dino_cosine=0.95,
            local_group_min_color_similarity=0.9,
            local_group_min_shape_similarity=0.9,
            local_group_min_combined_score=0.93,
            local_group_min_shared_neighbors=1,
        )

        def make_row(pid: str, x0: int, x1: int, emb: List[float]) -> dict:
            mask = np.zeros((120, 120), dtype=bool)
            mask[20:40, x0:x1] = True
            row = {
                "proposal_id": pid,
                "mask": mask,
                "bbox_xyxy": [x0, 20, x1, 40],
                "mask_area_px": int(mask.sum()),
                "sam_pred_iou": 0.95,
                "sam_stability_score": 0.94,
                "dino_embedding": emb,
                "color_histogram": [1.0, 0.0, 0.0, 0.0],
                "contrast_color_histogram": [1.0, 0.0, 0.0, 0.0],
                "shape_features": {"aspect_ratio": 1.0, "solidity": 0.95, "fill_ratio": 0.9, "eccentricity": 0.1},
            }
            return row

        proposals = dmc._annotate_basic_proposal_geometry(
            [
                make_row("a", 10, 30, [1.0, 0.0, 0.0]),
                make_row("b", 32, 52, [0.99, 0.01, 0.0]),
                make_row("c", 54, 74, [0.98, 0.02, 0.0]),
            ]
        )
        groups, _edge_debug = dmc._build_local_island_groups(proposals, cfg)
        member_counts = sorted(group["member_count"] for group in groups)
        self.assertEqual(member_counts, [1, 2])

    def test_should_merge_pair_allows_similarity_only_merge(self) -> None:
        mask_a = np.zeros((120, 120), dtype=bool)
        mask_a[18:48, 12:42] = True
        mask_b = np.zeros((120, 120), dtype=bool)
        mask_b[18:48, 84:114] = True

        base_shape = {"aspect_ratio": 1.0, "solidity": 0.95, "fill_ratio": 0.88, "eccentricity": 0.10}
        cfg = dmc.build_config(
            merge_min_dino_cosine=0.98,
            merge_min_color_similarity=0.98,
            merge_min_shape_similarity=0.98,
            merge_combined_feature_score=0.98,
            merge_soft_min_dino_cosine=0.98,
            merge_soft_min_shape_similarity=0.98,
            merge_iou_thresh=0.20,
            merge_containment_thresh=0.95,
            merge_bbox_gap_px=10.0,
            merge_bbox_growth_max=0.20,
            merge_similarity_only_min_dino_cosine=0.94,
            merge_similarity_only_min_color_similarity=0.90,
            merge_similarity_only_min_shape_similarity=0.85,
            merge_similarity_only_min_score=0.92,
            merge_similarity_only_bbox_gap_px=80.0,
        )
        ok, metrics = dmc._should_merge_pair(
            {
                "proposal_id": "p1",
                "mask": mask_a,
                "bbox_xyxy": [12, 18, 42, 48],
                "dino_embedding": [1.0, 0.0, 0.0],
                "color_histogram": [0.4, 0.6, 0.0, 0.0],
                "contrast_color_histogram": [0.42, 0.58, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            {
                "proposal_id": "p2",
                "mask": mask_b,
                "bbox_xyxy": [84, 18, 114, 48],
                "dino_embedding": [0.995, 0.005, 0.0],
                "color_histogram": [0.39, 0.61, 0.0, 0.0],
                "contrast_color_histogram": [0.41, 0.59, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            cfg,
        )
        self.assertTrue(ok)
        self.assertEqual(metrics["similarity_only_ok"], 1.0)

    def test_should_merge_pair_rejects_similarity_only_merge_when_too_far(self) -> None:
        mask_a = np.zeros((240, 240), dtype=bool)
        mask_a[18:48, 12:42] = True
        mask_b = np.zeros((240, 240), dtype=bool)
        mask_b[18:48, 190:220] = True

        base_shape = {"aspect_ratio": 1.0, "solidity": 0.95, "fill_ratio": 0.88, "eccentricity": 0.10}
        cfg = dmc.build_config(
            merge_min_dino_cosine=0.98,
            merge_min_color_similarity=0.98,
            merge_min_shape_similarity=0.98,
            merge_combined_feature_score=0.98,
            merge_soft_min_dino_cosine=0.98,
            merge_soft_min_shape_similarity=0.98,
            merge_iou_thresh=0.20,
            merge_containment_thresh=0.95,
            merge_bbox_gap_px=10.0,
            merge_bbox_growth_max=0.20,
            merge_similarity_only_min_dino_cosine=0.94,
            merge_similarity_only_min_color_similarity=0.90,
            merge_similarity_only_min_shape_similarity=0.85,
            merge_similarity_only_min_score=0.92,
            merge_similarity_only_bbox_gap_px=80.0,
        )
        ok, metrics = dmc._should_merge_pair(
            {
                "proposal_id": "p1",
                "mask": mask_a,
                "bbox_xyxy": [12, 18, 42, 48],
                "dino_embedding": [1.0, 0.0, 0.0],
                "color_histogram": [0.4, 0.6, 0.0, 0.0],
                "contrast_color_histogram": [0.42, 0.58, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            {
                "proposal_id": "p2",
                "mask": mask_b,
                "bbox_xyxy": [190, 18, 220, 48],
                "dino_embedding": [0.995, 0.005, 0.0],
                "color_histogram": [0.39, 0.61, 0.0, 0.0],
                "contrast_color_histogram": [0.41, 0.59, 0.0, 0.0],
                "shape_features": dict(base_shape),
            },
            cfg,
        )
        self.assertFalse(ok)
        self.assertGreater(metrics["bbox_gap_px"], cfg.merge_similarity_only_bbox_gap_px)


class RefinerTests(unittest.TestCase):
    def _make_cluster_row(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        *,
        feature_id: str,
        dino_embedding: list[float],
        color_histogram: list[float],
        contrast_color_histogram: list[float],
        shape_features: dict[str, float],
    ) -> dict:
        row = dmc._build_cluster_row_from_mask(
            img,
            mask,
            [
                {
                    "proposal_id": feature_id,
                    "feature_cluster_id": feature_id,
                    "sam_pred_iou": 0.95,
                    "sam_stability_score": 0.93,
                    "stroke_indexes": [],
                    "merged_from_mask_ids": [feature_id],
                    "dino_embedding": dino_embedding,
                }
            ],
            dmc.build_config(crop_pad_px=4),
            proposal_id=feature_id,
            feature_cluster_id=feature_id,
        )
        assert row is not None
        row["dino_embedding"] = list(dino_embedding)
        row["color_histogram"] = list(color_histogram)
        row["contrast_color_histogram"] = list(contrast_color_histogram)
        row["shape_features"] = dict(shape_features)
        row["feature_cluster_id"] = feature_id
        row["proposal_id"] = feature_id
        row["merged_from_mask_ids"] = [feature_id]
        return row

    def test_refiner_dedupes_duplicate_clusters(self) -> None:
        img = np.full((80, 80, 3), 255, dtype=np.uint8)
        mask = np.zeros((80, 80), dtype=bool)
        mask[18:48, 12:42] = True
        img[mask] = np.array([210, 120, 90], dtype=np.uint8)
        shape = {"aspect_ratio": 1.0, "solidity": 0.95, "fill_ratio": 0.90, "eccentricity": 0.08}
        row_a = self._make_cluster_row(
            img,
            mask,
            feature_id="fc_a",
            dino_embedding=[1.0, 0.0, 0.0],
            color_histogram=[0.4, 0.6, 0.0, 0.0],
            contrast_color_histogram=[0.42, 0.58, 0.0, 0.0],
            shape_features=shape,
        )
        row_b = dict(row_a)
        row_b["feature_cluster_id"] = "fc_b"
        row_b["proposal_id"] = "fc_b"
        row_b["merged_from_mask_ids"] = ["fc_b"]
        row_b["sam_pred_iou"] = 0.90

        cfg = dmc.build_config(
            refiner_enabled=True,
            refiner_duplicate_iou_thresh=0.5,
            refiner_duplicate_containment_thresh=0.8,
        )

        def _unused_runner(*_args, **_kwargs):
            raise AssertionError("runner should not be called for pure dedupe")

        final_rows, refine_debug = dmc._refine_merged_clusters_samrefiner_style(
            img,
            img,
            np.zeros(mask.shape, dtype=np.uint8),
            [row_a, row_b],
            cfg,
            refinement_runner=_unused_runner,
        )
        self.assertEqual(len(final_rows), 1)
        self.assertTrue(any(str(x.get("dropped_feature_cluster_id", "")) == "fc_b" for x in refine_debug))

    def test_refiner_groups_similar_clusters_into_one_final_cluster(self) -> None:
        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        mask_a = np.zeros((120, 120), dtype=bool)
        mask_a[18:48, 12:42] = True
        mask_b = np.zeros((120, 120), dtype=bool)
        mask_b[18:48, 54:84] = True
        img[mask_a | mask_b] = np.array([210, 120, 90], dtype=np.uint8)

        shape = {"aspect_ratio": 1.0, "solidity": 0.95, "fill_ratio": 0.90, "eccentricity": 0.08}
        row_a = self._make_cluster_row(
            img,
            mask_a,
            feature_id="fc_a",
            dino_embedding=[1.0, 0.0, 0.0],
            color_histogram=[0.4, 0.6, 0.0, 0.0],
            contrast_color_histogram=[0.42, 0.58, 0.0, 0.0],
            shape_features=shape,
        )
        row_b = self._make_cluster_row(
            img,
            mask_b,
            feature_id="fc_b",
            dino_embedding=[0.99, 0.01, 0.0],
            color_histogram=[0.39, 0.61, 0.0, 0.0],
            contrast_color_histogram=[0.41, 0.59, 0.0, 0.0],
            shape_features=shape,
        )

        cfg = dmc.build_config(
            refiner_enabled=True,
            refiner_candidate_bbox_gap_px=24.0,
            refiner_candidate_min_dino_cosine=0.90,
            refiner_candidate_min_color_similarity=0.85,
            refiner_candidate_min_shape_similarity=0.80,
            refiner_candidate_min_combined_score=0.85,
        )

        def _runner(image_rgb, _cleaned_rgb, _removed_mask, members, cfg_obj):
            union_mask = np.logical_or.reduce([np.asarray(m["mask"], dtype=bool) for m in members])
            return dmc._build_cluster_row_from_mask(
                image_rgb,
                union_mask,
                members,
                cfg_obj,
                proposal_id="tmp_refined",
                feature_cluster_id="tmp_refined",
                refine_meta={"refine_stage": "samrefiner_style", "refine_score": 0.91},
            )

        final_rows, _ = dmc._refine_merged_clusters_samrefiner_style(
            img,
            img,
            np.zeros(mask_a.shape, dtype=np.uint8),
            [row_a, row_b],
            cfg,
            refinement_runner=_runner,
        )
        self.assertEqual(len(final_rows), 1)
        self.assertEqual(set(final_rows[0]["refined_from_cluster_ids"]), {"fc_a", "fc_b"})

    def test_refiner_preserves_cluster_entry_contract(self) -> None:
        img = np.full((120, 120, 3), 255, dtype=np.uint8)
        mask = np.zeros((120, 120), dtype=bool)
        mask[20:60, 18:62] = True
        img[mask] = np.array([220, 110, 90], dtype=np.uint8)
        shape = {"aspect_ratio": 1.1, "solidity": 0.95, "fill_ratio": 0.88, "eccentricity": 0.12}
        row = self._make_cluster_row(
            img,
            mask,
            feature_id="fc_contract",
            dino_embedding=[1.0, 0.0, 0.0],
            color_histogram=[0.4, 0.6, 0.0, 0.0],
            contrast_color_histogram=[0.42, 0.58, 0.0, 0.0],
            shape_features=shape,
        )
        row["refined_from_cluster_ids"] = ["fc_contract"]
        row["refine_stage"] = "samrefiner_style"
        row["refine_score"] = 0.88

        contract = dmc._build_in_memory_cluster_contract([row])
        self.assertEqual(len(contract["cluster_entries"]), 1)
        entry = contract["cluster_entries"][0]
        self.assertEqual(entry["crop_file_mask"], "mask_0000.png")
        self.assertEqual(entry["bbox_xyxy"], [int(v) for v in row["crop_bbox_xyxy"]])
        self.assertEqual(entry["refined_from_cluster_ids"], ["fc_contract"])
        self.assertEqual(entry["refine_stage"], "samrefiner_style")


if __name__ == "__main__":
    unittest.main()
