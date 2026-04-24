from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import qwentest


class LabelingUpgradeUnitTests(unittest.TestCase):
    def test_component_catalog_keeps_visual_signature_and_fallbacks(self) -> None:
        catalog = qwentest.normalize_component_catalog(
            [
                {
                    "label": "nucleus",
                    "refined_visual_description": "large round enclosed region",
                    "synonyms": ["cell nucleus"],
                    "source": ["c2"],
                }
            ],
            fallback_labels=["mitochondria", "nucleus"],
        )
        self.assertEqual([row["label"] for row in catalog], ["nucleus", "mitochondria"])
        self.assertEqual(catalog[0]["visual_signature"], "large round enclosed region")
        self.assertEqual(catalog[0]["synonyms"], ["cell nucleus"])

    def test_siglip_rank_shape_and_margin(self) -> None:
        ranked = qwentest._rank_siglip_scores(
            catalog=[
                {"label": "nucleus"},
                {"label": "mitochondria"},
                {"label": "golgi"},
            ],
            scores=[0.25, 0.9, 0.4],
            top_k=2,
        )
        self.assertEqual(ranked[0]["label"], "mitochondria")
        self.assertEqual(ranked[0]["rank"], 1)
        self.assertAlmostEqual(ranked[0]["margin_to_next"], 0.5)
        self.assertEqual(ranked[1]["label"], "golgi")

    def test_visual_parser_returns_vlm_visual(self) -> None:
        parsed = qwentest._parse_cluster_visual_summary(
            job={},
            parsed_obj={
                "geometry": "rounded blob with small branch",
                "dominant_colors": ["purple", "pink"],
                "boundary_shape": "smooth oval",
                "internal_pattern": "dense center",
                "surrounding_relation": "near larger boundary",
                "salient_parts": ["center", "rim"],
                "notes_without_naming": "looks enclosed",
            },
            raw_text="{}",
        )
        self.assertEqual(parsed["vlm_visual"]["geometry"], "rounded blob with small branch")
        self.assertEqual(parsed["vlm_visual"]["dominant_colors"], ["purple", "pink"])

    def test_final_match_parser_keeps_evidence_fields(self) -> None:
        parsed = qwentest._parse_diagram_final_matches(
            job={
                "component_catalog": [{"label": "nucleus"}, {"label": "mitochondria"}],
                "refined_labels": [],
            },
            parsed_obj={
                "matches": [
                    {
                        "label": "nucleus",
                        "source_type": "cluster",
                        "source_key": "C001",
                        "stroke_ids": [1, 2],
                        "cluster_ids": ["C001"],
                        "confidence": 0.87,
                        "reason": "rounded and enclosed",
                        "alternatives": ["mitochondria"],
                        "evidence_used": {"siglip": ["rank 1"], "vlm_visual": ["rounded"]},
                    }
                ]
            },
            raw_text="{}",
        )
        first = parsed["matches"][0]
        self.assertEqual(first["label"], "nucleus")
        self.assertEqual(first["alternatives"], ["mitochondria"])
        self.assertEqual(first["evidence_used"]["siglip"], ["rank 1"])


if __name__ == "__main__":
    unittest.main()
