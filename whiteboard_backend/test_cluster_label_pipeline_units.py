"""Lightweight unit tests for the cluster-label evidence pipeline.

These tests intentionally avoid loading any heavyweight models. They validate:

1. C2 catalog merge precedence (visual-stage > stage1 terminology).
2. Source tracking: fallback-labels duplicates do not pollute provenance.
3. Visual-stage manifest fallback loading from disk.
4. Malformed VLM output produces ``parse_ok=False`` with an error string.
5. Evidence-packet arbiter prompt contains the catalog signatures and SigLIP
   top-k rankings.
6. Empty arbiter output records ``arbiter_error`` and does not silently produce
   blank confidences.

Run with::

    python -m unittest whiteboard_backend/test_cluster_label_pipeline_units.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# The backend module must be importable directly.
BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Ensure heavy torch/vllm imports don't explode when the module is imported
# at parse time. The modules below only use pure-Python helpers; we don't call
# anything that touches GPU here.
os.environ.setdefault("QWEN_TEXT_BACKEND", "none")

import qwentest  # noqa: E402
import test_cluster_label_report as tester  # noqa: E402
try:
    from LLMstuff import qwen_vllm_server  # noqa: E402
    _SERVER_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on local FastAPI install
    qwen_vllm_server = None  # type: ignore[assignment]
    _SERVER_IMPORT_ERROR = exc


class CatalogMergeTests(unittest.TestCase):
    def test_visual_stage_beats_stage1(self) -> None:
        # Two rows for the same label -- one is stage-1 terminology, the other
        # has a refined visual description from the visual stage.
        rows = [
            {
                "label": "nucleus",
                "description": "membrane-bounded organelle of eukaryotic cells",
                "source": ["c2:accepted_components"],
            },
            {
                "label": "nucleus",
                "refined_visual_description": "Diagram of the nucleus showing the outer membrane, nuclear pores and nucleolus.",
                "source": ["c2:visual_stage"],
            },
        ]
        catalog = tester._clean_component_catalog(rows)
        self.assertEqual(len(catalog), 1)
        row = catalog[0]
        self.assertEqual(row["label"], "nucleus")
        self.assertIn("outer membrane", row["visual_signature"])
        self.assertEqual(row["visual_signature_source"], "c2:visual_stage:refined")
        # Stage-1 terminology is preserved separately, not lost.
        self.assertIn("membrane-bounded", row["stage1_description"])

    def test_fallback_duplicate_does_not_pollute_source(self) -> None:
        catalog = qwentest.normalize_component_catalog(
            [
                {
                    "label": "mitochondrion",
                    "refined_visual_description": "Double-membrane organelle",
                    "source": ["c2:visual_stage"],
                },
            ],
            fallback_labels=["mitochondrion"],  # duplicate!
        )
        self.assertEqual(len(catalog), 1)
        sources = catalog[0].get("source", [])
        # fallback_labels must NOT pollute the provenance of a real C2 row.
        self.assertNotIn("fallback_labels", sources)
        self.assertIn("c2:visual_stage", sources)

    def test_manifest_fallback_loaded_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "visual_stage_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "components": [
                            {
                                "label": "ribosome",
                                "refined_visual_description": "Small dotted structures on ER",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            fake_result = {
                "prompts": {
                    "x": {
                        "visual_stage": {
                            "manifest_path": str(manifest_path),
                        }
                    }
                }
            }
            manifest = tester._load_visual_stage_manifest(fake_result)
            self.assertIsInstance(manifest, dict)
            catalog = tester._clean_component_catalog([], visual_stage_manifest=manifest)
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["label"], "ribosome")
            self.assertIn("dotted", catalog[0]["visual_signature"])


class ParseClusterVisualTests(unittest.TestCase):
    def test_malformed_vlm_flagged(self) -> None:
        out = qwentest._parse_cluster_visual("face对象. . . . . . . . . . .")
        self.assertFalse(out.get("parse_ok"))
        self.assertTrue(out.get("parse_error"))

    def test_empty_text_flagged_empty(self) -> None:
        out = qwentest._parse_cluster_visual("")
        self.assertFalse(out.get("parse_ok"))
        self.assertEqual(out.get("parse_error"), "empty_vlm_output")

    def test_valid_json_accepted(self) -> None:
        text = json.dumps(
            {
                "full_visual_LEFT": "a circle with two lines",
                "LEFT_SURROUNDINGS": "empty",
                "geometry_keywords": ["circle", "line"],
            }
        )
        out = qwentest._parse_cluster_visual(text)
        self.assertTrue(out.get("parse_ok"))
        self.assertIn("circle", out["full_visual_LEFT"])


class VisionWorkerConfigTests(unittest.TestCase):
    def test_tester_does_not_send_internvl_dynamic_patch(self) -> None:
        self.assertFalse(tester._vision_supports_mm_dynamic_patch("cyankiwi/InternVL3_5-8B-AWQ-4bit"))
        self.assertFalse(tester._vision_supports_mm_dynamic_patch("OpenGVLab/InternVL3_5-8B-HF"))

    def test_server_request_supports_fast_vision_knobs(self) -> None:
        if qwen_vllm_server is None:
            self.skipTest("qwen_vllm_server dependencies are not installed locally")
        req = qwen_vllm_server.LoadWorkerRequest(
            model="cyankiwi/InternVL3_5-8B-AWQ-4bit",
            processor="OpenGVLab/InternVL3_5-8B-HF",
            limit_mm_per_prompt={"image": 1},
            disable_log_stats=True,
        )
        self.assertEqual(req.limit_mm_per_prompt, {"image": 1})
        self.assertTrue(req.disable_log_stats)

    def test_internvl_prompt_keeps_vllm_image_placeholder(self) -> None:
        if qwen_vllm_server is None:
            self.skipTest("qwen_vllm_server dependencies are not installed locally")
        cfg = qwen_vllm_server.WorkerConfig(
            model="cyankiwi/InternVL3_5-8B-AWQ-4bit",
            processor="OpenGVLab/InternVL3_5-8B-HF",
            language_model_only=False,
        )
        worker = object.__new__(qwen_vllm_server.QwenVLLMWorker)
        worker.config = cfg
        worker.model_id = cfg.model
        prompt = qwen_vllm_server.QwenVLLMWorker.build_vision_prompt(
            worker,
            "Return JSON.",
            num_images=1,
            system_prompt="System text.",
        )
        self.assertIn("<image>", prompt)
        self.assertNotIn("<IMG_CONTEXT>", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    @unittest.skipIf(qwen_vllm_server is None, "qwen_vllm_server dependencies are not installed locally")
    def test_server_strips_internvl_dynamic_patch_kwargs(self) -> None:
        sanitized, warnings = qwen_vllm_server._sanitize_mm_processor_kwargs(
            model="cyankiwi/InternVL3_5-8B-AWQ-4bit",
            processor="OpenGVLab/InternVL3_5-8B-HF",
            mm_processor_kwargs={"max_dynamic_patch": 1, "min_dynamic_patch": 1, "other": True},
        )
        self.assertEqual(sanitized, {"other": True})
        self.assertEqual(len(warnings), 2)


class ArbiterPromptTests(unittest.TestCase):
    def test_prompt_contains_signatures_and_siglip(self) -> None:
        catalog = [
            {
                "label": "nucleus",
                "visual_signature": "circular double-membrane organelle with pores",
                "visual_signature_source": "c2:visual_stage:refined",
                "stage1_description": "membrane-bounded organelle",
                "synonyms": ["cell nucleus"],
            },
            {
                "label": "mitochondrion",
                "visual_signature": "bean-shaped with cristae",
                "visual_signature_source": "c2:visual_stage",
                "stage1_description": "organelle that generates ATP",
                "synonyms": [],
            },
        ]
        cluster_evidence = [
            {
                "cluster_key": "c1",
                "colour_hint": "red",
                "vlm_parse_ok": True,
                "vlm_visual": {
                    "full_visual_LEFT": "circle with internal dots",
                    "LEFT_SURROUNDINGS": "",
                    "geometry_keywords": ["circle"],
                },
                "siglip_top_k": [
                    {"rank": 1, "label": "nucleus", "score": 0.81, "margin_to_next": 0.12},
                    {"rank": 2, "label": "mitochondrion", "score": 0.69, "margin_to_next": 0.04},
                ],
            },
        ]
        prompt = qwentest.build_cluster_evidence_arbiter_prompt(
            base_context="eukaryotic cell diagram",
            component_catalog=catalog,
            cluster_evidence=cluster_evidence,
        )
        self.assertIn("nucleus", prompt)
        self.assertIn("double-membrane", prompt)
        self.assertIn("siglip_top_k", prompt)
        self.assertIn("final_label", prompt)
        self.assertIn("evidence_used", prompt)
        self.assertIn("ALLOWED_LABELS", prompt)

    def test_empty_arbiter_output_records_error(self) -> None:
        # Force the generator to return empty by monkey-patching.
        original_gen = qwentest._generate_json_objects_from_prompts

        def fake_gen(**kwargs):
            return [{}], [""]

        qwentest._generate_json_objects_from_prompts = fake_gen  # type: ignore[assignment]
        try:
            debug: dict = {}
            out = qwentest.evidence_packet_match_labels_with_qwen(
                model=object(),
                processor=object(),
                device="cpu",
                base_context="ctx",
                component_catalog=[{"label": "nucleus", "visual_signature": "foo"}],
                cluster_visual_map={
                    "c1": {"full_visual_LEFT": "", "LEFT_SURROUNDINGS": "", "geometry_keywords": []}
                },
                cluster_status_map={"c1": {"vlm_parse_ok": False, "vlm_error": "e", "model_raw_preview": ""}},
                siglip_by_cluster={"c1": []},
                debug_sink=debug,
            )
        finally:
            qwentest._generate_json_objects_from_prompts = original_gen  # type: ignore[assignment]
        self.assertEqual(debug.get("mode"), "evidence_packet_arbiter")
        # No SigLIP signal + empty arbiter -> explicit null (no fallback).
        self.assertEqual(out, {})

    def test_arbiter_prose_recovery_parses_markdown_labels(self) -> None:
        raw = """
Based on the provided evidence packets:

### Cluster 0001 (Red, Tubular)
**Final Label:** `endoplasmic reticulum`
**Reasoning:** The VLM describes curved tubular structures and SigLIP rank 1 is endoplasmic reticulum.

### Cluster 0002
**Final Label:** nucleus
**Reasoning:** SigLIP rank 1 is nucleus and the visual read says circular foreground.
"""
        out = qwentest._parse_cluster_evidence_arbiter_prose(
            raw,
            allowed_labels=["nucleus", "endoplasmic reticulum", "Golgi apparatus"],
            cluster_keys=["0001_red_2_mask.png", "0002_black_4_mask.png"],
        )
        self.assertEqual(out["0001_red_2_mask.png"]["label"], "endoplasmic reticulum")
        self.assertEqual(out["0002_black_4_mask.png"]["label"], "nucleus")
        self.assertIn("siglip_top_k", out["0001_red_2_mask.png"]["evidence_used"])

    def test_evidence_arbiter_recovers_qwen_markdown_output(self) -> None:
        original_gen = qwentest._generate_json_objects_from_prompts
        raw = """
### Cluster 0001 (Red, Tubular)
**Final Label:** `endoplasmic reticulum`
**Reasoning:** The VLM describes curved tubular structures and SigLIP rank 1 is endoplasmic reticulum.
"""

        def fake_gen(**kwargs):
            return [{}], [raw]

        qwentest._generate_json_objects_from_prompts = fake_gen  # type: ignore[assignment]
        try:
            debug: dict = {}
            out = qwentest.evidence_packet_match_labels_with_qwen(
                model=object(),
                processor=object(),
                device="cpu",
                base_context="eukaryotic cell diagram",
                component_catalog=[
                    {"label": "endoplasmic reticulum", "visual_signature": "curved tubular membrane network"},
                    {"label": "nucleus", "visual_signature": "large round enclosed region"},
                ],
                cluster_visual_map={
                    "0001_red_2_mask.png": {
                        "full_visual_LEFT": "curved red tubular structures",
                        "LEFT_SURROUNDINGS": "gray background",
                        "geometry_keywords": ["curved", "tubular"],
                    }
                },
                cluster_status_map={"0001_red_2_mask.png": {"vlm_parse_ok": True}},
                siglip_by_cluster={
                    "0001_red_2_mask.png": [
                        {"rank": 1, "label": "endoplasmic reticulum", "score": 0.18, "margin_to_next": 0.01}
                    ]
                },
                debug_sink=debug,
            )
        finally:
            qwentest._generate_json_objects_from_prompts = original_gen  # type: ignore[assignment]
        self.assertEqual(out["0001_red_2_mask.png"]["label"], "endoplasmic reticulum")
        self.assertIn("prose", debug.get("parse_mode", ""))

    def test_duplicate_labels_are_not_dropped(self) -> None:
        matches = {
            "c1": {"label": "nucleus", "confidence": 0.9, "reason": "a"},
            "c2": {"label": "nucleus", "confidence": 0.8, "reason": "b"},
        }
        out = qwentest._dedupe_matches_by_label(matches, ["c1", "c2"])
        self.assertEqual(out["c1"]["label"], "nucleus")
        self.assertEqual(out["c2"]["label"], "nucleus")
        self.assertNotIn("duplicate_label_dropped", str(out["c2"].get("reason", "")))

    def test_unmatched_report_still_shows_top_candidate_c2_context(self) -> None:
        with TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.html"
            qwentest._write_cluster_label_visual_report(
                report_path=report_path,
                image_index=1,
                base_context="eukaryotic cell diagram",
                renders_mask_rgb={},
                results={
                    "cluster_order": ["c1"],
                    "component_catalog": [
                        {
                            "label": "nucleus",
                            "visual_signature": "large rounded membrane-bound region with pores",
                            "visual_signature_source": "c2:visual_stage:refined",
                        }
                    ],
                    "clusters": {
                        "c1": {
                            "mask_name": "missing.png",
                            "matched_label": None,
                            "match_confidence": 0.0,
                            "match_reason": "",
                            "vlm_parse_ok": True,
                            "vlm_error": "",
                            "model_raw_preview": "",
                            "visual": {
                                "full_visual_LEFT": "rounded central region",
                                "LEFT_SURROUNDINGS": "gray background",
                                "geometry_keywords": ["rounded"],
                            },
                            "siglip_top_k": [
                                {"rank": 1, "label": "nucleus", "score": 0.81, "margin_to_next": 0.12}
                            ],
                            "arbiter_row": {"alternatives": [], "evidence_used": ["siglip_top_k.rank1"]},
                        }
                    },
                    "postfacto_debug": {},
                },
            )
            html = report_path.read_text(encoding="utf-8")
        self.assertIn("<strong>nucleus</strong>:", html)
        self.assertIn("large rounded membrane-bound region with pores", html)
        self.assertIn("Top candidate C2 contexts", html)


if __name__ == "__main__":
    unittest.main()
