import unittest

import qwentest
import timeline


class BatchPlanningTests(unittest.TestCase):
    def test_parse_chapter_timeline_steps_top_and_substeps(self):
        chapter = """
        CHAPTER: Demo
        1. First step
           - bullet one
           - bullet two
        2. Second step
           a. sub alpha
        """
        steps = timeline.parse_chapter_timeline_steps(chapter)
        keys = [s["key"] for s in steps]
        self.assertEqual(keys, ["1", "1.sub1", "1.sub2", "2", "2.a"])

    def test_flatten_speech_and_ranges(self):
        flat, ranges = timeline.build_flat_speech_and_ranges(
            full_speech="fallback speech",
            step_order=["1", "2"],
            step_speech_map={"1": "hello >1.0 world", "2": "next part"},
        )
        self.assertEqual(flat, "hello >1.0 world next part")
        self.assertEqual(ranges["1"]["start_word_index"], 0)
        self.assertEqual(ranges["1"]["end_word_index"], 2)
        self.assertEqual(ranges["2"]["start_word_index"], 3)
        self.assertEqual(ranges["2"]["end_word_index"], 4)

    def test_build_chunk_sync_map_has_boundary_silence(self):
        speech = "alpha beta >2.0 gamma"
        maps = [
            {
                "content": "img1",
                "query": "img1",
                "range_start": 0,
                "range_end": 1,
                "diagram": 0,
                "text_tag": 0,
                "write_text": "",
            }
        ]
        chunk = timeline.build_chunk_sync_map_for_chapter(
            chapter_index=1,
            speech_text=speech,
            image_text_maps=maps,
            speech_step_order=["1"],
            speech_step_ranges={"1": {"start_word_index": 0, "end_word_index": 3}},
            assets_by_name={},
            board_w=4000,
            board_h=4000,
            include_inter_chunk_silence=True,
        )
        entries = chunk["entries"]
        silences = [e for e in entries if e["type"] == "silence"]
        self.assertTrue(any(abs(float(s.get("duration_sec", 0.0)) - 2.0) < 1e-6 for s in silences))
        self.assertTrue(any(bool(s.get("chunk_boundary_silence", False)) for s in silences))
        boundary = [s for s in silences if bool(s.get("chunk_boundary_silence", False))]
        self.assertTrue(boundary and bool(boundary[0].get("delete_all", False)))

    def test_split_segments_and_batch_chunking(self):
        entries = [
            {"type": "image", "name": "a", "range_start": 1, "delete_all": False},
            {"type": "image", "name": "b", "range_start": 2, "delete_all": False},
            {"type": "silence", "name": "s1", "range_start": 3, "delete_all": True},
            {"type": "image", "name": "c", "range_start": 4, "delete_all": False},
            {"type": "silence", "name": "s2", "range_start": 5, "delete_all": True},
        ]
        segs = timeline.split_entries_by_deletion_silence(entries)
        self.assertEqual(len(segs), 2)
        self.assertEqual([x["name"] for x in segs[0]["visual_entries"]], ["a", "b"])
        self.assertEqual([x["name"] for x in segs[1]["visual_entries"]], ["c"])

        batches = timeline._chunk_list(list(range(17)), 8)
        self.assertEqual([len(x) for x in batches], [8, 8, 1])

    def test_convert_local_sync_to_absolute_clamps(self):
        action = {"type": "draw_image", "sync_local": {"start_word_offset": -4, "end_word_offset": 99}}
        out = timeline.convert_local_sync_to_absolute(action, event_start_word=10, event_end_word=14)
        self.assertEqual(out["sync_local"]["start_word_offset"], 0)
        self.assertEqual(out["sync_local"]["end_word_offset"], 4)
        self.assertEqual(out["sync_absolute"]["start_word_index"], 10)
        self.assertEqual(out["sync_absolute"]["end_word_index"], 14)

    def test_space_plan_repair_non_overlap_and_clamp(self):
        chunk = {
            "chapter_index": 1,
            "entries": [
                {"entry_index": 0, "type": "image", "range_start": 0, "bbox_px": {"w": 500, "h": 500}},
                {"entry_index": 1, "type": "image", "range_start": 1, "bbox_px": {"w": 500, "h": 500}},
                {"entry_index": 2, "type": "silence", "range_start": 2, "chunk_boundary_silence": False},
            ],
        }
        bad_plan = [
            {"entry_index": 0, "type": "image", "print_bbox": {"x": -100, "y": -100, "w": 9999, "h": 9999}},
            {"entry_index": 1, "type": "image", "print_bbox": {"x": 0, "y": 0, "w": 9999, "h": 9999}},
            {"entry_index": 2, "type": "silence", "delete_all": True},
        ]
        repaired = qwentest._repair_space_plan_chunk(
            chunk=chunk,
            planned_entries=bad_plan,
            board_w=1200,
            board_h=1200,
        )
        entries = repaired["entries"]
        visuals = [e for e in entries if e["type"] in ("image", "text")]
        self.assertEqual(len(visuals), 2)

        a = visuals[0]["print_bbox"]
        b = visuals[1]["print_bbox"]
        self.assertGreaterEqual(a["x"], 0)
        self.assertGreaterEqual(a["y"], 0)
        self.assertLessEqual(a["x"] + a["w"], 1200)
        self.assertLessEqual(a["y"] + a["h"], 1200)
        self.assertFalse(qwentest._bbox_overlap_xywh(a, b))

    def test_normalize_static_visual_action_schema(self):
        req = {
            "name": "cell_diagram",
            "type": "image",
            "diagram": 1,
            "text_tag": 0,
            "content": "cell",
            "write_text": "",
            "range_start": 10,
            "range_end": 20,
            "print_bbox": {"x": 100, "y": 200, "w": 900, "h": 700},
            "objects_that_comprise_image": ["nucleus", "membrane"],
        }
        draw = timeline.normalize_static_visual_action(
            {"type": "draw_image", "location": {"x": 150, "y": 220}, "sync_local": {"start_word_offset": 1, "end_word_offset": 3}},
            req=req,
            action_index=0,
        )
        self.assertEqual(draw["type"], "draw_image")
        self.assertEqual(draw["target"], "cell_diagram")
        self.assertEqual(draw["x"], 150)
        self.assertEqual(draw["y"], 220)

        highlight = timeline.normalize_static_visual_action(
            {"type": "highlight_cluster", "cluster_name": "nucleus", "sync_local": {"start_word_offset": 4, "end_word_offset": 5}},
            req=req,
            action_index=1,
        )
        self.assertEqual(highlight["type"], "highlight_cluster")
        self.assertEqual(highlight["cluster_name"], "nucleus")

    def test_sorted_active_objects_for_cleanup_top_to_bottom(self):
        pool = {
            "b": {"name": "B", "bbox": {"x": 50, "y": 400, "w": 100, "h": 50}, "created_order": 2},
            "a": {"name": "A", "bbox": {"x": 10, "y": 100, "w": 100, "h": 50}, "created_order": 1},
            "c": {"name": "C", "bbox": {"x": 20, "y": 100, "w": 100, "h": 50}, "created_order": 3},
        }
        out = timeline.sorted_active_objects_for_cleanup(pool)
        self.assertEqual([x["name"] for x in out], ["A", "C", "B"])


if __name__ == "__main__":
    unittest.main()
