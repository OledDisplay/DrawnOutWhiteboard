#!/usr/bin/env python3
"""
Stage-2 visual description pipeline for accepted Wikidata components.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html import unescape
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover - optional dependency at runtime
    DDGS = None

try:
    from .QwenWorker import ServerQwenWorker
except Exception:
    from QwenWorker import ServerQwenWorker


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIMEDIA_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_WORKER = str(
    os.environ.get("QWEN_VLLM_SERVER_URL", os.environ.get("QWEN_WORKER_URL", "http://127.0.0.1:8009"))
    or "http://127.0.0.1:8009"
).strip().rstrip("/")
DEFAULT_OUTPUT_ROOT = os.environ.get("VISUAL_STAGE_OUTPUT_ROOT", ".dist")
USER_AGENT = os.environ.get(
    "VISUAL_STAGE_UA",
    "WikidataVisualDescriptionStage/1.0 (component-visual-stage; contact: local-script)",
)
REFINED_VISUAL_DESCRIPTION_MAX_CHARS = max(120, int(os.environ.get("C2_REFINED_VISUAL_DESCRIPTION_MAX_CHARS", "700") or 700))


def _clean_text(value: Any) -> str:
    text = unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _slugify(value: str, fallback: str, limit: int = 56) -> str:
    text = _clean_text(value)
    if not text:
        text = fallback
    original = text
    text = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    if not text:
        text = fallback
    limit = max(12, int(limit or 56))
    if len(text) > limit:
        digest = hashlib.sha1(original.encode("utf-8", errors="ignore")).hexdigest()[:8]
        prefix_limit = max(1, limit - len(digest) - 1)
        text = f"{text[:prefix_limit].strip('._ ')}_{digest}"
    return text[:limit].strip("._ ") or fallback


def _shorten(text: str, limit: int) -> str:
    text = _clean_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].strip() or text[:limit].strip()


def _sentence_chunks(text: str, take: int = 2, from_end: bool = False) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return ""
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
    if not parts:
        return cleaned[:280]
    selected = parts[-take:] if from_end else parts[:take]
    return " ".join(selected)[:320].strip()


def _component_key(label: str, qid: str) -> str:
    base = _slugify(label or qid, qid or "component", limit=64)
    if qid:
        return f"{qid}:{base}"
    return base


class WorkerClient:
    def __init__(self, url: str, timeout: int = 240):
        self.url = str(url or DEFAULT_WORKER).rstrip("/")
        self.timeout = int(timeout)
        self.worker = ServerQwenWorker(
            model_name=str(os.environ.get("QWEN_TEXT_MODEL_ID", os.environ.get("QWEN_MODEL", "cyankiwi/Qwen3.5-4B-AWQ-4bit")) or "cyankiwi/Qwen3.5-4B-AWQ-4bit"),
            max_new_tokens=max(96, int(os.environ.get("C2_QWEN_MAX_NEW_TOKENS", "256") or 256)),
            server_base_url=self.url,
            stage_io_dir=str(os.environ.get("QWEN_STAGE_IO_DIR", "") or ""),
        )

    def infer(self, task: str, mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.worker.infer({"task": task, "mode": mode, "payload": payload})

    def infer_many(self, task: str, mode: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.worker.infer_many(
            [
                {"task": task, "mode": mode, "payload": payload}
                for payload in (payloads or [])
            ]
        )


class ReferenceClient:
    def __init__(self, user_agent: str = USER_AGENT, timeout: int = 30):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        adapter = HTTPAdapter(pool_connections=24, pool_maxsize=24)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _api_get(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def get_entity(self, qid: str) -> Dict[str, Any]:
        data = self._api_get(
            WIKIDATA_API,
            {
                "action": "wbgetentities",
                "ids": qid,
                "languages": "en",
                "props": "labels|descriptions|claims|sitelinks/urls",
                "format": "json",
            },
        )
        return (data.get("entities") or {}).get(qid, {})

    def component_profile(self, qid: str) -> Dict[str, Any]:
        entity = self.get_entity(qid)
        wikipedia_url = self._entity_wikipedia_url(entity)
        wikipedia_title = self._wikipedia_title_from_url(wikipedia_url)
        wikipedia_lead = ""
        wikipedia_sections: List[Dict[str, str]] = []
        if wikipedia_title:
            try:
                wikipedia_lead = self.fetch_wikipedia_extract(wikipedia_title, sentences=2)
            except Exception:
                wikipedia_lead = ""
            try:
                wikipedia_sections = self.fetch_wikipedia_sections(wikipedia_title)
            except Exception:
                wikipedia_sections = [{"id": "0", "title": "Lead"}]
        commons_file = self._extract_commons_filename(entity)
        return {
            "qid": qid,
            "label": self._entity_label(entity),
            "description": self._entity_description(entity),
            "wikipedia_url": wikipedia_url,
            "wikipedia_title": wikipedia_title,
            "wikipedia_lead": wikipedia_lead,
            "wikipedia_sections": wikipedia_sections,
            "commons_file": commons_file,
        }

    def fetch_wikipedia_extract(self, title: str, sentences: int = 0, section: Optional[str] = None) -> str:
        if not title:
            return ""
        if section is not None and str(section).strip() not in {"", "0"}:
            data = self._api_get(
                WIKIPEDIA_API,
                {
                    "action": "parse",
                    "page": title,
                    "redirects": 1,
                    "prop": "text",
                    "section": str(section),
                    "format": "json",
                },
            )
            html = (((data.get("parse") or {}).get("text") or {}).get("*", "")) if isinstance((data.get("parse") or {}).get("text"), dict) else ""
            if not html:
                return ""
            soup = BeautifulSoup(html, "html.parser")
            return _clean_text(soup.get_text(" ", strip=True))

        params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "redirects": 1,
            "explaintext": 1,
            "format": "json",
        }
        if sentences > 0:
            params["exsentences"] = int(sentences)
            params["exintro"] = 1
        data = self._api_get(WIKIPEDIA_API, params)
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            if isinstance(page, dict):
                return _clean_text(page.get("extract", ""))
        return ""

    def fetch_wikipedia_sections(self, title: str) -> List[Dict[str, str]]:
        if not title:
            return []
        data = self._api_get(
            WIKIPEDIA_API,
            {
                "action": "parse",
                "page": title,
                "redirects": 1,
                "prop": "sections",
                "format": "json",
            },
        )
        rows = [{"id": "0", "title": "Lead"}]
        for row in (data.get("parse") or {}).get("sections", [])[:16]:
            if not isinstance(row, dict):
                continue
            index = str(row.get("index", "")).strip()
            line = _clean_text(row.get("line", ""))
            if index and line:
                rows.append({"id": index, "title": line})
        return rows

    def fetch_commons_file_candidate(self, filename: str, source_kind: str = "wikidata_p18") -> Optional[Dict[str, Any]]:
        clean_name = _clean_text(filename).replace("File:", "").strip()
        if not clean_name:
            return None
        data = self._api_get(
            WIKIMEDIA_API,
            {
                "action": "query",
                "titles": f"File:{clean_name}",
                "prop": "imageinfo|info",
                "iiprop": "url|extmetadata|mime",
                "iiurlwidth": 1200,
                "inprop": "url",
                "format": "json",
            },
        )
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            candidate = self._commons_page_to_candidate(page, source_kind=source_kind, query="")
            if candidate:
                return candidate
        return None

    def search_commons(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()
        clean_query = _clean_text(query)
        query_variants = []
        if clean_query:
            query_variants.append(f"filetype:bitmap {clean_query}")
            query_variants.append(clean_query)
        for search_query in query_variants:
            data = self._api_get(
                WIKIMEDIA_API,
                {
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": search_query,
                    "gsrnamespace": 6,
                    "gsrlimit": min(max(limit * 2, 10), 20),
                    "prop": "imageinfo|info",
                    "iiprop": "url|extmetadata|mime",
                    "iiurlwidth": 1200,
                    "inprop": "url",
                    "format": "json",
                },
            )
            pages = list(((data.get("query") or {}).get("pages") or {}).values())
            pages.sort(key=lambda item: int(item.get("index", 999999)) if isinstance(item, dict) else 999999)
            for page in pages:
                candidate = self._commons_page_to_candidate(page, source_kind="wikimedia", query=search_query)
                if not candidate:
                    continue
                key = (candidate.get("image_url", ""), candidate.get("page_url", ""))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(candidate)
                if len(candidates) >= limit:
                    return candidates
        return candidates

    def _commons_page_to_candidate(self, page: Dict[str, Any], source_kind: str, query: str) -> Optional[Dict[str, Any]]:
        if not isinstance(page, dict):
            return None
        imageinfo = (page.get("imageinfo") or [{}])[0]
        image_url = imageinfo.get("url") or imageinfo.get("thumburl")
        if not image_url:
            return None
        extmetadata = imageinfo.get("extmetadata") or {}
        description = " ".join(
            item
            for item in [
                _clean_text(self._meta_value(extmetadata, "ObjectName")),
                _clean_text(self._meta_value(extmetadata, "ImageDescription")),
                _clean_text(self._meta_value(extmetadata, "Artist")),
                _clean_text(self._meta_value(extmetadata, "Credit")),
                _clean_text(self._meta_value(extmetadata, "LicenseShortName")),
            ]
            if item
        ).strip()
        pageid = page.get("pageid")
        title = _clean_text(page.get("title", ""))
        page_url = f"https://commons.wikimedia.org/?curid={pageid}" if pageid else page.get("fullurl", "")
        return {
            "source_kind": source_kind,
            "title": title,
            "page_url": page_url,
            "image_url": image_url,
            "thumbnail_url": imageinfo.get("thumburl", ""),
            "description": description or title.replace("File:", "").replace("_", " "),
            "query": query,
            "license": _clean_text(self._meta_value(extmetadata, "LicenseShortName")),
        }

    @staticmethod
    def _meta_value(extmetadata: Dict[str, Any], key: str) -> str:
        value = extmetadata.get(key)
        if isinstance(value, dict):
            return value.get("value", "")
        if isinstance(value, str):
            return value
        return ""

    @staticmethod
    def _entity_label(entity: Dict[str, Any]) -> str:
        labels = entity.get("labels", {})
        row = labels.get("en") if isinstance(labels, dict) else None
        if isinstance(row, dict):
            return row.get("value", "")
        return ""

    @staticmethod
    def _entity_description(entity: Dict[str, Any]) -> str:
        descriptions = entity.get("descriptions", {})
        row = descriptions.get("en") if isinstance(descriptions, dict) else None
        if isinstance(row, dict):
            return row.get("value", "")
        return ""

    @staticmethod
    def _entity_wikipedia_url(entity: Dict[str, Any]) -> str:
        sitelinks = entity.get("sitelinks", {})
        enwiki = sitelinks.get("enwiki") if isinstance(sitelinks, dict) else None
        if isinstance(enwiki, dict):
            return str(enwiki.get("url", "") or "")
        return ""

    @staticmethod
    def _wikipedia_title_from_url(url: str) -> str:
        parsed = urlparse(url or "")
        if not parsed.path:
            return ""
        title = parsed.path.rsplit("/", 1)[-1]
        return unquote(title)

    @staticmethod
    def _extract_commons_filename(entity: Dict[str, Any]) -> str:
        claims = entity.get("claims", {})
        p18_rows = claims.get("P18", []) if isinstance(claims, dict) else []
        for row in p18_rows:
            mainsnak = row.get("mainsnak", {}) if isinstance(row, dict) else {}
            datavalue = mainsnak.get("datavalue", {}) if isinstance(mainsnak, dict) else {}
            value = datavalue.get("value") if isinstance(datavalue, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""


class VisualDescriptionStage:
    def __init__(
        self,
        worker: WorkerClient,
        references: ReferenceClient,
        output_root: str = DEFAULT_OUTPUT_ROOT,
        mode: str = "normal",
        component_batch_size: int = 4,
        logger: Optional[logging.Logger] = None,
    ):
        self.worker = worker
        self.references = references
        self.output_root = os.path.abspath(output_root)
        self.mode = mode
        self.component_batch_size = max(1, int(component_batch_size or 4))
        self.network_workers = max(
            self.component_batch_size,
            int(os.environ.get("VISUAL_STAGE_NETWORK_WORKERS", str(max(6, self.component_batch_size * 3))) or max(6, self.component_batch_size * 3)),
        )
        self.fast_query_refinement = str(os.environ.get("VISUAL_STAGE_FAST_QUERY_REFINEMENT", "1")).strip().lower() not in {"0", "false", "no", "off"}
        self.fast_section_selection = str(os.environ.get("VISUAL_STAGE_FAST_SECTION_SELECTION", "1")).strip().lower() not in {"0", "false", "no", "off"}
        self.fast_refinement_skip = str(os.environ.get("VISUAL_STAGE_FAST_REFINEMENT_SKIP", "1")).strip().lower() not in {"0", "false", "no", "off"}
        self.wikimedia_target_count = max(1, int(os.environ.get("VISUAL_STAGE_WIKIMEDIA_LIMIT", "3") or 3))
        self.ddg_target_count = max(0, int(os.environ.get("VISUAL_STAGE_DDG_LIMIT", "2") or 2))
        self.ddg_raw_limit = max(self.ddg_target_count, int(os.environ.get("VISUAL_STAGE_DDG_RAW_LIMIT", "12") or 12))
        self.ddg_context_count = max(0, int(os.environ.get("VISUAL_STAGE_DDG_CONTEXT_LIMIT", "0") or 0))
        self.wikipedia_text_limit = max(320, int(os.environ.get("VISUAL_STAGE_WIKIPEDIA_TEXT_LIMIT", "700") or 700))
        self.refine_candidate_limit = max(1, int(os.environ.get("VISUAL_STAGE_REFINE_CANDIDATES", "3") or 3))
        self.image_download_timeout = (
            max(1, int(os.environ.get("VISUAL_STAGE_IMG_CONNECT_TIMEOUT", "2") or 2)),
            max(2, int(os.environ.get("VISUAL_STAGE_IMG_READ_TIMEOUT", "4") or 4)),
        )
        self.logger = logger or logging.getLogger("visual_stage")

    def run(self, stage1_report: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        accepted = [row for row in stage1_report.get("accepted_components", []) if isinstance(row, dict)]
        target = stage1_report.get("target", {}) or {}
        self.logger.info(
            "visual stage start prompt=%r accepted=%s target=%s output_root=%s batch_size=%s",
            str(original_prompt or "")[:160],
            len(accepted),
            target.get("label") if isinstance(target, dict) else "",
            self.output_root,
            self.component_batch_size,
        )
        if not accepted:
            self.logger.warning("visual stage empty: no accepted components from C2 stage1 prompt=%r", str(original_prompt or "")[:160])
            return {
                "prompt_dir": "",
                "manifest_path": "",
                "components": [],
                "query_overrides": {},
                "error": "",
            }

        prompt_dir = os.path.join(
            self.output_root,
            _slugify(original_prompt or target.get("label", ""), target.get("qid", "prompt"), limit=40),
        )
        os.makedirs(prompt_dir, exist_ok=True)
        self.logger.info("visual stage prompt_dir=%s", os.path.abspath(prompt_dir))

        t0 = time.perf_counter()
        query_overrides = self._refine_queries(
            original_prompt=original_prompt,
            target=target,
            components=accepted,
        )
        self.logger.info("visual query refinement complete elapsed_s=%.3f overrides=%s", time.perf_counter() - t0, len(query_overrides))
        parts_context = self._parts_context(accepted)
        states = self._build_component_states(accepted, prompt_dir, query_overrides)
        self.logger.info(
            "visual states built count=%s skipped=%s labels=%s",
            len(states),
            sum(1 for state in states if str(state.get("skip_reason", "") or "").strip()),
            [str(state.get("label", "")) for state in states[:24]],
        )
        component_rows = []

        for chunk_index, chunk in enumerate(self._chunked(states, self.component_batch_size), start=1):
            labels = [str(state.get("label", "")) for state in chunk]
            self.logger.info("visual chunk start index=%s size=%s labels=%s", chunk_index, len(chunk), labels)
            chunk_t0 = time.perf_counter()
            phase_t0 = time.perf_counter()
            self._load_profiles_parallel(chunk)
            self.logger.info("visual chunk phase profiles_done index=%s elapsed_s=%.3f loaded=%s errors=%s", chunk_index, time.perf_counter() - phase_t0, sum(1 for s in chunk if s.get("profile")), [s.get("error") for s in chunk if s.get("error")][:8])
            phase_t0 = time.perf_counter()
            self._batch_select_sections(chunk, original_prompt)
            self.logger.info("visual chunk phase section_select_done index=%s elapsed_s=%.3f", chunk_index, time.perf_counter() - phase_t0)
            phase_t0 = time.perf_counter()
            self._load_selected_wikipedia_text_parallel(chunk)
            self.logger.info("visual chunk phase wikipedia_text_done index=%s elapsed_s=%.3f text_chars=%s", chunk_index, time.perf_counter() - phase_t0, [len(str(s.get("selected_wikipedia_text", "") or "")) for s in chunk])
            phase_t0 = time.perf_counter()
            self._batch_extract_wikipedia_descriptions(chunk, original_prompt, parts_context)
            self.logger.info("visual chunk phase extract_descriptions_done index=%s elapsed_s=%.3f desc_chars=%s", chunk_index, time.perf_counter() - phase_t0, [len(str(s.get("wikipedia_visual_description", "") or "")) for s in chunk])
            phase_t0 = time.perf_counter()
            self._collect_candidates_parallel(chunk)
            self.logger.info("visual chunk phase collect_candidates_done index=%s elapsed_s=%.3f candidate_counts=%s", chunk_index, time.perf_counter() - phase_t0, [len(s.get("image_candidates") or []) for s in chunk])
            phase_t0 = time.perf_counter()
            self._batch_refine_descriptions(chunk, original_prompt)
            self.logger.info("visual chunk phase refine_descriptions_done index=%s elapsed_s=%.3f refined_chars=%s", chunk_index, time.perf_counter() - phase_t0, [len(str(s.get("refined_visual_description", "") or "")) for s in chunk])
            for state in chunk:
                row = self._finalize_component_state(
                    state=state,
                    target=target,
                    original_prompt=original_prompt,
                )
                self.logger.info(
                    "visual component finalized label=%s qid=%s candidates=%s wiki_desc_chars=%s refined_chars=%s error=%s json=%s",
                    row.get("label"),
                    row.get("qid"),
                    len(row.get("image_candidates") or []),
                    len(str(row.get("wikipedia_visual_description", "") or "")),
                    len(str(row.get("refined_visual_description", "") or "")),
                    row.get("error"),
                    row.get("json_path"),
                )
                component_rows.append(row)
            self.logger.info("visual chunk complete index=%s elapsed_s=%.3f", chunk_index, time.perf_counter() - chunk_t0)

        manifest = {
            "target_prompt": original_prompt,
            "target": target,
            "prompt_dir": os.path.abspath(prompt_dir),
            "component_batch_size": self.component_batch_size,
            "query_overrides": query_overrides,
            "components": component_rows,
        }
        manifest_path = os.path.join(prompt_dir, "visual_stage_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        self.logger.info("visual stage complete components=%s manifest=%s", len(component_rows), os.path.abspath(manifest_path))

        return {
            "prompt_dir": os.path.abspath(prompt_dir),
            "manifest_path": os.path.abspath(manifest_path),
            "component_batch_size": self.component_batch_size,
            "components": component_rows,
            "query_overrides": query_overrides,
            "error": "",
        }

    def _refine_queries(self, original_prompt: str, target: Dict[str, Any], components: List[Dict[str, Any]]) -> Dict[str, str]:
        if self.fast_query_refinement:
            return self._heuristic_query_refinements(original_prompt=original_prompt, target=target, components=components)
        payload = {
            "target": {
                "prompt": original_prompt,
                "label": target.get("label", ""),
                "description": target.get("description", ""),
            },
            "components": [
                {
                    "qid": row.get("qid", ""),
                    "label": row.get("label", ""),
                    "description": row.get("description", ""),
                    "note": row.get("note", ""),
                }
                for row in components[:24]
            ],
        }
        try:
            result = self.worker.infer(task="visual_query_refinement", mode=self.mode, payload=payload)
        except Exception:
            self.logger.exception("visual query refinement failed; using defaults")
            return {}

        out: Dict[str, str] = {}
        for row in result.get("queries", []):
            if not isinstance(row, dict):
                continue
            qid = str(row.get("qid", "")).strip()
            query = _shorten(str(row.get("query", "")).strip(), 120)
            if qid and query:
                out[qid] = query
        return out

    def _heuristic_query_refinements(self, original_prompt: str, target: Dict[str, Any], components: List[Dict[str, Any]]) -> Dict[str, str]:
        prompt = _clean_text(original_prompt)
        target_label = _clean_text(target.get("label", ""))
        seed = target_label or prompt
        prompt_l = prompt.lower()
        seed_l = seed.lower()
        out: Dict[str, str] = {}
        for row in components[:24]:
            if not isinstance(row, dict):
                continue
            qid = str(row.get("qid", "")).strip()
            label = _clean_text(row.get("label", ""))
            if not qid or not label:
                continue
            label_l = label.lower()
            if not seed:
                out[qid] = _shorten(label, 120)
                continue
            if label_l in prompt_l or label_l == seed_l:
                continue
            out[qid] = _shorten(f"{seed} {label}".strip(), 120)
        return out

    def _parts_context(self, components: Iterable[Dict[str, Any]]) -> str:
        rows = []
        for row in list(components)[:5]:
            label = _clean_text(row.get("label", ""))
            desc = _shorten(_clean_text(row.get("description", "")), 48)
            piece = label
            if desc:
                piece += f": {desc}"
            rows.append(piece)
        return " | ".join(rows)

    def _build_component_states(self, components: List[Dict[str, Any]], prompt_dir: str, query_overrides: Dict[str, str]) -> List[Dict[str, Any]]:
        states = []
        for component in components:
            qid = str(component.get("qid", "")).strip()
            label = str(component.get("label", "")).strip() or qid
            component_dir = os.path.join(prompt_dir, _slugify(label, qid or "component", limit=32))
            os.makedirs(component_dir, exist_ok=True)
            skipped_reason = self._visual_skip_reason(component)
            states.append(
                {
                    "component": component,
                    "qid": qid,
                    "label": label,
                    "component_key": _component_key(label, qid),
                    "component_dir": component_dir,
                    "search_query": query_overrides.get(qid) or label or qid,
                    "profile": {},
                    "selected_sections": [],
                    "selected_wikipedia_text": "",
                    "wikipedia_description": _clean_text(component.get("description", "")),
                    "all_candidates": [],
                    "refined_description": "",
                    "error": "",
                    "skipped_reason": skipped_reason,
                }
            )
        return states

    def _chunked(self, rows: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
        step = max(1, int(size or 1))
        for idx in range(0, len(rows), step):
            yield rows[idx:idx + step]

    def _mark_state_error(self, state: Dict[str, Any], message: str) -> None:
        if not state.get("error"):
            state["error"] = str(message)

    def _visual_skip_reason(self, component: Dict[str, Any]) -> str:
        label = _clean_text(component.get("label", "")).lower()
        desc = _clean_text(component.get("description", "")).lower()
        note = _clean_text(component.get("note", "")).lower()
        if "class" in note:
            return "skipped generic class-like candidate"
        if label in {"entity", "member of a group", "organelle", "cellular component", "component", "part", "element", "structure", "system"}:
            return "skipped generic candidate"
        bad_desc_terms = (
            "anything that can be considered",
            "group element",
            "component of a cell",
            "organized cell-level biological structure",
            "maximum traffic flow",
            "time management tool",
            "ratio between",
            "force resisting",
            "class of",
            "type of relation",
        )
        if any(term in desc for term in bad_desc_terms):
            return "skipped abstract candidate"
        return ""

    def _load_profiles_parallel(self, states: List[Dict[str, Any]]) -> None:
        todo = [state for state in states if not state.get("error") and not state.get("skipped_reason") and state.get("qid")]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=max(1, min(self.network_workers, len(todo)))) as pool:
            future_map = {pool.submit(self.references.component_profile, state["qid"]): state for state in todo}
            for future in as_completed(future_map):
                state = future_map[future]
                try:
                    state["profile"] = future.result() or {}
                except Exception as exc:
                    self.logger.exception("visual stage profile fetch failed for %s", state.get("qid", ""))
                    self._mark_state_error(state, exc)

    def _batch_select_sections(self, states: List[Dict[str, Any]], original_prompt: str) -> None:
        active = [state for state in states if not state.get("error") and not state.get("skipped_reason")]
        if not active:
            return
        if self.fast_section_selection:
            for state in active:
                profile = state.get("profile", {}) or {}
                state["selected_sections"] = self._select_sections_heuristically(profile)
                state["profile"]["selected_wikipedia_sections"] = state["selected_sections"]
            return
        payloads = []
        mapped = []
        for state in active:
            profile = state.get("profile", {}) or {}
            sections = [row for row in profile.get("wikipedia_sections", []) if isinstance(row, dict)]
            if not sections:
                state["selected_sections"] = ["0"]
                continue
            payloads.append(
                {
                    "target_prompt": original_prompt,
                    "component": {
                        "qid": state["component"].get("qid", ""),
                        "label": state["component"].get("label", ""),
                        "description": state["component"].get("description", ""),
                    },
                    "page": {
                        "title": profile.get("wikipedia_title", ""),
                        "lead": _shorten(profile.get("wikipedia_lead", ""), 220),
                    },
                    "sections": sections[:12],
                }
            )
            mapped.append(state)
        if not payloads:
            return
        try:
            results = self.worker.infer_many(task="wikipedia_section_select", mode=self.mode, payloads=payloads)
        except Exception as exc:
            self.logger.exception("wikipedia section selection batch failed")
            results = [{"sections": ["0"]} for _ in payloads]
        for state, result in zip(mapped, results):
            allowed = {
                str(row.get("id", "")).strip()
                for row in (state.get("profile", {}) or {}).get("wikipedia_sections", [])
                if isinstance(row, dict) and str(row.get("id", "")).strip()
            }
            chosen = []
            for sid in result.get("sections", []):
                sid = str(sid).strip()
                if sid and sid in allowed and sid not in chosen:
                    chosen.append(sid)
                if len(chosen) >= 2:
                    break
            state["selected_sections"] = chosen or ["0"]
            state["profile"]["selected_wikipedia_sections"] = state["selected_sections"]

    def _select_sections_heuristically(self, profile: Dict[str, Any]) -> List[str]:
        sections = [row for row in profile.get("wikipedia_sections", []) if isinstance(row, dict)]
        if not sections:
            return ["0"]
        preferred_terms = (
            "description", "design", "structure", "construction", "components", "layout",
            "anatomy", "morphology", "form", "configuration", "body", "frame", "wheel",
            "appearance", "overview",
        )
        allowed = []
        titles_by_id: Dict[str, str] = {}
        for row in sections[:12]:
            sid = str(row.get("id", "")).strip()
            title = _clean_text(row.get("title", "")).lower()
            if not sid:
                continue
            allowed.append(sid)
            titles_by_id[sid] = title
        if not allowed:
            return ["0"]
        picked = ["0"] if "0" in allowed else [allowed[0]]
        for sid in allowed:
            if sid in picked:
                continue
            if any(term in titles_by_id.get(sid, "") for term in preferred_terms):
                picked.append(sid)
                break
        return picked[:2]

    def _load_selected_wikipedia_text_parallel(self, states: List[Dict[str, Any]]) -> None:
        todo = [state for state in states if not state.get("error") and not state.get("skipped_reason")]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=max(1, min(self.network_workers, len(todo)))) as pool:
            future_map = {pool.submit(self._fetch_selected_wikipedia_text, state.get("profile", {}), state.get("selected_sections", [])): state for state in todo}
            for future in as_completed(future_map):
                state = future_map[future]
                try:
                    state["selected_wikipedia_text"] = future.result() or ""
                except Exception as exc:
                    self.logger.exception("selected wikipedia text fetch failed for %s", state.get("qid", ""))
                    state["selected_wikipedia_text"] = ""

    def _batch_extract_wikipedia_descriptions(self, states: List[Dict[str, Any]], original_prompt: str, parts_context: str) -> None:
        payloads = []
        mapped = []
        for state in states:
            if state.get("error") or state.get("skipped_reason"):
                continue
            text = _clean_text(state.get("selected_wikipedia_text", ""))
            if not text:
                state["wikipedia_description"] = _clean_text(state["component"].get("description", ""))
                continue
            profile = state.get("profile", {}) or {}
            payloads.append(
                {
                    "target_prompt": original_prompt,
                    "parts_context": parts_context,
                    "component": {
                        "qid": state["component"].get("qid", ""),
                        "label": state["component"].get("label", ""),
                        "description": state["component"].get("description", ""),
                        "note": state["component"].get("note", ""),
                    },
                    "page": {
                        "title": profile.get("wikipedia_title", ""),
                        "url": profile.get("wikipedia_url", ""),
                        "sections": ", ".join(state.get("selected_sections", [])),
                        "text": text[:self.wikipedia_text_limit],
                    },
                }
            )
            mapped.append(state)
        if not payloads:
            return
        try:
            results = self.worker.infer_many(task="wikipedia_visual_extract", mode=self.mode, payloads=payloads)
        except Exception:
            self.logger.exception("wikipedia visual extract batch failed")
            results = [{"visual_description": ""} for _ in payloads]
        for state, result in zip(mapped, results):
            state["wikipedia_description"] = _clean_text(result.get("visual_description", "")) or _clean_text(state["component"].get("description", ""))

    def _collect_candidates_parallel(self, states: List[Dict[str, Any]]) -> None:
        todo = [state for state in states if not state.get("error") and not state.get("skipped_reason")]
        if not todo:
            return
        with ThreadPoolExecutor(max_workers=max(1, min(self.network_workers, len(todo)))) as pool:
            future_map = {pool.submit(self._collect_all_candidates_for_state, state): state for state in todo}
            for future in as_completed(future_map):
                state = future_map[future]
                try:
                    state["all_candidates"] = future.result()
                except Exception as exc:
                    self.logger.exception("image candidate collection failed for %s", state.get("qid", ""))
                    state["all_candidates"] = []

    def _collect_all_candidates_for_state(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        profile = state.get("profile", {}) or {}
        search_query = state.get("search_query", "") or state.get("label", "") or state.get("qid", "")
        component_dir = state.get("component_dir", "")
        wikimedia_candidates = self._collect_wikimedia_candidates(profile, search_query, component_dir)
        ddg_candidates = self._collect_ddg_candidates(search_query, component_dir)
        return self._assign_candidate_ids((wikimedia_candidates + ddg_candidates)[:10])

    def _batch_refine_descriptions(self, states: List[Dict[str, Any]], original_prompt: str) -> None:
        payloads = []
        mapped = []
        for state in states:
            if state.get("error") or state.get("skipped_reason"):
                continue
            candidates = state.get("all_candidates", []) or []
            if not candidates or self._should_skip_refinement(state):
                state["refined_description"] = _shorten(state.get("wikipedia_description", ""), REFINED_VISUAL_DESCRIPTION_MAX_CHARS)
                continue
            payloads.append(
                {
                    "target_prompt": original_prompt,
                    "component": {
                        "qid": state["component"].get("qid", ""),
                        "label": state["component"].get("label", ""),
                        "description": state["component"].get("description", ""),
                        "note": state["component"].get("note", ""),
                    },
                    "base_description": state.get("wikipedia_description", ""),
                    "candidates": [
                        {
                            "id": row.get("id", ""),
                            "source_kind": row.get("source_kind", ""),
                            "description": row.get("description", ""),
                        }
                        for row in candidates[:self.refine_candidate_limit]
                    ],
                }
            )
            mapped.append(state)
        if not payloads:
            return
        try:
            results = self.worker.infer_many(task="visual_description_refinement", mode=self.mode, payloads=payloads)
        except Exception:
            self.logger.exception("visual description refinement batch failed")
            results = [{"refined_description": ""} for _ in payloads]
        for state, result in zip(mapped, results):
            state["refined_description"] = _shorten(
                _clean_text(result.get("refined_description", "")) or state.get("wikipedia_description", ""),
                REFINED_VISUAL_DESCRIPTION_MAX_CHARS,
            )

    def _should_skip_refinement(self, state: Dict[str, Any]) -> bool:
        if not self.fast_refinement_skip:
            return False
        base_description = _clean_text(state.get("wikipedia_description", ""))
        candidates = [row for row in (state.get("all_candidates", []) or []) if isinstance(row, dict)]
        if not candidates:
            return True
        if not base_description:
            return False
        base_words = {word for word in re.findall(r"[a-z0-9]+", base_description.lower()) if len(word) > 3}
        novel_candidates = 0
        for row in candidates[:self.refine_candidate_limit]:
            desc = _clean_text(row.get("description", ""))
            if len(desc) < 32:
                continue
            desc_words = {word for word in re.findall(r"[a-z0-9]+", desc.lower()) if len(word) > 3}
            if not desc_words:
                continue
            overlap = len(base_words & desc_words)
            if overlap < max(3, int(len(desc_words) * 0.5)):
                novel_candidates += 1
            if novel_candidates >= 1:
                return False
        return True

    def _finalize_component_state(self, *, state: Dict[str, Any], target: Dict[str, Any], original_prompt: str) -> Dict[str, Any]:
        if state.get("skipped_reason"):
            error_path = os.path.join(state["component_dir"], "visual_profile.json")
            payload = {
                "target_prompt": original_prompt,
                "component": state["component"],
                "component_key": state.get("component_key", ""),
                "error": state["skipped_reason"],
            }
            with open(error_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            return {
                "qid": state.get("qid", ""),
                "label": state.get("label", ""),
                "component": dict(state.get("component") or {}),
                "component_key": str(state.get("component_key", "") or ""),
                "component_dir": os.path.abspath(state["component_dir"]),
                "json_path": os.path.abspath(error_path),
                "search_query": state.get("search_query", ""),
                "query": state.get("search_query", ""),
                "wikipedia_visual_description": "",
                "refined_visual_description": "",
                "image_candidates": [],
                "candidate_count": 0,
                "error": state["skipped_reason"],
            }
        if state.get("error"):
            error_path = os.path.join(state["component_dir"], "visual_profile.json")
            error_payload = {
                "target_prompt": original_prompt,
                "component": state["component"],
                "component_key": state.get("component_key", ""),
                "error": state["error"],
            }
            with open(error_path, "w", encoding="utf-8") as handle:
                json.dump(error_payload, handle, ensure_ascii=False, indent=2)
            return {
                "qid": state.get("qid", ""),
                "label": state.get("label", ""),
                "component": dict(state.get("component") or {}),
                "component_key": str(state.get("component_key", "") or ""),
                "component_dir": os.path.abspath(state["component_dir"]),
                "json_path": os.path.abspath(error_path),
                "search_query": state.get("search_query", ""),
                "query": state.get("search_query", ""),
                "wikipedia_visual_description": "",
                "refined_visual_description": "",
                "image_candidates": [],
                "candidate_count": 0,
                "error": state["error"],
            }
        return self._write_component_output(
            component_dir=state["component_dir"],
            target=target,
            original_prompt=original_prompt,
            component=state["component"],
            profile=state.get("profile", {}),
            search_query=state.get("search_query", ""),
            wikipedia_description=state.get("wikipedia_description", ""),
            refined_description=state.get("refined_description", "") or state.get("wikipedia_description", ""),
            candidates=state.get("all_candidates", []),
        )

    def _extract_wikipedia_visual_description(
        self,
        *,
        original_prompt: str,
        parts_context: str,
        component: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> str:
        selected_sections = self._select_wikipedia_sections(
            original_prompt=original_prompt,
            component=component,
            profile=profile,
        )
        profile["selected_wikipedia_sections"] = selected_sections
        wikipedia_text = self._fetch_selected_wikipedia_text(profile, selected_sections)
        if not wikipedia_text:
            return _clean_text(component.get("description", ""))

        payload = {
            "target_prompt": original_prompt,
            "parts_context": parts_context,
            "component": {
                "qid": component.get("qid", ""),
                "label": component.get("label", ""),
                "description": component.get("description", ""),
                "note": component.get("note", ""),
            },
            "page": {
                "title": profile.get("wikipedia_title", ""),
                "url": profile.get("wikipedia_url", ""),
                "sections": ", ".join(selected_sections),
                "text": wikipedia_text[:self.wikipedia_text_limit],
            },
        }
        result = self.worker.infer(task="wikipedia_visual_extract", mode=self.mode, payload=payload)
        return _clean_text(result.get("visual_description", "")) or _clean_text(component.get("description", ""))

    def _select_wikipedia_sections(
        self,
        *,
        original_prompt: str,
        component: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> List[str]:
        sections = [row for row in profile.get("wikipedia_sections", []) if isinstance(row, dict)]
        if not sections:
            return ["0"]
        payload = {
            "target_prompt": original_prompt,
            "component": {
                "qid": component.get("qid", ""),
                "label": component.get("label", ""),
                "description": component.get("description", ""),
            },
            "page": {
                "title": profile.get("wikipedia_title", ""),
                "lead": _shorten(profile.get("wikipedia_lead", ""), 220),
            },
            "sections": sections[:12],
        }
        try:
            result = self.worker.infer(task="wikipedia_section_select", mode=self.mode, payload=payload)
        except Exception:
            self.logger.exception("wikipedia section selection failed for %s", component.get("qid", ""))
            return ["0"]
        chosen = []
        allowed = {str(row.get("id", "")).strip() for row in sections if str(row.get("id", "")).strip()}
        for sid in result.get("sections", []):
            sid = str(sid).strip()
            if sid and sid in allowed and sid not in chosen:
                chosen.append(sid)
            if len(chosen) >= 2:
                break
        return chosen or ["0"]

    def _fetch_selected_wikipedia_text(self, profile: Dict[str, Any], selected_sections: List[str]) -> str:
        title = str(profile.get("wikipedia_title", "")).strip()
        if not title:
            return ""
        bits = []
        chosen_map = {
            str(row.get("id", "")).strip(): _clean_text(row.get("title", ""))
            for row in profile.get("wikipedia_sections", [])
            if isinstance(row, dict)
        }
        for sid in selected_sections[:2]:
            sid = str(sid).strip()
            if not sid:
                continue
            if sid == "0":
                text = _shorten(profile.get("wikipedia_lead", ""), 320)
                heading = chosen_map.get("0", "Lead")
            else:
                try:
                    text = _shorten(self.references.fetch_wikipedia_extract(title, section=sid), 700)
                except Exception:
                    text = ""
                heading = chosen_map.get(sid, sid)
            if text:
                bits.append(f"{heading}: {text}")
        return " ".join(bits).strip()

    def _collect_wikimedia_candidates(self, profile: Dict[str, Any], query: str, component_dir: str) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        seen = set()

        direct = self.references.fetch_commons_file_candidate(profile.get("commons_file", ""))
        if direct:
            candidates.append(direct)

        if len(candidates) < self.wikimedia_target_count:
            search_limit = max(self.wikimedia_target_count, 1)
            for row in self.references.search_commons(query, limit=search_limit):
                candidates.append(row)

        out = []
        for idx, candidate in enumerate(candidates, start=1):
            key = (candidate.get("image_url", ""), candidate.get("page_url", ""))
            if key in seen:
                continue
            seen.add(key)
            local_path = self._download_image(
                candidate.get("thumbnail_url", "") or candidate.get("image_url", ""),
                component_dir,
                stem=f"wikimedia_{idx:02d}",
            )
            enriched = dict(candidate)
            enriched["local_path"] = local_path
            enriched["description"] = _shorten(enriched.get("description", ""), 360)
            out.append(enriched)
            if len(out) >= self.wikimedia_target_count:
                break
        return out

    def _collect_ddg_candidates(self, query: str, component_dir: str) -> List[Dict[str, Any]]:
        if self.ddg_target_count <= 0:
            return []
        if DDGS is None:
            self.logger.warning("duckduckgo_search is not available; DDG candidates skipped")
            return []

        headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
        raw_rows = []
        try:
            with DDGS(headers=headers) as ddgs:
                for item in ddgs.images(query, max_results=self.ddg_raw_limit, safesearch="moderate", region="wt-wt"):
                    if isinstance(item, dict):
                        raw_rows.append(item)
                    if len(raw_rows) >= self.ddg_raw_limit:
                        break
        except Exception:
            self.logger.exception("DDG image search failed for query=%s", query)
            return []

        out = []
        seen = set()
        for idx, item in enumerate(raw_rows, start=1):
            page_url = str(item.get("url") or item.get("source") or "").strip()
            image_url = str(item.get("image") or "").strip()
            if not page_url or not image_url:
                continue
            key = (image_url, page_url)
            if key in seen:
                continue
            seen.add(key)
            context = self._extract_ddg_page_context(page_url=page_url, image_url=image_url) if len(out) < self.ddg_context_count else {"description": "", "context_above": "", "context_below": "", "page_title": ""}
            local_path = self._download_image(str(item.get("thumbnail") or image_url), component_dir, stem=f"ddg_{idx:02d}")
            description = context.get("description", "") or _clean_text(item.get("title", "")) or query
            out.append(
                {
                    "source_kind": "ddg",
                    "title": _clean_text(item.get("title", "")),
                    "page_url": page_url,
                    "image_url": image_url,
                    "thumbnail_url": str(item.get("thumbnail") or "").strip(),
                    "description": _shorten(description, 320),
                    "context_above": context.get("context_above", ""),
                    "context_below": context.get("context_below", ""),
                    "page_title": context.get("page_title", ""),
                    "local_path": local_path,
                }
            )
            if len(out) >= self.ddg_target_count:
                break
        return out

    def _extract_ddg_page_context(self, page_url: str, image_url: str) -> Dict[str, str]:
        try:
            response = self.references.session.get(page_url, timeout=(3, 5))
            response.raise_for_status()
        except Exception:
            return {"description": "", "context_above": "", "context_below": "", "page_title": ""}

        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        if "html" not in content_type and "<html" not in (response.text or "").lower():
            return {"description": "", "context_above": "", "context_below": "", "page_title": ""}

        soup = BeautifulSoup(response.text or "", "html.parser")
        page_title = _clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
        tag = self._find_matching_image_tag(soup, page_url, image_url)
        above = self._nearest_text(tag, direction="previous") if tag else ""
        below = self._nearest_text(tag, direction="next") if tag else ""
        caption = ""
        if tag:
            figure = tag.find_parent("figure")
            if figure:
                caption_tag = figure.find("figcaption")
                if caption_tag:
                    caption = _sentence_chunks(caption_tag.get_text(" ", strip=True), take=2)

        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if meta_tag:
            meta_desc = _sentence_chunks(meta_tag.get("content", ""), take=2)

        description_bits = [item for item in [caption, above, below, meta_desc] if item]
        description = " ".join(description_bits).strip()
        if not description:
            description = meta_desc or page_title
        return {
            "description": description,
            "context_above": above,
            "context_below": below,
            "page_title": page_title,
        }

    def _find_matching_image_tag(self, soup: BeautifulSoup, page_url: str, image_url: str):
        target_norm = self._normalize_media_url(image_url, page_url)
        target_name = os.path.basename(urlparse(target_norm).path).lower()
        best = None

        for tag in soup.find_all("img"):
            candidate_url = (
                tag.get("src")
                or tag.get("data-src")
                or tag.get("data-original")
                or tag.get("data-lazy-src")
                or ""
            )
            if not candidate_url:
                continue
            candidate_norm = self._normalize_media_url(candidate_url, page_url)
            if not candidate_norm:
                continue
            if candidate_norm == target_norm:
                return tag
            candidate_name = os.path.basename(urlparse(candidate_norm).path).lower()
            if target_name and candidate_name and target_name == candidate_name:
                best = tag
        return best

    @staticmethod
    def _normalize_media_url(url: str, base_url: str) -> str:
        joined = urljoin(base_url, str(url or "").strip())
        parsed = urlparse(joined)
        if not parsed.scheme or not parsed.netloc:
            return ""
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    def _nearest_text(self, tag, direction: str) -> str:
        if tag is None:
            return ""
        iterator = tag.find_all_previous if direction == "previous" else tag.find_all_next
        pieces = []
        seen = set()
        for node in iterator(["figcaption", "p", "li", "dd", "dt", "blockquote", "div"]):
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) < 40:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            chunk = _sentence_chunks(text, take=2, from_end=(direction == "previous"))
            if chunk:
                pieces.append(chunk)
            if len(pieces) >= 2:
                break
        if direction == "previous":
            pieces.reverse()
        return " ".join(pieces).strip()

    def _refine_visual_description(
        self,
        *,
        original_prompt: str,
        component: Dict[str, Any],
        wikipedia_description: str,
        candidates: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "target_prompt": original_prompt,
            "component": {
                "qid": component.get("qid", ""),
                "label": component.get("label", ""),
                "description": component.get("description", ""),
                "note": component.get("note", ""),
            },
            "base_description": wikipedia_description,
            "candidates": [
                {
                    "id": row.get("id", ""),
                    "source_kind": row.get("source_kind", ""),
                    "description": row.get("description", ""),
                }
                for row in candidates[:10]
            ],
        }
        result = self.worker.infer(task="visual_description_refinement", mode=self.mode, payload=payload)
        return _shorten(
            _clean_text(result.get("refined_description", "")) or wikipedia_description,
            REFINED_VISUAL_DESCRIPTION_MAX_CHARS,
        )

    def _assign_candidate_ids(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for idx, row in enumerate(candidates, start=1):
            item = dict(row)
            item["id"] = f"IMG{idx}"
            out.append(item)
        return out

    def _download_image(self, url: str, dest_dir: str, stem: str) -> str:
        clean_url = str(url or "").strip()
        if not clean_url:
            return ""
        try:
            response = self.references.session.get(clean_url, timeout=self.image_download_timeout, stream=True)
            response.raise_for_status()
        except Exception:
            return ""

        content_type = str(response.headers.get("Content-Type", "") or "").lower()
        ext = os.path.splitext(urlparse(clean_url).path)[1].lower()
        if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
            if "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"
            elif "gif" in content_type:
                ext = ".gif"
            else:
                ext = ".jpg"
        path = os.path.join(dest_dir, f"{stem}{ext}")
        try:
            with open(path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        handle.write(chunk)
        except Exception:
            return ""
        return os.path.abspath(path)

    def _write_component_output(
        self,
        *,
        component_dir: str,
        target: Dict[str, Any],
        original_prompt: str,
        component: Dict[str, Any],
        profile: Dict[str, Any],
        search_query: str,
        wikipedia_description: str,
        refined_description: str,
        candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        json_path = os.path.join(component_dir, "visual_profile.json")
        payload = {
            "target_prompt": original_prompt,
            "target": target,
            "component_key": _component_key(
                str(component.get("label", "") or ""),
                str(component.get("qid", "") or ""),
            ),
            "component": {
                "qid": component.get("qid", ""),
                "label": component.get("label", ""),
                "stage1_description": component.get("description", ""),
                "stage1_note": component.get("note", ""),
                "wikipedia_url": profile.get("wikipedia_url", ""),
                "wikipedia_title": profile.get("wikipedia_title", ""),
                "selected_wikipedia_sections": profile.get("selected_wikipedia_sections", []),
                "search_query": search_query,
            },
            "wikipedia_visual_description": wikipedia_description,
            "refined_visual_description": _shorten(refined_description, REFINED_VISUAL_DESCRIPTION_MAX_CHARS),
            "image_candidates": candidates,
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return {
            "qid": component.get("qid", ""),
            "label": component.get("label", ""),
            "component_key": payload.get("component_key", ""),
            "component": payload.get("component", {}),
            "component_dir": os.path.abspath(component_dir),
            "json_path": os.path.abspath(json_path),
            "search_query": search_query,
            "query": search_query,
            "wikipedia_visual_description": wikipedia_description,
            "refined_visual_description": _shorten(refined_description, REFINED_VISUAL_DESCRIPTION_MAX_CHARS),
            "image_candidates": candidates,
            "candidate_count": len(candidates),
            "error": "",
        }


def run_visual_description_stage(
    *,
    stage1_report: Dict[str, Any],
    original_prompt: str,
    worker_url: str = DEFAULT_WORKER,
    worker_client: Optional[Any] = None,
    mode: str = "normal",
    output_root: str = DEFAULT_OUTPUT_ROOT,
    component_batch_size: int = 6,
    worker_timeout: int = 120,
    http_timeout: int = 30,
) -> Dict[str, Any]:
    worker = worker_client or WorkerClient(worker_url, timeout=worker_timeout)
    references = ReferenceClient(timeout=http_timeout)
    stage = VisualDescriptionStage(
        worker=worker,
        references=references,
        output_root=output_root,
        mode=mode,
        component_batch_size=component_batch_size,
    )
    return stage.run(stage1_report=stage1_report, original_prompt=original_prompt)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stage-2 visual description pipeline")
    parser.add_argument("--report", default="", help="Path to a stage-1 report JSON file. If omitted, read stdin.")
    parser.add_argument("--target-prompt", required=True, help="Original stage-1 mother object prompt")
    parser.add_argument("--worker", default=DEFAULT_WORKER, help="Qwen worker /infer URL")
    parser.add_argument("--worker-timeout", type=int, default=120, help="HTTP timeout for worker calls")
    parser.add_argument("--mode", default="normal", choices=["normal", "thinking"], help="worker mode")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root folder for Prompt/Component outputs")
    parser.add_argument("--component-batch-size", type=int, default=4, help="How many components to process in parallel per chunk")
    parser.add_argument("--timeout", type=int, default=30, help="HTTP timeout for Wikimedia/Wikipedia calls")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print final JSON")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.report:
        with open(args.report, "r", encoding="utf-8") as handle:
            stage1_report = json.load(handle)
    else:
        stage1_report = json.load(os.fdopen(0, "r", encoding="utf-8"))

    result = run_visual_description_stage(
        stage1_report=stage1_report,
        original_prompt=args.target_prompt,
        worker_url=args.worker,
        mode=args.mode,
        output_root=args.output_root,
        component_batch_size=args.component_batch_size,
        worker_timeout=args.worker_timeout,
        http_timeout=args.timeout,
    )

    if args.pretty:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
