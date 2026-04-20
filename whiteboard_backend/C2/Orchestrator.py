#!/usr/bin/env python3
"""
Wikidata / WDQS component-finding orchestrator.

This build uses a strict MCP-like action contract with the worker, keeps the dynamic payload compact,
and removes deterministic / heuristic fallbacks for Qwen failures.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests
from requests.adapters import HTTPAdapter

try:
    from .VisualDescriptionStage import run_visual_description_stage
    from .QwenWorker import ServerQwenWorker
except Exception:
    from VisualDescriptionStage import run_visual_description_stage
    from QwenWorker import ServerQwenWorker


WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_WORKER = str(
    os.environ.get("QWEN_VLLM_SERVER_URL", os.environ.get("QWEN_WORKER_URL", "http://127.0.0.1:8009"))
    or "http://127.0.0.1:8009"
).strip().rstrip("/")
USER_AGENT = os.environ.get(
    "WIKIDATA_COMPONENT_AGENT_UA",
    "WikidataComponentAgent/1.2 (graph-part-orchestrator; contact: local-script)",
)
SCAN_ROUTES = {"dir", "rev", "cls", "char", "up"}
ENTITY_QUERY_SIMPLIFIER_SYSTEM_PROMPT = """
You rewrite failed Wikidata entity search queries into one alternative short search query.

Goal:
- make the next entity-search attempt meaningfully different
- simplify, shorten, or lightly rephrase the query
- preserve the core concrete object when possible

Rules:
- return one plain-text query only
- no explanation
- no markdown
- no quotes
- keep it short, usually 1 to 5 words
- you may drop modifiers, remove trailing qualifiers, replace hyphens, or use a simpler wording
- do not add facts that are not already implied by the input
""".strip()

_WIKIDATA_REQUEST_LIMIT = max(1, int(os.environ.get("WIKIDATA_HTTP_MAX_CONCURRENCY", "4") or 4))
_WIKIDATA_REQUEST_SEMAPHORE = threading.BoundedSemaphore(_WIKIDATA_REQUEST_LIMIT)
_WIKIDATA_RATE_LOCK = threading.Lock()
_WIKIDATA_COOLDOWN_UNTIL = 0.0


@dataclass
class MemoryEntry:
    mid: str
    action: Dict[str, Any]
    priority: int = 0
    tries: int = 0


@dataclass
class CandidateRecord:
    qid: str
    label: str = ""
    description: str = ""
    urls: List[str] = field(default_factory=list)
    direct_has_part_refs: int = 0
    direct_part_of_refs: int = 0
    class_routes: List[str] = field(default_factory=list)
    class_via: List[str] = field(default_factory=list)
    routes: Set[str] = field(default_factory=set)
    evidence_only_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    opened: bool = False
    sitelinks: int = 0
    last_open_summary: Dict[str, Any] = field(default_factory=dict)


class WorkerClient:
    def __init__(self, url: str, timeout: int = 180):
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


class LocalWorkerClient:
    def __init__(self, worker: Any):
        self.worker = worker

    def infer(self, task: str, mode: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self.worker.infer({"task": task, "mode": mode, "payload": payload})

    def infer_many(self, task: str, mode: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self.worker.infer_many(
            [
                {"task": task, "mode": mode, "payload": payload}
                for payload in (payloads or [])
            ]
        )


class WikidataClient:
    def __init__(self, user_agent: str = USER_AGENT, timeout: int = 30, language: str = "en"):
        self.user_agent = user_agent
        self.timeout = timeout
        self.sparql_retry_count = max(0, int(os.environ.get("WDQS_RETRY_COUNT", "1") or 1))
        self.sparql_retry_backoff = 1.25
        self.sparql_timeout_cap = max(self.timeout, int(os.environ.get("WDQS_TIMEOUT_CAP", "45") or 45))
        self.language = language
        self.max_workers = max(4, int(os.environ.get("WIKIDATA_MAX_WORKERS", "8") or 8))
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        adapter = HTTPAdapter(pool_connections=self.max_workers * 2, pool_maxsize=self.max_workers * 2)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.entity_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.seed_cache: Dict[str, Dict[str, Any]] = {}
        self.open_cache: Dict[str, Dict[str, Any]] = {}
        self.scan_cache: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
        self.check_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.survey_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        self.expand_cache: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

    def _acquire_request_slot(self) -> None:
        global _WIKIDATA_COOLDOWN_UNTIL
        while True:
            with _WIKIDATA_RATE_LOCK:
                wait_for = max(0.0, float(_WIKIDATA_COOLDOWN_UNTIL) - time.monotonic())
            if wait_for <= 0.0:
                break
            time.sleep(min(wait_for, 0.5))
        _WIKIDATA_REQUEST_SEMAPHORE.acquire()

    def _release_request_slot(self) -> None:
        try:
            _WIKIDATA_REQUEST_SEMAPHORE.release()
        except Exception:
            pass

    def _note_rate_limit(self, response: Optional[requests.Response], *, attempt: int) -> None:
        global _WIKIDATA_COOLDOWN_UNTIL
        retry_after = 0.0
        if response is not None:
            raw = str(response.headers.get("Retry-After", "") or "").strip()
            if raw:
                try:
                    retry_after = max(retry_after, float(raw))
                except Exception:
                    retry_after = retry_after
        retry_after = max(retry_after, self.sparql_retry_backoff * (attempt + 1), 1.0)
        with _WIKIDATA_RATE_LOCK:
            _WIKIDATA_COOLDOWN_UNTIL = max(float(_WIKIDATA_COOLDOWN_UNTIL), time.monotonic() + retry_after)

    def resolve_entity(self, text: str, limit: int = 10) -> List[Dict[str, Any]]:
        params = {
            "action": "wbsearchentities",
            "search": text,
            "language": self.language,
            "limit": limit,
            "format": "json",
            "type": "item",
        }
        data = self._api_get(params)
        rows = []
        for row in data.get("search", []):
            qid = row.get("id", "")
            rows.append(
                {
                    "id": qid,
                    "label": row.get("label", ""),
                    "description": row.get("description", ""),
                    "match": row.get("match", {}).get("text", ""),
                    "url": row.get("concepturi", self.page_url(qid)),
                }
            )
        if not rows:
            return []
        meta = self.get_entities([row["id"] for row in rows], props="labels|descriptions|claims|sitelinks/urls")
        enriched = []
        for row in rows:
            entity = meta.get(row["id"], {})
            p31 = self._extract_item_claims(entity, "P31")[:3] if entity else []
            p31_meta = self._meta_for_ids([item.get("id", "") for item in p31]) if p31 else {}
            enriched.append(
                {
                    "id": row["id"],
                    "label": row.get("label") or self._label(entity),
                    "description": row.get("description") or self._description(entity),
                    "match": row.get("match", ""),
                    "url": row.get("url", self.page_url(row["id"])),
                    "sitelinks": self._sitelinks_count(entity),
                    "inst": [p31_meta.get(item.get("id", ""), {}).get("label", item.get("id", "")) for item in p31],
                }
            )
        return enriched

    def expand_neighbors(self, qid: str, limit: int = 8) -> List[Dict[str, Any]]:
        cache_key = (qid, int(limit))
        cached = self.expand_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        query = self._query_neighbors(qid, limit)
        rows = self._run_sparql(query)

        cand_ids = [row.get("cand", "") for row in rows if row.get("cand")]
        meta = self._meta_for_ids(cand_ids)

        for row in rows:
            cand = row.get("cand", "")
            info = meta.get(cand, {})
            row["candLabel"] = info.get("label", row.get("candLabel", ""))
            row["candDescription"] = info.get("description", row.get("candDescription", ""))
            row["candUrl"] = info.get("url", self.page_url(cand) if cand else "")
            row["candSitelinks"] = info.get("sitelinks", 0)

        self.expand_cache[cache_key] = copy.deepcopy(rows)
        return rows

    def _query_neighbors(self, qid: str, limit: int) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?cand ?candLabel ?edge ?dir WHERE {{
  {{
    wd:{qid} wdt:P527 ?cand .
    BIND("P527" AS ?edge)
    BIND("out" AS ?dir)
  }}
  UNION
  {{
    wd:{qid} wdt:P361 ?cand .
    BIND("P361" AS ?edge)
    BIND("out" AS ?dir)
  }}
  UNION
  {{
    wd:{qid} wdt:P31 ?cand .
    BIND("P31" AS ?edge)
    BIND("out" AS ?dir)
  }}
  UNION
  {{
    wd:{qid} wdt:P279 ?cand .
    BIND("P279" AS ?edge)
    BIND("out" AS ?dir)
  }}
  UNION
  {{
    wd:{qid} wdt:P1552 ?cand .
    BIND("P1552" AS ?edge)
    BIND("out" AS ?dir)
  }}
  UNION
  {{
    ?cand wdt:P361 wd:{qid} .
    BIND("P361" AS ?edge)
    BIND("in" AS ?dir)
  }}
  UNION
  {{
    ?cand wdt:P527 wd:{qid} .
    BIND("P527" AS ?edge)
    BIND("in" AS ?dir)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
}}
LIMIT {int(limit)}
""".strip()

    def get_entities(self, ids: Sequence[str], props: str = "labels|descriptions|claims|sitelinks/urls") -> Dict[str, Any]:
        clean_ids = [item for item in ids if item]
        if not clean_ids:
            return {}
        unique_ids: List[str] = []
        seen = set()
        cached: Dict[str, Any] = {}
        missing: List[str] = []
        for qid in clean_ids:
            if qid in seen:
                continue
            seen.add(qid)
            unique_ids.append(qid)
            key = (qid, props)
            if key in self.entity_cache:
                cached[qid] = copy.deepcopy(self.entity_cache[key])
            else:
                missing.append(qid)

        if missing:
            for start in range(0, len(missing), 50):
                chunk = missing[start:start + 50]
                params = {
                    "action": "wbgetentities",
                    "ids": "|".join(chunk),
                    "languages": self.language,
                    "props": props,
                    "format": "json",
                }
                data = self._api_get(params)
                entities = data.get("entities", {})
                for qid, entity in entities.items():
                    self.entity_cache[(qid, props)] = copy.deepcopy(entity)
                    cached[qid] = copy.deepcopy(entity)

        return {qid: cached.get(qid, {}) for qid in unique_ids}

    def seed_fields(self, qid: str) -> Dict[str, Any]:
        cached = self.seed_cache.get(qid)
        if cached is not None:
            return copy.deepcopy(cached)
        entities = self.get_entities([qid], props="labels|descriptions|claims|sitelinks/urls")
        entity = entities.get(qid)
        if not entity:
            raise RuntimeError(f"entity not found: {qid}")

        parts = self._extract_item_claims(entity, "P527")
        characteristics = self._extract_item_claims(entity, "P1552")
        all_ids = [row["id"] for row in parts + characteristics if row.get("id")]
        meta = self._meta_for_ids(all_ids)
        result = {
            "subject": self._compact_entity_meta(entity),
            "parts": [self._merge_claim_with_meta(row, meta) for row in parts],
            "characteristics": [self._merge_claim_with_meta(row, meta) for row in characteristics],
        }
        self.seed_cache[qid] = copy.deepcopy(result)
        return result

    def open_node(self, qid: str) -> Dict[str, Any]:
        cached = self.open_cache.get(qid)
        if cached is not None:
            return copy.deepcopy(cached)
        entities = self.get_entities([qid], props="labels|descriptions|claims|sitelinks/urls")
        entity = entities.get(qid)
        if not entity:
            raise RuntimeError(f"entity not found: {qid}")

        selected_props = ["P31", "P279", "P361", "P527", "P1552"]
        raw_ids: List[str] = []
        by_prop: Dict[str, List[Dict[str, Any]]] = {}
        for pid in selected_props:
            claims = self._extract_item_claims(entity, pid)
            by_prop[pid] = claims
            raw_ids.extend([row["id"] for row in claims if row.get("id")])

        meta = self._meta_for_ids(raw_ids)
        result = {
            "id": qid,
            "label": self._label(entity),
            "description": self._description(entity),
            "url": self._entity_url(entity, qid),
            "sitelinks": self._sitelinks_count(entity),
            "P31": [self._merge_claim_with_meta(row, meta) for row in by_prop["P31"][:8]],
            "P279": [self._merge_claim_with_meta(row, meta) for row in by_prop["P279"][:8]],
            "P361": [self._merge_claim_with_meta(row, meta) for row in by_prop["P361"][:8]],
            "P527": [self._merge_claim_with_meta(row, meta) for row in by_prop["P527"][:8]],
            "P1552": [self._merge_claim_with_meta(row, meta) for row in by_prop["P1552"][:8]],
        }
        self.open_cache[qid] = copy.deepcopy(result)
        return result

    def open_many(self, qids: Sequence[str]) -> List[Dict[str, Any]]:
        clean = []
        seen = set()
        for qid in qids:
            qid = str(qid).strip()
            if qid and qid not in seen:
                clean.append(qid)
                seen.add(qid)
            if len(clean) >= 4:
                break
        if not clean:
            return []

        out: Dict[str, Dict[str, Any]] = {}
        missing: List[str] = []
        for qid in clean:
            cached = self.open_cache.get(qid)
            if cached is not None:
                out[qid] = copy.deepcopy(cached)
            else:
                missing.append(qid)

        if missing:
            entities = self.get_entities(missing, props="labels|descriptions|claims|sitelinks/urls")
            raw_ids: List[str] = []
            claim_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for qid in missing:
                entity = entities.get(qid, {})
                claim_map[qid] = {}
                for pid in ("P31", "P279", "P361", "P527", "P1552"):
                    rows = self._extract_item_claims(entity, pid)[:8]
                    claim_map[qid][pid] = rows
                    raw_ids.extend([row["id"] for row in rows if row.get("id")])
            meta = self._meta_for_ids(raw_ids)
            for qid in missing:
                entity = entities.get(qid, {})
                profile = {
                    "id": qid,
                    "label": self._label(entity),
                    "description": self._description(entity),
                    "url": self._entity_url(entity, qid),
                    "sitelinks": self._sitelinks_count(entity),
                    "P31": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P31", [])],
                    "P279": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P279", [])],
                    "P361": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P361", [])],
                    "P527": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P527", [])],
                    "P1552": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P1552", [])],
                }
                self.open_cache[qid] = copy.deepcopy(profile)
                out[qid] = profile

        return [copy.deepcopy(out[qid]) for qid in clean if qid in out]

    def preview_nodes(self, qids: Sequence[str]) -> List[Dict[str, Any]]:
        clean = []
        seen = set()
        for qid in qids:
            qid = str(qid).strip()
            if qid and qid not in seen:
                clean.append(qid)
                seen.add(qid)
            if len(clean) >= 12:
                break
        if not clean:
            return []
        entities = self.get_entities(clean, props="labels|descriptions|claims|sitelinks/urls")
        raw_ids: List[str] = []
        claim_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for qid in clean:
            entity = entities.get(qid, {})
            claim_map[qid] = {}
            for pid in ("P31", "P279", "P361", "P527"):
                rows = self._extract_item_claims(entity, pid)[:4]
                claim_map[qid][pid] = rows
                raw_ids.extend([row.get("id", "") for row in rows if row.get("id")])
        meta = self._meta_for_ids(raw_ids)
        previews = []
        for qid in clean:
            entity = entities.get(qid, {})
            previews.append({
                "id": qid,
                "label": self._label(entity),
                "description": self._description(entity),
                "url": self._entity_url(entity, qid),
                "sitelinks": self._sitelinks_count(entity),
                "P31": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P31", [])[:3]],
                "P279": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P279", [])[:3]],
                "P361": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P361", [])[:3]],
                "P527": [self._merge_claim_with_meta(row, meta) for row in claim_map[qid].get("P527", [])[:3]],
            })
        return previews

    def survey(self, qid: str, limit: int = 8) -> List[Dict[str, Any]]:
        cache_key = (qid, int(limit))
        cached = self.survey_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)

        jobs = {
            "dir": ("dir", min(max(4, limit), 10)),
            "rev": ("rev", min(max(4, limit), 8)),
            "cls": ("cls", min(max(3, limit // 2), 6)),
            "char": ("char", min(max(3, limit // 2), 6)),
        }
        results_by_name: Dict[str, List[Dict[str, Any]]] = {"dir": [], "rev": [], "cls": [], "char": []}
        with ThreadPoolExecutor(max_workers=4) as pool:
            future_map = {
                pool.submit(self.scan_route, qid=qid, route=route, limit=row_limit): name
                for name, (route, row_limit) in jobs.items()
            }
            for future in as_completed(future_map):
                name = future_map[future]
                try:
                    results_by_name[name] = future.result()
                except requests.exceptions.Timeout:
                    if name == "char":
                        logging.getLogger("wikidata_client").warning("char survey scan timed out for %s; continuing without char rows", qid)
                    else:
                        raise

        dir_rows = results_by_name["dir"]
        rev_rows = results_by_name["rev"]
        cls_rows = results_by_name["cls"]
        char_rows = results_by_name["char"]
        merged: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for source_name, rows in (("dir", dir_rows), ("rev", rev_rows), ("cls", cls_rows), ("char", char_rows)):
            for row in rows:
                cand = row.get("cand", "")
                if not cand or cand == qid:
                    continue
                if cand not in merged:
                    merged[cand] = {
                        "cand": cand,
                        "candLabel": row.get("candLabel", ""),
                        "candDescription": row.get("candDescription", ""),
                        "candUrl": row.get("candUrl", self.page_url(cand)),
                        "candSitelinks": row.get("candSitelinks", 0),
                        "sources": [],
                        "routes": [],
                        "refs": 0,
                        "via": [],
                    }
                    order.append(cand)
                target = merged[cand]
                target["candLabel"] = target["candLabel"] or row.get("candLabel", "")
                target["candDescription"] = target["candDescription"] or row.get("candDescription", "")
                target["candUrl"] = row.get("candUrl", target["candUrl"])
                target["candSitelinks"] = max(int(target.get("candSitelinks", 0) or 0), int(row.get("candSitelinks", 0) or 0))
                target["sources"].append(source_name)
                target["routes"].append(row.get("route", source_name))
                target["refs"] = max(int(target.get("refs", 0) or 0), self._as_int(row.get("refs", 0)))
                via = row.get("viaLabel", row.get("via", ""))
                if via:
                    target["via"].append(via)
        out = []
        for cand in order[:limit]:
            row = merged[cand]
            row["sources"] = sorted(set(row.get("sources", [])))
            row["routes"] = sorted(set(row.get("routes", [])))
            row["via"] = sorted(set(row.get("via", [])))[:3]
            out.append(row)
        self.survey_cache[cache_key] = copy.deepcopy(out)
        return out

    def probe_many(self, subject_qid: str, candidate_qids: Sequence[str]) -> List[Dict[str, Any]]:
        with ThreadPoolExecutor(max_workers=2) as pool:
            future_open = pool.submit(self.open_many, candidate_qids)
            future_check = pool.submit(self.check_many, subject_qid, candidate_qids)
            opens = {row.get("id", ""): row for row in future_open.result()}
            checks = {row.get("candidate", ""): row for row in future_check.result()}
        rows = []
        for qid in candidate_qids:
            qid = str(qid).strip()
            if not qid:
                continue
            rows.append({
                "candidate": qid,
                "open": opens.get(qid, {}),
                "check": checks.get(qid, {"subject": subject_qid, "candidate": qid, "direct_has_part_refs": 0, "direct_part_of_refs": 0, "direct_routes": [], "class_routes": [], "class_via": []}),
            })
        return rows

    def scan_route(self, qid: str, route: str, limit: int = 6) -> List[Dict[str, Any]]:
        route = route.strip().lower()
        cache_key = (qid, route, int(limit))
        cached = self.scan_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        if route == "dir":
            query = self._query_dir(qid, limit)
        elif route == "rev":
            query = self._query_rev(qid, limit)
        elif route == "cls":
            query = self._query_cls(qid, limit)
        elif route == "char":
            query = self._query_char(qid, limit)
        elif route == "up":
            query = self._query_up(qid, limit)
        else:
            raise ValueError(f"unsupported route: {route}")

        rows = self._run_sparql(query)
        cand_ids = [row.get("cand", "") for row in rows if row.get("cand")]
        meta = self._meta_for_ids(cand_ids)
        for row in rows:
            qid_value = row.get("cand", "")
            if qid_value in meta:
                row["candLabel"] = meta[qid_value].get("label", row.get("candLabel", ""))
                row["candDescription"] = meta[qid_value].get("description", "")
                row["candUrl"] = meta[qid_value].get("url", self.page_url(qid_value))
                row["candSitelinks"] = meta[qid_value].get("sitelinks", 0)
        self.scan_cache[cache_key] = copy.deepcopy(rows)
        return rows

    def check_relation(self, subject_qid: str, candidate_qid: str) -> Dict[str, Any]:
        cache_key = (subject_qid, candidate_qid)
        cached = self.check_cache.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
        direct_query = self._query_check_direct(subject_qid, candidate_qid)
        class_query = self._query_check_class(subject_qid, candidate_qid)
        try:
            direct_rows = self._run_sparql(direct_query)
            class_rows = self._run_sparql(class_query)
        except requests.exceptions.RequestException as exc:
            logging.getLogger("orchestrator").warning(
                "check_relation failed for subject=%s candidate=%s; treating as unsupported: %s",
                subject_qid,
                candidate_qid,
                exc,
            )
            result = {
                "subject": subject_qid,
                "candidate": candidate_qid,
                "direct_has_part_refs": 0,
                "direct_part_of_refs": 0,
                "direct_routes": [],
                "class_routes": [],
                "class_via": [],
                "error": str(exc),
            }
            self.check_cache[cache_key] = copy.deepcopy(result)
            return result

        direct_has_part_refs = 0
        direct_part_of_refs = 0
        direct_routes: List[str] = []
        for row in direct_rows:
            route = row.get("route", "")
            ref_count = self._as_int(row.get("refs", 0))
            if route == "direct_p527":
                direct_has_part_refs += max(1, ref_count)
                direct_routes.append(route)
            elif route == "inverse_p361":
                direct_part_of_refs += max(1, ref_count)
                direct_routes.append(route)

        class_routes: List[str] = []
        class_via: List[str] = []
        for row in class_rows:
            route = row.get("route", "")
            via_label = row.get("viaLabel", row.get("via", ""))
            if route:
                class_routes.append(route)
            if via_label:
                class_via.append(via_label)

        result = {
            "subject": subject_qid,
            "candidate": candidate_qid,
            "direct_has_part_refs": direct_has_part_refs,
            "direct_part_of_refs": direct_part_of_refs,
            "direct_routes": sorted(set(direct_routes)),
            "class_routes": sorted(set(class_routes)),
            "class_via": sorted(set(class_via)),
        }
        self.check_cache[cache_key] = copy.deepcopy(result)
        return result

    def check_many(self, subject_qid: str, candidate_qids: Sequence[str]) -> List[Dict[str, Any]]:
        clean = []
        seen = set()
        for qid in candidate_qids:
            qid = str(qid).strip()
            if qid and qid not in seen:
                clean.append(qid)
                seen.add(qid)
            if len(clean) >= 4:
                break
        if not clean:
            return []
        out: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max(1, min(self.max_workers, len(clean)))) as pool:
            future_map = {pool.submit(self.check_relation, subject_qid, qid): qid for qid in clean}
            for future in as_completed(future_map):
                qid = future_map[future]
                try:
                    out[qid] = future.result()
                except requests.exceptions.RequestException as exc:
                    logging.getLogger("orchestrator").warning(
                        "check_many future failed for subject=%s candidate=%s; skipping: %s",
                        subject_qid,
                        qid,
                        exc,
                    )
        return [copy.deepcopy(out[qid]) for qid in clean if qid in out]

    def page_url(self, qid: str) -> str:
        return f"https://www.wikidata.org/wiki/{qid}"

    def _api_get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._acquire_request_slot()
        try:
            response = self.session.get(WIKIDATA_API, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in {429, 500, 502, 503, 504}:
                self._note_rate_limit(getattr(exc, "response", None), attempt=0)
            raise
        finally:
            self._release_request_slot()

    def _run_sparql(self, query: str) -> List[Dict[str, Any]]:
        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": self.user_agent,
        }
        last_exc: Optional[Exception] = None
        payload: Dict[str, Any] = {}
        for attempt in range(self.sparql_retry_count + 1):
            request_timeout = min(self.sparql_timeout_cap, self.timeout + attempt * 20)
            try:
                self._acquire_request_slot()
                try:
                    response = self.session.get(
                        WDQS_ENDPOINT,
                        params={"query": query, "format": "json"},
                        headers=headers,
                        timeout=request_timeout,
                    )
                finally:
                    self._release_request_slot()
                response.raise_for_status()
                payload = response.json()
                last_exc = None
                break
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout) as exc:
                last_exc = exc
                self._note_rate_limit(None, attempt=attempt)
                if attempt >= self.sparql_retry_count:
                    raise
                time.sleep(self.sparql_retry_backoff * (attempt + 1))
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status in {429, 500, 502, 503, 504}:
                    self._note_rate_limit(getattr(exc, "response", None), attempt=attempt)
                if status not in {429, 500, 502, 503, 504} or attempt >= self.sparql_retry_count:
                    raise
                time.sleep(self.sparql_retry_backoff * (attempt + 1))
        if last_exc is not None:
            raise last_exc
        rows = []
        for binding in payload.get("results", {}).get("bindings", []):
            row = {}
            for key, value in binding.items():
                raw = value.get("value", "")
                if raw.startswith("http://www.wikidata.org/entity/"):
                    raw = raw.rsplit("/", 1)[-1]
                row[key] = raw
            rows.append(row)
        return rows

    def _meta_for_ids(self, ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
        clean = []
        seen = set()
        for qid in ids:
            if qid and qid not in seen:
                seen.add(qid)
                clean.append(qid)
        if not clean:
            return {}
        entities = self.get_entities(clean, props="labels|descriptions|sitelinks/urls")
        meta = {}
        for qid, entity in entities.items():
            meta[qid] = self._compact_entity_meta(entity)
        return meta

    def _merge_claim_with_meta(self, claim: Dict[str, Any], meta: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        qid = claim.get("id", "")
        info = meta.get(qid, {})
        merged = dict(claim)
        merged["label"] = info.get("label", merged.get("label", ""))
        merged["description"] = info.get("description", merged.get("description", ""))
        merged["url"] = info.get("url", self.page_url(qid) if qid else "")
        merged["sitelinks"] = info.get("sitelinks", 0)
        return merged

    def _compact_entity_meta(self, entity: Dict[str, Any]) -> Dict[str, Any]:
        qid = entity.get("id", "")
        return {
            "id": qid,
            "label": self._label(entity),
            "description": self._description(entity),
            "url": self._entity_url(entity, qid),
            "sitelinks": self._sitelinks_count(entity),
        }

    def _entity_url(self, entity: Dict[str, Any], qid: str) -> str:
        sitelinks = entity.get("sitelinks", {})
        enwiki = sitelinks.get("enwiki") if isinstance(sitelinks, dict) else None
        if isinstance(enwiki, dict) and enwiki.get("url"):
            return enwiki["url"]
        return self.page_url(qid)

    def _sitelinks_count(self, entity: Dict[str, Any]) -> int:
        sitelinks = entity.get("sitelinks", {})
        return len(sitelinks) if isinstance(sitelinks, dict) else 0

    def _label(self, entity: Dict[str, Any]) -> str:
        labels = entity.get("labels", {})
        if isinstance(labels, dict):
            row = labels.get(self.language) or next(iter(labels.values()), {})
            if isinstance(row, dict):
                return row.get("value", "")
        return ""

    def _description(self, entity: Dict[str, Any]) -> str:
        descriptions = entity.get("descriptions", {})
        if isinstance(descriptions, dict):
            row = descriptions.get(self.language) or next(iter(descriptions.values()), {})
            if isinstance(row, dict):
                return row.get("value", "")
        return ""

    def _extract_item_claims(self, entity: Dict[str, Any], pid: str) -> List[Dict[str, Any]]:
        claims = entity.get("claims", {}).get(pid, [])
        rows: List[Dict[str, Any]] = []
        for claim in claims:
            mainsnak = claim.get("mainsnak", {})
            if mainsnak.get("snaktype") != "value":
                continue
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if not isinstance(value, dict):
                continue
            qid = value.get("id")
            if not qid:
                continue
            refs = claim.get("references", [])
            rows.append({"id": qid, "rank": claim.get("rank", "normal"), "refs": len(refs)})
        return rows

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0

    def _query_dir(self, qid: str, limit: int) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX prov: <http://www.w3.org/ns/prov#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT ?cand ?candLabel ?route (COUNT(?ref) AS ?refs) WHERE {{
  {{
    wd:{qid} p:P527 ?st .
    ?st ps:P527 ?cand .
    OPTIONAL {{ ?st prov:wasDerivedFrom ?ref . }}
    BIND("direct_p527" AS ?route)
  }}
  UNION
  {{
    ?cand p:P361 ?st .
    ?st ps:P361 wd:{qid} .
    OPTIONAL {{ ?st prov:wasDerivedFrom ?ref . }}
    BIND("inverse_p361" AS ?route)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
}}
GROUP BY ?cand ?candLabel ?route
LIMIT {int(limit)}
""".strip()

    def _query_cls(self, qid: str, limit: int) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?cand ?candLabel ?route ?via ?viaLabel WHERE {{
  {{
    wd:{qid} (wdt:P31|wdt:P279) ?seed .
    ?seed wdt:P279* ?via .
    ?via wdt:P527 ?cand .
    BIND("class_p527" AS ?route)
  }}
  UNION
  {{
    wd:{qid} (wdt:P31|wdt:P279) ?seed .
    ?seed wdt:P279* ?via .
    ?via wdt:P2670 ?cand .
    BIND("class_p2670" AS ?route)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
}}
LIMIT {int(limit)}
""".strip()

    def _query_rev(self, qid: str, limit: int) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?cand ?candLabel ?route ?via ?viaLabel WHERE {{
  {{
    wd:{qid} wdt:P279 ?seed .
    ?seed wdt:P279* ?via .
    ?cand wdt:P361 ?via .
    BIND("class_inverse_p361" AS ?route)
  }}
  UNION
  {{
    wd:{qid} wdt:P31 ?seed .
    ?seed wdt:P279* ?via .
    ?cand wdt:P361 ?via .
    BIND("class_inverse_p361" AS ?route)
  }}
  FILTER(?cand != wd:{qid})
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
}}
LIMIT {int(limit)}
""".strip()

    def _query_char(self, qid: str, limit: int) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?cand ?candLabel ?route WHERE {{
  wd:{qid} wdt:P1552 ?cand .
  BIND("char_node" AS ?route)
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
}}
LIMIT {int(limit)}
""".strip()

    def _query_up(self, qid: str, limit: int) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?cand ?candLabel ?route WHERE {{
  {{ wd:{qid} wdt:P31 ?cand . BIND("P31" AS ?route) }}
  UNION
  {{ wd:{qid} wdt:P279 ?cand . BIND("P279" AS ?route) }}
  UNION
  {{ wd:{qid} wdt:P361 ?cand . BIND("P361" AS ?route) }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
}}
LIMIT {int(limit)}
""".strip()

    def _query_check_direct(self, subject_qid: str, candidate_qid: str) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX prov: <http://www.w3.org/ns/prov#>
SELECT ?route (COUNT(?ref) AS ?refs) WHERE {{
  {{
    wd:{subject_qid} p:P527 ?st .
    ?st ps:P527 wd:{candidate_qid} .
    OPTIONAL {{ ?st prov:wasDerivedFrom ?ref . }}
    BIND("direct_p527" AS ?route)
  }}
  UNION
  {{
    wd:{candidate_qid} p:P361 ?st .
    ?st ps:P361 wd:{subject_qid} .
    OPTIONAL {{ ?st prov:wasDerivedFrom ?ref . }}
    BIND("inverse_p361" AS ?route)
  }}
}}
GROUP BY ?route
""".strip()

    def _query_check_class(self, subject_qid: str, candidate_qid: str) -> str:
        return f"""
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?route ?via ?viaLabel WHERE {{
  {{
    wd:{subject_qid} (wdt:P31|wdt:P279) ?seed .
    ?seed wdt:P279* ?via .
    ?via wdt:P527 wd:{candidate_qid} .
    BIND("class_p527" AS ?route)
  }}
  UNION
  {{
    wd:{subject_qid} (wdt:P31|wdt:P279) ?seed .
    ?seed wdt:P279* ?via .
    ?via wdt:P2670 wd:{candidate_qid} .
    BIND("class_p2670" AS ?route)
  }}
  UNION
  {{
    wd:{subject_qid} wdt:P279 ?seed .
    ?seed wdt:P279* ?via .
    wd:{candidate_qid} wdt:P361 ?via .
    BIND("class_inverse_p361" AS ?route)
  }}
  UNION
  {{
    wd:{subject_qid} wdt:P31 ?seed .
    ?seed wdt:P279* ?via .
    wd:{candidate_qid} wdt:P361 ?via .
    BIND("class_inverse_p361" AS ?route)
  }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "{self.language}". }}
}}
LIMIT 8
""".strip()


class ComponentAgentOrchestrator:
    def __init__(
        self,
        target_text: str,
        worker: WorkerClient,
        wikidata: WikidataClient,
        max_steps: int = 5,
        mode: str = "normal",
        limit: int = 6,
        qid: Optional[str] = None,
    ):
        self.target_text = target_text
        self.worker = worker
        self.wikidata = wikidata
        self.max_steps = max_steps
        self.mode = mode
        self.limit = limit
        self.target_qid = qid or ""
        self.target_label = ""
        self.target_description = ""
        self.target_resolution: Dict[str, Any] = {}
        self.subject_seed: Dict[str, Any] = {}
        self.memory: List[MemoryEntry] = []
        self.memory_counter = 1
        self.history: List[Dict[str, Any]] = []
        self.last_results: List[Dict[str, Any]] = []
        self.candidates: Dict[str, CandidateRecord] = {}
        self.rejected: Dict[str, str] = {}
        self.opened_nodes: Dict[str, Dict[str, Any]] = {}
        self.executed_signatures: Set[str] = set()
        self.logger = logging.getLogger("orchestrator")
        self.accepted_qids: Set[str] = set()
        self.held_qids: Set[str] = set()
        self.rejected_qids: Set[str] = set()
        self.fast_stage1 = str(os.environ.get("ORCHESTRATOR_FAST_STAGE1", "0")).strip().lower() not in {"0", "false", "no", "off"}

    def _resolve_candidates_once(self, query: str, limit: int) -> List[Dict[str, Any]]:
        try:
            return self.wikidata.resolve_entity(query, limit=limit)
        except Exception as exc:
            self.logger.warning("resolve_entity failed for query=%r: %s", query, exc)
            return []

    def _iter_suffix_trim_queries(self, phrase: str) -> List[str]:
        text = re.sub(r"\s+", " ", str(phrase or "")).strip(" -")
        if not text:
            return []
        out: List[str] = []
        seen: Set[str] = set()
        current = text
        while True:
            cut = max(current.rfind(" "), current.rfind("-"))
            if cut < 0:
                break
            current = current[:cut].rstrip(" -")
            if not current:
                break
            key = current.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(current)
        return out

    def _parse_simplified_query_text(self, raw_text: str) -> str:
        text = str(raw_text or "").strip()
        if not text:
            return ""
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                text = str(parsed.get("query", "") or "").strip()
        except Exception:
            pass
        if not text:
            return ""
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().startswith("```")]
        if lines:
            text = lines[0]
        text = re.sub(r"^\s*query\s*:\s*", "", text, flags=re.IGNORECASE).strip()
        text = text.strip("`\"' ")
        text = re.sub(r"\s+", " ", text).strip(" -")
        if len(text) > 120:
            text = text[:120].rsplit(" ", 1)[0].strip() or text[:120].strip()
        return text

    def _qwen_simplify_entity_query(self, phrase: str, attempted_queries: Sequence[str]) -> str:
        local_worker = getattr(self.worker, "worker", None)
        generate = getattr(local_worker, "_generate", None)
        if local_worker is None or not callable(generate):
            return ""

        payload = {
            "query": phrase,
            "attempted": [str(item).strip() for item in (attempted_queries or []) if str(item).strip()][:8],
        }
        user_prompt = (
            "TASK=entity_query_simplify\n"
            "Return one alternative simplified Wikidata entity search query.\n"
            "It must differ from the already tried queries.\n"
            "INPUT=" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        )

        try:
            bundle = generate(
                task="entity_query_simplify",
                system_prompt=ENTITY_QUERY_SIMPLIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                mode=self.mode,
            )
        except Exception as exc:
            self.logger.warning("entity_query_simplify prompt failed for %r: %s", phrase, exc)
            return ""

        raw_text = str(bundle.get("parse_text", "") or bundle.get("final_text", "") or bundle.get("raw_text", "")).strip()
        candidate = self._parse_simplified_query_text(raw_text)
        if not candidate:
            return ""

        attempted = {str(item).strip().casefold() for item in (attempted_queries or []) if str(item).strip()}
        if candidate.casefold() in attempted:
            return ""
        return candidate

    def _resolve_entity_with_fallbacks(self, phrase: str, limit: int = 10) -> Dict[str, Any]:
        original = re.sub(r"\s+", " ", str(phrase or "")).strip(" -")
        tried: List[str] = []
        tried_keys: Set[str] = set()

        def _try_query(query: str, strategy: str) -> Optional[Dict[str, Any]]:
            query = re.sub(r"\s+", " ", str(query or "")).strip(" -")
            if not query:
                return None
            key = query.casefold()
            if key in tried_keys:
                return None
            tried_keys.add(key)
            tried.append(query)
            candidates = self._resolve_candidates_once(query, limit=limit)
            if not candidates:
                return None
            return {
                "query": query,
                "strategy": strategy,
                "candidates": candidates,
                "tried_queries": list(tried),
            }

        found = _try_query(original, "direct")
        if found is not None:
            return found

        for query in self._iter_suffix_trim_queries(original):
            found = _try_query(query, "trimmed_suffix")
            if found is not None:
                return found

        simplified = self._qwen_simplify_entity_query(original, attempted_queries=tried)
        if simplified:
            found = _try_query(simplified, "qwen_simplified")
            if found is not None:
                return found
            for query in self._iter_suffix_trim_queries(simplified):
                found = _try_query(query, "qwen_simplified_trimmed_suffix")
                if found is not None:
                    return found

        return {
            "query": original,
            "strategy": "unresolved",
            "candidates": [],
            "tried_queries": list(tried),
            "qwen_query": simplified,
        }

    def _bucketed_qids(self) -> Set[str]:
        return set(self.accepted_qids) | set(self.held_qids) | set(self.rejected_qids)

    def _seed_size(self) -> int:
        return len(self.subject_seed.get("parts", []) or []) + len(self.subject_seed.get("characteristics", []) or [])

    def _route_action_count(self) -> int:
        count = 0
        for row in self.history:
            action = row.get("action", {}) if isinstance(row, dict) else {}
            result = row.get("result", {}) if isinstance(row, dict) else {}
            if str(action.get("tool", "")).strip() in {"survey", "scan", "expand_neighbors"} and bool(result.get("ok", False)):
                count += 1
        return count

    def _expanded_anchor_qids(self) -> Set[str]:
        out: Set[str] = set()
        for row in self.history:
            action = row.get("action", {}) if isinstance(row, dict) else {}
            if str(action.get("tool", "")).strip() == "expand_neighbors":
                qid = str((action.get("args", {}) or {}).get("qid", "")).strip()
                if qid:
                    out.add(qid)
        return out

    def _best_expand_anchor_qid(self) -> Optional[str]:
        generic = {"organelle", "component", "part", "structure", "system", "cellular component", "aircraft component"}
        expanded = self._expanded_anchor_qids()
        for qid in self._sorted_candidates_qids(set(self.accepted_qids)):
            if qid in expanded:
                continue
            row = self.candidates.get(qid)
            if row is None:
                continue
            name = (row.label or "").strip().lower()
            if name and name not in generic:
                return qid
        for qid in self._sorted_candidates_qids(set(self.accepted_qids)):
            if qid not in expanded:
                return qid
        return None

    def _active_frontier_qids(self) -> List[str]:
        bucketed = self._bucketed_qids()
        return [row.qid for row in self._sorted_candidates() if row.qid not in bucketed]

    def _has_successful_scan_route(self, route: str) -> bool:
        route = str(route).strip().lower()
        for row in self.history:
            action = row.get("action", {}) if isinstance(row, dict) else {}
            result = row.get("result", {}) if isinstance(row, dict) else {}
            if str(action.get("tool", "")).strip() != "scan" or not bool(result.get("ok", False)):
                continue
            args = action.get("args", {}) if isinstance(action.get("args", {}), dict) else {}
            if str(args.get("route", "")).strip().lower() == route:
                return True
        return False

    def _forced_pre_worker_action(self, step: int) -> Optional[Dict[str, Any]]:
        if self._seed_size() >= 20:
            return None
        route_count = self._route_action_count()
        if route_count == 0:
            return {"tool": "survey", "args": {"qid": self.target_qid, "k": min(max(6, self.limit), 8)}, "why": "forced early exploration"}
        if route_count == 1 and step <= 2:
            route = "rev" if not self._has_successful_scan_route("rev") else "dir"
            why = "forced reverse class scan" if route == "rev" else "forced target scan"
            return {"tool": "scan", "args": {"qid": self.target_qid, "route": route, "k": min(6, self.limit)}, "why": why}
        if not self._active_frontier_qids():
            if not self._has_successful_scan_route("rev"):
                return {"tool": "scan", "args": {"qid": self.target_qid, "route": "rev", "k": min(6, self.limit)}, "why": "forced reverse fallback"}
            anchor = self._best_expand_anchor_qid()
            if anchor:
                return {"tool": "expand_neighbors", "args": {"qid": anchor, "k": 6}, "why": "forced child exploration"}
        return None

    def run(self) -> Dict[str, Any]:
        self._resolve_target_if_needed()
        self._run_seed()

        for step in range(1, self.max_steps + 1):
            forced_action = self._forced_pre_worker_action(step)
            if forced_action is not None:
                result = self._execute_action(forced_action)
                self.last_results = [result]
                continue

            payload, alias_to_qid, _ = self._build_agent_payload(step)
            agent_result = self._choose_actions_fast(payload)
            self.logger.info("step=%s decision=%s why=%s", step, agent_result.get("decision"), agent_result.get("why"))

            decision = agent_result.get("decision", "stop")
            if decision == "stop":
                break

            raw_actions = agent_result.get("actions", [])
            if not isinstance(raw_actions, list):
                raw_actions = []
            if decision == "fallback":
                memory_id = str(agent_result.get("memory_id", "")).strip()
                if not memory_id:
                    raise RuntimeError("worker returned fallback without memory_id")
                raw_actions = [{"tool": "memory", "args": {"memory_id": memory_id}, "why": agent_result.get("why", "")}] 

            action_results = []
            for raw_action in raw_actions:
                try:
                    action = self._translate_action(raw_action, alias_to_qid)
                except Exception as exc:
                    bad_tool = str(raw_action.get("tool", "invalid_action")).strip() if isinstance(raw_action, dict) else "invalid_action"
                    skipped = {
                        "tool": bad_tool,
                        "ok": False,
                        "brief": f"invalid action skipped: {exc}",
                        "data": {"raw": raw_action},
                    }
                    self.history.append({"action": {"tool": bad_tool, "args": {}}, "result": skipped})
                    action_results.append(skipped)
                    self.logger.info("RESULT %s", json.dumps(skipped, ensure_ascii=False, separators=(",", ":")))
                    continue

                signature = self._action_signature(action)
                if signature in self.executed_signatures and action.get("tool") != "memory":
                    skipped = {
                        "tool": action.get("tool", ""),
                        "ok": False,
                        "brief": "duplicate action skipped",
                        "data": {"signature": signature},
                    }
                    self.history.append({"action": action, "result": skipped})
                    action_results.append(skipped)
                    self.logger.info("RESULT %s", json.dumps(skipped, ensure_ascii=False, separators=(",", ":")))
                    continue

                result = self._execute_action(action)
                action_results.append(result)
                if action.get("tool") != "memory":
                    self.executed_signatures.add(signature)

            self.last_results = action_results[-3:]
            if not action_results:
                break

        final_review = self._run_final_review()
        return self._build_final_report(final_review)

    def _parse_dir_pair(self, value: Any) -> tuple[int, int]:
        text = str(value or "").strip()
        if "/" not in text:
            return 0, 0
        left, right = text.split("/", 1)
        try:
            return int(left), int(right)
        except Exception:
            return 0, 0

    def _common_sense_terms(self, text: Any) -> Set[str]:
        raw = str(text or "").lower()
        tokens = re.findall(r"[a-z0-9]+", raw)
        stop = {
            "the", "and", "for", "with", "from", "into", "that", "this", "these", "those", "their",
            "its", "our", "your", "are", "was", "were", "been", "being", "have", "has", "had",
            "not", "but", "use", "used", "using", "type", "kind", "form", "object", "thing",
            "part", "parts", "component", "components", "structure", "system", "item", "items",
        }
        return {token for token in tokens if len(token) >= 3 and token not in stop}

    def _review_common_sense_flags(
        self,
        *,
        name: str,
        desc: str,
        class_via: List[str],
        evidence_only: List[str],
        prompt_text: str,
        direct_support: bool,
    ) -> Dict[str, bool]:
        name_l = name.lower()
        desc_l = desc.lower()
        prompt_terms = self._common_sense_terms(prompt_text)
        candidate_terms = self._common_sense_terms(" ".join([name, desc] + list(class_via)))
        relation_overlap = bool(prompt_terms & candidate_terms)
        physical_nouns = {
            "assembly", "apparatus", "axle", "bearing", "blade", "body", "bogie", "bracket", "brake",
            "cabin", "car", "carriage", "chamber", "chassis", "coachwork", "compartment", "coupler",
            "cover", "device", "door", "engine", "filter", "frame", "gear", "handle", "hood",
            "housing", "instrument", "landing", "lever", "locomotive", "machine", "mechanism",
            "membrane", "mirror", "module", "motor", "panel", "pantograph", "pedal", "pipe",
            "piston", "rod", "rotor", "seat", "sensor", "shaft", "shell", "spring", "tank",
            "tube", "valve", "vehicle", "wagon", "wheel", "window", "wing",
        }
        physical_phrases = {
            "part of", "component of", "mounted on", "attached to", "located in", "located on",
            "forms part", "consists of", "made of", "visible", "outer", "inner", "surface",
            "pair of", "set of", "mechanical", "physical", "structural",
        }
        action_object_phrases = {
            "used to", "used for", "designed to", "serves to", "allows", "supports", "holds",
            "carries", "connects", "rotates", "opens", "closes", "protects", "moves", "steers",
            "drives", "brakes", "transmits",
        }
        abstract_phrases = {
            "process", "property", "quality", "function", "activity", "ability", "capacity", "metric",
            "rate", "ratio", "schedule", "time management", "role", "theory", "relation", "class of",
            "type of relation", "measure of", "field of study", "phenomenon",
        }
        generic_only_names = {"component", "part", "structure", "system", "element", "organelle", "assembly", "module"}
        looks_physical = (
            any(term in name_l or term in desc_l for term in physical_nouns)
            or any(phrase in desc_l for phrase in physical_phrases)
            or any(phrase in desc_l for phrase in action_object_phrases)
            or relation_overlap
        )
        return {
            "looks_physical": looks_physical,
            "relation_overlap": relation_overlap,
            "generic_name": name_l.strip() in generic_only_names,
            "abstract": any(phrase in desc_l for phrase in abstract_phrases),
            "evidence_only": bool(evidence_only) and not direct_support,
        }

    def _repair_agent_result(self, agent_result: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        if str(agent_result.get("decision", "")).strip().lower() != "act":
            return agent_result

        state = payload.get("state", {})
        if not isinstance(state, dict):
            return agent_result

        cand_rows = [row for row in state.get("cand", []) if isinstance(row, dict)]
        seed_rows = [row for row in state.get("seed", []) if isinstance(row, dict)]
        accepted_rows = [row for row in state.get("accepted", []) if isinstance(row, dict)]
        hold_rows = [row for row in state.get("hold", []) if isinstance(row, dict)]
        rejected_rows = [row for row in state.get("rejected", []) if isinstance(row, dict)]
        last_rows = [row for row in payload.get("last", []) if isinstance(row, dict)]

        accepted = {str(row.get("id", "")).strip() for row in accepted_rows if str(row.get("id", "")).strip()}
        held = {str(row.get("id", "")).strip() for row in hold_rows if str(row.get("id", "")).strip()}
        rejected = {str(row.get("id", "")).strip() for row in rejected_rows if str(row.get("id", "")).strip()}
        bucketed = accepted | held | rejected

        strong_unaccepted: List[str] = []
        for row in cand_rows:
            rid = str(row.get("id", "")).strip()
            if not rid or rid in bucketed:
                continue
            d1, d2 = self._parse_dir_pair(row.get("dir", "0/0"))
            if d1 + d2 > 0:
                strong_unaccepted.append(rid)

        seed_unbucketed: List[str] = []
        for row in seed_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid not in bucketed:
                seed_unbucketed.append(rid)
        seed_unbucketed = self._unique_keep_order(seed_unbucketed)

        actions = agent_result.get("actions", [])
        if not isinstance(actions, list):
            actions = []

        def extract_ids(action: Dict[str, Any]) -> List[str]:
            if not isinstance(action, dict):
                return []
            tool = str(action.get("tool", "")).strip()
            args = action.get("args", {})
            if not isinstance(args, dict):
                return []
            if tool in {"accept_many", "hold_many", "reject_many", "check_many", "probe_many"}:
                vals = args.get("candidates", [])
                out = []
                if isinstance(vals, list):
                    for v in vals:
                        if isinstance(v, dict):
                            v = v.get("id", "")
                        v = str(v).strip()
                        if v:
                            out.append(v)
                return out
            if tool in {"check", "open", "expand_neighbors"}:
                key = "candidate" if tool == "check" else "node" if tool == "open" else "anchor"
                v = str(args.get(key, "")).strip()
                return [v] if v else []
            return []

        def repeats_bucketed_only() -> bool:
            if not actions:
                return False
            ok = False
            for action in actions:
                tool = str(action.get("tool", "")).strip()
                if tool not in {"accept_many", "hold_many", "reject_many", "check", "check_many", "probe_many"}:
                    return False
                ids = extract_ids(action)
                if not ids or any(rid not in bucketed for rid in ids):
                    return False
                ok = True
            return ok

        route_recent = any(str(row.get("tool", "")).strip() in {"survey", "scan", "expand_neighbors"} and bool(row.get("ok", False)) for row in last_rows)
        small_seed = len(seed_rows) < 20

        if small_seed and not route_recent:
            return {
                "decision": "act",
                "why": "explore early frontier",
                "actions": [{"tool": "survey", "args": {"anchor": "T", "k": 8}, "why": "explore more parts"}],
                "memory_id": "",
            }

        if strong_unaccepted:
            if len(strong_unaccepted) == 1:
                return {"decision": "act", "why": "confirm direct candidate", "actions": [{"tool": "check", "args": {"candidate": strong_unaccepted[0]}, "why": "confirm support"}], "memory_id": ""}
            return {"decision": "act", "why": "confirm direct candidates", "actions": [{"tool": "check_many", "args": {"candidates": strong_unaccepted[:4]}, "why": "confirm support"}], "memory_id": ""}

        if seed_unbucketed:
            return {"decision": "act", "why": "probe unresolved seed", "actions": [{"tool": "probe_many", "args": {"candidates": seed_unbucketed[:4]}, "why": "resolve seed nodes"}], "memory_id": ""}

        if repeats_bucketed_only():
            anchor_qid = self._best_expand_anchor_qid()
            if anchor_qid:
                anchor_alias = anchor_qid
                for alias, qid in self._build_visible_alias_maps()[0].items():
                    if qid == anchor_qid:
                        anchor_alias = alias
                        break
                return {"decision": "act", "why": "explore accepted child", "actions": [{"tool": "expand_neighbors", "args": {"anchor": anchor_alias, "k": 6}, "why": "look at children"}], "memory_id": ""}
            if small_seed:
                return {"decision": "act", "why": "keep exploring target", "actions": [{"tool": "scan", "args": {"anchor": "T", "route": "dir", "k": 4}, "why": "search more parts"}], "memory_id": ""}
            return {"decision": "stop", "why": "frontier exhausted", "actions": [], "memory_id": ""}

        return agent_result

    def _resolve_target_if_needed(self) -> None:
        if self.target_qid:
            profile = self.wikidata.open_node(self.target_qid)
            self.target_label = profile.get("label", self.target_qid)
            self.target_description = profile.get("description", "")
            self.target_resolution = {
                "picked": {
                    "qid": self.target_qid,
                    "label": self.target_label,
                    "description": self.target_description,
                    "why": "explicit qid provided",
                },
                "alts": [],
                "candidates": [],
            }
            self.opened_nodes[self.target_qid] = profile
            return

        resolution = self._resolve_entity_with_fallbacks(self.target_text, limit=10)
        candidates = resolution.get("candidates", [])
        if not candidates:
            raise RuntimeError(f"no Wikidata entity found for: {self.target_text}")

        compact_rows = []
        alias_to_row = {}
        for idx, row in enumerate(candidates[:10], start=1):
            alias = f"K{idx}"
            compact = self._compact_resolution_candidate(alias, row)
            compact_rows.append(compact)
            alias_to_row[alias] = row

        payload = {"proto": "graph-part-mcp-v2", "target_text": self.target_text, "candidates": compact_rows}
        choice = self._choose_target_fast(payload)
        picked_alias = str(choice.get("id", "")).strip()
        if picked_alias not in alias_to_row:
            raise RuntimeError("worker returned unknown target candidate id")
        selected = alias_to_row[picked_alias]

        self.target_qid = selected["id"]
        self.target_label = selected.get("label", self.target_qid)
        self.target_description = selected.get("description", "")
        self.target_resolution = {
            "picked": {
                "qid": selected["id"],
                "label": selected.get("label", ""),
                "description": selected.get("description", ""),
                "sitelinks": selected.get("sitelinks", 0),
                "why": choice.get("why", ""),
            },
            "alts": choice.get("alts", []),
            "candidates": compact_rows,
            "lookup_query": resolution.get("query", self.target_text),
            "lookup_strategy": resolution.get("strategy", "direct"),
            "tried_queries": resolution.get("tried_queries", [self.target_text]),
        }
        self.logger.info("resolved target=%s label=%s", self.target_qid, self.target_label)
        self.history.append({"action": {"tool": "resolve", "args": {"text": self.target_text}}, "result": {"brief": f"resolved {self.target_qid}", "data": self.target_resolution}})

    def _run_seed(self) -> None:
        result = self._execute_action({"tool": "seed", "args": {"qid": self.target_qid, "k": self.limit}, "why": "initial seed"})
        self.last_results = [result]

    def _choose_target_fast(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = [row for row in payload.get("candidates", []) if isinstance(row, dict)]
        if self.fast_stage1 and rows:
            target_text = str(payload.get("target_text", "")).strip().lower()

            def score(row: Dict[str, Any]) -> Tuple[int, int, int, str]:
                rid = str(row.get("id", "")).strip()
                name = str(row.get("name", "") or "").strip().lower()
                desc = str(row.get("d", row.get("description", "")) or "").strip().lower()
                inst = [str(item).strip().lower() for item in (row.get("i", row.get("inst", [])) or []) if str(item).strip()]
                sitelinks = int(row.get("sl", row.get("sitelinks", 0)) or 0)

                bad_desc_terms = {
                    "scientific article", "scholarly article", "scientific journal", "journal",
                    "patent", "book", "film", "song", "album", "episode", "paper",
                }
                good_desc_terms = {
                    "type of cell", "cell type", "cell", "organelle", "part of", "component",
                    "railway vehicle", "vehicle", "anatomical structure",
                }
                bad_inst_terms = {
                    "scholarly article", "scientific journal", "united states patent", "patent", "journal",
                }

                bad = 1 if any(term in desc for term in bad_desc_terms) or any(term in item for term in bad_inst_terms for item in inst) else 0
                good = 1 if any(term in desc for term in good_desc_terms) else 0
                exact = 1 if target_text and name == target_text else 0
                return (bad, -good, -exact, -sitelinks, rid)

            ranked = sorted(rows, key=score)
            if ranked:
                picked = str(ranked[0].get("id", "")).strip()
                if picked:
                    return {"id": picked, "why": "fast local target ranking", "alts": []}

        return self.worker.infer(task="choose_target", mode=self.mode, payload=payload)

    def _choose_actions_fast(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.fast_stage1:
            local = self._deterministic_choose_actions(payload)
            if local is not None:
                return local
        agent_result = self.worker.infer(task="choose_actions", mode=self.mode, payload=payload)
        return self._repair_agent_result(agent_result, payload)

    def _deterministic_choose_actions(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        state = payload.get("state", {}) or {}
        if not isinstance(state, dict):
            return None
        cand_rows = [row for row in state.get("cand", []) if isinstance(row, dict)]
        seed_rows = [row for row in state.get("seed", []) if isinstance(row, dict)]
        accepted = {str(row.get("id", "")).strip() for row in state.get("accepted", []) if isinstance(row, dict) and str(row.get("id", "")).strip()}
        held = {str(row.get("id", "")).strip() for row in state.get("hold", []) if isinstance(row, dict) and str(row.get("id", "")).strip()}
        rejected = {str(row.get("id", "")).strip() for row in state.get("rejected", []) if isinstance(row, dict) and str(row.get("id", "")).strip()}
        bucketed = accepted | held | rejected
        last_rows = [row for row in payload.get("last", []) if isinstance(row, dict)]
        step = int(payload.get("step", 0) or 0)
        total_seed = len(seed_rows)
        small_seed = total_seed < 20
        route_recent = any(str(row.get("tool", "")).strip() in {"survey", "scan", "expand_neighbors"} and bool(row.get("ok", False)) for row in last_rows)

        strong = []
        for row in cand_rows:
            rid = str(row.get("id", "")).strip()
            if not rid or rid in bucketed:
                continue
            d1, d2 = self._parse_dir_pair(row.get("dir", "0/0"))
            if d1 + d2 > 0:
                strong.append(rid)
        if strong:
            if len(strong) == 1:
                return {"decision": "act", "why": "fast local confirm direct candidate", "actions": [{"tool": "check", "args": {"candidate": strong[0]}, "why": "confirm support"}], "memory_id": ""}
            return {"decision": "act", "why": "fast local confirm direct candidates", "actions": [{"tool": "check_many", "args": {"candidates": strong[:4]}, "why": "confirm support"}], "memory_id": ""}

        unresolved_seed = []
        for row in seed_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid not in bucketed:
                unresolved_seed.append(rid)
        unresolved_seed = self._unique_keep_order(unresolved_seed)

        if small_seed and not route_recent and step <= 2:
            return {"decision": "act", "why": "fast local early exploration", "actions": [{"tool": "survey", "args": {"anchor": "T", "k": 8}, "why": "explore more parts"}], "memory_id": ""}
        if unresolved_seed:
            return {"decision": "act", "why": "fast local probe unresolved seed", "actions": [{"tool": "probe_many", "args": {"candidates": unresolved_seed[:4]}, "why": "resolve seed nodes"}], "memory_id": ""}
        if small_seed:
            accepted_names = {str(row.get("id", "")).strip(): str(row.get("name", "")).strip().lower() for row in cand_rows if isinstance(row, dict)}
            generic = {"organelle", "component", "part", "structure", "system", "cellular component", "aircraft component"}
            for rid in [str(row.get("id", "")).strip() for row in state.get("accepted", []) if isinstance(row, dict)]:
                if rid and accepted_names.get(rid, "") not in generic:
                    return {"decision": "act", "why": "fast local explore accepted child", "actions": [{"tool": "expand_neighbors", "args": {"anchor": rid, "k": 6}, "why": "look at children"}], "memory_id": ""}
            return {"decision": "act", "why": "fast local keep exploring target", "actions": [{"tool": "scan", "args": {"anchor": "T", "route": "dir", "k": 4}, "why": "search more parts"}], "memory_id": ""}
        return {"decision": "stop", "why": "fast local frontier exhausted", "actions": [], "memory_id": ""}

    def _deterministic_final_review(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {"accept": [], "reject": [], "hold": [], "notes": {}}
        rows = [row for row in self._sorted_candidates() if row.qid in self.candidates]
        concrete: List[Tuple[int, str, Set[str]]] = []
        early_reject: List[Tuple[int, str, str]] = []
        prompt_text = " ".join([self.target_text, self.target_label, self.target_description]).strip()
        generic_name_terms = {"component", "part", "structure", "element", "organelle", "system", "complex"}
        abstract_desc_terms = {
            "process", "property", "quality", "function", "activity", "force", "ratio", "efficiency",
            "capacity", "traffic flow", "schedule", "time management", "rate", "metric", "ability",
            "relation", "role", "theory", "field of study", "phenomenon", "class of", "type of relation",
            "measure of", "maximum traffic flow", "resisting the relative motion",
        }
        concrete_desc_terms = {
            "vehicle", "apparatus", "mechanism", "assembly", "framework", "wheel", "axle", "bearing",
            "device", "machine", "housing", "chamber", "membrane", "rod", "tube", "wagon", "car",
            "bogie", "pantograph", "coupler", "locomotive", "body", "frame", "organ", "organelle",
            "compartment", "module", "brake", "gear", "sensor", "filter", "tank", "panel",
            "window", "cover", "chassis", "shaft", "steering", "dashboard", "hood", "windshield",
            "carburetor", "coachwork",
        }

        for row in rows:
            rid = str(row.qid).strip()
            if not rid:
                continue
            name = str(row.label or "").strip()
            desc = str(row.description or "").strip()
            name_l = name.lower()
            desc_l = desc.lower()
            route_set = {str(item).strip() for item in row.routes if str(item).strip()}
            direct_support = bool(route_set & {"seed_p527", "scan_direct_p527", "scan_inverse_p361", "preview_p361_target", "open_structural_support", "check_supported"}) or ((row.direct_has_part_refs + row.direct_part_of_refs) > 0)
            seed_like = bool(route_set & {"seed_p527", "seed_p1552", "check_supported", "scan_direct_p527", "scan_inverse_p361", "preview_p361_target", "open_structural_support"})
            commonsense = self._review_common_sense_flags(
                name=name,
                desc=desc,
                class_via=row.class_via,
                evidence_only=row.evidence_only_reasons,
                prompt_text=prompt_text,
                direct_support=direct_support,
            )

            if not name and not desc:
                early_reject.append((-100, rid, "empty candidate"))
                continue

            score = 0
            tags: Set[str] = set()
            is_generic_name = name_l in generic_name_terms
            is_abstract = any(term in desc_l for term in abstract_desc_terms)
            looks_concrete = any(term in desc_l or term in name_l for term in concrete_desc_terms)
            if direct_support:
                score += 6
                tags.add("target_support")
            if seed_like:
                score += 3
                tags.add("seed_like")
            if looks_concrete:
                score += 3
                tags.add("concrete_shape")
            if commonsense["looks_physical"]:
                score += 4
                tags.add("physical")
            if commonsense["relation_overlap"]:
                score += 2
                tags.add("prompt_related")
            if len(name.split()) >= 2 and not is_generic_name:
                score += 1
            if len(desc) >= 40:
                score += 1
            if is_generic_name or commonsense["generic_name"]:
                score -= 3
                tags.add("generic")
            if is_abstract or commonsense["abstract"]:
                score -= 6
                tags.add("abstract")
            if commonsense["evidence_only"]:
                score -= 4
                tags.add("evidence_only")
            if "complex" in desc_l or "multisubunit complex" in desc_l:
                score -= 2
            if any(term.startswith("neighbor_") for term in route_set) and not (direct_support or seed_like):
                score -= 2
            if any(term in name_l for term in ["envelope", "granule"]) or any(term in desc_l for term in ["attachment point", "centromeric region"]):
                score -= 3

            concrete.append((score, rid, tags))

        accept_rows: List[Tuple[int, str, str]] = []
        reject_rows: List[Tuple[int, str, str]] = list(early_reject)

        for score, rid, tags in concrete:
            has_support = "target_support" in tags or "seed_like" in tags
            looks_physical = "physical" in tags or "concrete_shape" in tags
            prompt_related = "prompt_related" in tags
            if looks_physical and "abstract" not in tags and "generic" not in tags and "evidence_only" not in tags and (has_support or prompt_related):
                accept_rows.append((score, rid, "physical object related to target"))
            elif "abstract" in tags and not has_support:
                reject_rows.append((score, rid, "abstract not a physical object"))
            elif "generic" in tags and not has_support and not prompt_related:
                reject_rows.append((score, rid, "generic umbrella item"))
            elif "evidence_only" in tags and not (looks_physical and prompt_related):
                reject_rows.append((score, rid, "evidence-only node"))
            elif not looks_physical and not has_support and not prompt_related:
                reject_rows.append((score, rid, "no clear physical object tie"))
            elif looks_physical and "abstract" not in tags and "generic" not in tags:
                accept_rows.append((score, rid, "physical object plausibly related to target"))
            else:
                if "generic" in tags:
                    note_text = "too generic for final accept"
                elif "abstract" in tags:
                    note_text = "description stays abstract"
                elif "evidence_only" in tags:
                    note_text = "indirect evidence only"
                elif has_support:
                    note_text = "support exists but not a clean physical part"
                else:
                    note_text = "not enough target relation"
                reject_rows.append((score, rid, note_text))

        accept_rows.sort(key=lambda item: (-item[0], item[1]))
        reject_rows.sort(key=lambda item: (-item[0], item[1]))

        result["accept"] = self._unique_keep_order([rid for _, rid, _ in accept_rows])
        result["hold"] = []
        result["reject"] = self._unique_keep_order([rid for _, rid, _ in reject_rows if rid not in result["accept"]])

        for _, rid, note in accept_rows:
            result["notes"][rid] = note
        for _, rid, note in reject_rows:
            if rid not in result["accept"]:
                result["notes"].setdefault(rid, note)

        return result

    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        tool = action.get("tool")
        args = action.get("args", {}) or {}
        self.logger.info("ACTION %s %s", tool, json.dumps(args, ensure_ascii=False, separators=(",", ":")))

        if tool == "seed":
            qid = str(args.get("qid", self.target_qid))
            data = self.wikidata.seed_fields(qid)
            self.subject_seed = data
            self.target_label = data["subject"].get("label", self.target_label)
            self.target_description = data["subject"].get("description", self.target_description)
            self._ingest_seed(data)
            result = {"tool": "seed", "ok": True, "brief": self._brief_seed(data), "data": self._compact_seed(data)}
        elif tool == "open":
            qid = str(args.get("qid", ""))
            data = self.wikidata.open_node(qid)
            self.opened_nodes[qid] = data
            self._ingest_open(data)
            result = {"tool": "open", "ok": True, "brief": self._brief_open(data), "data": self._compact_open(data)}
        elif tool == "open_many":
            qids = args.get("qids", [])
            data = self.wikidata.open_many(qids if isinstance(qids, list) else [])
            for profile in data:
                self.opened_nodes[profile["id"]] = profile
                self._ingest_open(profile)
            result = {"tool": "open_many", "ok": True, "brief": f"open_many count={len(data)}", "data": {"rows": [self._compact_open(row) for row in data]}}
        elif tool == "survey":
            qid = str(args.get("qid", self.target_qid))
            k = min(max(3, int(args.get("k", self.limit) or self.limit)), 10)
            data = self.wikidata.survey(qid=qid, limit=k)
            self._ingest_survey(qid, data)
            result = {"tool": "survey", "ok": True, "brief": f"survey on {qid} rows={len(data)}", "data": self._compact_survey(qid, data)}
        elif tool == "scan":
            qid = str(args.get("qid", self.target_qid))
            route = str(args.get("route", "dir"))
            k = min(max(1, int(args.get("k", 3))), 6)
            data = self.wikidata.scan_route(qid=qid, route=route, limit=k)
            self._ingest_scan(qid, route, data)
            result = {"tool": "scan", "ok": True, "brief": self._brief_scan(qid, route, data), "data": self._compact_scan(qid, route, data)}
        elif tool == "check":
            subject = str(args.get("subject", self.target_qid))
            candidate = str(args.get("candidate", ""))
            data = self.wikidata.check_relation(subject, candidate)
            self._ingest_check(candidate, data)
            result = {"tool": "check", "ok": True, "brief": self._brief_check(data), "data": data}
        elif tool == "check_many":
            subject = str(args.get("subject", self.target_qid))
            candidates = args.get("candidates", [])
            data = self.wikidata.check_many(subject, candidates if isinstance(candidates, list) else [])
            for row in data:
                self._ingest_check(row.get("candidate", ""), row)
            result = {"tool": "check_many", "ok": True, "brief": f"check_many count={len(data)}", "data": {"rows": data}}
        elif tool == "probe_many":
            candidates = args.get("candidates", [])
            data = self.wikidata.probe_many(self.target_qid, candidates if isinstance(candidates, list) else [])
            for row in data:
                if row.get("open"):
                    self.opened_nodes[row["open"]["id"]] = row["open"]
                    self._ingest_open(row["open"])
                if row.get("check"):
                    self._ingest_check(row["candidate"], row["check"])
            result = {"tool": "probe_many", "ok": True, "brief": f"probe_many count={len(data)}", "data": {"rows": [self._compact_probe_row(row) for row in data]}}
        elif tool == "accept_many":
            candidates = args.get("candidates", [])
            if isinstance(candidates, str):
                candidates = [item.strip() for item in candidates.split(",") if item.strip()]
            qids = [qid for qid in candidates if qid in self.candidates]
            self._mark_bucket("accept", qids)
            result = {"tool": "accept_many", "ok": True, "brief": f"accepted count={len(qids)}", "data": {"nodes": qids}}
        elif tool == "hold_many":
            candidates = args.get("candidates", [])
            if isinstance(candidates, str):
                candidates = [item.strip() for item in candidates.split(",") if item.strip()]
            qids = [qid for qid in candidates if qid in self.candidates]
            self._mark_bucket("hold", qids)
            result = {"tool": "hold_many", "ok": True, "brief": f"held count={len(qids)}", "data": {"nodes": qids}}
        elif tool == "reject_many":
            candidates = args.get("candidates", [])
            if isinstance(candidates, str):
                candidates = [item.strip() for item in candidates.split(",") if item.strip()]
            qids = [qid for qid in candidates if qid in self.candidates]
            self._mark_bucket("reject", qids)
            result = {"tool": "reject_many", "ok": True, "brief": f"rejected count={len(qids)}", "data": {"nodes": qids}}
        elif tool == "save_memory":
            route = args.get("route", {})
            label = str(args.get("label", "")).strip()
            if isinstance(route, dict) and route.get("tool"):
                self._add_memory(route, priority=7)
                result = {"tool": "save_memory", "ok": True, "brief": f"saved memory {label or route.get('tool', '')}", "data": {"label": label}}
            else:
                result = {"tool": "save_memory", "ok": False, "brief": "invalid route", "data": {}}
        elif tool == "expand_neighbors":
            qid = str(args.get("qid", ""))
            k = min(max(1, int(args.get("k", 6) or 6)), 8)
            data = self.wikidata.expand_neighbors(qid, limit=k)
            self._ingest_neighbor_expand(qid, data)
            result = {
                "tool": "expand_neighbors",
                "ok": True,
                "brief": f"expand_neighbors on {qid} rows={len(data)}",
                "data": {
                    "anchor": qid,
                    "rows": [{"cand": row.get("cand", ""), "label": row.get("candLabel", ""), "edge": row.get("edge", ""), "dir": row.get("dir", "")} for row in data[:8]],
                },
            }
        elif tool == "memory":
            mid = str(args.get("mid", ""))
            entry = self._get_memory(mid)
            if entry is None:
                raise RuntimeError(f"memory route not found: {mid}")
            entry.tries += 1
            forwarded = self._execute_action(entry.action)
            result = {"tool": "memory", "ok": True, "brief": f"fallback {mid} -> {forwarded.get('brief', '')}", "data": {"mid": mid, "forward": forwarded.get("data", {})}}
        else:
            raise ValueError(f"unsupported tool: {tool}")

        self.history.append({"action": action, "result": result})
        self.logger.info("RESULT %s", json.dumps(result, ensure_ascii=False, separators=(",", ":")))
        return result

    def _ingest_neighbor_expand(self, source_qid: str, rows: List[Dict[str, Any]]) -> None:
        focus: List[str] = []
        for item in rows:
            cand = item.get("cand", "")
            if not cand or cand == self.target_qid:
                continue
            merged = {
                "id": cand,
                "label": item.get("candLabel", ""),
                "description": item.get("candDescription", ""),
                "url": item.get("candUrl", self.wikidata.page_url(cand)),
                "sitelinks": int(item.get("candSitelinks", 0) or 0),
            }
            row = self._candidate(cand, merged)

            edge = str(item.get("edge", ""))
            direction = str(item.get("dir", ""))
            row.routes.add(f"neighbor_{edge}_{direction}")

            if edge == "P361" and direction == "in":
                row.direct_part_of_refs += 1
                focus.append(cand)
            elif edge == "P527" and direction == "out":
                row.direct_has_part_refs += 1
                focus.append(cand)
            else:
                row.notes.append(f"neighbor via {edge} {direction}")

        if focus:
            self._add_memory({"tool": "probe_many", "args": {"candidates": focus[:4]}, "why": "probe expanded neighbors"}, priority=10)

    def _mark_bucket(self, bucket: str, qids: List[str]) -> None:
        for qid in qids:
            if not qid or qid == self.target_qid:
                continue
            if bucket == "accept":
                self.accepted_qids.add(qid)
                self.held_qids.discard(qid)
                self.rejected_qids.discard(qid)
            elif bucket == "hold":
                if qid not in self.accepted_qids:
                    self.held_qids.add(qid)
                    self.rejected_qids.discard(qid)
            elif bucket == "reject":
                if qid not in self.accepted_qids:
                    self.rejected_qids.add(qid)
                    self.held_qids.discard(qid)

    def _build_agent_payload(self, step: int) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, str]]:
        alias_to_qid, qid_to_alias = self._build_visible_alias_maps()
        bucketed = self._bucketed_qids()

        seed_rows = []
        for row in self.subject_seed.get("parts", [])[:6]:
            qid = row.get("id", "")
            if qid in bucketed:
                continue
            alias = qid_to_alias.get(qid)
            if alias:
                seed_rows.append({"id": alias, "kind": "part", "name": self._short(row.get("label", ""), 24), "d": self._short(row.get("description", ""), 36)})
        for row in self.subject_seed.get("characteristics", [])[:6]:
            qid = row.get("id", "")
            if qid in bucketed:
                continue
            alias = qid_to_alias.get(qid)
            if alias:
                seed_rows.append({"id": alias, "kind": "char", "name": self._short(row.get("label", ""), 24), "d": self._short(row.get("description", ""), 36)})

        cand_rows = []
        for candidate in self._sorted_candidates():
            if candidate.qid in bucketed:
                continue
            alias = qid_to_alias.get(candidate.qid)
            if not alias:
                continue
            cand_rows.append({
                "id": alias,
                "name": self._short(candidate.label, 24),
                "d": self._short(candidate.description, 36),
                "dir": f"{candidate.direct_has_part_refs}/{candidate.direct_part_of_refs}",
                "rt": sorted(candidate.routes)[:3],
                "cls": candidate.class_via[:1],
                "eo": candidate.evidence_only_reasons[:1],
            })
            if len(cand_rows) >= 8:
                break

        opened_rows = []
        for qid, data in list(self.opened_nodes.items()):
            if qid in bucketed:
                continue
            alias = qid_to_alias.get(qid)
            if not alias:
                continue
            opened_rows.append({
                "id": alias,
                "name": self._short(data.get("label", ""), 24),
                "p361": [self._alias_or_label(item.get("id", ""), qid_to_alias, item.get("label", item.get("id", ""))) for item in data.get("P361", [])[:2]],
                "p527": [self._alias_or_label(item.get("id", ""), qid_to_alias, item.get("label", item.get("id", ""))) for item in data.get("P527", [])[:2]],
            })
            if len(opened_rows) >= 2:
                break

        memory_rows = []
        for entry in self._ranked_memory()[:3]:
            memory_rows.append({"id": entry.mid, "go": self._memory_synopsis(entry.action, qid_to_alias), "p": entry.priority})

        last_rows = [self._compact_result_for_model(item, qid_to_alias) for item in self.last_results[-1:]]

        payload = {
            "proto": "gpm2",
            "step": step,
            "target": {"id": "T", "name": self._short(self.target_label, 28), "d": self._short(self.target_description, 42)},
            "tools": ["seed", "open", "open_many", "scan", "survey", "check", "check_many", "probe_many", "accept_many", "hold_many", "reject_many", "save_memory", "expand_neighbors", "memory"],
            "state": {
                "seed": seed_rows,
                "cand": cand_rows,
                "opened": opened_rows,
                "accepted": [{"id": qid_to_alias[qid]} for qid in list(self.accepted_qids)[:16] if qid in qid_to_alias],
                "hold": [{"id": qid_to_alias[qid]} for qid in list(self.held_qids)[:16] if qid in qid_to_alias],
                "rejected": [{"id": qid_to_alias[qid]} for qid in list(self.rejected_qids)[:16] if qid in qid_to_alias],
            },
            "memory": memory_rows,
            "last": last_rows,
        }
        return payload, alias_to_qid, qid_to_alias

    def _build_visible_alias_maps(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        qids: List[str] = []
        seen = set()

        def add(qid: str) -> None:
            qid = str(qid).strip()
            if qid and qid not in seen and qid != self.target_qid:
                seen.add(qid)
                qids.append(qid)

        for row in self.subject_seed.get("parts", [])[:6]:
            add(row.get("id", ""))
        for row in self.subject_seed.get("characteristics", [])[:8]:
            add(row.get("id", ""))
        for row in self._sorted_candidates()[:12]:
            add(row.qid)
        for qid in list(self.opened_nodes.keys())[:6]:
            add(qid)
        for entry in self._ranked_memory()[:10]:
            for qid in self._extract_qids_from_action(entry.action):
                add(qid)

        alias_to_qid = {}
        qid_to_alias = {}
        for idx, qid in enumerate(qids, start=1):
            alias = f"N{idx}"
            alias_to_qid[alias] = qid
            qid_to_alias[qid] = alias
        return alias_to_qid, qid_to_alias

    def _translate_action(self, raw_action: Dict[str, Any], alias_to_qid: Dict[str, str]) -> Dict[str, Any]:
        if not isinstance(raw_action, dict):
            raise ValueError("worker action must be an object")

        tool = str(raw_action.get("tool", "")).strip()
        args = raw_action.get("args", {}) or {}
        why = str(raw_action.get("why", "")).strip()
        if not isinstance(args, dict):
            raise ValueError("worker action args must be an object")

        def node_to_qid(alias: Any, allow_target: bool) -> str:
            alias = str(alias).strip()
            if not alias:
                raise ValueError(f"tool {tool} received empty node alias")
            if alias == "T":
                if not allow_target:
                    raise ValueError(f"tool {tool} does not allow target alias T here")
                return self.target_qid
            if alias not in alias_to_qid:
                raise ValueError(f"worker action references unknown alias: {alias}")
            return alias_to_qid[alias]

        def split_aliasish_values(values: Any) -> List[Any]:
            if isinstance(values, list):
                return values
            if isinstance(values, str):
                text = values
                for ch in "[]{}()":
                    text = text.replace(ch, " ")
                text = text.replace("|", ",")
                parts = [part.strip().strip("'").strip('"') for part in text.split(",")]
                return [part for part in parts if part]
            return []

        def nodes_to_qids(values: Any, allow_target: bool, min_len: int, max_len: int) -> List[str]:
            raw_items = split_aliasish_values(values)
            qids: List[str] = []
            seen = set()
            for value in raw_items:
                if isinstance(value, dict):
                    value = value.get("id", value.get("node", value.get("candidate", "")))
                try:
                    qid = node_to_qid(value, allow_target=allow_target)
                except ValueError:
                    continue
                if qid not in seen:
                    qids.append(qid)
                    seen.add(qid)
            if len(qids) < min_len or len(qids) > max_len:
                raise ValueError(f"tool {tool} received invalid list length")
            return qids

        def translate_nested_route(route_obj: Any) -> Dict[str, Any]:
            if not isinstance(route_obj, dict):
                raise ValueError("save_memory route must be an object")
            nested_tool = str(route_obj.get("tool", "")).strip()
            if nested_tool in {"save_memory", "memory"}:
                raise ValueError("save_memory route cannot nest save_memory or memory")
            return self._translate_action(route_obj, alias_to_qid)

        if tool == "memory":
            memory_id = str(args.get("memory_id", args.get("mid", ""))).strip()
            if not memory_id:
                raise ValueError("memory action missing memory_id")
            return {"tool": "memory", "args": {"mid": memory_id}, "why": why or "memory"}

        if tool == "seed":
            if str(args.get("target", "")).strip() != "T":
                raise ValueError("seed action must target T")
            return {"tool": "seed", "args": {"qid": self.target_qid, "k": self.limit}, "why": why or "seed"}

        if tool == "open":
            qid = node_to_qid(args.get("node", ""), allow_target=True)
            return {"tool": "open", "args": {"qid": qid}, "why": why or "open"}

        if tool == "open_many":
            qids = nodes_to_qids(args.get("nodes", []), allow_target=False, min_len=1, max_len=4)
            return {"tool": "open_many", "args": {"qids": qids}, "why": why or "open_many"}

        if tool == "scan":
            qid = node_to_qid(args.get("anchor", ""), allow_target=True)
            route = str(args.get("route", "")).strip().lower()
            if route not in SCAN_ROUTES:
                raise ValueError("scan route must be dir, rev, cls, char, or up")
            try:
                k = int(args.get("k", 4) or 4)
            except Exception:
                k = 4
            k = min(max(1, k), 6)
            return {"tool": "scan", "args": {"qid": qid, "route": route, "k": k}, "why": why or "scan"}

        if tool == "survey":
            qid = node_to_qid(args.get("anchor", ""), allow_target=True)
            try:
                k = int(args.get("k", 6) or 6)
            except Exception:
                k = 6
            k = min(max(3, k), 8)
            return {"tool": "survey", "args": {"qid": qid, "k": k}, "why": why or "survey"}

        if tool == "check":
            qid = node_to_qid(args.get("candidate", ""), allow_target=False)
            return {"tool": "check", "args": {"subject": self.target_qid, "candidate": qid}, "why": why or "check"}

        if tool == "check_many":
            qids = nodes_to_qids(args.get("candidates", []), allow_target=False, min_len=1, max_len=4)
            return {"tool": "check_many", "args": {"subject": self.target_qid, "candidates": qids}, "why": why or "check_many"}

        if tool == "probe_many":
            qids = nodes_to_qids(args.get("candidates", []), allow_target=False, min_len=1, max_len=4)
            return {"tool": "probe_many", "args": {"candidates": qids}, "why": why or "probe_many"}

        if tool == "accept_many":
            qids = nodes_to_qids(args.get("candidates", []), allow_target=False, min_len=1, max_len=8)
            return {"tool": "accept_many", "args": {"candidates": qids}, "why": why or "accept_many"}

        if tool == "hold_many":
            qids = nodes_to_qids(args.get("candidates", []), allow_target=False, min_len=1, max_len=8)
            return {"tool": "hold_many", "args": {"candidates": qids}, "why": why or "hold_many"}

        if tool == "reject_many":
            qids = nodes_to_qids(args.get("candidates", []), allow_target=False, min_len=1, max_len=8)
            return {"tool": "reject_many", "args": {"candidates": qids}, "why": why or "reject_many"}

        if tool == "save_memory":
            route = translate_nested_route(args.get("route", {}))
            label = str(args.get("label", "")).strip()[:80]
            return {"tool": "save_memory", "args": {"route": route, "label": label}, "why": why or "save_memory"}

        if tool == "expand_neighbors":
            qid = node_to_qid(args.get("anchor", ""), allow_target=False)
            try:
                k = int(args.get("k", 6) or 6)
            except Exception:
                k = 6
            k = min(max(1, k), 8)
            return {"tool": "expand_neighbors", "args": {"qid": qid, "k": k}, "why": why or "expand_neighbors"}

        raise ValueError(f"unsupported worker tool: {tool}")

    def _run_final_review(self) -> Dict[str, Any]:
        if self.fast_stage1:
            return self._deterministic_final_review()
        alias_to_qid: Dict[str, str] = {}
        compact_candidates: List[Dict[str, Any]] = []
        visible_qids = self._unique_keep_order(
            [row.qid for row in self._sorted_candidates()] +
            list(self.accepted_qids) +
            list(self.held_qids) +
            list(self.rejected_qids)
        )
        for idx, qid in enumerate(visible_qids[:64], start=1):
            candidate = self.candidates.get(qid)
            if candidate is None:
                continue
            alias = f"R{idx}"
            alias_to_qid[alias] = qid
            compact_candidates.append({
                "id": alias,
                "name": candidate.label,
                "d": candidate.description,
                "dir": f"{candidate.direct_has_part_refs}/{candidate.direct_part_of_refs}",
                "cls": candidate.class_via[:2],
                "eo": candidate.evidence_only_reasons[:2],
                "rt": sorted(candidate.routes)[:6],
            })
        if not compact_candidates:
            return {"accept": [], "reject": [], "hold": [], "notes": {}}
        payload = {
            "proto": "graph-part-mcp-v2",
            "target_prompt": self.target_text,
            "target": {"id": "T", "name": self.target_label, "d": self.target_description},
            "candidates": compact_candidates,
        }
        review = self.worker.infer(task="final_review", mode=self.mode, payload=payload)
        mapped = {"accept": [], "reject": [], "hold": [], "notes": {}}
        for key in ("accept", "reject", "hold"):
            for alias in review.get(key, []):
                qid = alias_to_qid.get(str(alias).strip(), "")
                if not qid:
                    raise RuntimeError("worker returned unknown final review candidate id")
                mapped[key].append(qid)
        notes = review.get("notes", {})
        if isinstance(notes, dict):
            for alias, reason in notes.items():
                qid = alias_to_qid.get(str(alias).strip(), "")
                if not qid:
                    continue
                mapped["notes"][qid] = str(reason)
        return mapped

    def _build_final_report(self, review: Dict[str, Any]) -> Dict[str, Any]:
        accepted_set = set(review.get("accept", []))
        rejected_set = set(review.get("reject", []))
        rejected_set -= accepted_set

        accepted = [self._candidate_to_output(qid, review.get("notes", {}).get(qid, "accepted in final rearrange")) for qid in self._sorted_candidates_qids(accepted_set) if qid in self.candidates]
        rejected = [self._candidate_to_output(qid, review.get("notes", {}).get(qid, "rejected in final rearrange")) for qid in self._sorted_candidates_qids(rejected_set) if qid in self.candidates]

        return {
            "input_prompt": self.target_text,
            "target": {"qid": self.target_qid, "label": self.target_label, "description": self.target_description, "url": self.wikidata.page_url(self.target_qid)},
            "resolution": self.target_resolution,
            "accepted_components": accepted,
            "rejected_candidates": rejected,
            "memory": [self._compact_memory(entry, {}) for entry in self._ranked_memory()],
            "history": [self._compact_history(row) for row in self.history[-20:]],
        }

    def resolve_requested_components(self, requested_components: Sequence[Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for raw in requested_components or []:
            phrase = str(raw or "").strip()
            if not phrase:
                continue
            phrase_key = phrase.casefold()
            if phrase_key in seen:
                continue
            seen.add(phrase_key)

            resolved_row: Optional[Dict[str, Any]] = None
            resolution = self._resolve_entity_with_fallbacks(phrase, limit=8)
            candidates = resolution.get("candidates", [])

            if candidates:
                compact_rows = []
                alias_to_row = {}
                for idx, row in enumerate(candidates[:8], start=1):
                    alias = f"K{idx}"
                    compact = self._compact_resolution_candidate(alias, row)
                    compact_rows.append(compact)
                    alias_to_row[alias] = row
                payload = {"proto": "graph-part-mcp-v2", "target_text": phrase, "candidates": compact_rows}
                try:
                    choice = self._choose_target_fast(payload)
                    picked_alias = str(choice.get("id", "")).strip()
                    picked_row = alias_to_row.get(picked_alias)
                except Exception:
                    picked_row = candidates[0]
                if isinstance(picked_row, dict):
                    qid = str(picked_row.get("id", "") or "").strip()
                    resolved_row = {
                        "qid": qid,
                        "label": str(picked_row.get("label", "") or phrase).strip() or phrase,
                        "description": str(picked_row.get("description", "") or "").strip(),
                        "routes": ["diagram_requested_component"],
                        "direct_p527_refs": 0,
                        "direct_p361_refs": 0,
                        "class_via": [],
                        "sitelinks": int(picked_row.get("sitelinks", 0) or 0),
                        "url": str(picked_row.get("url", "") or self.wikidata.page_url(qid)).strip() if qid else "",
                        "note": (
                            f"requested component from diagram prompt: {phrase}"
                            if resolution.get("strategy") == "direct"
                            else f"requested component from diagram prompt: {phrase} (resolved via {resolution.get('strategy')} query {resolution.get('query', phrase)!r})"
                        ),
                    }

            if resolved_row is None:
                attempted = resolution.get("tried_queries", [])
                tried_note = f"; tried={attempted}" if attempted else ""
                resolved_row = {
                    "qid": "",
                    "label": phrase,
                    "description": "",
                    "routes": ["diagram_requested_component_unresolved"],
                    "direct_p527_refs": 0,
                    "direct_p361_refs": 0,
                    "class_via": [],
                    "sitelinks": 0,
                    "url": "",
                    "note": f"requested component from diagram prompt (unresolved): {phrase}{tried_note}",
                }

            out.append(resolved_row)
        return out

    def merge_requested_components_into_report(
        self,
        report: Dict[str, Any],
        requested_components: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        accepted = [row for row in (report.get("accepted_components") or []) if isinstance(row, dict)]
        merged: List[Dict[str, Any]] = []
        seen_keys = set()

        def _merge_one(row: Dict[str, Any]) -> None:
            qid = str(row.get("qid", "") or "").strip()
            label = str(row.get("label", "") or "").strip()
            key = (qid or "", label.casefold())
            if not qid and not label:
                return
            if key in seen_keys:
                return
            seen_keys.add(key)
            merged.append(row)

        for row in accepted:
            _merge_one(dict(row))
        for row in requested_components or []:
            if not isinstance(row, dict):
                continue
            _merge_one(dict(row))

        report["accepted_components"] = merged
        report["requested_components_resolved"] = [dict(row) for row in requested_components or [] if isinstance(row, dict)]
        return report

    def _candidate_to_output(self, qid: str, note: str) -> Dict[str, Any]:
        row = self.candidates[qid]
        self._clean_candidate(row)
        return {
            "qid": row.qid,
            "label": row.label,
            "description": row.description,
            "routes": sorted(row.routes),
            "direct_p527_refs": row.direct_has_part_refs,
            "direct_p361_refs": row.direct_part_of_refs,
            "class_via": row.class_via[:4],
            "sitelinks": row.sitelinks,
            "url": row.urls[0] if row.urls else self.wikidata.page_url(row.qid),
            "note": note,
        }

    def _sorted_candidates(self) -> List[CandidateRecord]:
        for row in self.candidates.values():
            self._clean_candidate(row)

        generic_labels = {"organelle", "component", "part", "structure", "system", "element"}

        def key_fn(row: CandidateRecord) -> Tuple[int, int, int, int, int, str]:
            direct_score = row.direct_has_part_refs + row.direct_part_of_refs
            confirmed_score = 1 if "check_supported" in row.routes else 0
            seed_score = 1 if "seed_p527" in row.routes else 0
            class_score = len(row.class_routes) + len(row.class_via)
            generic_penalty = 1 if (row.label or "").strip().lower() in generic_labels else 0
            return (-confirmed_score, -direct_score, generic_penalty, -seed_score, -class_score, row.label or row.qid)

        return sorted(self.candidates.values(), key=key_fn)

    def _sorted_candidates_qids(self, qids: Set[str]) -> List[str]:
        ranked = []
        for row in self._sorted_candidates():
            if row.qid in qids:
                ranked.append(row.qid)
        return ranked

    def _ranked_memory(self) -> List[MemoryEntry]:
        return sorted(self.memory, key=lambda item: (1 if item.tries > 0 else 0, -item.priority, item.tries, item.mid))

    def _ingest_seed(self, data: Dict[str, Any]) -> None:
        part_ids = []
        char_ids = []
        char_id_set = set()
        for part in data.get("parts", [])[:6]:
            qid = part.get("id", "")
            if not qid:
                continue
            part_ids.append(qid)
            row = self._candidate(qid, part)
            row.direct_has_part_refs += max(1, int(part.get("refs", 0) or 0))
            row.routes.add("seed_p527")
            row.notes.append("direct seed part")
            self._clean_candidate(row)
            self._add_memory({"tool": "check", "args": {"subject": self.target_qid, "candidate": qid}, "why": "confirm seed"}, priority=11)

        for char in data.get("characteristics", [])[:8]:
            qid = char.get("id", "")
            if qid:
                char_ids.append(qid)
                char_id_set.add(qid)
                self._add_memory({"tool": "open", "args": {"qid": qid}, "why": "inspect char"}, priority=5)

        previews = self.wikidata.preview_nodes(part_ids + char_ids[:6])
        for preview in previews:
            self._ingest_preview(preview, is_seed=preview.get("id") in char_id_set)

        focus = part_ids[:2] + char_ids[:3]
        if len(focus) >= 2:
            self._add_memory({"tool": "probe_many", "args": {"candidates": focus[:4]}, "why": "probe seeded frontier"}, priority=12)
        elif len(focus) == 1:
            self._add_memory({"tool": "check", "args": {"subject": self.target_qid, "candidate": focus[0]}, "why": "confirm seeded frontier"}, priority=12)
        if len(char_ids) >= 2:
            self._add_memory({"tool": "open_many", "args": {"qids": char_ids[:4]}, "why": "compare char nodes"}, priority=6)

        survey_k = min(max(5, self.limit), 8)
        self._add_memory({"tool": "survey", "args": {"qid": self.target_qid, "k": survey_k}, "why": "broad frontier survey"}, priority=13)
        self._add_memory({"tool": "scan", "args": {"route": "dir", "qid": self.target_qid, "k": min(6, self.limit)}, "why": "strongest scan"}, priority=10)
        self._add_memory({"tool": "scan", "args": {"route": "rev", "qid": self.target_qid, "k": min(6, self.limit)}, "why": "reverse class scan"}, priority=9)
        self._add_memory({"tool": "scan", "args": {"route": "cls", "qid": self.target_qid, "k": min(5, self.limit)}, "why": "class scan"}, priority=8)
        self._add_memory({"tool": "scan", "args": {"route": "up", "qid": self.target_qid, "k": min(4, self.limit)}, "why": "up scan"}, priority=6)

    def _ingest_preview(self, data: Dict[str, Any], is_seed: bool = False) -> None:
        qid = data.get("id", "")
        if not qid:
            return
        row = self._candidate(qid, data)
        row.sitelinks = max(row.sitelinks, int(data.get("sitelinks", 0) or 0))
        type_labels = [item.get("label", item.get("id", "")) for item in data.get("P31", [])[:2]]
        parent_ids = {item.get("id") for item in data.get("P361", [])}
        if type_labels:
            row.notes.append("types: " + ", ".join(type_labels))
        if self.target_qid in parent_ids:
            row.direct_part_of_refs += 1
            row.routes.add("preview_p361_target")
            row.notes.append("preview shows part-of target")
        if any(item.get("id") == self.target_qid for item in data.get("P527", [])):
            row.direct_has_part_refs += 1
            row.routes.add("preview_p527_target")
            row.notes.append("preview shows target in has-part")
        self._clean_candidate(row)
        if is_seed and self.target_qid in parent_ids:
            self._add_memory({"tool": "check", "args": {"subject": self.target_qid, "candidate": qid}, "why": "confirm preview tie"}, priority=11)

    def _ingest_survey(self, source_qid: str, rows: List[Dict[str, Any]]) -> None:
        focus = []
        weak = []
        for item in rows:
            cand = item.get("cand", "")
            if not cand or cand == self.target_qid:
                continue
            merged = {
                "id": cand,
                "label": item.get("candLabel", ""),
                "description": item.get("candDescription", ""),
                "url": item.get("candUrl", self.wikidata.page_url(cand)),
                "sitelinks": int(item.get("candSitelinks", 0) or 0),
            }
            row = self._candidate(cand, merged)
            for route_name in item.get("routes", []):
                row.routes.add(f"survey_{route_name}")
                if route_name == "direct_p527":
                    row.direct_has_part_refs += max(1, self._as_int(item.get("refs", 0)))
                elif route_name == "inverse_p361":
                    row.direct_part_of_refs += max(1, self._as_int(item.get("refs", 0)))
                elif route_name.startswith("class_"):
                    row.class_routes.append(route_name)
            for via in item.get("via", [])[:2]:
                row.class_via.append(via)
            self._clean_candidate(row)
            if row.direct_has_part_refs > 0 or row.direct_part_of_refs > 0 or row.class_routes:
                focus.append(cand)
            else:
                weak.append(cand)
        if len(focus) >= 2:
            self._add_memory({"tool": "probe_many", "args": {"candidates": focus[:4]}, "why": "probe survey frontier"}, priority=11)
        elif len(focus) == 1:
            self._add_memory({"tool": "check", "args": {"subject": self.target_qid, "candidate": focus[0]}, "why": "confirm survey frontier"}, priority=11)
        elif len(weak) >= 2:
            self._add_memory({"tool": "open_many", "args": {"qids": weak[:4]}, "why": "inspect survey weak nodes"}, priority=5)
        elif len(weak) == 1:
            self._add_memory({"tool": "open", "args": {"qid": weak[0]}, "why": "inspect survey weak node"}, priority=5)

    def _ingest_open(self, data: Dict[str, Any]) -> None:
        qid = data.get("id", "")
        if not qid:
            return
        row = self._candidate(qid, data)
        row.opened = True
        row.last_open_summary = data
        row.urls = [data.get("url", self.wikidata.page_url(qid))]
        row.sitelinks = max(row.sitelinks, int(data.get("sitelinks", 0) or 0))

        parents = {item.get("id") for item in data.get("P361", [])}
        if parents and self.target_qid not in parents and qid != self.target_qid:
            row.evidence_only_reasons.append("opened node lacks direct parent tie to target")
        if qid != self.target_qid and self.target_qid in parents:
            row.routes.add("open_structural_support")
            row.direct_part_of_refs += 1
            self._add_memory({"tool": "check", "args": {"subject": self.target_qid, "candidate": qid}, "why": "confirm opened tie"}, priority=10)

        context_nodes = []
        for prop in ("P31", "P279", "P361"):
            for item in data.get(prop, [])[:1]:
                iqid = item.get("id", "")
                if iqid and iqid != self.target_qid:
                    context_nodes.append(iqid)
                    self._add_memory({"tool": "open", "args": {"qid": iqid}, "why": f"inspect {prop}"}, priority=3)
        self._clean_candidate(row)
        if len(context_nodes) >= 2:
            self._add_memory({"tool": "open_many", "args": {"qids": context_nodes[:2]}, "why": "compare context"}, priority=3)

    def _ingest_scan(self, source_qid: str, route: str, rows: List[Dict[str, Any]]) -> None:
        batch_check_candidates: List[str] = []
        batch_open_candidates: List[str] = []
        for item in rows:
            cand = item.get("cand", "")
            if not cand or cand == self.target_qid:
                continue
            merged = {
                "id": cand,
                "label": item.get("candLabel", ""),
                "description": item.get("candDescription", ""),
                "url": item.get("candUrl", self.wikidata.page_url(cand)),
                "sitelinks": int(item.get("candSitelinks", 0) or 0),
            }
            row = self._candidate(cand, merged)
            route_name = item.get("route", route)
            row.routes.add(f"scan_{route_name}")
            if route == "dir":
                refs = self._as_int(item.get("refs", 0))
                if route_name == "direct_p527":
                    row.direct_has_part_refs += max(1, refs)
                elif route_name == "inverse_p361":
                    row.direct_part_of_refs += max(1, refs)
                batch_check_candidates.append(cand)
            elif route in {"cls", "rev"}:
                via = item.get("viaLabel", item.get("via", ""))
                if via:
                    row.class_via.append(via)
                    row.class_routes.append(route_name)
                batch_check_candidates.append(cand)
            elif route == "char":
                row.evidence_only_reasons.append("characteristic node")
                batch_open_candidates.append(cand)
            elif route == "up":
                row.evidence_only_reasons.append("context node")
                batch_open_candidates.append(cand)
            self._clean_candidate(row)

        top_check = self._unique_keep_order(batch_check_candidates)[:4]
        top_open = self._unique_keep_order(batch_open_candidates)[:4]
        if len(top_check) >= 2:
            self._add_memory({"tool": "probe_many", "args": {"candidates": top_check}, "why": "probe candidate set"}, priority=11 if route == "dir" else 8)
            self._add_memory({"tool": "check_many", "args": {"subject": self.target_qid, "candidates": top_check}, "why": "confirm candidate set"}, priority=9 if route == "dir" else 6)
        elif len(top_check) == 1:
            self._add_memory({"tool": "check", "args": {"subject": self.target_qid, "candidate": top_check[0]}, "why": "confirm candidate"}, priority=10 if route == "dir" else 7)
        if len(top_open) >= 2:
            self._add_memory({"tool": "open_many", "args": {"qids": top_open}, "why": "inspect weak nodes"}, priority=5)
        elif len(top_open) == 1:
            self._add_memory({"tool": "open", "args": {"qid": top_open[0]}, "why": "inspect weak node"}, priority=5)

    def _ingest_check(self, candidate_qid: str, data: Dict[str, Any]) -> None:
        if not candidate_qid:
            return

        row = self._candidate(candidate_qid, {"id": candidate_qid})
        row.direct_has_part_refs += int(data.get("direct_has_part_refs", 0) or 0)
        row.direct_part_of_refs += int(data.get("direct_part_of_refs", 0) or 0)

        for route in data.get("class_routes", []):
            row.class_routes.append(route)
            row.routes.add(f"check_{route}")

        for via in data.get("class_via", []):
            row.class_via.append(via)

        if row.direct_has_part_refs > 0 or row.direct_part_of_refs > 0:
            row.routes.add("check_supported")
            self._mark_bucket("accept", [candidate_qid])
        elif row.class_routes:
            self._mark_bucket("hold", [candidate_qid])
        else:
            row.evidence_only_reasons.append("check found no structural support")
            self.rejected[candidate_qid] = "no structural support"
            self._mark_bucket("reject", [candidate_qid])

        self._clean_candidate(row)

    def _candidate(self, qid: str, data: Dict[str, Any]) -> CandidateRecord:
        row = self.candidates.get(qid)
        if row is None:
            row = CandidateRecord(qid=qid)
            self.candidates[qid] = row
        label = data.get("label")
        if label:
            row.label = label
        description = data.get("description")
        if description:
            row.description = description
        row.sitelinks = max(row.sitelinks, int(data.get("sitelinks", 0) or 0))
        url = data.get("url")
        if url and url not in row.urls:
            row.urls.append(url)
        self._clean_candidate(row)
        return row

    def _add_memory(self, action: Dict[str, Any], priority: int = 0) -> None:
        for existing in self.memory:
            if existing.action == action:
                existing.priority = max(existing.priority, priority)
                return
        mid = f"M{self.memory_counter}"
        self.memory_counter += 1
        self.memory.append(MemoryEntry(mid=mid, action=action, priority=priority))

    def _get_memory(self, mid: str) -> Optional[MemoryEntry]:
        for entry in self.memory:
            if entry.mid == mid:
                return entry
        return None

    def _compact_resolution_candidate(self, alias: str, row: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": alias, "name": row.get("label", ""), "d": row.get("description", ""), "sl": row.get("sitelinks", 0), "i": row.get("inst", [])[:2], "match": row.get("match", "")}

    def _compact_memory(self, entry: MemoryEntry, qid_to_alias: Dict[str, str]) -> Dict[str, Any]:
        return {"id": entry.mid, "go": self._memory_synopsis(entry.action, qid_to_alias), "why": self._short(str(entry.action.get("why", "")), 48), "p": entry.priority, "t": entry.tries}

    def _memory_synopsis(self, action: Dict[str, Any], qid_to_alias: Dict[str, str]) -> str:
        tool = action.get("tool", "")
        args = action.get("args", {}) or {}

        def alias(qid: str) -> str:
            if not qid:
                return "?"
            if qid == self.target_qid:
                return "T"
            return qid_to_alias.get(qid, "N?")

        if tool == "scan":
            return f"scan({alias(args.get('qid', self.target_qid))},{args.get('route', 'dir')},{min(6, int(args.get('k', 3) or 3))})"
        if tool == "survey":
            return f"survey({alias(args.get('qid', self.target_qid))},{min(8, int(args.get('k', 6) or 6))})"
        if tool == "check":
            return f"check({alias(args.get('candidate', ''))})"
        if tool == "open":
            return f"open({alias(args.get('qid', ''))})"
        if tool == "expand_neighbors":
            return f"expand_neighbors({alias(args.get('qid', ''))},{min(8, int(args.get('k', 6) or 6))})"
        if tool == "probe_many":
            nodes = [alias(q) for q in (args.get("candidates", []) or [])[:4]]
            return f"probe_many([{','.join(nodes)}])"
        if tool == "check_many":
            nodes = [alias(q) for q in (args.get("candidates", []) or [])[:4]]
            return f"check_many([{','.join(nodes)}])"
        if tool == "open_many":
            nodes = [alias(q) for q in (args.get("qids", []) or [])[:4]]
            return f"open_many([{','.join(nodes)}])"
        if tool == "accept_many":
            nodes = [alias(q) for q in (args.get("candidates", []) or [])[:4]]
            return f"accept_many([{','.join(nodes)}])"
        if tool == "hold_many":
            nodes = [alias(q) for q in (args.get("candidates", []) or [])[:4]]
            return f"hold_many([{','.join(nodes)}])"
        if tool == "reject_many":
            nodes = [alias(q) for q in (args.get("candidates", []) or [])[:4]]
            return f"reject_many([{','.join(nodes)}])"
        if tool == "save_memory":
            route = args.get("route", {})
            if isinstance(route, dict):
                return f"save_memory({self._memory_synopsis(route, qid_to_alias)})"
            return "save_memory(?)"
        if tool == "seed":
            return "seed(T)"
        return tool or "action"

    def _compact_result_for_model(self, item: Dict[str, Any], qid_to_alias: Dict[str, str]) -> Dict[str, Any]:
        tool = item.get("tool", "")
        data = item.get("data", {}) or {}
        out = {"tool": tool, "ok": bool(item.get("ok", False)), "sum": self._short(item.get("brief", ""), 96)}

        if tool == "seed":
            out["parts"] = [qid_to_alias.get(row.get("id", ""), "") for row in data.get("parts", [])[:4] if qid_to_alias.get(row.get("id", ""), "")]
            out["chars"] = [qid_to_alias.get(row.get("id", ""), "") for row in data.get("chars", [])[:4] if qid_to_alias.get(row.get("id", ""), "")]
            return out
        if tool == "scan":
            out["route"] = data.get("route", "")
            out["anchor"] = self._alias_or_blank(data.get("qid", ""), qid_to_alias)
            out["nodes"] = [qid_to_alias.get(row.get("cand", ""), "") for row in data.get("rows", [])[:4] if qid_to_alias.get(row.get("cand", ""), "")]
            return out
        if tool == "survey":
            out["anchor"] = self._alias_or_blank(data.get("qid", ""), qid_to_alias)
            out["nodes"] = [qid_to_alias.get(row.get("cand", ""), "") for row in data.get("rows", [])[:4] if qid_to_alias.get(row.get("cand", ""), "")]
            return out
        if tool == "open":
            out["node"] = self._alias_or_blank(data.get("id", ""), qid_to_alias)
            out["p361"] = [self._alias_or_blank(item_id.get("id", ""), qid_to_alias) for item_id in data.get("P361", [])[:2] if self._alias_or_blank(item_id.get("id", ""), qid_to_alias)]
            return out
        if tool == "open_many":
            out["nodes"] = [self._alias_or_blank(row.get("id", ""), qid_to_alias) for row in data.get("rows", [])[:4] if self._alias_or_blank(row.get("id", ""), qid_to_alias)]
            return out
        if tool == "check":
            out["candidate"] = self._alias_or_blank(data.get("candidate", ""), qid_to_alias)
            out["dir"] = f"{data.get('direct_has_part_refs', 0)}/{data.get('direct_part_of_refs', 0)}"
            out["cls"] = data.get("class_via", [])[:2]
            return out
        if tool == "check_many":
            out["nodes"] = [self._alias_or_blank(row.get("candidate", ""), qid_to_alias) for row in data.get("rows", [])[:4] if self._alias_or_blank(row.get("candidate", ""), qid_to_alias)]
            return out
        if tool == "probe_many":
            out["nodes"] = [self._alias_or_blank(row.get("candidate", ""), qid_to_alias) for row in data.get("rows", [])[:4] if self._alias_or_blank(row.get("candidate", ""), qid_to_alias)]
            return out
        if tool == "accept_many":
            out["nodes"] = [self._alias_or_blank(q, qid_to_alias) for q in data.get("nodes", [])[:6] if self._alias_or_blank(q, qid_to_alias)]
            return out
        if tool == "hold_many":
            out["nodes"] = [self._alias_or_blank(q, qid_to_alias) for q in data.get("nodes", [])[:6] if self._alias_or_blank(q, qid_to_alias)]
            return out
        if tool == "reject_many":
            out["nodes"] = [self._alias_or_blank(q, qid_to_alias) for q in data.get("nodes", [])[:6] if self._alias_or_blank(q, qid_to_alias)]
            return out
        if tool == "expand_neighbors":
            out["anchor"] = self._alias_or_blank(data.get("anchor", ""), qid_to_alias)
            out["nodes"] = [qid_to_alias.get(row.get("cand", ""), "") for row in data.get("rows", [])[:4] if qid_to_alias.get(row.get("cand", ""), "")]
            return out
        if tool == "memory":
            out["memory_id"] = data.get("mid", "")
            return out
        return out

    def _brief_seed(self, data: Dict[str, Any]) -> str:
        return f"seed parts={len(data.get('parts', []))} chars={len(data.get('characteristics', []))}"

    def _brief_open(self, data: Dict[str, Any]) -> str:
        return f"open {data.get('id', '')} p31={len(data.get('P31', []))} p361={len(data.get('P361', []))} p527={len(data.get('P527', []))}"

    def _brief_scan(self, qid: str, route: str, rows: List[Dict[str, Any]]) -> str:
        return f"scan {route} on {qid} rows={len(rows)}"

    def _brief_check(self, data: Dict[str, Any]) -> str:
        return f"check {data.get('candidate', '')} direct={data.get('direct_has_part_refs', 0)}/{data.get('direct_part_of_refs', 0)} class={len(data.get('class_routes', []))}"

    def _compact_seed(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {"subject": data.get("subject", {}), "parts": [self._tiny_node(row) for row in data.get("parts", [])[:6]], "chars": [self._tiny_node(row) for row in data.get("characteristics", [])[:6]]}

    def _compact_open(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": data.get("id", ""),
            "label": data.get("label", ""),
            "desc": data.get("description", ""),
            "sitelinks": data.get("sitelinks", 0),
            "P31": [self._tiny_node(row) for row in data.get("P31", [])[:4]],
            "P279": [self._tiny_node(row) for row in data.get("P279", [])[:4]],
            "P361": [self._tiny_node(row) for row in data.get("P361", [])[:4]],
            "P527": [self._tiny_node(row) for row in data.get("P527", [])[:4]],
            "P1552": [self._tiny_node(row) for row in data.get("P1552", [])[:4]],
        }

    def _compact_scan(self, qid: str, route: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        compact_rows = []
        for row in rows[:6]:
            compact_rows.append({
                "cand": row.get("cand", ""),
                "label": row.get("candLabel", ""),
                "route": row.get("route", route),
                "via": row.get("viaLabel", row.get("via", "")),
                "desc": self._short(row.get("candDescription", "")),
                "refs": row.get("refs", "0"),
            })
        return {"qid": qid, "route": route, "rows": compact_rows}

    def _compact_survey(self, qid: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        compact_rows = []
        for row in rows[:8]:
            compact_rows.append({
                "cand": row.get("cand", ""),
                "label": row.get("candLabel", ""),
                "desc": self._short(row.get("candDescription", "")),
                "routes": row.get("routes", [])[:3],
                "via": row.get("via", [])[:2],
                "refs": row.get("refs", 0),
            })
        return {"qid": qid, "rows": compact_rows}

    def _compact_probe_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        open_row = row.get("open", {}) or {}
        check_row = row.get("check", {}) or {}
        return {
            "candidate": row.get("candidate", ""),
            "label": open_row.get("label", ""),
            "desc": self._short(open_row.get("description", "")),
            "support": f"{check_row.get('direct_has_part_refs', 0)}/{check_row.get('direct_part_of_refs', 0)}",
            "class_via": check_row.get("class_via", [])[:2],
            "p361": [item.get("id", "") for item in open_row.get("P361", [])[:2]],
            "p31": [item.get("label", item.get("id", "")) for item in open_row.get("P31", [])[:2]],
        }

    def _short(self, text: str, n: int = 72) -> str:
        text = str(text or "").strip()
        if len(text) <= n:
            return text
        return text[: max(0, n - 1)].rstrip() + "…"

    def _tiny_node(self, row: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": row.get("id", ""), "label": row.get("label", ""), "refs": row.get("refs", 0)}

    def _compact_history(self, row: Dict[str, Any]) -> Dict[str, Any]:
        action = row.get("action") or {"tool": "resolve", "args": {}}
        result = row.get("result", {})
        return {"tool": action.get("tool", "resolve"), "args": action.get("args", {}), "brief": result.get("brief", "")}

    def _action_signature(self, action: Dict[str, Any]) -> str:
        key = {"tool": action.get("tool", ""), "args": action.get("args", {}) or {}}
        return json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))

    def _extract_qids_from_action(self, action: Dict[str, Any]) -> List[str]:
        tool = str(action.get("tool", "")).strip()
        args = action.get("args", {}) or {}
        qids: List[str] = []

        if tool in {"seed", "scan", "survey", "expand_neighbors"}:
            qid = str(args.get("qid", "")).strip()
            if qid:
                qids.append(qid)
        elif tool in {"open", "check"}:
            for key in ("qid", "candidate"):
                qid = str(args.get(key, "")).strip()
                if qid:
                    qids.append(qid)
        elif tool in {"open_many", "check_many", "probe_many", "accept_many", "hold_many", "reject_many"}:
            for key in ("qids", "candidates"):
                value = args.get(key, [])
                if isinstance(value, list):
                    qids.extend(str(item).strip() for item in value if str(item).strip())
        elif tool == "save_memory":
            route = args.get("route", {})
            if isinstance(route, dict):
                qids.extend(self._extract_qids_from_action(route))

        return self._unique_keep_order(qids)

    def _alias_or_label(self, qid: str, qid_to_alias: Dict[str, str], fallback: str) -> str:
        if qid == self.target_qid:
            return "T"
        return qid_to_alias.get(qid, fallback)

    def _alias_or_blank(self, qid: str, qid_to_alias: Dict[str, str]) -> str:
        if qid == self.target_qid:
            return "T"
        return qid_to_alias.get(qid, "")

    def _clean_candidate(self, row: CandidateRecord) -> None:
        row.urls = self._unique_keep_order(row.urls)
        row.class_routes = self._unique_keep_order(row.class_routes)
        row.class_via = self._unique_keep_order(row.class_via)
        row.evidence_only_reasons = self._unique_keep_order(row.evidence_only_reasons)
        row.notes = self._unique_keep_order(row.notes)

    @staticmethod
    def _unique_keep_order(items: Sequence[Any]) -> List[Any]:
        out = []
        seen = set()
        for item in items:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
            if key not in seen:
                seen.add(key)
                out.append(item)
        return out

    @staticmethod
    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except Exception:
            return 0


def run_orchestrator_bundle(
    *,
    prompts: Sequence[Dict[str, Any]],
    worker_url: str = DEFAULT_WORKER,
    worker_client: Optional[Any] = None,
    mode: str = "normal",
    steps: int = 4,
    limit: int = 8,
    timeout: int = 18,
    output_root: str = ".dist",
    visual_component_batch_size: int = 4,
    skip_visual_stage: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    prompt_rows = [row for row in (prompts or []) if isinstance(row, dict)]
    if not prompt_rows:
        return {"ok": True, "prompts": {}, "errors": {}, "count": 0}

    shared_worker = worker_client or WorkerClient(worker_url, timeout=max(60, int(timeout) * 4))
    default_bundle_workers = max(1, int(os.environ.get("C2_DIAGRAM_BUNDLE_WORKERS", "6") or 6))
    worker_count = max_workers if max_workers is not None else max(1, min(len(prompt_rows), default_bundle_workers))
    reports_by_prompt: Dict[str, Dict[str, Any]] = {}
    errors_by_prompt: Dict[str, str] = {}

    def _run_one(row: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        prompt = str(row.get("prompt", "") or row.get("target", "") or "").strip()
        if not prompt:
            raise ValueError("bundle item missing prompt")

        requested_components = row.get("requested_components") or row.get("diagram_required_objects") or []
        if not isinstance(requested_components, list):
            requested_components = []

        wikidata = WikidataClient(timeout=timeout)
        orchestrator = ComponentAgentOrchestrator(
            target_text=prompt,
            worker=shared_worker,
            wikidata=wikidata,
            max_steps=steps,
            mode=mode,
            limit=limit,
            qid=(str(row.get("qid", "") or "").strip() or None),
        )
        report = orchestrator.run()
        requested_rows = orchestrator.resolve_requested_components(requested_components)
        report = orchestrator.merge_requested_components_into_report(report, requested_rows)
        report["requested_components_input"] = [
            str(item).strip()
            for item in requested_components
            if str(item).strip()
        ]
        if skip_visual_stage:
            report["visual_stage"] = {
                "prompt_dir": "",
                "manifest_path": "",
                "components": [],
                "query_overrides": {},
                "error": "skipped by flag",
            }
        else:
            report["visual_stage"] = run_visual_description_stage(
                stage1_report=report,
                original_prompt=prompt,
                worker_url=worker_url,
                worker_client=shared_worker,
                mode=mode,
                output_root=output_root,
                component_batch_size=visual_component_batch_size,
                worker_timeout=max(60, int(timeout) * 4),
                http_timeout=timeout,
            )
        return prompt, report

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="c2_bundle") as pool:
        future_map = {
            pool.submit(_run_one, row): str(row.get("prompt", "") or row.get("target", "") or "").strip()
            for row in prompt_rows
        }
        for future in as_completed(future_map):
            prompt = future_map[future]
            try:
                key, report = future.result()
                reports_by_prompt[key] = report
            except Exception as exc:
                errors_by_prompt[prompt] = f"{type(exc).__name__}: {exc}"
                reports_by_prompt[prompt] = {
                    "input_prompt": prompt,
                    "accepted_components": [],
                    "requested_components_input": [],
                    "requested_components_resolved": [],
                    "visual_stage": {
                        "prompt_dir": "",
                        "manifest_path": "",
                        "components": [],
                        "query_overrides": {},
                        "error": str(exc),
                    },
                }

    return {
        "ok": not bool(errors_by_prompt),
        "prompts": reports_by_prompt,
        "errors": errors_by_prompt,
        "count": len(reports_by_prompt),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wikidata component orchestrator")
    parser.add_argument("target", help="target object text, e.g. 'train' or 'cell'")
    parser.add_argument("--qid", default="", help="explicit target QID")
    parser.add_argument("--worker", default=DEFAULT_WORKER, help="Qwen worker /infer URL")
    parser.add_argument("--worker-timeout", type=int, default=60, help="HTTP timeout for worker requests")
    parser.add_argument("--mode", default="normal", choices=["normal", "thinking"], help="worker mode")
    parser.add_argument("--steps", type=int, default=4, help="max agent cycles")
    parser.add_argument("--limit", type=int, default=8, help="default row limit for scans")
    parser.add_argument("--timeout", type=int, default=18, help="HTTP timeout for Wikidata calls")
    parser.add_argument("--output-root", default=".dist", help="root folder for stage-2 Prompt/Component outputs")
    parser.add_argument("--visual-component-batch-size", type=int, default=4, help="how many stage-2 components to process in parallel")
    parser.add_argument("--skip-visual-stage", action="store_true", help="skip the stage-2 visual description pipeline")
    parser.add_argument("--pretty", action="store_true", help="pretty-print final JSON")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    worker = WorkerClient(args.worker, timeout=args.worker_timeout)
    wikidata = WikidataClient(timeout=args.timeout)
    orchestrator = ComponentAgentOrchestrator(
        target_text=args.target,
        worker=worker,
        wikidata=wikidata,
        max_steps=args.steps,
        mode=args.mode,
        limit=args.limit,
        qid=args.qid or None,
    )

    report = orchestrator.run()
    if args.skip_visual_stage:
        report["visual_stage"] = {
            "prompt_dir": "",
            "manifest_path": "",
            "components": [],
            "query_overrides": {},
            "error": "skipped by flag",
        }
    else:
        try:
            report["visual_stage"] = run_visual_description_stage(
                stage1_report=report,
                original_prompt=args.target,
                worker_url=args.worker,
                mode=args.mode,
                output_root=args.output_root,
                component_batch_size=args.visual_component_batch_size,
                worker_timeout=args.worker_timeout,
                http_timeout=args.timeout,
            )
        except Exception as exc:
            logging.getLogger("visual_stage").exception("stage-2 visual description pipeline failed")
            report["visual_stage"] = {
                "prompt_dir": "",
                "manifest_path": "",
                "components": [],
                "query_overrides": {},
                "error": str(exc),
            }
    if args.pretty:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
