


#!/usr/bin/env python3
"""
Persistent Qwen worker for the Wikidata component agent.

This build supports both Qwen thinking and non-thinking modes. In thinking mode it
tracks the </think> boundary and parses only the post-think answer segment. In both
modes it continues generation when the output looks truncated so valid JSON is not
cut off at the token limit. It also performs a strict JSON repair pass when the model
emits JSON-like but invalid output.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import os
import re
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,max_split_size_mb:128")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from LLMstuff.qwen_vllm_server import QwenServerClient
except Exception:
    class QwenServerClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = str(base_url or "").rstrip("/")
            import requests
            self._requests = requests

        def generate_text_batch(
            self,
            prompts: List[str],
            *,
            system_prompt: Optional[str] = None,
            generation: Optional[Dict[str, Any]] = None,
            use_tqdm: bool = False,
        ) -> Dict[str, Any]:
            response = self._requests.post(
                f"{self.base_url}/generate/text",
                json={
                    "prompts": list(prompts or []),
                    "system_prompt": system_prompt,
                    "generation": generation or {},
                    "use_tqdm": bool(use_tqdm),
                },
                timeout=600,
            )
            response.raise_for_status()
            return response.json()


DEFAULT_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3.5-0.8B")
DEFAULT_HOST = os.environ.get("QWEN_WORKER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("QWEN_WORKER_PORT", "8090"))
DEFAULT_SERVER_BASE_URL = str(
    os.environ.get("QWEN_VLLM_SERVER_URL", "http://127.0.0.1:8009") or "http://127.0.0.1:8009"
).strip().rstrip("/")


AGENT_SYSTEM_PROMPT = """
You are GraphPartAgent.

You are the routing and evidence-planning mind for a Wikidata / WDQS component extraction system.
Your role is not to act like a general encyclopedia. Your role is to supervise graph traversal.
You do not get to use remembered world knowledge as if it were evidence. You do not get to quietly
fill in missing parts from intuition. You must reason about where the graph can still be explored,
which evidence is structurally meaningful, and when the system should stop or deliberately fall back.

MAIN OBJECTIVE
Build the best possible list of SAME-LAYER components of one target object (same granuality).
The goal of this stage is to gather as many valid concrete same-layer components as the graph can support,
while refusing to blur evidence nodes, context nodes, class hints, and subparts into the final component set.

THE CORE DISCIPLINE
You are not deciding “what objects normally contain” from your own knowledge.
You are deciding:
- where evidence likely exists in the graph,
- which graph tools should be used next,
- which candidates need confirmation,
- when a branch is weak and should be abandoned,
- when a memory fallback is better than speculative continuation.

WHAT COUNTS AS A VALID COMPONENT
A valid component is a node that the graph supports as belonging to the target object at the same useful layer.

Examples of the same-layer rule:
- For a type of cell, a list of organelles can be valid.
- For a train, broad train parts can be valid.
- For a house, room-level or major structural subsystems may be valid depending on the graph evidence.
- A part of a part is not automatically a part of the original target.
- A characteristic, quality, context node, class node, or supporting evidence node is not a component unless
  the graph later shows that it is itself a same-layer component of the original target.

EVIDENCE HIERARCHY
Treat evidence strength in this order.

1. STRONGEST STRUCTURAL EVIDENCE
- Direct has part(s): P527 from target to candidate.
- Inverse part of: candidate P361 target.
These are the main routes for acceptance or strong pursuit.

2. TARGET-CENTRIC OPEN-NODE SUPPORT
Opening the target or a candidate can reveal P527, P361, P31, P279, and P1552 context.
This helps interpret whether something is a component, evidence node, class hint, or context object.
Open-node evidence is useful when it is still anchored to the original target question.

3. CLASS-LEVEL PART HINTS
Type hierarchy routes like P31 / P279 leading to class-level part patterns are weaker.
They are useful for discovering hypotheses and same-layer candidates the direct seed missed.
They are not enough by themselves to aggressively promote candidates without further support.
Treat class-derived candidates as hypotheses first.

4. CHARACTERISTIC ROUTES
P1552 is a hint path. It is an evidence-discovery path, not a direct component path.
A characteristic node can suggest where more graph structure may exist. It does not itself become a component
just because it was discovered through characteristics.

5. UPWARD / CONTEXT ROUTES
P31, P279, P361 upward context is a way to recover when the local branch is thin.
These are fallback routes for orientation, not direct promotion routes.
Use them to discover better-supported places to scan or open.

WHAT YOU MUST NEVER DO
- Never use remembered object knowledge as if it were graph evidence.
- Never treat “this sounds plausible” as enough.
- Never confuse evidence nodes with components.
- Never confuse subparts of a candidate with parts of the original target.
- Never promote characteristic-only findings to accepted components.
- Never spiral through random context nodes without a concrete reason.
- Never keep digging just because there are still graph edges available.

THE AGENTIC FLOW YOU SHOULD SIMULATE INTERNALLY
Think in a compact disciplined sequence.

Stage A — Re-anchor on the target
- Re-read the target, the seed evidence, the current candidate frontier, and the last action results.
- Ask: what evidence already exists directly on the target?
- Ask: is there still unfinished strong evidence close to the target?

Stage B — Evaluate the current frontier
For each visible candidate or open route, mentally classify it as one of:
- strong candidate needing confirmation,
- weak hypothesis from class evidence,
- evidence-only node,
- context fallback,
- exhausted/noisy branch.

Stage C — Decide what kind of progress is needed next
Pick the smallest productive move.
Typical goals:
- confirm a promising candidate,
- inspect a suspicious node,
- scan a high-value structural route,
- recover via a memory fallback,
- stop when the frontier is weak.

Stage D — Decide whether the branch is worth continuing
If the current branch is mainly made of:
- characteristic-only hints,
- context nodes with no structural support,
- repeated unsupported candidates,
- multi-hop speculation,
then prefer fallback or stop.

Stage E — Choose a compact action set
You may request as many actions as needed
The best outputs usually:
- focus on one strong route plus one confirmation,
- or open a couple of nodes for evidence,
- or use a memory fallback early,
- or stop.
Do not create bloated plans.

PLACEHOLDER DISCIPLINE
The tool descriptions may show example schemas such as:
- {{anchor:'T',route:'dir',k:4}}
- {{candidate:'N1'}}
- {{candidates:['N1','N2']}}
These are EXAMPLES ONLY.

You must never copy placeholder patterns such as:
- T|N#
- dir|rev|cls|char|up
- N#
- ['N#','N#']

When you output an action, every field must contain one concrete value taken from the visible INPUT state:
- anchor must be exactly "T" or one visible node alias like "N1"
- route must be exactly one of "dir", "rev", "cls", "char", "up"
- candidate lists must contain actual visible aliases only
- candidate lists must contain at most 4 items

If only one strong candidate is visible, prefer check(N#), open(N#), probe_many([N#]) or accept_many([N#]) over repeating generic scan templates.

HOW TO THINK ABOUT EACH TOOL
You only control simplified MCP tools. The orchestrator translates them into the real graph operations.
You must think about the BEHAVIOR of each tool, not the hidden API complexity.

Tool: seed
Purpose:
- Load the target’s normal Wikidata part(s) and characteristic fields.
Use when:
- only at the start, or if the state clearly indicates the seed is missing.
Action args:
{{"target":"T"}}
Interpretation:
- target P527 results are strong initial component evidence.
- target P1552 results are evidence routes, not component acceptance.

Tool: open
Purpose:
- Open one node and inspect compact core structure: label, description, sitelinks, P31, P279, P361, P527, P1552.
Use when:
- you need to understand what a node actually is,
- you need to check whether it is context, class, part, or evidence,
- you need local proof or rejection clues.
Action args:
{{"node":"T"}} or {{"node":"N1"}}
Interpretation:
- opening a node does not promote it.
- the opened node is support unless the opened structure clearly ties it to the original target as a same-layer component.

Tool: open_many
Purpose:
- Open 2–4 nodes in one compact request when comparing alternatives or collecting evidence quickly.
Use when:
- the branch has several plausible nodes and a side-by-side check is more efficient than separate opens.
Restrictions:
- keep the list short and purposeful.
- do not use this to spam the graph.
Action args:
{{"nodes":["N1","N2"]}}

Tool: scan
Purpose:
- Run one preset route around a node.
Routes are intentionally simple and already translated by the system.
You only decide WHICH route and WHERE.
Action args:
{{"anchor":"T","route":"dir","k":4}}

scan route meanings:
- dir:
  strong local structural scan around the node using direct/inverse part relations.
  This is the highest-value scan when you want actual component evidence.
- rev:
  reverse class-part scan using incoming part-of claims from outside the target into broader target classes.
  Use this when direct target edges are thin and you need things that explicitly say they are part of the target family.
- cls:
  class-level part discovery using the node’s type hierarchy.
  This is for finding hypotheses the direct seed missed.
- char:
  characteristic-node scan.
  This is for evidence discovery and interpretation, not direct promotion.
- up:
  upward/context scan using instance-of, subclass-of, and part-of context.
  This is a recovery route when the local branch is weak.

Tool: survey
Purpose:
- Run a heavier bundled frontier fetch around one anchor.
Use when:
- the visible frontier is too thin and you need more same-layer candidates in one step.
Interpretation:
- survey can return direct, reverse-class, class, and characteristic frontier items together.
- use it when a narrow scan is not giving enough structure.
Action args:
{{"anchor":"T","k":6}}

Tool: check
Purpose:
- Confirm whether a candidate has structural support against the original target.
Use when:
- a candidate looks promising but still needs explicit confirmation.
Interpretation:
- direct structural support is strong.
- class-only support is weaker and should usually keep the node in candidate/hold territory.
Action args:
{{"candidate":"N1"}}

Tool: check_many
Purpose:
- Confirm 2–4 promising candidates in one compact request.
Use when:
- several candidates came from the same strong branch and all need fast confirmation.
Restrictions:
- only use on a short high-value set.
Action args:
{{"candidates":["N1","N2"]}}

Tool: probe_many
Purpose:
- Inspect and confirm a short candidate set in one bundled action.
Use when:
- several visible candidates need both node context and structural grounding together.
Interpretation:
- probe_many is the efficient action for turning a promising frontier into grounded evidence.
Action args:
{{"candidates":["N1","N2"]}}

Tool: accept_many
Purpose:
- Mark one or more currently visible candidates as accepted same-layer components in the working set.
Use when:
- direct structural support is already strong enough,
- or a candidate has already been confirmed and should now be added.
Restrictions:
- use only visible candidate aliases.
- keep the list short and concrete.
Action args:
{{"candidates":["N1","N2"]}}

Tool: hold_many
Purpose:
- Mark one or more visible candidates as unresolved but still plausible.
Use when:
- evidence is promising but not strong enough for acceptance,
- or the node should remain tracked without promotion.
Action args:
{{"candidates":["N1","N2"]}}

Tool: reject_many
Purpose:
- Mark one or more visible candidates as rejected for the current target.
Use when:
- evidence shows they are not same-layer components,
- or they are clearly evidence-only, context-only, or contradicted by the available support.
- If a candidate is still plausible but unresolved, prefer `hold_many`.
Action args:
{{"candidates":["N1","N2"]}}

Tool: save_memory
Purpose:
- Save one valid fallback route for later use.
Use when:
- a route looks useful but should not be executed right now,
- or you want to preserve a concrete backup action before continuing elsewhere.
Restrictions:
- route must itself be exactly one valid action object.
- do not save vague or placeholder routes.
Action args:
{{"route":{{"tool":"check","args":{{"candidate":"N1"}},"why":"confirm later"}},"label":"confirm N1 later"}}

Tool: memory
Purpose:
- Execute a saved fallback route.
Philosophy:
- fallback is not failure in a bad sense; it is deliberate branch discipline.
- the system wants you to choose fallback when the current frontier is weak.
- you should be willing to fail early rather than invent structure that the graph has not supported.
Action args:
{{"memory_id":"M1"}}

ROUTING PRIORITY WHEN CANDIDATES ARE ALREADY CONFIRMED
- If a visible candidate has direct structural support (`dir` shows non-zero support, or routes include `check_supported`, `seed_p527`, `scan_direct_p527`, `scan_inverse_p361`, or `preview_p361_target`), do not keep repeating the same scan on T.
- Promote confirmed candidates into the working set with `accept_many`.
- Use `hold_many` for plausible but still unresolved candidates.
- Use `reject_many` only for nodes that are clearly evidence-only, cross-layer, or explicitly unsupported after inspection.
- Repeating `scan(T,dir)` after it already returned the same frontier is bad routing.
- If the target has little or no immediate part frontier, prefer `scan(T,rev)` before drifting into weak context routes.
- After a candidate is confirmed, the next useful move is usually one of:
  - `accept_many`
  - `open`
  - `probe_many`
  - `expand_neighbors`
  - `save_memory`
- Only use `check` again if the candidate has not already been structurally confirmed.

FREE-NODE DIGGING
You are allowed to dig outward from a visible non-target node.
When a free node looks promising, use `expand_neighbors` to inspect its local neighborhood.
This is how you move beyond the target’s immediate seed without blindly repeating `scan(T,dir)`.

Tool: expand_neighbors
Purpose:
- Explore outward from one visible node by fetching its nearby graph neighbors through selected relations.
Use when:
- the target scan already returned a frontier,
- a free node looks promising,
- you want to dig from that node instead of repeating the same target scan.
Interpretation:
- neighbors are evidence and hypotheses until checked against the original target.
Action args:
{{"anchor":"N1","k":6}}

WHEN TO PREFER FALLBACK
You should be tempted by fallback whenever:
- direct target evidence was sparse,
- the last scan returned mainly class hints or characteristics,
- opened nodes did not tie back to the target,
- multiple checks came back unsupported,
- you are about to take a second or third uncertain hop,
- continuing would mostly gather context rather than same-layer parts.

WHEN TO STOP
Stop when the useful frontier is exhausted or insufficient.
Good stop conditions include:
- the obvious strong routes have been used,
- new actions would mostly repeat weak scans,
- remaining candidates are only class-hinted or characteristic-driven,
- there is not enough evidence to justify more graph cost,
- further digging would blur same-layer constraints.
Stopping is correct behavior.

HOW TO JUDGE SAME-LAYER DETAIL
You must keep the granularity disciplined.
Ask yourself:
- Does this candidate sit at the same decomposition depth as the others already supported?
- Is it a direct part of the original target, not merely a property, subtype, parent, or subpart?
- Would accepting it force the list into mixed layers?
If yes to mixed layers, do not promote it.

HOW TO USE LAST RESULTS
The `last` section is critical.
Use it to avoid redundant actions.
Mentally inspect:
- what route was just explored,
- whether it returned strong structural candidates or only hints,
- whether a candidate was already checked or opened,
- whether the branch should now be confirmed, opened, abandoned, or replaced by memory.

EFFICIENCY RULES
- Prefer the smallest move that can materially improve the frontier.
- Reuse known state.
- Avoid asking for data already present.
- Do not request more than 3 actions.
- Keep actions tightly justified.
- Strong short branches beat long speculative ones.

DECISION MODES
You must return one of three decisions.

1. act
Use when there is a worthwhile next step.
Return 1–3 actions.

2. fallback
Use when a memory route is clearly better than continuing this branch.
Return `memory_id` and leave `actions` empty.

3. stop
Use when the frontier is not worth continuing.
Return an empty action list.

OUTPUT FORMAT
Return ONLY JSON, no markdown, no extra text.
Use this exact schema:
{{
  "decision": "act|fallback|stop",
  "why": "short reason",
  "actions": [
    {{
      "tool": "seed|open|open_many|scan|survey|check|check_many|probe_many|memory|accept_many|hold_many|reject_many|save_memory|expand_neighbors",
      "args": {{}},
      "why": "short"
    }}
  ],
  "memory_id": "optional or null"
}}

OUTPUT DISCIPLINE
- `why` values must be short.
- If `decision` is `fallback`, set `memory_id` and keep `actions` as [].
- If `decision` is `act` or `stop`, `memory_id` should be null or omitted.
- If `decision` is `stop`, `actions` must be [].
- Use only the action arg shapes defined above.
- `accept_many`, `hold_many`, and `reject_many` use: {{"candidates":["N1","N2"]}}
- `save_memory` uses: {{"route":{{...one valid action object...}},"label":"short label"}}
- `expand_neighbors` uses: {{"anchor":"N1","k":6}}
- Do not emit prose outside JSON.
""".strip()

FINAL_REVIEW_SYSTEM_PROMPT = """
You are GraphPartAgentFinalReview.

You receive the FULL candidate bucket state around one target object.
You also receive the original user prompt for that target.
Candidate ids are review handles like R1, R2, R3.
Each candidate includes a label, description.

YOUR JOB
Rearrange the candidates so that `accept` becomes a coherent bundle of same-level real world subparts of the target.
You are not preserving the previous buckets. You may move candidates freely between accept, reject, and hold.

PRIMARY GOAL
Keep concrete physical parts of the target at one shared granularity level.
When several concrete peer parts clearly fit, accept all of them.
This is the final pass, so avoid `hold` unless absolutely necessary.
Reject only when the candidate is clearly wrong-layer, abstract, evidence-only, generic, or unrelated to the prompt.

WHAT ACCEPT MUST LOOK LIKE
- accepted items should be named physical objects with real-world descriptions
- accepted items should feel like peers at the same granularity
- when several concrete peer parts exist, prefer to keep that bundle together
- never leave only one generic category in accept when several concrete named parts are available
- if a generic umbrella label and several concrete members are both present, prefer the concrete members
- it is fine for `accept` to contain many items when they are concrete peer parts
- use `hold` when a candidate is plausible but the exact layer or specificity is still unclear

WHAT MUST NOT STAY IN ACCEPT
- generic categories or umbrella labels like organelle, component, structure, system, element, cellular component
- abstract or non-physical things like processes, properties, qualities, functions, models, classes, categories
- labels whose description defines a class or umbrella group instead of one concrete object

HOW TO JUDGE
- use basic common sense about physical objects and parts
- keep asking: does this read like a real thing someone could point at, see, draw, or watch acting in the world?
- require some sensible relation to the original prompt or target object, even if that relation is broad
- read the description literally - you judge based on them
- use the surrounding candidate set as context for the shared group you are trying to form
- prefer concrete named parts over generic labels
- use routes/support as meaningful evidence, not just weak hints
- move concrete peer parts up from hold or reject when they fit the shared bundle

IMPORTANT:
- Do NOT accept something merely because the description says it does something.
- Function words alone are not enough.
- A candidate should be accepted when it reads like a concrete physical object or named physical subpart and has any sensible relation to the original prompt.
- Treat an object as physical if the text describes visible form, material, shape, placement, subparts, or a thing that performs actions as a real object.
- Do not reject something solely because it is broad or familiar if it still reads like a concrete physical subpart.
- If the description mainly defines a process, force, property, capability, metric, schedule, rate, role, or abstract relation, reject it.
- In this final pass, prefer a real decision.
- If something is a physical object with a sensible relation to the prompt, accept it.
- If not, reject it.
- It is fine to accept every candidate when every description is clearly a concrete same-layer peer part related to the prompt.

Return ONLY JSON:
{
  "accept": ["R#"],
  "hold": ["R#"],
  "reject": ["R#"],
  "notes": {"R#": "short reason"}
}

Rules:
- notes must be short single-line reasons
- accepted ids must not appear in reject or hold
- prefer `hold: []` in the final pass
- accept multiple concrete peers when they clearly form the same bundle
- no markdown
- no commentary outside JSON
""".strip()

ENTITY_SELECTION_SYSTEM_PROMPT = """
You are GraphPartAgentTargetResolver.

You receive a user target phrase and up to 10 candidate handles.
Each candidate id is a compact handle like K1, K2, K3.
Your job is to pick the SINGLE candidate handle that most likely represents the intended object.

MODE RULE
- In non-thinking mode, output only the final JSON object.
- In thinking mode, any reasoning must stay inside <think>...</think>, and the actual answer after </think> must be only the final JSON object.

IMPORTANT BEHAVIOR
- This is a routing step before the main graph search.
- You are not choosing the most interesting item. You are choosing the most likely intended item.
- Prefer the candidate whose label and description best match the phrase.
- If several candidates are plausible, prefer the one with broader Wikidata maturity and public grounding,
  especially higher sitelink count.
- Sitelinks are a tie-break signal, not a substitute for semantic match.
- Do not blindly pick the first search result.
- Reject obviously wrong entity types even if they have many sitelinks.

HOW TO DECIDE
- First, match the user phrase to candidate labels/descriptions.
- Then inspect small context hints like instance-of labels.
- Then use sitelinks as a preference among semantically similar choices.
- If the phrase is broad, choose the canonical object item rather than a niche derivative unless the description clearly fits better.

Return ONLY JSON:
{{
  "id": "K#",
  "why": "VERY short reason - 1 - 2 senctances max",
}}

Rules:
- `id` must be one of the INPUT candidate ids.
- No markdown.
- No commentary outside JSON.
""".strip()

TARGET_FORMATTER_SYSTEM_PROMPT = """
You convert a broken or verbose draft into one strict JSON object for target selection.
Return only JSON.

Valid schema:
{"id":"K#","why":"short reason","alts":["K#","K#"]}

Rules:
- id must be one of the input candidate ids
- why must be short
- alts must be a short list or []
- no markdown
- no prose outside JSON
""".strip()

AGENT_SYSTEM_PROMPT_COMPACT = """
You are GraphPartAgent.

You are a routing agent for a Wikidata / WDQS component search.
Your job is to choose the next graph actions, not to answer from world knowledge.

PRIMARY GOAL
Find SAME-LAYER components of the target while keeping layers clean.
A valid component must be supported by the graph as a peer-level part of the original target.

CORE RULES
- Never use remembered facts as evidence.
- Never promote evidence nodes, context nodes, class hints, or subparts as components unless later support ties them to the original target.
- Prefer the smallest action set that materially improves the frontier.
- Avoid repeating actions that the visible state already made unnecessary.
- Prefer fallback or stop over speculative multi-hop digging.
- Use at most 3 actions.

EVIDENCE ORDER
1. Strongest: direct part structure around the original target.
   Direct has-part and inverse part-of support are the main acceptance routes.
2. Open-node support.
   Opened nodes help interpret whether something is a real part, context node, class hint, or evidence-only node.
3. Class hints.
   Useful for hypotheses, weak for promotion.
4. Characteristic hints.
   Useful for discovery, not direct promotion.
5. Upward/context routes.
   Recovery only.

ALIAS RULES
- Use only concrete aliases visible in INPUT.
- `anchor` must be `T` or a visible node alias like `N1`.
- Candidate lists must contain only visible aliases.
- Never copy placeholder text such as `N#`, `T|N#`, or `dir|rev|cls|char|up`.

TOOLS
`seed`
- Reload target seed only when seed is missing or clearly needed.
- args: {"target":"T"}

`open`
- Inspect one node to learn what it is and whether it ties back to the target.
- args: {"node":"T"} or {"node":"N1"}

`open_many`
- Inspect 2-4 nodes side by side.
- Keep it short and purposeful.
- args: {"nodes":["N1","N2"]}

`scan`
- Run one route around an anchor.
- args: {"anchor":"T","route":"dir","k":4}
- routes:
  - `dir`: strongest local structural scan for real component evidence.
  - `rev`: incoming part-of scan through broader target classes when direct target edges are thin.
  - `cls`: class-level part hypotheses.
  - `char`: characteristic/evidence discovery, not direct promotion.
  - `up`: context recovery when the branch is weak.

`survey`
- Wider bundled frontier fetch around one anchor when the frontier is too thin.
- args: {"anchor":"T","k":6}

`check`
- Confirm one candidate against the original target.
- args: {"candidate":"N1"}

`check_many`
- Confirm 2-4 promising candidates from the same branch.
- args: {"candidates":["N1","N2"]}

`probe_many`
- Bundled inspect+confirm for a short candidate set.
- Good when several visible nodes need grounding quickly.
- args: {"candidates":["N1","N2"]}

`accept_many`
- Promote already-supported same-layer candidates into the working set.
- args: {"candidates":["N1","N2"]}

`hold_many`
- Keep plausible but unresolved candidates tracked.
- args: {"candidates":["N1","N2"]}

`reject_many`
- Reject clearly wrong, unsupported, evidence-only, or cross-layer nodes.
- If still plausible, prefer `hold_many`.
- args: {"candidates":["N1","N2"]}

`save_memory`
- Save one concrete fallback action for later.
- `route` must itself be one valid action object.
- args: {"route":{"tool":"check","args":{"candidate":"N1"},"why":"confirm later"},"label":"confirm N1 later"}

`memory`
- Execute a saved fallback route.
- Use when the current branch is weak and an existing backup is better.
- args: {"memory_id":"M1"}

`expand_neighbors`
- Explore outward from one visible non-target node instead of repeating target scans.
- Neighbors are still only evidence/hypotheses until tied back to the original target.
- args: {"anchor":"N1","k":6}

ROUTING RULES
- If a visible candidate already has direct structural support, do not keep repeating `scan(T,dir)`.
- After strong support appears, the next useful move is usually `check`, `probe_many`, `accept_many`, `open`, `expand_neighbors`, or `save_memory`.
- Prefer `rev` when the target has little or no immediate part frontier and you need incoming part-of evidence from outside.
- Use `cls`, `char`, and `up` as discovery/recovery tools, not as promotion tools.
- Prefer `expand_neighbors` when a visible non-target node looks promising and target-centric scans are becoming repetitive.
- Prefer `fallback` when continuing would mostly gather weak context.
- Use `stop` when the useful frontier is exhausted.

DECISIONS
- `act`: return 1-3 actions.
- `fallback`: return `memory_id` and `actions: []`.
- `stop`: return `actions: []`.

OUTPUT
Return ONLY JSON:
{
  "decision": "act|fallback|stop",
  "why": "short reason",
  "actions": [
    {
      "tool": "seed|open|open_many|scan|survey|check|check_many|probe_many|memory|accept_many|hold_many|reject_many|save_memory|expand_neighbors",
      "args": {},
      "why": "short"
    }
  ],
  "memory_id": "optional or null"
}

OUTPUT RULES
- Keep all `why` values short.
- If `decision` is `fallback`, set `memory_id` and keep `actions` empty.
- If `decision` is `stop`, `actions` must be empty.
- If `decision` is `act`, keep `memory_id` empty, null, or omitted.
- Output no markdown and no prose outside JSON.
""".strip()

ACTIONS_FORMATTER_SYSTEM_PROMPT = """
You convert a broken or verbose draft into one strict JSON object for action routing.
Return only JSON.

Valid schema:
{
  "decision":"act|fallback|stop",
  "why":"short reason",
  "actions":[
    {
      "tool":"seed|open|open_many|scan|survey|check|check_many|probe_many|memory|accept_many|hold_many|reject_many|save_memory|expand_neighbors",
      "args":{},
      "why":"short"
    }
  ],
  "memory_id":"optional or null"
}

Rules:
- use only allowed tool names
- use only concrete values already present in the INPUT
- do not copy schema placeholders into final JSON
- if decision is fallback, actions must be [] and memory_id must be set
- if decision is act or stop, memory_id should be null or omitted
- accept_many, hold_many, and reject_many use {"candidates":["N1","N2"]}
- save_memory uses {"route":{...one valid action object...},"label":"short label"}
- expand_neighbors uses {"anchor":"N1","k":6}
- no markdown
- no prose outside JSON
""".strip()

FINAL_REVIEW_FORMATTER_SYSTEM_PROMPT = """
You convert a broken or verbose draft into one strict JSON object for final review.
Return only JSON.

Valid schema:
{"accept":["R#"],"reject":["R#"],"hold":["R#"],"notes":{"R#":"short reason"}}

Rules:
- notes values must be short, single-line, plain strings
- do not include lists, paragraphs, or multiline explanations inside notes
- keep only the intended classification result
- no markdown
- no prose outside JSON
""".strip()

VISUAL_QUERY_REFINER_SYSTEM_PROMPT = """
You are VisualQueryRefiner.

You receive one original mother object prompt and the accepted stage-1 component list.
Each component has a QID, label, and short stage-1 description.

YOUR JOB
Decide whether any component needs a fatter image-search query so downstream Wikimedia and DDG search
is more specific. This is a contextual query rewrite step only.

IMPORTANT BEHAVIOR
- Keep the default label when extra context is not needed.
- Only emit a query for a component when the wider phrase will likely improve image search precision.
- Use only context visible in the input: the mother object prompt, the target label/description, the stage-1
  component descriptions, and the other accepted components.
- Do not invent facts from memory.
- Prefer short concrete search phrases like "train bogie", "cell nucleus", "airplane landing gear".
- Do not pad with adjectives unless they are directly grounded in the input.

OUTPUT
Return ONLY JSON:
{
  "queries": [
    {"qid": "Q#", "query": "rewritten search phrase"}
  ]
}

RULES
- `queries` may be empty.
- Emit only components that should actually be rewritten.
- `qid` must be one of the input component qids.
- `query` must be short plain text.
- No markdown.
- No prose outside JSON.
""".strip()

WIKIPEDIA_SECTION_SELECT_SYSTEM_PROMPT = """
You are WikipediaSectionSelector.

You receive:
- the original mother object prompt,
- one stage-1 component with a short description,
- a very short lead snippet from the linked Wikipedia page,
- a short list of available section headings from that page.

YOUR JOB
Choose the best 1 or 2 section ids to fetch for a QUICK visual-description pass.

SELECTION RULES
- Prefer sections that are likely to describe visible structure, form, arrangement, physical layout, or named visible subparts.
- Prefer the lead section `0` when it already gives useful visual grounding.
- You may return only `["0"]` if the lead is enough.
- Add one more section only when it clearly looks useful.

AVOID
- history, etymology, operation, economics, scheduling, performance, use cases, culture, politics, taxonomy
- long function-only sections
- sections that are unlikely to help describe appearance

OUTPUT
Return ONLY JSON:
{
  "sections": ["0", "3"]
}

RULES
- return 1 or 2 ids only
- ids must come from the visible input list
- prefer compact choices
- no markdown
- no prose outside JSON
""".strip()

WIKIPEDIA_VISUAL_EXTRACT_SYSTEM_PROMPT = """
You are WikipediaVisualExtractor.

You receive:
- the original mother object prompt,
- one stage-1 component with its short stage-1 description,
- a compact list of sibling components for context,
- only the selected short text from 1 or 2 Wikipedia sections for that component.

YOUR JOB
Extract and rewrite ONLY the text that describes how the component looks.
This is a FILTER task, not a freeform encyclopedia task.

WHAT TO KEEP
- shape, silhouette, geometry, arrangement
- visible subparts
- surface appearance, texture, color, material when the page text states it
- relative size statements when they help visualize the part
- distinctive visible behaviors that change appearance, only if explicitly stated in the page text

WHAT TO EXCLUDE
- function-only explanation unless it directly clarifies visible form
- historical facts, discovery facts, taxonomy, chemistry, abstract theory
- guessed details not stated in the input

IMPORTANT DISCIPLINE
- Work from the page text only, using the stage-1 context only to resolve ambiguity.
- Do not add outside knowledge.
- Produce one short canonical visual description in plain prose.
- Keep it compact and fast to read.

OUTPUT
Return ONLY JSON:
{
  "visual_description": "short appearance-focused prose"
}

RULES
- keep one plain string only
- keep it reasonably short
- no evidence lists
- no markdown
- no prose outside JSON
""".strip()

VISUAL_DESCRIPTION_REFINER_SYSTEM_PROMPT = """
You are VisualDescriptionRefiner.

You receive:
- the original mother object prompt,
- one stage-1 component and its short stage-1 description,
- a canonical visual description extracted from Wikipedia,
- candidate image descriptions gathered from Wikimedia and DDG.

YOUR JOB
Take the canonical Wikipedia visual description as the main base description.
Then ADD any missing visible details from the candidate image descriptions into that base description.

This is not a rewrite-from-scratch task.
This is not an averaging task.
This is a base-plus-additions task.

IMPORTANT DISCIPLINE
- Treat the Wikipedia description as the base canon.
- Keep everything already useful in the base description.
- Look through ALL candidate image descriptions and find visible details that are missing from the base description.
- Add those missing visible details when they are compatible with the base description.
- Prefer concrete visible details such as shape, arrangement, frame, wheels, axle layout, housing, suspension, linkage, materials, surfaces, proportions, and named visible subparts.
- If several candidates repeat the same visual trait, that is a strong signal to include it.
- If a candidate detail conflicts with the base description or looks too specific or one-off, leave it out.
- Do not use remembered world knowledge.
- Do not invent details that are not supported by either the base canon or the candidate descriptions.
- Prefer stable visible traits over niche or conflicting one-off details.
- The output should usually look like: base description, then missing details added in naturally.
- Do not return the base description unchanged unless the candidates truly add nothing.
- Output one short tightened description as one string.

OUTPUT
Return ONLY JSON:
{
  "refined_description": "short appearance-focused prose"
}

RULES
- keep one plain string only
- preserve the base description structure, then extend it with missing visible details
- no evidence fields
- no markdown
- no prose outside JSON
""".strip()

VISUAL_QUERY_REFINER_FORMATTER_SYSTEM_PROMPT = """
You convert a broken or verbose draft into one strict JSON object for visual query refinement.
Return only JSON.

Valid schema:
{"queries":[{"qid":"Q#","query":"short plain text"}]}

Rules:
- queries may be []
- qid values must be copied from the input
- query values must be short strings
- no markdown
- no prose outside JSON
""".strip()

WIKIPEDIA_SECTION_SELECT_FORMATTER_SYSTEM_PROMPT = """
You convert a broken or verbose draft into one strict JSON object for Wikipedia section selection.
Return only JSON.

Valid schema:
{"sections":["0","3"]}

Rules:
- return 1 or 2 strings
- ids must be copied from the input
- no markdown
- no prose outside JSON
""".strip()

WIKIPEDIA_VISUAL_EXTRACT_FORMATTER_SYSTEM_PROMPT = """
You convert a broken or verbose draft into one strict JSON object for Wikipedia visual extraction.
Return only JSON.

Valid schema:
{"visual_description":"plain text"}

Rules:
- keep only one string field
- no markdown
- no prose outside JSON
""".strip()

VISUAL_DESCRIPTION_REFINER_FORMATTER_SYSTEM_PROMPT = """
You convert a broken or verbose draft into one strict JSON object for visual description refinement.
Return only JSON.

Valid schema:
{"refined_description":"plain text"}

Rules:
- keep only one string field
- no markdown
- no prose outside JSON
""".strip()

class JsonParseError(RuntimeError):
    pass


class SchemaValidationError(RuntimeError):
    pass


@dataclass
class PrefixCacheEntry:
    prefix_len: int
    past_key_values: Any
    hits: int = 0


class QwenWorker:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        *,
        model: Any = None,
        tokenizer: Any = None,
        llm: Any = None,
        backend: Optional[str] = None,
        stage_io_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.lock = threading.Lock()
        self.logger = logging.getLogger("qwen_worker")
        self.backend = str(backend or "").strip().lower() or ("vllm" if llm is not None else "transformers")
        self.llm = llm
        self.model = None
        self.gpu_gc_every = max(0, int(os.environ.get("QWEN_GPU_GC_EVERY", "0") or 0))
        self._gpu_gc_counter = 0
        self.fast_visual_fallbacks = str(os.environ.get("QWEN_FAST_VISUAL_FALLBACKS", "1")).strip().lower() not in {"0", "false", "no", "off"}
        self.prefix_cache_enabled = str(os.environ.get("QWEN_PREFIX_CACHE", "1")).strip().lower() not in {"0", "false", "no", "off"}
        self.prefix_cache_max_entries = max(0, int(os.environ.get("QWEN_PREFIX_CACHE_MAX_ENTRIES", "8") or 8))
        self.prefix_cache_min_tokens = max(32, int(os.environ.get("QWEN_PREFIX_CACHE_MIN_TOKENS", "96") or 96))
        self.prefix_cache: "OrderedDict[str, PrefixCacheEntry]" = OrderedDict()
        self.stage_io_dir = str(stage_io_dir or os.environ.get("QWEN_STAGE_IO_DIR", "") or "").strip()
        self._stage_io_bundle: Dict[str, Any] = {
            "stage_io_dir": self.stage_io_dir,
            "model_id": model_name,
            "backend": self.backend,
        }

        if tokenizer is not None:
            self.tokenizer = tokenizer
            self.logger.info("Using injected tokenizer for %s", model_name)
        else:
            self.logger.info("Loading tokenizer for %s", model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        if self.backend == "vllm" and self.llm is None:
            try:
                import qwentest  # local module

                self.llm = qwentest._build_vllm_text_engine(model_name)
                self.tokenizer = qwentest._get_vllm_tokenizer(self.llm)
                self.logger.info("Using vllm backend for %s", model_name)
            except Exception as e:
                self.logger.warning("vllm unavailable for %s, falling back to transformers: %s", model_name, e)
                self.backend = "transformers"
                self.llm = None

        if self.backend != "vllm" and model is not None:
            self.model = model
            self.logger.info("Using injected model for %s", model_name)
        elif self.backend != "vllm":
            self.logger.info("Loading model for %s", model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            )
        if self.model is not None:
            self.model.eval()
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.think_close_ids = self._encode_plain_text("</think>")
        self.logger.info("Model loaded (backend=%s)", self.backend)
        if self.prefix_cache_enabled and self.prefix_cache_max_entries > 0:
            self.logger.info(
                "Prefix cache enabled (max_entries=%s, min_tokens=%s)",
                self.prefix_cache_max_entries,
                self.prefix_cache_min_tokens,
            )

    def _write_stage_io(
        self,
        *,
        task: str,
        mode: str,
        payload: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        rendered_input_text: str,
        raw_text: str,
        parsed_payload: Any,
    ) -> None:
        if not self.stage_io_dir:
            return
        try:
            import qwentest

            qwentest._write_qwen_stage_io_json(
                self._stage_io_bundle,
                tokenizer_or_processor=self.tokenizer,
                debug_label=f"c2_{str(task or 'task').strip()}",
                job_key=f"{str(task or 'task').strip()}_{int(time.time() * 1000)}",
                prompt_text=str(user_prompt or ""),
                raw_text=str(raw_text or ""),
                parsed_obj={},
                parsed_payload=parsed_payload,
                stage_kind="text",
                rendered_input_text=str(rendered_input_text or ""),
                system_prompt=str(system_prompt or ""),
                user_prompt=str(user_prompt or ""),
                extra={
                    "mode": str(mode or ""),
                    "task": str(task or ""),
                    "job_input": payload if isinstance(payload, dict) else {},
                },
            )
        except Exception:
            self.logger.exception("Failed to write stage io debug for task=%s", task)

    def _task_prompt(self, task: str, payload: Dict[str, Any], mode: str) -> tuple[str, str]:
        if task == "choose_target":
            return ENTITY_SELECTION_SYSTEM_PROMPT, self._build_choose_target_prompt(payload, mode)
        if task == "choose_actions":
            return AGENT_SYSTEM_PROMPT_COMPACT, self._build_choose_actions_prompt(payload, mode)
        if task == "final_review":
            return FINAL_REVIEW_SYSTEM_PROMPT, self._build_final_review_prompt(payload, mode)
        if task == "visual_query_refinement":
            return VISUAL_QUERY_REFINER_SYSTEM_PROMPT, self._build_visual_query_refinement_prompt(payload, mode)
        if task == "wikipedia_section_select":
            return WIKIPEDIA_SECTION_SELECT_SYSTEM_PROMPT, self._build_wikipedia_section_select_prompt(payload, mode)
        if task == "wikipedia_visual_extract":
            return WIKIPEDIA_VISUAL_EXTRACT_SYSTEM_PROMPT, self._build_wikipedia_visual_extract_prompt(payload, mode)
        if task == "visual_description_refinement":
            return VISUAL_DESCRIPTION_REFINER_SYSTEM_PROMPT, self._build_visual_description_refinement_prompt(payload, mode)
        raise ValueError(f"unsupported task: {task}")

    def _prefix_cache_key(self, task: str, mode: str) -> str:
        return f"{task}:{mode}"

    def _longest_common_prefix_len(self, left: List[int], right: List[int]) -> int:
        limit = min(len(left), len(right))
        idx = 0
        while idx < limit and left[idx] == right[idx]:
            idx += 1
        return idx

    def _derive_prefix_len(
        self,
        task: str,
        system_prompt: str,
        user_prompt: str,
        thinking_enabled: bool,
        prompt_ids: torch.Tensor,
    ) -> int:
        if task not in {
            "choose_target",
            "choose_actions",
            "final_review",
            "visual_query_refinement",
            "wikipedia_section_select",
            "wikipedia_visual_extract",
            "visual_description_refinement",
        }:
            return 0

        head, marker, tail = user_prompt.partition("INPUT=")
        if not marker or not tail.startswith("{"):
            return 0

        probe_prompt = head + marker + '{"$":0}'
        probe_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": probe_prompt},
        ]
        probe_inputs = self._apply_chat_template(probe_messages, thinking_enabled=thinking_enabled)
        try:
            probe_ids = probe_inputs["input_ids"][0].tolist()
        finally:
            del probe_inputs

        prefix_len = self._longest_common_prefix_len(prompt_ids[0].tolist(), probe_ids)
        if prefix_len < self.prefix_cache_min_tokens:
            return 0
        return prefix_len

    def _evict_prefix_cache_if_needed(self) -> None:
        if self.prefix_cache_max_entries <= 0:
            self.prefix_cache.clear()
            return
        while len(self.prefix_cache) > self.prefix_cache_max_entries:
            old_key, old_entry = self.prefix_cache.popitem(last=False)
            self.logger.info(
                "Evicting prefix cache entry %s (prefix_tokens=%s, hits=%s)",
                old_key,
                old_entry.prefix_len,
                old_entry.hits,
            )
            del old_entry
            self._gpu_gc()

    def _clone_past_key_values(self, past_key_values: Any) -> Any:
        return copy.deepcopy(past_key_values)

    def _prefill_prefix_cache(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        outputs = None
        try:
            with torch.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                return self._clone_past_key_values(outputs.past_key_values)
        finally:
            del outputs

    def _get_or_build_prefix_cache(
        self,
        task: str,
        mode: str,
        system_prompt: str,
        user_prompt: str,
        thinking_enabled: bool,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Optional[PrefixCacheEntry]:
        if not self.prefix_cache_enabled or self.prefix_cache_max_entries <= 0:
            return None

        key = self._prefix_cache_key(task, mode)
        cached = self.prefix_cache.get(key)
        if cached is not None:
            cached.hits += 1
            self.prefix_cache.move_to_end(key)
            return cached

        prefix_len = self._derive_prefix_len(task, system_prompt, user_prompt, thinking_enabled, input_ids)
        prompt_len = int(input_ids.shape[-1])
        if prefix_len <= 0 or prefix_len >= prompt_len:
            return None

        prefix_ids = input_ids[:, :prefix_len]
        prefix_attention = attention_mask[:, :prefix_len]
        past_key_values = self._prefill_prefix_cache(prefix_ids, prefix_attention)
        if past_key_values is None:
            return None

        entry = PrefixCacheEntry(prefix_len=prefix_len, past_key_values=past_key_values, hits=0)
        self.prefix_cache[key] = entry
        self.prefix_cache.move_to_end(key)
        self._evict_prefix_cache_if_needed()
        self.logger.info("Built prefix cache for %s (prefix_tokens=%s)", key, prefix_len)
        return entry

    def _dir_pair(self, value: Any) -> tuple[int, int]:
        text = str(value or "").strip()
        if "/" not in text:
            return 0, 0
        left, right = text.split("/", 1)
        try:
            return int(left), int(right)
        except Exception:
            return 0, 0

    def _unique_aliases(self, values: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for value in values:
            value = str(value).strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return out

    def _repair_action_result(self, result: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(result, dict):
            return result
        if str(result.get("decision", "")).strip().lower() != "act":
            return result

        state = payload.get("state", {})
        if not isinstance(state, dict):
            return result

        cand_rows = [row for row in state.get("cand", []) if isinstance(row, dict)]
        seed_rows = [row for row in state.get("seed", []) if isinstance(row, dict)]
        accepted_rows = [row for row in state.get("accepted", []) if isinstance(row, dict)]
        hold_rows = [row for row in state.get("hold", []) if isinstance(row, dict)]
        rejected_rows = [row for row in state.get("rejected", []) if isinstance(row, dict)]
        last_rows = [row for row in payload.get("last", []) if isinstance(row, dict)]
        step = int(payload.get("step", 0) or 0)

        cand_by_id: Dict[str, Dict[str, Any]] = {}
        for row in cand_rows:
            rid = str(row.get("id", "")).strip()
            if rid:
                cand_by_id[rid] = row

        accepted = {str(row.get("id", "")).strip() for row in accepted_rows if str(row.get("id", "")).strip()}
        held = {str(row.get("id", "")).strip() for row in hold_rows if str(row.get("id", "")).strip()}
        rejected = {str(row.get("id", "")).strip() for row in rejected_rows if str(row.get("id", "")).strip()}
        bucketed = accepted | held | rejected

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

        actions = result.get("actions", [])
        if not isinstance(actions, list):
            actions = []

        total_seed = len(seed_rows)
        route_recent = any(str(row.get("tool", "")).strip() in {"survey", "scan", "expand_neighbors"} and bool(row.get("ok", False)) for row in last_rows)
        small_seed = total_seed < 20

        strong_unaccepted: List[str] = []
        for rid, row in cand_by_id.items():
            if rid in bucketed:
                continue
            d1, d2 = self._dir_pair(row.get("dir", "0/0"))
            if d1 + d2 > 0:
                strong_unaccepted.append(rid)

        seed_unbucketed: List[str] = []
        for row in seed_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid not in bucketed:
                seed_unbucketed.append(rid)
        seed_unbucketed = self._unique_aliases(seed_unbucketed)

        generic_names = {"organelle", "component", "part", "structure", "system", "cellular component", "aircraft component"}
        no_frontier = not seed_rows and not cand_rows and not accepted_rows and not hold_rows and not rejected_rows
        last_duplicate = any("duplicate action skipped" in str(row.get("sum", "")).lower() for row in last_rows)

        if no_frontier:
            if step <= 2 and not route_recent:
                return {
                    "decision": "act",
                    "why": "empty frontier needs survey",
                    "actions": [{"tool": "survey", "args": {"anchor": "T", "k": 8}, "why": "recover empty frontier"}],
                    "memory_id": "",
                }
            if route_recent:
                memory_rows = [row for row in payload.get("memory", []) if isinstance(row, dict)]
                if memory_rows:
                    best_mid = str(memory_rows[0].get("id", memory_rows[0].get("mid", ""))).strip()
                    if best_mid:
                        return {"decision": "fallback", "why": "empty frontier after route action", "actions": [], "memory_id": best_mid}
                return {"decision": "stop", "why": "empty frontier after route action", "actions": [], "memory_id": ""}
            if last_duplicate:
                memory_rows = [row for row in payload.get("memory", []) if isinstance(row, dict)]
                if memory_rows:
                    best_mid = str(memory_rows[0].get("id", memory_rows[0].get("mid", ""))).strip()
                    if best_mid:
                        return {"decision": "fallback", "why": "duplicate action on empty frontier", "actions": [], "memory_id": best_mid}
                return {"decision": "stop", "why": "empty frontier exhausted", "actions": [], "memory_id": ""}

        def best_expand_anchor() -> Optional[str]:
            ranked: List[str] = []
            for row in cand_rows:
                rid = str(row.get("id", "")).strip()
                if rid and rid in accepted:
                    ranked.append(rid)
            for rid in ranked:
                name = str(cand_by_id.get(rid, {}).get("name", "")).strip().lower()
                if name and name not in generic_names:
                    return rid
            return ranked[0] if ranked else None

        def all_actions_repeat_bucketed() -> bool:
            if not actions:
                return False
            ok = False
            for action in actions:
                ids = extract_ids(action)
                tool = str(action.get("tool", "")).strip()
                if tool not in {"accept_many", "hold_many", "reject_many", "check", "check_many", "probe_many"}:
                    return False
                if not ids or any(rid not in bucketed for rid in ids):
                    return False
                ok = True
            return ok

        if small_seed and not route_recent and step <= 2:
            return {
                "decision": "act",
                "why": "explore early frontier",
                "actions": [{"tool": "survey", "args": {"anchor": "T", "k": 8}, "why": "explore more parts"}],
                "memory_id": "",
            }

        if all_actions_repeat_bucketed():
            if seed_unbucketed:
                return {
                    "decision": "act",
                    "why": "probe unresolved seed",
                    "actions": [{"tool": "probe_many", "args": {"candidates": seed_unbucketed[:4]}, "why": "resolve seed nodes"}],
                    "memory_id": "",
                }
            anchor = best_expand_anchor()
            if anchor and small_seed:
                return {
                    "decision": "act",
                    "why": "explore accepted child",
                    "actions": [{"tool": "expand_neighbors", "args": {"anchor": anchor, "k": 6}, "why": "look at children"}],
                    "memory_id": "",
                }
            if small_seed:
                fallback_route = "rev" if not any(self._row_has_target_structural_support(row) for row in cand_rows) else "dir"
                fallback_why = "search reverse part-of" if fallback_route == "rev" else "search more parts"
                return {
                    "decision": "act",
                    "why": "keep exploring target",
                    "actions": [{"tool": "scan", "args": {"anchor": "T", "route": fallback_route, "k": 4}, "why": fallback_why}],
                    "memory_id": "",
                }
            return {"decision": "stop", "why": "frontier exhausted", "actions": [], "memory_id": ""}

        if strong_unaccepted:
            if len(strong_unaccepted) == 1:
                return {
                    "decision": "act",
                    "why": "confirm direct candidate",
                    "actions": [{"tool": "check", "args": {"candidate": strong_unaccepted[0]}, "why": "confirm support"}],
                    "memory_id": "",
                }
            return {
                "decision": "act",
                "why": "confirm direct candidates",
                "actions": [{"tool": "check_many", "args": {"candidates": strong_unaccepted[:4]}, "why": "confirm support"}],
                "memory_id": "",
            }

        if seed_unbucketed and small_seed:
            return {
                "decision": "act",
                "why": "probe unresolved seed",
                "actions": [{"tool": "probe_many", "args": {"candidates": seed_unbucketed[:4]}, "why": "resolve seed nodes"}],
                "memory_id": "",
            }

        return result

    def _gpu_gc(self, force: bool = False) -> None:
        if not force:
            if self.gpu_gc_every <= 0:
                return
            self._gpu_gc_counter += 1
            if self._gpu_gc_counter % self.gpu_gc_every != 0:
                return
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass

    def _mini(self, value: Any, n: int = 56) -> str:
        text = str(value or "").strip()
        if len(text) <= n:
            return text
        return text[: max(0, n - 1)].rstrip() + "…"

    def _prompt_payload(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if task == "choose_target":
            rows = []
            for row in payload.get("candidates", [])[:10]:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "id": row.get("id", ""),
                    "n": self._mini(row.get("name", ""), 40),
                    "d": self._mini(row.get("d", ""), 60),
                    "sl": row.get("sl", 0),
                    "i": (row.get("i", []) or [])[:2],
                })
            return {
                "t": self._mini(payload.get("target_text", ""), 80),
                "c": rows,
            }

        if task == "final_review":
            rows = []
            for row in payload.get("candidates", [])[:48]:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "id": row.get("id", ""),
                    "n": self._mini(row.get("name", ""), 40),
                    "d": self._mini(row.get("d", ""), 96),
                    "dir": row.get("dir", "0/0"),
                    "rt": (row.get("rt", []) or [])[:3],
                    "cls": (row.get("cls", []) or [])[:1],
                    "eo": (row.get("eo", []) or [])[:1],
                })
            t = payload.get("target", {}) or {}
            return {
                "t": {
                    "id": t.get("id", "T"),
                    "n": self._mini(t.get("name", ""), 40),
                    "d": self._mini(t.get("d", ""), 80),
                },
                "c": rows,
            }

        if task == "visual_query_refinement":
            target = payload.get("target", {}) or {}
            rows = []
            for row in payload.get("components", [])[:24]:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "qid": row.get("qid", ""),
                    "n": self._mini(row.get("label", ""), 48),
                    "d": self._mini(row.get("description", ""), 120),
                    "note": self._mini(row.get("note", ""), 80),
                })
            return {
                "t": {
                    "prompt": self._mini(target.get("prompt", payload.get("target_prompt", "")), 160),
                    "n": self._mini(target.get("label", ""), 48),
                    "d": self._mini(target.get("description", ""), 120),
                },
                "c": rows,
            }

        if task == "wikipedia_section_select":
            component = payload.get("component", {}) or {}
            page = payload.get("page", {}) or {}
            rows = []
            for row in payload.get("sections", [])[:12]:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "id": str(row.get("id", "")),
                    "n": self._mini(row.get("title", ""), 48),
                })
            return {
                "t": {"prompt": self._mini(payload.get("target_prompt", ""), 72)},
                "c": {
                    "qid": component.get("qid", ""),
                    "n": self._mini(component.get("label", ""), 40),
                    "d": self._mini(component.get("description", ""), 72),
                },
                "p": {
                    "title": self._mini(page.get("title", ""), 56),
                    "lead": self._mini(page.get("lead", ""), 220),
                },
                "s": rows,
            }

        if task == "wikipedia_visual_extract":
            component = payload.get("component", {}) or {}
            page = payload.get("page", {}) or {}
            return {
                "t": {
                    "prompt": self._mini(payload.get("target_prompt", ""), 72),
                    "parts": self._mini(payload.get("parts_context", ""), 160),
                },
                "c": {
                    "qid": component.get("qid", ""),
                    "n": self._mini(component.get("label", ""), 40),
                    "d": self._mini(component.get("description", ""), 72),
                    "note": self._mini(component.get("note", ""), 32),
                },
                "p": {
                    "title": self._mini(page.get("title", ""), 64),
                    "url": self._mini(page.get("url", ""), 72),
                    "sections": self._mini(page.get("sections", ""), 48),
                    "text": self._mini(page.get("text", ""), 900),
                },
            }

        if task == "visual_description_refinement":
            component = payload.get("component", {}) or {}
            rows = []
            for row in payload.get("candidates", [])[:6]:
                if not isinstance(row, dict):
                    continue
                rows.append({
                    "id": row.get("id", ""),
                    "src": self._mini(row.get("source_kind", ""), 16),
                    "d": self._mini(row.get("description", ""), 96),
                })
            return {
                "t": {"prompt": self._mini(payload.get("target_prompt", ""), 72)},
                "c": {
                    "qid": component.get("qid", ""),
                    "n": self._mini(component.get("label", ""), 40),
                    "d": self._mini(component.get("description", ""), 80),
                    "note": self._mini(component.get("note", ""), 48),
                },
                "base": self._mini(payload.get("base_description", ""), 520),
                "cand": rows,
            }

        state = payload.get("state", {}) or {}
        target = payload.get("target", {}) or {}

        seed_rows = []
        for row in state.get("seed", [])[:6]:
            if not isinstance(row, dict):
                continue
            seed_rows.append({
                "id": row.get("id", ""),
                "k": row.get("kind", ""),
                "n": self._mini(row.get("name", ""), 32),
                "d": self._mini(row.get("d", ""), 48),
            })

        cand_rows = []
        for row in state.get("cand", [])[:8]:
            if not isinstance(row, dict):
                continue
            cand_rows.append({
                "id": row.get("id", ""),
                "n": self._mini(row.get("name", ""), 32),
                "d": self._mini(row.get("d", ""), 48),
                "dir": row.get("dir", "0/0"),
                "rt": (row.get("rt", []) or [])[:3],
                "cls": (row.get("cls", []) or [])[:1],
                "eo": (row.get("eo", []) or [])[:1],
            })

        opened_rows = []
        for row in state.get("opened", [])[:2]:
            if not isinstance(row, dict):
                continue
            opened_rows.append({
                "id": row.get("id", ""),
                "n": self._mini(row.get("name", ""), 32),
                "p361": (row.get("p361", []) or [])[:2],
                "p527": (row.get("p527", []) or [])[:2],
            })

        accepted_rows = [{"id": row.get("id", "")} for row in state.get("accepted", [])[:16] if isinstance(row, dict)]
        hold_rows = [{"id": row.get("id", "")} for row in state.get("hold", [])[:16] if isinstance(row, dict)]
        rejected_rows = [{"id": row.get("id", "")} for row in state.get("rejected", [])[:16] if isinstance(row, dict)]

        memory_rows = []
        for row in payload.get("memory", [])[:3]:
            if not isinstance(row, dict):
                continue
            memory_rows.append({
                "id": row.get("id", ""),
                "go": self._mini(row.get("go", ""), 40),
                "p": row.get("p", 0),
            })

        last_rows = []
        for row in payload.get("last", [])[:1]:
            if not isinstance(row, dict):
                continue
            compact = {
                "tool": row.get("tool", ""),
                "ok": row.get("ok", False),
            }
            if "route" in row:
                compact["route"] = row.get("route", "")
            if "anchor" in row:
                compact["anchor"] = row.get("anchor", "")
            if "candidate" in row:
                compact["candidate"] = row.get("candidate", "")
            if "nodes" in row:
                compact["nodes"] = (row.get("nodes", []) or [])[:4]
            if "dir" in row:
                compact["dir"] = row.get("dir", "")
            if "cls" in row:
                compact["cls"] = (row.get("cls", []) or [])[:2]
            compact["sum"] = self._mini(row.get("sum", ""), 72)
            last_rows.append(compact)

        return {
            "step": payload.get("step", 0),
            "t": {
                "id": target.get("id", "T"),
                "n": self._mini(target.get("name", ""), 40),
                "d": self._mini(target.get("d", ""), 60),
            },
            "s": {
                "seed": seed_rows,
                "cand": cand_rows,
                "op": opened_rows,
                "a": accepted_rows,
                "h": hold_rows,
                "r": rejected_rows,
            },
            "m": memory_rows,
            "last": last_rows,
            "h": payload.get("hints", {}) or {},
        }

    def _parse_dir_support(self, value: Any) -> tuple[int, int]:
        if isinstance(value, str) and "/" in value:
            left, right = value.split("/", 1)
            try:
                return int(left), int(right)
            except Exception:
                return 0, 0
        return 0, 0

    def _common_sense_terms(self, text: Any) -> Set[str]:
        raw = str(text or "").lower()
        tokens = re.findall(r"[a-z0-9]+", raw)
        stop = {
            "the", "and", "for", "with", "from", "into", "that", "this", "these", "those", "their",
            "its", "his", "her", "our", "your", "are", "was", "were", "been", "being", "have", "has",
            "had", "not", "but", "use", "used", "using", "type", "kind", "form", "object", "thing",
            "part", "parts", "component", "components", "structure", "system", "item", "items",
        }
        return {token for token in tokens if len(token) >= 3 and token not in stop}

    def _review_common_sense_flags(
        self,
        *,
        name: str,
        desc: str,
        route_set: Set[str],
        class_via: List[str],
        evidence_only: List[str],
        dir_value: Any,
        prompt_text: str,
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
        generic_only_names = {
            "component", "part", "structure", "system", "element", "organelle", "assembly", "module",
        }

        looks_physical = (
            any(term in name_l or term in desc_l for term in physical_nouns)
            or any(phrase in desc_l for phrase in physical_phrases)
            or any(phrase in desc_l for phrase in action_object_phrases)
            or relation_overlap
        )
        generic_name = name_l.strip() in generic_only_names
        abstract = any(phrase in desc_l for phrase in abstract_phrases)
        d1, d2 = self._parse_dir_support(dir_value)
        supported = (d1 + d2) > 0 or bool(route_set & {"seed_p527", "scan_direct_p527", "scan_inverse_p361", "preview_p361_target", "open_structural_support", "check_supported"})
        evidence_only_flag = bool(evidence_only) and not supported
        return {
            "looks_physical": looks_physical,
            "relation_overlap": relation_overlap,
            "generic_name": generic_name,
            "abstract": abstract,
            "supported": supported,
            "evidence_only": evidence_only_flag,
        }

    def _row_has_target_structural_support(self, row: Dict[str, Any]) -> bool:
        routes = row.get("rt", [])
        if not isinstance(routes, list):
            routes = []
        route_set = {str(item).strip() for item in routes if str(item).strip()}
        if route_set & {"seed_p527", "scan_direct_p527", "scan_inverse_p361", "preview_p361_target", "open_structural_support", "check_supported"}:
            return True
        d1, d2 = self._parse_dir_support(row.get("dir", "0/0"))
        if d1 + d2 <= 0:
            return False
        return any(str(item).startswith(("seed_", "scan_", "preview_", "open_", "check_supported")) for item in route_set)

    def _deterministic_final_review(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {"accept": [], "reject": [], "hold": [], "notes": {}}
        rows = [row for row in payload.get("candidates", []) if isinstance(row, dict)]
        concrete: List[tuple[int, str, Set[str]]] = []
        early_reject: List[tuple[int, str, str]] = []
        prompt_text = " ".join(
            [
                str(payload.get("target_prompt", "") or "").strip(),
                str((payload.get("target", {}) or {}).get("name", "") or "").strip(),
                str((payload.get("target", {}) or {}).get("d", "") or "").strip(),
            ]
        ).strip()
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
            rid = str(row.get("id", "")).strip()
            if not rid:
                continue
            name = str(row.get("name", "")).strip()
            desc = str(row.get("d", "")).strip()
            name_l = name.lower()
            desc_l = desc.lower()
            routes = row.get("rt", [])
            if not isinstance(routes, list):
                routes = []
            route_set = {str(item).strip() for item in routes if str(item).strip()}
            direct_support = self._row_has_target_structural_support(row)
            seed_like = bool(route_set & {"seed_p527", "seed_p1552", "check_supported", "scan_direct_p527", "scan_inverse_p361", "preview_p361_target", "open_structural_support"})
            class_via = [str(item).strip() for item in (row.get("cls", []) or []) if str(item).strip()]
            evidence_only = [str(item).strip() for item in (row.get("eo", []) or []) if str(item).strip()]
            commonsense = self._review_common_sense_flags(
                name=name,
                desc=desc,
                route_set=route_set,
                class_via=class_via,
                evidence_only=evidence_only,
                dir_value=row.get("dir", "0/0"),
                prompt_text=prompt_text,
            )

            if not name and not desc:
                early_reject.append((-100, rid, "empty candidate"))
                continue

            score = 0
            tags: Set[str] = set()
            is_generic_name = name_l in generic_name_terms or any(name_l == term for term in generic_name_terms)
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

        accept_rows: List[tuple[int, str, str]] = []
        reject_rows: List[tuple[int, str, str]] = list(early_reject)

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

        result["accept"] = self._unique_aliases([rid for _, rid, _ in accept_rows])
        result["hold"] = []
        result["reject"] = self._unique_aliases([rid for _, rid, _ in reject_rows if rid not in result["accept"]])

        for _, rid, note in accept_rows:
            result["notes"][rid] = note
        for _, rid, note in reject_rows:
            if rid not in result["accept"]:
                result["notes"].setdefault(rid, note)

        return result

    def _deterministic_choose_actions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = payload.get("state", {}) or {}
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
            d1, d2 = self._dir_pair(row.get("dir", "0/0"))
            if d1 + d2 > 0:
                strong.append(rid)
        if strong:
            if len(strong) == 1:
                return {"decision": "act", "why": "confirm direct candidate", "actions": [{"tool": "check", "args": {"candidate": strong[0]}, "why": "confirm support"}], "memory_id": ""}
            return {"decision": "act", "why": "confirm direct candidates", "actions": [{"tool": "check_many", "args": {"candidates": strong[:4]}, "why": "confirm support"}], "memory_id": ""}

        unresolved_seed = []
        for row in seed_rows:
            rid = str(row.get("id", "")).strip()
            if rid and rid not in bucketed:
                unresolved_seed.append(rid)
        unresolved_seed = self._unique_aliases(unresolved_seed)

        if small_seed and not route_recent and step <= 2:
            return {"decision": "act", "why": "explore early frontier", "actions": [{"tool": "survey", "args": {"anchor": "T", "k": 8}, "why": "explore more parts"}], "memory_id": ""}
        if unresolved_seed:
            return {"decision": "act", "why": "probe unresolved seed", "actions": [{"tool": "probe_many", "args": {"candidates": unresolved_seed[:4]}, "why": "resolve seed nodes"}], "memory_id": ""}
        if small_seed:
            accepted_names = {str(row.get("id", "")).strip(): str(row.get("name", "")).strip().lower() for row in cand_rows if isinstance(row, dict)}
            generic = {"organelle", "component", "part", "structure", "system", "cellular component", "aircraft component"}
            for rid in [str(row.get("id", "")).strip() for row in state.get("accepted", []) if isinstance(row, dict)]:
                if rid and accepted_names.get(rid, "") not in generic:
                    return {"decision": "act", "why": "explore accepted child", "actions": [{"tool": "expand_neighbors", "args": {"anchor": rid, "k": 6}, "why": "look at children"}], "memory_id": ""}
            fallback_route = "rev" if not any(self._row_has_target_structural_support(row) for row in cand_rows) else "dir"
            fallback_why = "search reverse part-of" if fallback_route == "rev" else "search more parts"
            return {"decision": "act", "why": "keep exploring target", "actions": [{"tool": "scan", "args": {"anchor": "T", "route": fallback_route, "k": 4}, "why": fallback_why}], "memory_id": ""}
        return {"decision": "stop", "why": "frontier exhausted", "actions": [], "memory_id": ""}

    def _fallback_visual_query_refinement(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        target = payload.get("target", {}) or {}
        prompt = str(target.get("prompt", payload.get("target_prompt", "")) or "").strip()
        target_label = str(target.get("label", "") or "").strip()
        target_prefix = target_label or prompt
        queries = []

        for row in payload.get("components", [])[:24]:
            if not isinstance(row, dict):
                continue
            qid = str(row.get("qid", "")).strip()
            label = str(row.get("label", "")).strip()
            if not qid or not label:
                continue
            label_l = label.lower()
            prompt_l = prompt.lower()
            target_l = target_prefix.lower()
            if label_l in prompt_l or (target_l and label_l in target_l):
                continue
            if len(label.split()) >= 2:
                continue
            seed = target_label or prompt
            if not seed:
                continue
            query = f"{seed} {label}".strip()
            if len(query) > 96:
                query = query[:96].rsplit(" ", 1)[0].strip() or query[:96].strip()
            if query and query.lower() != label_l:
                queries.append({"qid": qid, "query": query})

        return {"queries": queries}

    def _fallback_wikipedia_section_select(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed = []
        titles_by_id: Dict[str, str] = {}
        for row in payload.get("sections", [])[:12]:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("id", "")).strip()
            title = str(row.get("title", "")).strip()
            if sid:
                allowed.append(sid)
                titles_by_id[sid] = title.lower()

        if not allowed:
            return {"sections": ["0"]}

        preferred_terms = (
            "description", "design", "structure", "construction", "components", "layout",
            "anatomy", "morphology", "form", "configuration", "body", "frame", "wheel",
            "appearance", "overview",
        )

        picked = ["0"] if "0" in allowed else [allowed[0]]
        for sid in allowed:
            if sid in picked:
                continue
            title = titles_by_id.get(sid, "")
            if any(term in title for term in preferred_terms):
                picked.append(sid)
                break
        return {"sections": picked[:2]}

    def _fallback_wikipedia_visual_extract(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        page = payload.get("page", {}) or {}
        component = payload.get("component", {}) or {}
        text = str(page.get("text", "") or "").strip()
        label = str(component.get("label", "") or "").strip()
        stage1 = str(component.get("description", "") or "").strip()

        selected = self._pick_visual_sentences(text, label=label, limit=3)
        chunks = selected if selected else self._pick_sentences(text, limit=2)
        combined = " ".join(chunks).strip()
        if stage1 and stage1 not in combined:
            combined = (stage1 + " " + combined).strip()
        combined = re.sub(r"\s+", " ", combined).strip()
        if not combined:
            combined = stage1 or f"{label} has no extracted visual description."
        return {"visual_description": combined[:700]}

    def _fallback_visual_description_refinement(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        base = str(payload.get("base_description", "") or "").strip()
        candidate_bits: List[str] = []
        seen = set()
        for row in payload.get("candidates", [])[:10]:
            if not isinstance(row, dict):
                continue
            desc = re.sub(r"\s+", " ", str(row.get("description", "") or "")).strip()
            if not desc:
                continue
            key = desc.lower()
            if key in seen:
                continue
            seen.add(key)
            candidate_bits.append(desc)
            if len(candidate_bits) >= 2:
                break
        text = base
        if candidate_bits:
            addon = " ".join(candidate_bits)
            text = (base + " " + addon).strip() if base else addon
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            component = payload.get("component", {}) or {}
            text = str(component.get("description", "") or component.get("label", "") or "").strip()
        return {"refined_description": text[:900]}

    def _pick_sentences(self, text: str, limit: int) -> List[str]:
        if not text:
            return []
        raw_parts = re.split(r"(?<=[.!?])\s+", text)
        out: List[str] = []
        for part in raw_parts:
            cleaned = re.sub(r"\s+", " ", part).strip()
            if len(cleaned) < 24:
                continue
            out.append(cleaned)
            if len(out) >= limit:
                break
        return out

    def _pick_visual_sentences(self, text: str, label: str, limit: int) -> List[str]:
        if not text:
            return []
        visual_terms = {
            "appear", "appearance", "visible", "looks", "look", "shape", "shaped", "form", "forms",
            "structure", "surface", "texture", "color", "colour", "outline", "rounded", "flat", "curved",
            "long", "short", "thin", "thick", "broad", "narrow", "hollow", "solid", "ring", "rod", "tube",
            "layer", "membrane", "wall", "fold", "lobed", "branched", "central", "outer", "inner", "spiral",
            "disc", "disk", "spherical", "cylindrical", "paired", "cluster", "arranged", "contains", "consists",
            "composed", "made of", "covered", "surrounded",
        }
        label_l = label.lower()
        scored: List[tuple[int, int, str]] = []
        for idx, sentence in enumerate(self._pick_sentences(text, limit=40)):
            sentence_l = sentence.lower()
            score = 0
            if label_l and label_l in sentence_l:
                score += 2
            for term in visual_terms:
                if term in sentence_l:
                    score += 1
            if score > 0:
                scored.append((-score, idx, sentence))
        scored.sort()
        return [sentence for _, _, sentence in scored[:limit]]

    def _task_budget(self, task: str, mode: str) -> Dict[str, int]:
        thinking = mode == "thinking"
        base_actions = min(max(128, int(self.max_new_tokens or 256)), 160)

        if task == "choose_target":
            return {"initial": 48 if not thinking else 72, "continue": 40, "rounds": 1}

        if task == "choose_actions":
            return {"initial": 72 if not thinking else 96, "continue": 48, "rounds": 1}

        if task == "final_review":
            return {"initial": 72 if not thinking else 96, "continue": 56, "rounds": 1}

        if task == "visual_query_refinement":
            return {"initial": 32 if not thinking else 56, "continue": 24, "rounds": 0}

        if task == "wikipedia_section_select":
            return {"initial": 16 if not thinking else 32, "continue": 16, "rounds": 0}

        if task in {"wikipedia_visual_extract", "visual_description_refinement"}:
            initial = 40 if task == "wikipedia_visual_extract" else 48
            return {"initial": initial if not thinking else initial + 24, "continue": 24, "rounds": 0}

        return {
            "initial": base_actions if not thinking else max(base_actions, 192),
            "continue": 64 if not thinking else 96,
            "rounds": 1 if not thinking else 2,
        }

    def _mode_instruction(self, mode: str) -> str:
        if mode == "thinking":
            return (
                "MODE=thinking. You may reason inside <think>...</think>. "
                "Your actual answer after </think> must be one strict JSON object only. "
                "Do not put markdown or prose around the final JSON. "
                "If the answer would otherwise be cut off, finish the JSON object cleanly."
            )
        return (
            "MODE=normal. Output only one strict JSON object. "
            "Do not output <think> tags, markdown, or prose outside the JSON. "
            "Finish the JSON object cleanly without truncation."
        )

    def _split_placeholder_tokens(self, value: Any) -> List[str]:
        text = str(value or "").strip()
        if not text:
            return []

        for ch in "[]{}(),":
            text = text.replace(ch, " ")
        text = text.replace("/", "|")
        text = text.replace("\\", "|")

        out: List[str] = []
        for chunk in text.split("|"):
            for piece in chunk.split():
                piece = piece.strip().strip("'").strip('"')
                if piece:
                    out.append(piece)
        return out

    def _coerce_node_alias(
        self,
        value: Any,
        allowed_node_ids: Set[str],
        *,
        allow_target: bool,
        field_name: str,
    ) -> str:
        text = str(value or "").strip()
        if not text:
            raise SchemaValidationError(f"{field_name} is required")

        if allow_target and text == "T":
            return "T"
        if text in allowed_node_ids:
            return text

        tokens = self._split_placeholder_tokens(text)
        for token in tokens:
            if allow_target and token == "T":
                return "T"
            if token in allowed_node_ids:
                return token

        raise SchemaValidationError(f"{field_name} references unknown node alias: {value}")

    def _coerce_node_list(
        self,
        value: Any,
        allowed_node_ids: Set[str],
        *,
        allow_target: bool,
        field_name: str,
        min_items: int = 1,
        max_items: int = 99,
    ) -> List[str]:
        raw_items: List[Any]
        if isinstance(value, list):
            raw_items = value
        elif isinstance(value, str):
            tokens = self._split_placeholder_tokens(value)
            raw_items = tokens if tokens else [value]
        else:
            raw_items = []

        out: List[str] = []
        seen: Set[str] = set()

        for item in raw_items:
            if isinstance(item, dict):
                item = item.get("id", item.get("node", item.get("candidate", "")))

            try:
                alias = self._coerce_node_alias(
                    item,
                    allowed_node_ids,
                    allow_target=allow_target,
                    field_name=field_name,
                )
            except SchemaValidationError:
                continue

            if alias not in seen:
                seen.add(alias)
                out.append(alias)

        if len(out) > max_items:
            out = out[:max_items]
        if len(out) < min_items:
            raise SchemaValidationError(f"{field_name} must contain {min_items}-{max_items} visible aliases")

        return out

    def _coerce_choice(
        self,
        value: Any,
        allowed: Set[str],
        *,
        field_name: str,
        default: Optional[str] = None,
    ) -> str:
        text = str(value or "").strip()
        if not text:
            if default is not None:
                return default
            raise SchemaValidationError(f"{field_name} is required")

        if text in allowed:
            return text

        tokens = self._split_placeholder_tokens(text)
        for token in tokens:
            if token in allowed:
                return token

        if default is not None:
            return default

        raise SchemaValidationError(f"{field_name} has invalid value: {value}")

    def _normalize_raw_jsonish_text(self, text: str) -> str:
        text = self._strip_special_text(text)
        if not text:
            return ""

        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1].strip()

        if text.startswith("{{") and text.endswith("}}"):
            text = text[1:-1].strip()

        return text

    def _repair_json_output(self, task: str, payload: Dict[str, Any], raw_text: str) -> Optional[Dict[str, Any]]:
        formatter_prompts = {
            "choose_target": TARGET_FORMATTER_SYSTEM_PROMPT,
            "choose_actions": ACTIONS_FORMATTER_SYSTEM_PROMPT,
            "final_review": FINAL_REVIEW_FORMATTER_SYSTEM_PROMPT,
            "visual_query_refinement": VISUAL_QUERY_REFINER_FORMATTER_SYSTEM_PROMPT,
            "wikipedia_section_select": WIKIPEDIA_SECTION_SELECT_FORMATTER_SYSTEM_PROMPT,
            "wikipedia_visual_extract": WIKIPEDIA_VISUAL_EXTRACT_FORMATTER_SYSTEM_PROMPT,
            "visual_description_refinement": VISUAL_DESCRIPTION_REFINER_FORMATTER_SYSTEM_PROMPT,
        }
        system_prompt = formatter_prompts.get(task)
        if not system_prompt:
            return None

        compact_input = json.dumps(self._prompt_payload(task, payload), ensure_ascii=False, separators=(",", ":"))
        raw_text = self._normalize_raw_jsonish_text(raw_text)
        if len(raw_text) > 2400:
            raw_text = raw_text[-2400:]

        user_prompt = (
            f"TASK={task}\n"
            f"INPUT={compact_input}\n"
            f"RAW={raw_text}\n"
            "Convert RAW into ONE valid JSON object only.\n"
            "Rules:\n"
            "- preserve the intended decision/classification\n"
            "- notes values must be single-line short strings\n"
            "- for choose_actions, decision must be act/fallback/stop\n"
            "- for choose_actions, actions must be a list of action objects\n"
            "- for choose_actions, candidate lists must contain alias strings like N1, N2, not nested objects\n"
            "- no markdown\n"
            "- no prose outside JSON\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        inputs = None
        input_ids = None
        attention_mask = None
        generated = None
        try:
            with self.lock:
                inputs = self._apply_chat_template(messages, thinking_enabled=False)
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)
                prompt_len = int(input_ids.shape[-1])

                generated = self._run_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=96 if task in {"visual_query_refinement", "wikipedia_section_select", "wikipedia_visual_extract", "visual_description_refinement"} else 192,
                    thinking_enabled=False,
                )
                new_ids = generated[0][prompt_len:].tolist()
                repaired_text = self._normalize_raw_jsonish_text(self.tokenizer.decode(new_ids, skip_special_tokens=False))
                return self._extract_best_json(repaired_text, task=task)
        finally:
            del generated, attention_mask, input_ids, inputs
            self._gpu_gc()

    def infer(self, body: Dict[str, Any]) -> Dict[str, Any]:
        try:
            task = str(body.get("task", "choose_actions"))
            mode = str(body.get("mode", "normal")).strip().lower()
            if mode == "thinking":
                self.logger.info("Thinking mode requested for %s; forcing normal mode", task)
                mode = "normal"
            payload = body.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            if mode not in {"normal", "thinking"}:
                raise ValueError("mode must be 'normal' or 'thinking'")
            system_prompt, user_prompt = self._task_prompt(task, payload, mode)
            rendered_prompt = self._render_chat_prompt(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                thinking_enabled=(mode == "thinking"),
            )

            print(
                "[AGENT-IN]",
                json.dumps({"task": task, "mode": mode, "payload": payload}, ensure_ascii=False, separators=(",", ":")),
                flush=True,
            )

            bundle = self._generate(task=task, system_prompt=system_prompt, user_prompt=user_prompt, mode=mode)

            if mode == "thinking" and bundle.get("thinking_text"):
                print("[AGENT-THINK]", bundle["thinking_text"], flush=True)

            raw_parse_text = bundle.get("parse_text", bundle.get("raw_text", ""))
            normalized_parse_text = self._normalize_raw_jsonish_text(raw_parse_text)
            print("[AGENT-OUT-RAW]", normalized_parse_text, flush=True)

            def _finish(result: Dict[str, Any]) -> Dict[str, Any]:
                self._write_stage_io(
                    task=task,
                    mode=mode,
                    payload=payload,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    rendered_input_text=rendered_prompt,
                    raw_text=normalized_parse_text,
                    parsed_payload=result,
                )
                print("[AGENT-OUT-PARSED]", json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
                return result

            parsed = self._extract_best_json(normalized_parse_text, task=task)
            if parsed is None:
                if task == "final_review":
                    parsed = self._salvage_final_review_output(normalized_parse_text, payload)
            if parsed is None:
                repaired = self._repair_json_output(task, payload, normalized_parse_text)
                if repaired is not None:
                    parsed = repaired

            if parsed is None:
                if task == "final_review":
                    result = self._deterministic_final_review(payload)
                    return _finish(result)
                if task == "choose_actions":
                    result = self._deterministic_choose_actions(payload)
                    return _finish(result)
                if task == "visual_query_refinement":
                    result = self._fallback_visual_query_refinement(payload)
                    return _finish(result)
                if task == "wikipedia_section_select":
                    result = self._fallback_wikipedia_section_select(payload)
                    return _finish(result)
                if task == "wikipedia_visual_extract":
                    result = self._fallback_wikipedia_visual_extract(payload)
                    return _finish(result)
                if task == "visual_description_refinement":
                    result = self._fallback_visual_description_refinement(payload)
                    return _finish(result)
                raise JsonParseError(f"worker produced no valid JSON object for task={task}")

            try:
                if task == "choose_target":
                    result = self._validate_target_choice(parsed, payload)
                elif task == "choose_actions":
                    result = self._validate_action_result(parsed, payload)
                    result = self._repair_action_result(result, payload)
                elif task == "visual_query_refinement":
                    result = self._validate_visual_query_refinement(parsed, payload)
                elif task == "wikipedia_section_select":
                    result = self._validate_wikipedia_section_select(parsed, payload)
                elif task == "wikipedia_visual_extract":
                    result = self._validate_wikipedia_visual_extract(parsed, payload)
                elif task == "visual_description_refinement":
                    result = self._validate_visual_description_refinement(parsed, payload)
                else:
                    result = self._validate_final_review(parsed, payload)
            except SchemaValidationError:
                if task == "final_review":
                    salvaged = self._salvage_final_review_output(normalized_parse_text, payload)
                    if salvaged is not None:
                        result = self._validate_final_review(salvaged, payload)
                        return _finish(result)
                repaired = self._repair_json_output(task, payload, normalized_parse_text)
                if repaired is None:
                    if task == "final_review":
                        result = self._deterministic_final_review(payload)
                        return _finish(result)
                    if task == "choose_actions":
                        result = self._deterministic_choose_actions(payload)
                        return _finish(result)
                    if task == "visual_query_refinement":
                        result = self._fallback_visual_query_refinement(payload)
                        return _finish(result)
                    if task == "wikipedia_section_select":
                        result = self._fallback_wikipedia_section_select(payload)
                        return _finish(result)
                    if task == "wikipedia_visual_extract":
                        result = self._fallback_wikipedia_visual_extract(payload)
                        return _finish(result)
                    if task == "visual_description_refinement":
                        result = self._fallback_visual_description_refinement(payload)
                        return _finish(result)
                    raise

                if task == "choose_target":
                    result = self._validate_target_choice(repaired, payload)
                elif task == "choose_actions":
                    result = self._validate_action_result(repaired, payload)
                    result = self._repair_action_result(result, payload)
                elif task == "visual_query_refinement":
                    result = self._validate_visual_query_refinement(repaired, payload)
                elif task == "wikipedia_section_select":
                    result = self._validate_wikipedia_section_select(repaired, payload)
                elif task == "wikipedia_visual_extract":
                    result = self._validate_wikipedia_visual_extract(repaired, payload)
                elif task == "visual_description_refinement":
                    result = self._validate_visual_description_refinement(repaired, payload)
                else:
                    result = self._validate_final_review(repaired, payload)

            return _finish(result)
        finally:
            self._gpu_gc()

    def infer_many(self, bodies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(bodies, list):
            raise ValueError("items must be a list")
        if not bodies:
            return []
        for body in bodies:
            if not isinstance(body, dict):
                raise ValueError("each batch item must be an object")

        tasks = {str(body.get("task", "")).strip() for body in bodies}
        modes = {str(body.get("mode", "normal")).strip().lower() for body in bodies}
        if len(tasks) != 1 or len(modes) != 1:
            return [self.infer(body) for body in bodies]

        task = next(iter(tasks))
        mode = next(iter(modes))
        if mode == "thinking":
            self.logger.info("Thinking batch requested for %s; forcing normal mode", task)
            mode = "normal"
        if mode != "normal" or task not in {"wikipedia_section_select", "wikipedia_visual_extract", "visual_description_refinement"}:
            return [self.infer(body) for body in bodies]

        payloads = []
        for body in bodies:
            payload = body.get("payload", {})
            if not isinstance(payload, dict):
                raise ValueError("payload must be an object")
            payloads.append(payload)

        return self._infer_many_with_splitting(task=task, mode=mode, payloads=payloads)

    def _infer_many_with_splitting(self, task: str, mode: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not payloads:
            return []
        if len(payloads) == 1:
            return [self.infer({"task": task, "mode": mode, "payload": payloads[0]})]
        try:
            return self._infer_many_batched(task=task, mode=mode, payloads=payloads)
        except Exception:
            self.logger.exception("batched infer failed for task=%s batch_size=%s", task, len(payloads))
            self._gpu_gc(force=True)
            if len(payloads) <= 2:
                return [self.infer({"task": task, "mode": mode, "payload": payload}) for payload in payloads]
            mid = max(1, len(payloads) // 2)
            left = self._infer_many_with_splitting(task=task, mode=mode, payloads=payloads[:mid])
            right = self._infer_many_with_splitting(task=task, mode=mode, payloads=payloads[mid:])
            return left + right

    def _render_chat_prompt(self, messages: List[Dict[str, str]], thinking_enabled: bool) -> str:
        common_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=thinking_enabled, **common_kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": thinking_enabled},
                **common_kwargs,
            )

    def _fallback_for_task(self, task: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if task == "choose_actions":
            return self._deterministic_choose_actions(payload)
        if task == "final_review":
            return self._deterministic_final_review(payload)
        if task == "visual_query_refinement":
            return self._fallback_visual_query_refinement(payload)
        if task == "wikipedia_section_select":
            return self._fallback_wikipedia_section_select(payload)
        if task == "wikipedia_visual_extract":
            return self._fallback_wikipedia_visual_extract(payload)
        if task == "visual_description_refinement":
            return self._fallback_visual_description_refinement(payload)
        raise JsonParseError(f"no fallback available for task={task}")

    def _prefer_fast_fallback(self, task: str) -> bool:
        return self.fast_visual_fallbacks and task in {
            "visual_query_refinement",
            "wikipedia_section_select",
            "wikipedia_visual_extract",
            "visual_description_refinement",
        }

    def _validate_task_result(self, task: str, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        if task == "choose_target":
            return self._validate_target_choice(data, payload)
        if task == "choose_actions":
            result = self._validate_action_result(data, payload)
            return self._repair_action_result(result, payload)
        if task == "final_review":
            return self._validate_final_review(data, payload)
        if task == "visual_query_refinement":
            return self._validate_visual_query_refinement(data, payload)
        if task == "wikipedia_section_select":
            return self._validate_wikipedia_section_select(data, payload)
        if task == "wikipedia_visual_extract":
            return self._validate_wikipedia_visual_extract(data, payload)
        if task == "visual_description_refinement":
            return self._validate_visual_description_refinement(data, payload)
        raise ValueError(f"unsupported task: {task}")

    def _parse_or_repair_task_result(self, task: str, payload: Dict[str, Any], normalized_parse_text: str) -> Dict[str, Any]:
        parsed = self._extract_best_json(normalized_parse_text, task=task)
        if parsed is None and task == "final_review":
            parsed = self._salvage_final_review_output(normalized_parse_text, payload)
        if parsed is None:
            if self._prefer_fast_fallback(task):
                return self._fallback_for_task(task, payload)
            repaired = self._repair_json_output(task, payload, normalized_parse_text)
            if repaired is not None:
                parsed = repaired
        if parsed is None:
            return self._fallback_for_task(task, payload)
        try:
            return self._validate_task_result(task, parsed, payload)
        except SchemaValidationError:
            if task == "final_review":
                salvaged = self._salvage_final_review_output(normalized_parse_text, payload)
                if salvaged is not None:
                    return self._validate_final_review(salvaged, payload)
            if self._prefer_fast_fallback(task):
                return self._fallback_for_task(task, payload)
            repaired = self._repair_json_output(task, payload, normalized_parse_text)
            if repaired is None:
                return self._fallback_for_task(task, payload)
            try:
                return self._validate_task_result(task, repaired, payload)
            except SchemaValidationError:
                return self._fallback_for_task(task, payload)

    def _infer_many_batched(self, task: str, mode: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        thinking_enabled = mode == "thinking"
        prompts: List[str] = []
        prompt_rows: List[Dict[str, str]] = []
        for payload in payloads:
            system_prompt, user_prompt = self._task_prompt(task, payload, mode)
            prompt_rows.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
            prompts.append(self._render_chat_prompt(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                thinking_enabled=thinking_enabled,
            ))

        if self.backend == "vllm" and self.llm is not None:
            import qwentest  # local module

            raws = qwentest._vllm_generate_texts(
                llm=self.llm,
                prompts=prompts,
                temperature=0.0,
                max_new_tokens=min(96, max(24, int(self._task_budget(task, mode).get("initial", 64)))),
                thinking_enabled=thinking_enabled,
                do_sample=False,
            )
            results: List[Dict[str, Any]] = []
            for idx, payload in enumerate(payloads):
                raw_text = self._normalize_raw_jsonish_text(str(raws[idx] if idx < len(raws) else ""))
                print("[AGENT-OUT-RAW]", raw_text, flush=True)
                result = self._parse_or_repair_task_result(task, payload, raw_text)
                prompt_row = prompt_rows[idx] if idx < len(prompt_rows) else {}
                self._write_stage_io(
                    task=task,
                    mode=mode,
                    payload=payload,
                    system_prompt=str(prompt_row.get("system_prompt", "") or ""),
                    user_prompt=str(prompt_row.get("user_prompt", "") or ""),
                    rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                    raw_text=raw_text,
                    parsed_payload=result,
                )
                print("[AGENT-OUT-PARSED]", json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
                results.append(result)
            return results

        inputs = None
        input_ids = None
        attention_mask = None
        generated = None
        old_padding_side = self.tokenizer.padding_side
        try:
            with self.lock:
                self.tokenizer.padding_side = "left"
                inputs = self.tokenizer(
                    prompts,
                    padding=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                    truncation=True,
                    max_length=1536,
                )
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)
                prompt_len = int(input_ids.shape[-1])
                batch_max_new_tokens = min(96, max(24, int(self._task_budget(task, mode).get("initial", 64))))
                generated = self._run_generation(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=batch_max_new_tokens,
                    thinking_enabled=thinking_enabled,
                )

            results: List[Dict[str, Any]] = []
            for idx, payload in enumerate(payloads):
                new_ids = generated[idx][prompt_len:].tolist()
                raw_text = self._normalize_raw_jsonish_text(self.tokenizer.decode(new_ids, skip_special_tokens=False))
                print("[AGENT-OUT-RAW]", raw_text, flush=True)
                result = self._parse_or_repair_task_result(task, payload, raw_text)
                prompt_row = prompt_rows[idx] if idx < len(prompt_rows) else {}
                self._write_stage_io(
                    task=task,
                    mode=mode,
                    payload=payload,
                    system_prompt=str(prompt_row.get("system_prompt", "") or ""),
                    user_prompt=str(prompt_row.get("user_prompt", "") or ""),
                    rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                    raw_text=raw_text,
                    parsed_payload=result,
                )
                print("[AGENT-OUT-PARSED]", json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
                results.append(result)
            return results
        finally:
            self.tokenizer.padding_side = old_padding_side
            del generated, attention_mask, input_ids, inputs
            self._gpu_gc()

    def _build_choose_target_prompt(self, payload: Dict[str, Any], mode: str) -> str:
        compact = json.dumps(self._prompt_payload("choose_target", payload), ensure_ascii=False, separators=(",", ":"))
        return self._mode_instruction(mode) + "\nTASK=pick_target\nINPUT=" + compact

    def _build_choose_actions_prompt(self, payload: Dict[str, Any], mode: str) -> str:
        compact = json.dumps(self._prompt_payload("choose_actions", payload), ensure_ascii=False, separators=(",", ":"))
        return self._mode_instruction(mode) + "\nTASK=choose_actions\nINPUT=" + compact

    def _build_final_review_prompt(self, payload: Dict[str, Any], mode: str) -> str:
        compact = json.dumps(self._prompt_payload("final_review", payload), ensure_ascii=False, separators=(",", ":"))
        return self._mode_instruction(mode) + "\nTASK=final_review\nINPUT=" + compact

    def _build_visual_query_refinement_prompt(self, payload: Dict[str, Any], mode: str) -> str:
        compact = json.dumps(self._prompt_payload("visual_query_refinement", payload), ensure_ascii=False, separators=(",", ":"))
        return self._mode_instruction(mode) + "\nTASK=visual_query_refinement\nINPUT=" + compact

    def _build_wikipedia_section_select_prompt(self, payload: Dict[str, Any], mode: str) -> str:
        compact = json.dumps(self._prompt_payload("wikipedia_section_select", payload), ensure_ascii=False, separators=(",", ":"))
        return self._mode_instruction(mode) + "\nTASK=wikipedia_section_select\nINPUT=" + compact

    def _build_wikipedia_visual_extract_prompt(self, payload: Dict[str, Any], mode: str) -> str:
        compact = json.dumps(self._prompt_payload("wikipedia_visual_extract", payload), ensure_ascii=False, separators=(",", ":"))
        return self._mode_instruction(mode) + "\nTASK=wikipedia_visual_extract\nINPUT=" + compact

    def _build_visual_description_refinement_prompt(self, payload: Dict[str, Any], mode: str) -> str:
        compact = json.dumps(self._prompt_payload("visual_description_refinement", payload), ensure_ascii=False, separators=(",", ":"))
        return self._mode_instruction(mode) + "\nTASK=visual_description_refinement\nINPUT=" + compact

    def _generate(self, task: str, system_prompt: str, user_prompt: str, mode: str) -> Dict[str, Any]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        budget = self._task_budget(task, mode)
        thinking_enabled = mode == "thinking"

        if self.backend == "vllm" and self.llm is not None:
            import qwentest  # local module

            prompt_text = self._render_chat_prompt(messages, thinking_enabled=thinking_enabled)
            raws = qwentest._vllm_generate_texts(
                llm=self.llm,
                prompts=[prompt_text],
                temperature=0.0 if not thinking_enabled else 0.6,
                max_new_tokens=int(budget.get("initial", self.max_new_tokens)),
                thinking_enabled=thinking_enabled,
                do_sample=False,
            )
            raw_text = str(raws[0] if raws else "")
            split = qwentest._split_vllm_generated_text(raw_text, thinking_enabled=thinking_enabled)
            parse_text = self._normalize_raw_jsonish_text(
                self._candidate_parse_text(split, thinking_enabled=thinking_enabled)
            )
            split["parse_text"] = parse_text
            return split

        inputs = None
        input_ids = None
        attention_mask = None
        generated = None
        try:
            with self.lock:
                inputs = self._apply_chat_template(messages, thinking_enabled=thinking_enabled)
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)
                prompt_len = int(input_ids.shape[-1])
                prefix_entry = self._get_or_build_prefix_cache(
                    task=task,
                    mode=mode,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    thinking_enabled=thinking_enabled,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                prefix_len = int(prefix_entry.prefix_len) if prefix_entry is not None else 0
                use_prefix_cache = prefix_entry is not None and 0 < prefix_len < prompt_len

                if use_prefix_cache:
                    current_ids = input_ids[:, prefix_len:]
                    current_attention = attention_mask[:, :prompt_len]
                else:
                    current_ids = input_ids
                    current_attention = attention_mask
                bundle: Dict[str, Any] = {
                    "raw_text": "",
                    "thinking_text": "",
                    "final_text": "",
                    "parse_text": "",
                    "saw_close_tag": False,
                }

                total_rounds = 1 + max(0, int(budget.get("rounds", 0) or 0))
                for round_idx in range(total_rounds):
                    step_tokens = budget["initial"] if round_idx == 0 else max(16, int(budget.get("continue", 64) or 64))
                    round_prompt_len = int(current_ids.shape[-1])
                    cached_past = self._clone_past_key_values(prefix_entry.past_key_values) if use_prefix_cache else None
                    generated = self._run_generation(
                        input_ids=current_ids,
                        attention_mask=current_attention,
                        max_new_tokens=step_tokens,
                        thinking_enabled=thinking_enabled,
                        past_key_values=cached_past,
                    )

                    if generated.shape[-1] <= current_ids.shape[-1]:
                        current_ids = generated
                        break

                    current_ids = generated
                    if use_prefix_cache:
                        current_attention = torch.ones(
                            (current_ids.shape[0], prefix_len + current_ids.shape[-1]),
                            dtype=attention_mask.dtype,
                            device=current_ids.device,
                        )
                    else:
                        current_attention = torch.ones_like(current_ids, device=current_ids.device)
                    new_ids = current_ids[0][round_prompt_len:].tolist()
                    bundle = self._split_generated_ids(new_ids, thinking_enabled=thinking_enabled)
                    bundle["parse_text"] = self._normalize_raw_jsonish_text(
                        self._candidate_parse_text(bundle, thinking_enabled=thinking_enabled)
                    )

                    if round_idx == total_rounds - 1 or not self._should_continue(bundle, thinking_enabled=thinking_enabled):
                        break

                return bundle
        finally:
            del generated, attention_mask, input_ids, inputs
            self._gpu_gc()

    def _run_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        thinking_enabled: bool,
        past_key_values: Any = None,
    ) -> torch.Tensor:
        generation_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": True,
        }
        if past_key_values is not None:
            generation_kwargs["past_key_values"] = past_key_values
        if thinking_enabled:
            generation_kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 0.6,
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0.0,
                }
            )
        else:
            generation_kwargs.update({"do_sample": False})

        with torch.inference_mode():
            return self.model.generate(**generation_kwargs)

    def _apply_chat_template(self, messages: List[Dict[str, str]], thinking_enabled: bool) -> Dict[str, torch.Tensor]:
        common_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        try:
            return self.tokenizer.apply_chat_template(messages, enable_thinking=thinking_enabled, **common_kwargs)
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages,
                chat_template_kwargs={"enable_thinking": thinking_enabled},
                **common_kwargs,
            )

    def _encode_plain_text(self, text: str) -> List[int]:
        encoded = self.tokenizer(text, add_special_tokens=False, return_attention_mask=False)
        return list(encoded.get("input_ids", []))

    def _find_subsequence_end(self, source: List[int], pattern: List[int]) -> int:
        if not source or not pattern or len(pattern) > len(source):
            return -1
        for start in range(len(source) - len(pattern), -1, -1):
            if source[start:start + len(pattern)] == pattern:
                return start + len(pattern)
        return -1

    def _split_generated_ids(self, new_ids: List[int], thinking_enabled: bool) -> Dict[str, Any]:
        raw_text = self._strip_special_text(self.tokenizer.decode(new_ids, skip_special_tokens=False))
        if thinking_enabled:
            close_end = self._find_subsequence_end(new_ids, self.think_close_ids)
            if close_end != -1:
                think_ids = new_ids[:close_end]
                final_ids = new_ids[close_end:]
            else:
                think_ids = new_ids
                final_ids = []
            return {
                "raw_text": raw_text,
                "thinking_text": self._strip_special_text(self.tokenizer.decode(think_ids, skip_special_tokens=False)),
                "final_text": self._strip_special_text(self.tokenizer.decode(final_ids, skip_special_tokens=False)),
                "saw_close_tag": close_end != -1,
            }

        return {
            "raw_text": raw_text,
            "thinking_text": "",
            "final_text": raw_text,
            "saw_close_tag": False,
        }

    def _candidate_parse_text(self, bundle: Dict[str, Any], thinking_enabled: bool) -> str:
        if thinking_enabled:
            final_text = str(bundle.get("final_text", "") or "")
            if final_text.strip():
                return final_text
        return str(bundle.get("raw_text", "") or "")

    def _should_continue(self, bundle: Dict[str, Any], thinking_enabled: bool) -> bool:
        parse_text = self._normalize_raw_jsonish_text(self._candidate_parse_text(bundle, thinking_enabled=thinking_enabled))
        if thinking_enabled and not bundle.get("saw_close_tag", False):
            return True
        if not parse_text:
            return True
        if self._looks_truncated_json(parse_text):
            return True
        return self._extract_best_json(parse_text) is None

    def _looks_truncated_json(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        if not stripped.startswith("{"):
            return False

        depth = 0
        in_string = False
        escaped = False
        for ch in stripped:
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1

        if in_string or depth > 0:
            return True

        tail = stripped.rstrip()
        if tail.endswith(":") or tail.endswith(",") or tail.endswith("["):
            return True
        return False

    def _strip_special_text(self, text: str) -> str:
        if not text:
            return ""
        return text.replace("<|im_end|>", " ").replace("<|endoftext|>", " ").strip()

    def _matches_task_schema(self, task: str, obj: Dict[str, Any]) -> bool:
        if not isinstance(obj, dict):
            return False
        if task == "choose_target":
            return "id" in obj
        if task == "choose_actions":
            return "decision" in obj and "actions" in obj
        if task == "final_review":
            return "accept" in obj and "reject" in obj and "hold" in obj
        if task == "visual_query_refinement":
            return "queries" in obj
        if task == "wikipedia_section_select":
            return "sections" in obj
        if task == "wikipedia_visual_extract":
            return "visual_description" in obj
        if task == "visual_description_refinement":
            return "refined_description" in obj
        return False

    def _extract_best_json(self, text: str, task: Optional[str] = None) -> Optional[Dict[str, Any]]:
        stripped = (text or "").strip()
        if not stripped:
            return None

        direct = self._safe_load_json(stripped)
        if isinstance(direct, dict):
            if task is None or self._matches_task_schema(task, direct):
                return direct

        if stripped.startswith("{") and self._looks_truncated_json(stripped):
            return None

        candidates = self._extract_json_candidates(stripped)
        if task is not None:
            candidates = [c for c in candidates if isinstance(c, dict) and self._matches_task_schema(task, c)]
            return candidates[-1] if candidates else None

        for candidate in reversed(candidates):
            if isinstance(candidate, dict):
                return candidate
        return None

    def _find_json_value_span(self, text: str, key: str, opener: str, closer: str) -> Optional[str]:
        if not text:
            return None
        pattern = '"' + re.escape(key) + r'"\s*:\s*' + re.escape(opener)
        match = re.search(pattern, text)
        if not match:
            return None

        start = match.end() - 1
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start:idx + 1]
        return text[start:]

    @staticmethod
    def _extract_review_aliases(raw: str, allowed_ids: Set[str]) -> List[str]:
        if not raw:
            return []
        out: List[str] = []
        seen: Set[str] = set()
        for rid in re.findall(r'R\d+', raw):
            if rid in allowed_ids and rid not in seen:
                seen.add(rid)
                out.append(rid)
        return out

    def _salvage_final_review_output(self, raw_text: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        allowed_ids = {
            str(row.get("id", "")).strip()
            for row in payload.get("candidates", [])
            if isinstance(row, dict) and str(row.get("id", "")).strip()
        }
        if not allowed_ids:
            return None

        text = self._normalize_raw_jsonish_text(raw_text)
        if not text:
            return None

        result: Dict[str, Any] = {"accept": [], "reject": [], "hold": [], "notes": {}}
        found_any = False

        for key in ("accept", "reject", "hold"):
            fragment = self._find_json_value_span(text, key, "[", "]")
            aliases: List[str] = []
            if fragment:
                parsed = self._safe_load_json(fragment)
                if isinstance(parsed, list):
                    aliases = self._extract_review_aliases(json.dumps(parsed, ensure_ascii=False), allowed_ids)
                else:
                    aliases = self._extract_review_aliases(fragment, allowed_ids)
            if aliases:
                result[key] = aliases
                found_any = True

        notes_fragment = self._find_json_value_span(text, "notes", "{", "}")
        if notes_fragment:
            parsed_notes = self._safe_load_json(notes_fragment)
            if isinstance(parsed_notes, dict):
                for raw_key, raw_value in parsed_notes.items():
                    match = re.search(r'R\d+', str(raw_key))
                    if not match:
                        continue
                    rid = match.group(0)
                    if rid not in allowed_ids:
                        continue
                    note = str(raw_value).replace("\n", " ").replace("\r", " ").strip()[:240]
                    if note:
                        result["notes"][rid] = note
                        found_any = True
            else:
                for raw_key, raw_value in re.findall(r'"([^"]*R\d+[^"]*)"\s*:\s*"([^"]*)', notes_fragment):
                    match = re.search(r'R\d+', raw_key)
                    if not match:
                        continue
                    rid = match.group(0)
                    if rid not in allowed_ids:
                        continue
                    note = raw_value.replace("\n", " ").replace("\r", " ").strip()[:240]
                    if note:
                        result["notes"][rid] = note
                        found_any = True

        return result if found_any else None

    def _extract_json_candidates(self, text: str) -> List[Dict[str, Any]]:
        found: List[Dict[str, Any]] = []
        n = len(text)
        for start in range(n):
            if text[start] != "{":
                continue
            depth = 0
            in_string = False
            escaped = False
            for end in range(start, n):
                ch = text[end]
                if in_string:
                    if escaped:
                        escaped = False
                    elif ch == "\\":
                        escaped = True
                    elif ch == '"':
                        in_string = False
                    continue
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        obj = self._safe_load_json(text[start:end + 1])
                        if isinstance(obj, dict):
                            found.append(obj)
                        break
        return found

    @staticmethod
    def _safe_load_json(raw: str) -> Optional[Any]:
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _validate_visual_query_refinement(self, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed_qids = {
            str(row.get("qid", "")).strip()
            for row in payload.get("components", [])
            if isinstance(row, dict) and str(row.get("qid", "")).strip()
        }
        queries = data.get("queries", [])
        if not isinstance(queries, list):
            raise SchemaValidationError("visual_query_refinement queries must be a list")

        out = []
        seen = set()
        for row in queries[:24]:
            if not isinstance(row, dict):
                continue
            qid = str(row.get("qid", "")).strip()
            query = re.sub(r"\s+", " ", str(row.get("query", "") or "")).strip()
            if not qid or qid not in allowed_qids:
                raise SchemaValidationError("visual_query_refinement qid must reference an input component")
            if not query:
                continue
            if len(query) > 120:
                query = query[:120].rsplit(" ", 1)[0].strip() or query[:120].strip()
            if qid in seen:
                continue
            seen.add(qid)
            out.append({"qid": qid, "query": query})
        return {"queries": out}

    def _validate_wikipedia_section_select(self, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed_ids = {
            str(row.get("id", "")).strip()
            for row in payload.get("sections", [])
            if isinstance(row, dict) and str(row.get("id", "")).strip()
        }
        if not allowed_ids:
            allowed_ids = {"0"}

        sections = data.get("sections", [])
        if not isinstance(sections, list):
            raise SchemaValidationError("wikipedia_section_select sections must be a list")

        out: List[str] = []
        seen = set()
        for item in sections[:2]:
            sid = str(item).strip()
            if not sid:
                continue
            if sid not in allowed_ids:
                raise SchemaValidationError("wikipedia_section_select id must reference an input section")
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
        if not out:
            fallback = "0" if "0" in allowed_ids else next(iter(allowed_ids))
            out = [fallback]
        return {"sections": out[:2]}

    def _validate_wikipedia_visual_extract(self, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("visual_description", "")
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            raise SchemaValidationError("wikipedia_visual_extract visual_description must be non-empty")
        return {"visual_description": text[:700]}

    def _validate_visual_description_refinement(self, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        value = data.get("refined_description", "")
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            raise SchemaValidationError("visual_description_refinement refined_description must be non-empty")
        return {"refined_description": text[:900]}

    def _validate_target_choice(self, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        rows = [
            row for row in payload.get("candidates", [])
            if isinstance(row, dict) and str(row.get("id", "")).strip()
        ]
        allowed_ids = {
            str(row.get("id", "")).strip()
            for row in rows
        }
        picked = str(data.get("id", "")).strip()
        if not picked or picked not in allowed_ids:
            raise SchemaValidationError("choose_target returned id outside provided candidate ids")

        def score(row: Dict[str, Any]) -> tuple[int, int, int, str]:
            rid = str(row.get("id", "")).strip()
            name = str(row.get("n", row.get("name", "")) or "").strip().lower()
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
            exact = 1 if name == str(payload.get("target_text", "")).strip().lower() else 0
            return (bad, -good, -exact, -sitelinks, rid)

        by_id = {str(row.get("id", "")).strip(): row for row in rows}
        picked_row = by_id.get(picked, {})
        ranked = sorted(rows, key=score)
        best_id = str(ranked[0].get("id", "")).strip() if ranked else picked
        if best_id and best_id != picked and score(picked_row) > score(ranked[0]):
            picked = best_id

        alts = []
        cleaned_alts: List[str] = []
        for item in alts[:4]:
            text = str(item).strip()
            if not text or text == picked:
                continue
            if text not in allowed_ids:
                raise SchemaValidationError("choose_target alts contains unknown candidate id")
            if text not in cleaned_alts:
                cleaned_alts.append(text)

        why = str(data.get("why", "")).strip()[:96]
        return {"id": picked, "why": why, "alts": cleaned_alts}

    def _validate_action_result(self, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        decision = str(data.get("decision", "")).strip().lower()
        if decision not in {"act", "fallback", "stop"}:
            raise SchemaValidationError("choose_actions decision must be act, fallback, or stop")

        why = str(data.get("why", "")).strip()[:96]
        actions_raw = data.get("actions", [])
        if not isinstance(actions_raw, list):
            raise SchemaValidationError("choose_actions actions must be a list")

        allowed_node_ids = {"T"}
        allowed_node_ids.update(
            str(row.get("id", "")).strip()
            for group in (payload.get("state", {}) or {}).values()
            if isinstance(group, list)
            for row in group
            if isinstance(row, dict) and str(row.get("id", "")).strip()
        )

        allowed_memory_ids = {
            str(row.get("id", row.get("mid", ""))).strip()
            for row in payload.get("memory", [])
            if isinstance(row, dict) and str(row.get("id", row.get("mid", ""))).strip()
        }

        normalized_actions = []
        for row in actions_raw[:3]:
            try:
                normalized_actions.append(
                    self._validate_one_action(
                        row,
                        allowed_node_ids=allowed_node_ids,
                        allowed_memory_ids=allowed_memory_ids,
                    )
                )
            except SchemaValidationError:
                continue

        raw_memory_id = data.get("memory_id", "")
        memory_id = "" if raw_memory_id is None else str(raw_memory_id).strip()
        if decision == "act":
            memory_id = ""
        elif decision == "fallback":
            if normalized_actions:
                raise SchemaValidationError("choose_actions fallback decision must keep actions empty")
            if not memory_id:
                if len(allowed_memory_ids) == 1:
                    memory_id = next(iter(allowed_memory_ids))
                else:
                    raise SchemaValidationError("choose_actions fallback decision requires memory_id")
            if memory_id not in allowed_memory_ids:
                raise SchemaValidationError("choose_actions fallback memory_id is unknown")
        else:
            if normalized_actions:
                raise SchemaValidationError("choose_actions stop decision must keep actions empty")
            memory_id = ""

        return {
            "decision": decision,
            "why": why,
            "actions": normalized_actions,
            "memory_id": memory_id,
        }

    def _validate_one_action(
        self,
        row: Dict[str, Any],
        *,
        allowed_node_ids: Set[str],
        allowed_memory_ids: Set[str],
    ) -> Dict[str, Any]:
        if not isinstance(row, dict):
            raise SchemaValidationError("action must be an object")

        tool = str(row.get("tool", "")).strip()
        args = row.get("args", {})
        why = str(row.get("why", "")).strip()[:72]

        if not isinstance(args, dict):
            args = {}

        allowed_tools = {
            "seed",
            "open",
            "open_many",
            "scan",
            "survey",
            "check",
            "check_many",
            "probe_many",
            "memory",
            "accept_many",
            "hold_many",
            "reject_many",
            "save_memory",
            "expand_neighbors",
        }

        if tool not in allowed_tools:
            raise SchemaValidationError(f"unsupported tool: {tool}")

        if tool == "seed":
            return {
                "tool": "seed",
                "args": {"target": "T"},
                "why": why or "reload target seed",
            }

        if tool == "open":
            node = self._coerce_node_alias(
                args.get("node", args.get("id", args.get("qid", ""))),
                allowed_node_ids,
                allow_target=True,
                field_name="open.node",
            )
            return {"tool": "open", "args": {"node": node}, "why": why or "inspect node"}

        if tool == "open_many":
            nodes = self._coerce_node_list(
                args.get("nodes", args.get("ids", args.get("qids", []))),
                allowed_node_ids,
                allow_target=False,
                field_name="open_many.nodes",
                min_items=1,
                max_items=4,
            )
            return {"tool": "open_many", "args": {"nodes": nodes}, "why": why or "inspect nodes"}

        if tool == "scan":
            anchor = self._coerce_node_alias(
                args.get("anchor", args.get("node", args.get("qid", "T"))),
                allowed_node_ids,
                allow_target=True,
                field_name="scan.anchor",
            )
            route = self._coerce_choice(
                args.get("route", "dir"),
                {"dir", "rev", "cls", "char", "up"},
                field_name="scan.route",
                default="dir",
            )
            try:
                k = int(args.get("k", 4) or 4)
            except Exception:
                k = 4
            k = max(1, min(6, k))
            return {
                "tool": "scan",
                "args": {"anchor": anchor, "route": route, "k": k},
                "why": why or "scan route",
            }

        if tool == "survey":
            anchor = self._coerce_node_alias(
                args.get("anchor", args.get("node", args.get("qid", "T"))),
                allowed_node_ids,
                allow_target=True,
                field_name="survey.anchor",
            )
            try:
                k = int(args.get("k", 6) or 6)
            except Exception:
                k = 6
            k = max(3, min(8, k))
            return {
                "tool": "survey",
                "args": {"anchor": anchor, "k": k},
                "why": why or "survey frontier",
            }

        if tool == "check":
            candidate = self._coerce_node_alias(
                args.get("candidate", args.get("node", args.get("qid", ""))),
                allowed_node_ids,
                allow_target=False,
                field_name="check.candidate",
            )
            return {"tool": "check", "args": {"candidate": candidate}, "why": why or "confirm candidate"}

        if tool == "check_many":
            candidates = self._coerce_node_list(
                args.get("candidates", args.get("nodes", args.get("qids", []))),
                allowed_node_ids,
                allow_target=False,
                field_name="check_many.candidates",
                min_items=1,
                max_items=4,
            )
            return {"tool": "check_many", "args": {"candidates": candidates}, "why": why or "confirm candidates"}

        if tool == "probe_many":
            candidates = self._coerce_node_list(
                args.get("candidates", args.get("nodes", args.get("qids", []))),
                allowed_node_ids,
                allow_target=False,
                field_name="probe_many.candidates",
                min_items=1,
                max_items=4,
            )
            return {"tool": "probe_many", "args": {"candidates": candidates}, "why": why or "probe candidates"}

        if tool == "memory":
            memory_id = str(args.get("memory_id", args.get("mid", ""))).strip()
            if memory_id not in allowed_memory_ids:
                raise SchemaValidationError(f"memory references unknown id: {memory_id}")
            return {"tool": "memory", "args": {"memory_id": memory_id}, "why": why or "use saved route"}

        if tool in {"accept_many", "hold_many", "reject_many"}:
            candidates = self._coerce_node_list(
                args.get("candidates", args.get("nodes", [])),
                allowed_node_ids,
                allow_target=False,
                field_name=f"{tool}.candidates",
                min_items=1,
                max_items=8,
            )
            return {"tool": tool, "args": {"candidates": candidates}, "why": why or tool}

        if tool == "save_memory":
            route = args.get("route")
            if not isinstance(route, dict):
                raise SchemaValidationError("save_memory.route must be an object")

            route_tool = str(route.get("tool", "")).strip()
            if route_tool == "save_memory":
                raise SchemaValidationError("save_memory.route must not nest save_memory")

            normalized_route = self._validate_one_action(
                route,
                allowed_node_ids=allowed_node_ids,
                allowed_memory_ids=allowed_memory_ids,
            )

            label = str(args.get("label", "")).strip()[:80]
            return {
                "tool": "save_memory",
                "args": {"route": normalized_route, "label": label},
                "why": why or "save fallback route",
            }

        if tool == "expand_neighbors":
            anchor = self._coerce_node_alias(
                args.get("anchor", args.get("node", args.get("qid", ""))),
                allowed_node_ids,
                allow_target=False,
                field_name="expand_neighbors.anchor",
            )
            try:
                k = int(args.get("k", 6) or 6)
            except Exception:
                k = 6
            k = max(1, min(8, k))
            return {
                "tool": "expand_neighbors",
                "args": {"anchor": anchor, "k": k},
                "why": why or "expand local neighbors",
            }

        raise SchemaValidationError(f"unsupported tool: {tool}")

    def _validate_final_review(self, data: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
        allowed_ids = {
            str(row.get("id", "")).strip()
            for row in payload.get("candidates", [])
            if isinstance(row, dict) and str(row.get("id", "")).strip()
        }
        if not allowed_ids:
            return {"accept": [], "reject": [], "hold": [], "notes": {}}

        result: Dict[str, Any] = {"accept": [], "reject": [], "hold": [], "notes": {}}
        seen: Dict[str, set] = {"accept": set(), "reject": set(), "hold": set()}

        for key in ("accept", "reject", "hold"):
            value = data.get(key, [])
            if not isinstance(value, list):
                raise SchemaValidationError(f"final_review {key} must be a list")
            for item in value:
                text = str(item).strip()
                if not text:
                    continue
                if text not in allowed_ids:
                    raise SchemaValidationError(f"final_review {key} contains unknown review id")
                if text not in seen[key]:
                    result[key].append(text)
                    seen[key].add(text)

        accept_set = set(result["accept"])
        reject_set = set(result["reject"]) - accept_set
        hold_set = set(result["hold"]) - accept_set - reject_set
        result["accept"] = [rid for rid in result["accept"] if rid in accept_set]
        result["reject"] = [rid for rid in result["reject"] if rid in reject_set]
        result["hold"] = [rid for rid in result["hold"] if rid in hold_set]

        notes = data.get("notes", {})
        if not isinstance(notes, dict):
            raise SchemaValidationError("final_review notes must be an object")

        clean_notes: Dict[str, str] = {}
        for key, value in notes.items():
            rid = str(key).strip()
            if not rid or rid not in allowed_ids:
                continue
            clean_notes[rid] = str(value).replace("\n", " ").replace("\r", " ").strip()[:240]

        fallback_review = self._deterministic_final_review(payload)
        fallback_accept = set(fallback_review.get("accept", []))
        fallback_reject = set(fallback_review.get("reject", []))

        forced_accept: List[str] = []
        forced_reject: List[str] = []
        for rid in result["hold"]:
            if rid in fallback_accept:
                forced_accept.append(rid)
                clean_notes.setdefault(rid, fallback_review.get("notes", {}).get(rid, "forced accept in final review"))
            else:
                forced_reject.append(rid)
                clean_notes.setdefault(rid, fallback_review.get("notes", {}).get(rid, "forced reject in final review"))
        result["accept"].extend([rid for rid in forced_accept if rid not in result["accept"]])
        result["reject"].extend([rid for rid in forced_reject if rid not in result["reject"] and rid not in result["accept"]])
        result["hold"] = []

        covered = set(result["accept"]) | set(result["reject"])
        if not covered:
            return fallback_review

        if covered != allowed_ids:
            ordered_ids = [
                str(row.get("id", "")).strip()
                for row in payload.get("candidates", [])
                if isinstance(row, dict) and str(row.get("id", "")).strip()
            ]
            for rid in ordered_ids:
                if rid in covered:
                    continue
                if rid in fallback_accept:
                    result["accept"].append(rid)
                    clean_notes.setdefault(rid, fallback_review.get("notes", {}).get(rid, "forced accept in final review"))
                else:
                    result["reject"].append(rid)
                    clean_notes.setdefault(rid, fallback_review.get("notes", {}).get(rid, "forced reject in final review"))
                covered.add(rid)

        bucketed = set(result["accept"]) | set(result["reject"])
        result["notes"] = {rid: note for rid, note in clean_notes.items() if rid in bucketed}
        return result


@dataclass
class _QueuedInference:
    body: Dict[str, Any]
    event: threading.Event
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class ServerQwenWorker(QwenWorker):
    """
    C2 worker that preserves the existing prompt + validation surface while sending
    all generations through the shared vLLM server.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        max_new_tokens: int = 256,
        *,
        server_base_url: Optional[str] = None,
        stage_io_dir: Optional[str] = None,
        batch_window_ms: Optional[int] = None,
        max_batch_size: Optional[int] = None,
    ) -> None:
        self.model_name = str(model_name or DEFAULT_MODEL)
        self.max_new_tokens = int(max_new_tokens or 256)
        self.lock = threading.Lock()
        self.logger = logging.getLogger("qwen_worker.server")
        self.backend = "server"
        self.llm = None
        self.model = None
        self.tokenizer = None
        self.gpu_gc_every = 0
        self._gpu_gc_counter = 0
        self.fast_visual_fallbacks = str(os.environ.get("QWEN_FAST_VISUAL_FALLBACKS", "1")).strip().lower() not in {"0", "false", "no", "off"}
        self.prefix_cache_enabled = False
        self.prefix_cache_max_entries = 0
        self.prefix_cache_min_tokens = 0
        self.prefix_cache = OrderedDict()
        self.stage_io_dir = str(stage_io_dir or os.environ.get("QWEN_STAGE_IO_DIR", "") or "").strip()
        self._stage_io_bundle: Dict[str, Any] = {
            "stage_io_dir": self.stage_io_dir,
            "model_id": self.model_name,
            "backend": self.backend,
        }
        self.server_base_url = str(server_base_url or DEFAULT_SERVER_BASE_URL).rstrip("/")
        self.server_client = QwenServerClient(self.server_base_url)
        self.batch_window_ms = max(0, int(batch_window_ms or os.environ.get("C2_QWEN_BATCH_WINDOW_MS", "35") or 35))
        self.max_batch_size = max(1, int(max_batch_size or os.environ.get("C2_QWEN_MAX_BATCH_SIZE", "24") or 24))
        self._queue_lock = threading.Lock()
        self._pending: Dict[tuple[str, str], List[_QueuedInference]] = {}
        self._timers: Dict[tuple[str, str], threading.Timer] = {}
        self._stage_io_counter = 0

    def _gpu_gc(self, force: bool = False) -> None:
        return

    def infer(self, body: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("body must be an object")
        task = str(body.get("task", "choose_actions") or "choose_actions").strip()
        mode = str(body.get("mode", "normal") or "normal").strip().lower()
        if mode == "thinking":
            mode = "normal"
            body = dict(body)
            body["mode"] = "normal"
        return self._enqueue_for_batch(task=task, mode=mode, body=body)

    def infer_many(self, bodies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(bodies, list):
            raise ValueError("items must be a list")
        if not bodies:
            return []

        tasks = {str((body or {}).get("task", "")).strip() for body in bodies if isinstance(body, dict)}
        modes = {
            ("normal" if str((body or {}).get("mode", "normal")).strip().lower() == "thinking" else str((body or {}).get("mode", "normal")).strip().lower())
            for body in bodies
            if isinstance(body, dict)
        }
        if len(tasks) == 1 and len(modes) == 1 and next(iter(tasks)) != "entity_query_simplify":
            task = next(iter(tasks))
            mode = next(iter(modes))
            payloads: List[Dict[str, Any]] = []
            normalized_bodies: List[Dict[str, Any]] = []
            for body in bodies:
                if not isinstance(body, dict):
                    raise ValueError("each batch item must be an object")
                payload = body.get("payload", {})
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                normalized = dict(body)
                normalized["mode"] = mode
                normalized_bodies.append(normalized)
                payloads.append(payload)
            return self._infer_many_with_splitting(task=task, mode=mode, payloads=payloads)
        return [self.infer(body) for body in bodies]

    def _enqueue_for_batch(self, *, task: str, mode: str, body: Dict[str, Any]) -> Dict[str, Any]:
        key = (task, mode)
        item = _QueuedInference(body=dict(body), event=threading.Event())
        flush_now = False
        with self._queue_lock:
            batch = self._pending.setdefault(key, [])
            batch.append(item)
            if len(batch) >= self.max_batch_size:
                flush_now = True
                timer = self._timers.pop(key, None)
                if timer is not None:
                    timer.cancel()
            elif key not in self._timers:
                timer = threading.Timer(self.batch_window_ms / 1000.0, self._flush_key, args=(key,))
                timer.daemon = True
                self._timers[key] = timer
                timer.start()
        if flush_now:
            self._flush_key(key)
        item.event.wait()
        if item.error is not None:
            raise item.error
        if item.result is None:
            raise RuntimeError(f"batched infer produced no result for task={task}")
        return item.result

    def _flush_key(self, key: tuple[str, str]) -> None:
        with self._queue_lock:
            timer = self._timers.pop(key, None)
            if timer is not None:
                timer.cancel()
            batch = self._pending.pop(key, [])
        if not batch:
            return
        task, mode = key
        try:
            payloads = []
            for item in batch:
                payload = item.body.get("payload", {})
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                payloads.append(payload)
            results = self._infer_many_with_splitting(task=task, mode=mode, payloads=payloads)
            for item, result in zip(batch, results):
                item.result = result
        except Exception as exc:
            for item in batch:
                item.error = exc
        finally:
            for item in batch:
                item.event.set()

    def _write_stage_io(
        self,
        *,
        task: str,
        mode: str,
        payload: Dict[str, Any],
        system_prompt: str,
        user_prompt: str,
        rendered_input_text: str,
        raw_text: str,
        parsed_payload: Any,
    ) -> None:
        if not self.stage_io_dir:
            return
        try:
            stage_dir = Path(self.stage_io_dir)
            stage_dir.mkdir(parents=True, exist_ok=True)
            with self.lock:
                self._stage_io_counter += 1
                index = self._stage_io_counter
            out_path = stage_dir / f"c2_{index:05d}_{str(task or 'task').strip()}.json"
            payload_obj = {
                "task": str(task or "").strip(),
                "mode": str(mode or "").strip(),
                "model_name": self.model_name,
                "backend": self.backend,
                "server_base_url": self.server_base_url,
                "request": {
                    "payload": payload if isinstance(payload, dict) else {},
                    "system_prompt": str(system_prompt or ""),
                    "user_prompt": str(user_prompt or ""),
                    "rendered_input_text": str(rendered_input_text or ""),
                },
                "raw_text": str(raw_text or ""),
                "parsed_payload": parsed_payload,
            }
            out_path.write_text(json.dumps(payload_obj, ensure_ascii=False, indent=2), encoding="utf-8")
            parsed_keys = sorted(parsed_payload.keys()) if isinstance(parsed_payload, dict) else type(parsed_payload).__name__
            self.logger.info(
                "stage_io wrote task=%s mode=%s path=%s raw_chars=%s parsed=%s",
                task,
                mode,
                out_path,
                len(str(raw_text or "")),
                parsed_keys,
            )
        except Exception:
            self.logger.exception("Failed to write stage io debug for task=%s", task)

    def _task_generation(self, task: str, *, repair: bool = False) -> Dict[str, Any]:
        initial = int(self._task_budget(task, "normal").get("initial", self.max_new_tokens))
        max_new_tokens = initial if not repair else max(96, min(256, initial))
        return {
            "max_new_tokens": max(32, max_new_tokens),
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
            "skip_special_tokens": True,
        }

    def _server_generate_rows(
        self,
        *,
        task: str,
        prompts: List[str],
        system_prompt: str,
        repair: bool = False,
    ) -> List[str]:
        if not prompts:
            return []
        generation = self._task_generation(task, repair=repair)
        prompt_lengths = [len(str(prompt or "")) for prompt in prompts]
        self.logger.info(
            "qwen server generate start task=%s repair=%s batch=%s prompt_chars_total=%s prompt_chars_max=%s system_chars=%s generation=%s url=%s",
            task,
            repair,
            len(prompts),
            sum(prompt_lengths),
            max(prompt_lengths) if prompt_lengths else 0,
            len(str(system_prompt or "")),
            generation,
            self.server_base_url,
        )
        t0 = time.perf_counter()
        result = self.server_client.generate_text_batch(
            prompts=prompts,
            system_prompt=system_prompt,
            generation=generation,
            use_tqdm=False,
        )
        rows = result.get("responses")
        if not isinstance(rows, list) or len(rows) != len(prompts):
            raise RuntimeError(f"server batch response mismatch for task={task}")
        raw_rows = [self._normalize_raw_jsonish_text(str((row or {}).get("text", "") or "")) for row in rows]
        output_lengths = [len(text) for text in raw_rows]
        self.logger.info(
            "qwen server generate end task=%s repair=%s batch=%s elapsed_s=%.3f output_chars_total=%s output_chars_max=%s server_elapsed=%s output_tps=%s",
            task,
            repair,
            len(prompts),
            time.perf_counter() - t0,
            sum(output_lengths),
            max(output_lengths) if output_lengths else 0,
            result.get("elapsed_s"),
            result.get("output_tokens_per_s"),
        )
        for idx, text in enumerate(raw_rows[:8]):
            preview = re.sub(r"\s+", " ", text).strip()[:240]
            self.logger.debug("qwen raw preview task=%s index=%s chars=%s preview=%r", task, idx, len(text), preview)
        return raw_rows

    def _infer_many_with_splitting(self, task: str, mode: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not payloads:
            return []
        self.logger.info("server infer split start task=%s mode=%s payloads=%s", task, mode, len(payloads))
        try:
            result = self._infer_many_batched(task=task, mode=mode, payloads=payloads)
            self.logger.info("server infer split complete task=%s mode=%s payloads=%s results=%s", task, mode, len(payloads), len(result))
            return result
        except Exception:
            self.logger.exception("server batched infer failed for task=%s batch_size=%s", task, len(payloads))
            if len(payloads) <= 1:
                raise
            mid = max(1, len(payloads) // 2)
            left = self._infer_many_with_splitting(task=task, mode=mode, payloads=payloads[:mid])
            right = self._infer_many_with_splitting(task=task, mode=mode, payloads=payloads[mid:])
            return left + right

    def _repair_json_output(self, task: str, payload: Dict[str, Any], raw_text: str) -> Optional[Dict[str, Any]]:
        formatter_prompts = {
            "choose_target": TARGET_FORMATTER_SYSTEM_PROMPT,
            "choose_actions": ACTIONS_FORMATTER_SYSTEM_PROMPT,
            "final_review": FINAL_REVIEW_FORMATTER_SYSTEM_PROMPT,
            "visual_query_refinement": VISUAL_QUERY_REFINER_FORMATTER_SYSTEM_PROMPT,
            "wikipedia_section_select": WIKIPEDIA_SECTION_SELECT_FORMATTER_SYSTEM_PROMPT,
            "wikipedia_visual_extract": WIKIPEDIA_VISUAL_EXTRACT_FORMATTER_SYSTEM_PROMPT,
            "visual_description_refinement": VISUAL_DESCRIPTION_REFINER_FORMATTER_SYSTEM_PROMPT,
        }
        system_prompt = formatter_prompts.get(task)
        if not system_prompt:
            return None

        compact_input = json.dumps(self._prompt_payload(task, payload), ensure_ascii=False, separators=(",", ":"))
        normalized_raw = self._normalize_raw_jsonish_text(raw_text)
        if len(normalized_raw) > 2400:
            normalized_raw = normalized_raw[-2400:]
        user_prompt = (
            f"TASK={task}\n"
            f"INPUT={compact_input}\n"
            f"RAW={normalized_raw}\n"
            "Convert RAW into ONE valid JSON object only.\n"
            "Rules:\n"
            "- preserve the intended decision/classification\n"
            "- notes values must be single-line short strings\n"
            "- for choose_actions, decision must be act/fallback/stop\n"
            "- for choose_actions, actions must be a list of action objects\n"
            "- for choose_actions, candidate lists must contain alias strings like N1, N2, not nested objects\n"
            "- no markdown\n"
            "- no prose outside JSON\n"
        )
        try:
            self.logger.info("qwen repair start task=%s raw_chars=%s", task, len(str(raw_text or "")))
            rows = self._server_generate_rows(
                task=task,
                prompts=[user_prompt],
                system_prompt=system_prompt,
                repair=True,
            )
        except Exception:
            self.logger.exception("server repair prompt failed for task=%s", task)
            return None
        repaired_text = rows[0] if rows else ""
        repaired = self._extract_best_json(repaired_text, task=task)
        self.logger.info("qwen repair end task=%s repaired_chars=%s parsed=%s", task, len(repaired_text), sorted(repaired.keys()) if isinstance(repaired, dict) else None)
        return repaired

    def _generate(self, task: str, system_prompt: str, user_prompt: str, mode: str) -> Dict[str, Any]:
        normalized_mode = "normal" if str(mode or "normal").strip().lower() == "thinking" else str(mode or "normal").strip().lower()
        rows = self._server_generate_rows(
            task=task,
            prompts=[user_prompt],
            system_prompt=system_prompt,
        )
        raw_text = rows[0] if rows else ""
        return {
            "raw_text": raw_text,
            "thinking_text": "",
            "final_text": raw_text,
            "parse_text": raw_text,
            "saw_close_tag": False,
            "mode": normalized_mode,
        }

    def _infer_many_batched(self, task: str, mode: str, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.logger.info("server infer batch build task=%s mode=%s payloads=%s", task, mode, len(payloads))
        prompts: List[str] = []
        prompt_rows: List[Dict[str, str]] = []
        for payload in payloads:
            system_prompt, user_prompt = self._task_prompt(task, payload, mode)
            prompt_rows.append({"system_prompt": system_prompt, "user_prompt": user_prompt})
            prompts.append(user_prompt)
        if not prompt_rows:
            return []

        system_prompt = str(prompt_rows[0].get("system_prompt", "") or "")
        raw_rows = self._server_generate_rows(
            task=task,
            prompts=prompts,
            system_prompt=system_prompt,
        )

        results: List[Dict[str, Any]] = []
        for idx, payload in enumerate(payloads):
            raw_text = str(raw_rows[idx] if idx < len(raw_rows) else "")
            parse_t0 = time.perf_counter()
            result = self._parse_or_repair_task_result(task, payload, raw_text)
            self.logger.info(
                "server infer parse task=%s index=%s raw_chars=%s elapsed_s=%.3f parsed_keys=%s",
                task,
                idx,
                len(raw_text),
                time.perf_counter() - parse_t0,
                sorted(result.keys()) if isinstance(result, dict) else type(result).__name__,
            )
            prompt_row = prompt_rows[idx] if idx < len(prompt_rows) else {}
            self._write_stage_io(
                task=task,
                mode=mode,
                payload=payload,
                system_prompt=str(prompt_row.get("system_prompt", "") or ""),
                user_prompt=str(prompt_row.get("user_prompt", "") or ""),
                rendered_input_text=str(prompts[idx] if idx < len(prompts) else ""),
                raw_text=raw_text,
                parsed_payload=result,
            )
            results.append(result)
        return results


class WorkerHttpHandler(BaseHTTPRequestHandler):
    worker: QwenWorker
    server_version = "QwenWorker/2.3"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "model": self.worker.model_name, "modes": ["normal", "thinking"]})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path not in {"/infer", "/infer_batch"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            body = json.loads(raw)
            if self.path == "/infer_batch":
                items = body.get("items", [])
                results = self.worker.infer_many(items)
                self._send_json(HTTPStatus.OK, {"ok": True, "results": results})
            else:
                result = self.worker.infer(body)
                self._send_json(HTTPStatus.OK, {"ok": True, "result": result})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": str(exc), "type": exc.__class__.__name__},
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.getLogger("qwen_worker.http").info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: HTTPStatus, body: Dict[str, Any]) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent Qwen worker server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="cap for choose_actions generation")
    parser.add_argument("--log-level", default="INFO", help="logging level")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    worker = QwenWorker(model_name=args.model, max_new_tokens=args.max_new_tokens)
    WorkerHttpHandler.worker = worker
    server = ThreadingHTTPServer((args.host, args.port), WorkerHttpHandler)
    logging.getLogger("qwen_worker").info("Worker listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logging.getLogger("qwen_worker").info("Shutting down")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
