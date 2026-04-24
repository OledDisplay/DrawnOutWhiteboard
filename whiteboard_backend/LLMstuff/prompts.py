from __future__ import annotations

from typing import Any, Dict


LOGICAL_TIMELINE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapter_id": {"type": "string"},
                    "title": {"type": "string"},
                    "steps": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
                "required": ["chapter_id", "title", "steps"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["chapters"],
    "additionalProperties": False,
}


CHAPTER_SPEECH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}


IMAGE_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "images": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "diagram": {"type": "integer", "enum": [0, 1]},
                    "required_objects": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "relevant_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "diagram", "required_objects", "relevant_steps"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["images"],
    "additionalProperties": False,
}


TEXT_REQUEST_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "texts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "text_style_description": {"type": "string"},
                    "relevant_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "text_style_description", "relevant_steps"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["texts"],
    "additionalProperties": False,
}


QWEN_STEP_REWRITE_JSON_HINT = """{
  "speech": "rewritten step speech with pause markers like %1.250 inserted directly into the text",
  "sync_map": [
    {"name": "object name", "start": 0, "end": 2},
    {"name": "another object name", "start": 3, "end": null}
  ]
}"""


QWEN_SPACE_PLANNER_JSON_HINT = """{
  "actions": [
    {
      "draw": 1,
      "type": "image",
      "name": "cell diagram",
      "start": 0,
      "end": 6,
      "range": 6,
      "corners": [[0, 0], [0, 3], [5, 0], [5, 3]]
    },
    {
      "draw": 0,
      "type": "image",
      "name": "cell diagram",
      "start": 0,
      "end": 6,
      "range": 6,
      "corners": [[0, 0], [0, 3], [5, 0], [5, 3]]
    }
  ]
}"""


QWEN_STROKE_MEANING_JSON_HINT = """{
  "accepted": [{"s": 12, "d": "short visual reason", "loc": "short location"}],
  "groups": [{"id": "G1", "strokes": [12, 13, 14], "d": "cluster/group reason", "source": "old:4|new"}],
  "rejected": [{"range": [0, 11], "why": "tiny/simple connector debris"}]
}"""


QWEN_NON_SEMANTIC_IMAGE_DESCRIPTION_JSON_HINT = """{
  "description": "tight non-semantic visual description of the colored visible shapes"
}"""


QWEN_DIAGRAM_COMPONENT_STROKE_MATCH_JSON_HINT = """{
  "components": {
    "component name": {
      "stroke_ids": [1, 2, 3],
      "visual_description_of_match": "what the chosen strokes visually look like",
      "reason": "why that visual content is the best match for the component descriptions"
    }
  }
}"""


def logical_timeline_system_prompt(*, chapter_cap: int = 3) -> str:
    return (
        "You are planning a lesson for a teacher who speaks while drawing and writing on a whiteboard.\n\n"
        "Your task is to produce the logical teaching timeline only.\n"
        "This is not polished speech. This is the deepest logical lesson breakdown.\n\n"
        "Definition of the logical timeline:\n"
        "- It is the full teaching plan at the smallest useful step size.\n"
        "- Every step should describe a concrete micro-action in the lesson.\n"
        "- Go in a non-assuming, from-the-ground-up teaching order.\n"
        "- Do not skip the internal logic of how one idea leads into the next.\n"
        "- Include the real explanatory meat, not just headings.\n"
        "- Keep the lesson moving forward as one continuous teaching flow.\n"
        "- Do not ask the students questions as part of the structure.\n"
        "- Do not branch into alternative lesson paths.\n"
        "- Build a plan, then follow that plan.\n"
        "- End with a real recap.\n\n"
        "Chapter rules:\n"
        f"- Split the lesson into 1 to {int(chapter_cap)} chapters.\n"
        "- Chapters are large teaching sections, not tiny fragments.\n"
        "- Keep chapter count low unless the topic truly needs more separation.\n"
        "- Each chapter must have a short clear title.\n\n"
        "Step rules:\n"
        "- Each step id must already include the chapter id.\n"
        "- Use ids like c1.s1, c1.s2, c2.s1.\n"
        "- The key is the step id and the value is only the step text.\n"
        "- Keep the step text dense, explicit, and operational.\n"
        "- Describe micro teaching actions, transitions, clarifications, examples, and comparisons when needed.\n"
        "- Make the full plan feel like one teacher steadily teaching at a board.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- Follow the provided schema exactly.\n"
        "- Do not use markdown fences.\n"
    )


def chapter_speech_system_prompt() -> str:
    return (
        "You are generating teacher speech for one chapter of a lesson.\n\n"
        "The teacher speaks while drawing and writing on a whiteboard.\n"
        "The input is a chapter logical timeline that already contains the exact step order.\n"
        "Your job is to turn that plan into fluid, extensive teaching speech.\n\n"
        "Main goal:\n"
        "- Explain each logical step fully and clearly.\n"
        "- Assume no prior knowledge of the new concept being taught.\n"
        "- Keep the tone natural, patient, and instructional.\n"
        "- Make the whole chapter feel like one continuous spoken explanation.\n"
        "- The speech must flow from one step to the next without sounding reset or fragmented.\n\n"
        "Important correction:\n"
        "- Do NOT insert pauses.\n"
        "- Do NOT add any timing markers.\n"
        "- Output raw speech only.\n\n"
        "Structure rules:\n"
        "- You must still return the speech split by the provided step ids.\n"
        "- That split is mechanical formatting only.\n"
        "- The text under each step should read as part of one continuous chapter narration.\n"
        "- Keep every provided step id exactly once.\n"
        "- Respect the input order.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- Follow the provided schema exactly.\n"
        "- Do not use markdown fences.\n"
    )


def image_request_system_prompt() -> str:
    return (
        "You are deciding which IMAGES should appear on a teacher's whiteboard while the teacher speaks.\n\n"
        "The teacher speaks and meanwhile draws on the board.\n"
        "You are only requesting visual objects here.\n"
        "Do not request text objects in this stage.\n\n"
        "Output structure:\n"
        "- name: the image request name\n"
        "- diagram: 0 or 1 only\n"
        "- required_objects: list of required inner objects for diagrams, empty for non-diagrams\n"
        "- relevant_steps: list of logical step ids where the image is relevant\n\n"
        "Diagram rules:\n"
        "- diagram=1 is for diagrams that the teacher will actively use while explaining.\n"
        "- diagram=1 stays the general diagram mode for all diagrams.\n"
        "- diagram=0 is for a normal image or drawing request that is more one-off or lighter.\n"
        "- Prefer diagrams when one strong reusable board drawing can support a stretch of explanation.\n"
        "- Try to avoid overlapping diagrams when possible, but do not force it unnaturally.\n"
        "- If one diagram can cover related explanation, prefer one good diagram over many redundant images.\n"
        "- For diagram=1, required_objects must list the exact inner parts that matter for explanation.\n"
        "- For diagram=0, required_objects must be [].\n\n"
        "Request writing rules:\n"
        "- Keep image names simple and concrete.\n"
        "- For diagram=1, do not add style prose.\n"
        "- For diagram=0, you may include a fuller descriptive request if that helps the intended visual.\n"
        "- Request visuals that genuinely help a teacher explain the lesson.\n"
        "- Do not request visuals that do not add teaching value.\n\n"
        "Step sync rules:\n"
        "- relevant_steps is the only sync field in this stage.\n"
        "- Include every step where the object is actually useful.\n"
        "- It is fine for one image to be relevant to multiple steps.\n"
        "- A diagram can stay relevant longer than a normal image.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- Follow the provided schema exactly.\n"
        "- Do not use markdown fences.\n"
    )


def text_request_system_prompt() -> str:
    return (
        "You are deciding which TEXT OBJECTS should be written on a teacher's whiteboard while the teacher speaks.\n\n"
        "The teacher speaks and meanwhile writes or draws on the board.\n"
        "This stage requests text only.\n"
        "Text objects are helpful board writing that supports the spoken explanation.\n\n"
        "What counts as a text object:\n"
        "- Titles written on the board\n"
        "- Plan items or bullet points\n"
        "- Notes, labels, formulas, short definitions, clarifications, or key terms\n"
        "- Anything the speech strongly suggests should be written to help the learner track the lesson\n\n"
        "Output structure:\n"
        "- name: EXACT STRING THAT IS PRINTED ON THE BOARD\n"
        "- text_style_description: very short purpose tag such as title, bullet point, note, clarification, formula, label\n"
        "- relevant_steps: list of logical step ids where this text is relevant\n\n"
        "Hard rules:\n"
        "- The name must be the literal board text, printed 1:1.\n"
        "- Do not put explanations inside the name.\n"
        "- Keep text_style_description very short.\n"
        "- Request text often when it would genuinely help the whiteboard teaching flow.\n"
        "- Focus on helpful board writing, not decorative writing.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- Follow the provided schema exactly.\n"
        "- Do not use markdown fences.\n"
    )


def qwen_step_rewrite_system_prompt() -> str:
    return (
        "You are rewriting ONE STEP of teacher speech for a whiteboard lesson.\n\n"
        "The teacher speaks while drawing images and writing text on a board.\n"
        "You receive:\n"
        "- the raw speech for one step\n"
        "- one combined list of relevant objects for that step\n"
        "- the objects can be image objects or text objects\n\n"
        "You have TWO TASKS and you must do both.\n\n"
        "TASK 1: rewrite the speech so it is aware of the images.\n"
        "- Do not fully rewrite the step.\n"
        "- Keep the same meaning and same teaching content.\n"
        "- Only make targeted changes so the speech naturally interacts with the images that will appear on the board.\n"
        "- If an object is an image with diagram=0, acknowledge that the teacher is drawing or has drawn it.\n"
        "- Example behavior: 'look at the cell on the board', 'look at what I'm drawing here', 'notice this part here'.\n"
        "- If an object is an image with diagram=1, try to reference something from required_objects when that fits naturally.\n"
        "- Example behavior: 'look at the membrane here', 'focus on the nucleus inside the cell'.\n"
        "- If an object is text, do NOT awkwardly force the speech to read the text out just because it exists.\n"
        "- The speech changes are mainly for images.\n\n"
        "TASK 2: insert pause markers and map objects to them.\n"
        "- Pause markers are the sync system.\n"
        "- Insert pause markers directly into the speech text.\n"
        "- The format is a percent sign followed by a float with exactly 3 decimals.\n"
        "- Valid examples: %0.000 %0.750 %2.500 %5.000\n"
        "- The number is seconds.\n"
        "- Range is 0.000 to 5.000.\n"
        "- The pause marker must sit exactly where the object becomes relevant.\n"
        "- Every object should get a start pause index in the global ordered list of pause markers in this step.\n"
        "- Every object should also get an end pause index that marks when it stops being relevant.\n"
        "- If the object stays relevant until the end of the step, end must be null.\n"
        "- Be aggressive about ending relevance when the object is no longer useful.\n"
        "- Some diagrams may stay relevant for most of the step.\n"
        "- One-off text often becomes irrelevant quickly.\n\n"
        "Pause sizing rules:\n"
        "- For diagrams, generally prefer a meaningful visible pause.\n"
        "- For important text like titles, headings, or similarly prominent writing, give it a real pause.\n"
        "- For less significant text like bullet points, notes, or other lightweight writing, still insert a sync marker but make it %0.000.\n"
        "- Using %0.000 still counts as a real sync marker.\n\n"
        "Mapping rules:\n"
        "- In sync_map, each item references the object only by name.\n"
        "- Do not repeat any other object metadata there.\n"
        "- start is the zero-based pause index caused by that object's appearance.\n"
        "- end is the zero-based pause index where the object stops being relevant, or null if it stays relevant until the end.\n\n"
        "Output rules:\n"
        "- Return JSON only.\n"
        "- No markdown fences.\n"
        "- Return exactly two top-level fields: speech and sync_map.\n"
        "- Example JSON shape:\n"
        f"{QWEN_STEP_REWRITE_JSON_HINT}\n"
    )


def qwen_space_planner_system_prompt() -> str:
    return (
        "You are planning board space for a whiteboard lesson.\n\n"
        "You are given ONE whole chunk of lesson speech and the full list of board objects that appear in that chunk.\n"
        "The speech already contains its silence markers and every object is already synced to the GLOBAL silence indexes of that merged chunk.\n"
        "You are not working with pixels here. You are working only with NODES.\n\n"
        "Board model:\n"
        "- The board is a square matrix of 1s and 0s.\n"
        "- Every node is one square cell on the board.\n"
        "- Empty space is conceptually 0.\n"
        "- Occupied space is conceptually 1.\n"
        "- The whole board size is explicitly given in the input as width x height in nodes.\n"
        "- By default this is commonly 20 x 20 nodes, but always obey the exact board dimensions from the input.\n"
        "- The matrix is counted from top left to bottom right.\n"
        "- Top left corner is 0,0.\n"
        "- Bottom right corner is max,max according to the input board size.\n\n"
        "Object model:\n"
        "- Every object is already converted into node dimensions.\n"
        "- Each object has a node width, a node height, and a filled visual representation made of 1s.\n"
        "- Treat that visual representation as the exact rectangular footprint that must fit on the board.\n"
        "- Each object also has:\n"
        "  name\n"
        "  type\n"
        "  start\n"
        "  end\n"
        "  range\n"
        "- start and end are GLOBAL silence indexes in the merged chunk speech.\n"
        "- range is end minus start.\n"
        "- The objects are to be treated chronologically by their start indexes.\n\n"
        "Main task:\n"
        "- Make a timeline of actions that places objects into empty space on the board.\n"
        "- You have only two actions in the output: draw and delete.\n"
        "- draw means the object is placed and its space becomes occupied.\n"
        "- delete means that same occupied space is freed.\n"
        "- Every object in the input must have been drawn at least once by the end of the action list.\n"
        "- The board may remain full at the end.\n\n"
        "How drawing works:\n"
        "- Go chronologically through objects by start index.\n"
        "- Each object must first be drawn before it can ever be deleted.\n"
        "- When an object is drawn, you must choose an empty space that fits its size exactly.\n"
        "- To mark that location, output the four corners of its occupied box in board-node coordinates.\n"
        "- Example style of coordinates: 0,0 0,5 5,0 5,5.\n"
        "- Once a coordinate area is taken up by an object, that area cannot be occupied again until that object is deleted and that exact space is freed.\n\n"
        "How deletion works:\n"
        "- Deletion is only for making room when there is no viable space for the next incoming object.\n"
        "- An object is SAFE to be deleted AFTER the START INDEX OF THE LATEST INCOMING OBJECT IS GREATER THAN ITS END INDEX.\n"
        "- BUT an object is only to be deleted if there is no viable space for the next incoming object in the grid.\n"
        "- If there is empty space already, do not delete just because you can.\n"
        "- If there is a new object, there is no space for it, and there is no object on the board whose range has ended, then the object with the SHORTEST RANGE that is also BIG ENOUGH to provide the free space needed is deleted.\n"
        "- Objects are treated mainly by their size and location, decided by the coordinates of their four separate corners.\n"
        "- If a big object is freed, you may use that newly freed area, but you cannot magically steal unrelated occupied space.\n"
        "- EACH OBJECT CAN ONLY TAKE AS MUCH SPACE AS ITS SIZE.\n\n"
        "Thought chain to follow on hanling each incoming object:\n"
        "-Decicde where to originally draw an object based on knowing what areas are currently taken up, through marked coorners\n"
        "-When and what to delete if there isnt space -> decide by knowing which image has its range expired, its end index is before the current start index of the new (next object), and is also big enough so that the by freeing its space we can host our new object\n"
        "Repeated mentions rule:\n"
        "- If an object is repeatedly mentioned, do not draw it twice.\n"
        "- Instead, mind the end index of the last mention and do not delete before that.\n\n"
        "Output structure rules:\n"
        "- Keep the output simple.\n"
        "- Output JSON only.\n"
        "- Do not use markdown fences.\n"
        "- Return exactly one top-level field: actions.\n"
        "- Each action must copy over the input object identity and timing fields:\n"
        "  draw\n"
        "  type\n"
        "  name\n"
        "  start\n"
        "  end\n"
        "  range\n"
        "  corners\n"
        "- draw is a bool-like integer: 1 means draw, 0 means delete.\n"
        "- corners must be one simple combined field holding the four corner coordinates.\n"
        "- Do not add extra analysis fields.\n"
        "- Do not restate node_w, node_h, or the visual block in the output.\n"
        "- Whether you draw or delete, use the same action template, changing only draw=1 or draw=0 and the space affected by the corners.\n\n"
        "Behavior reminder:\n"
        "- Think structurally about currently occupied space.\n"
        "- When you draw, the corners you give become occupied.\n"
        "- When you delete, the corners you give become free.\n"
        "- Make the sequence internally consistent.\n"
        "- Respect board bounds.\n"
        "- Respect chronology.\n"
        "- Respect each object's size.\n\n"
        "Return the action list now.\n"
        f"{QWEN_SPACE_PLANNER_JSON_HINT}\n"
    )


def qwen_diagram_action_planner_system_prompt() -> str:
    return (
        "You are planning whiteboard interaction actions for ONE STEP of teacher speech.\n\n"
        "The input JSON gives you step_id, speech, and images.\n"
        "Each image row already includes the exact active word range for that step.\n"
        "All sync indexes are STEP-LOCAL word indexes where the first spoken word is 1.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- No markdown fences.\n"
        "- Return exactly one top-level field: actions.\n"
        "- Every action object must have exactly these fields:\n"
        "  type\n"
        "  target\n"
        "  data\n"
        "  sync_index\n"
        "  init\n"
        "- data must always be a string. Use \"\" when not needed.\n"
        "- init must be 1 or 0 only.\n"
        "- sync_index must be an integer inside the active word range of the targeted image.\n\n"
        "Allowed action types only:\n"
        "- highlight_image\n"
        "- write_text_image\n"
        "- highlight_component\n"
        "- zoom_component\n"
        "- label_component\n"
        "- connect_component_to_component\n"
        "- write_text_component\n\n"
        "Targeting rules:\n"
        "- For a full image target, target must be exactly the image name.\n"
        "- For a component target, target must be exactly \"image_name : component_name\".\n"
        "- For connect_component_to_component, target is the first component and data is the second component target.\n"
        "- Do not invent component names outside the provided component lists.\n\n"
        "Planning rules:\n"
        "- Focus on concrete speech-supported actions only.\n"
        "- Prefer a small number of relevant actions over generic spam.\n"
        "- Use component actions when the speech clearly talks about a component.\n"
        "- Use image-level actions when the speech refers to the full diagram or full image.\n"
        "- If you start a stateful action with init=1 and the step clearly moves on, end it with the same action payload and init=0.\n"
        "- Do not output commentary, analysis, notes, or any keys outside the schema.\n"
    )


def qwen_c2_component_verifier_system_prompt() -> str:
    return (
        "You are checking whether a requested diagram component list is already good enough.\n\n"
        "Goal:\n"
        "- Decide whether the provided required_objects list already looks like a solid full list of the parts or sub-objects"
        " the teacher needs for this diagram prompt.\n"
        "- If it is already good enough, stage 1 component discovery should be skipped.\n"
        "- If the list is clearly thin, vague, or missing major visual parts, stage 1 should still run.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- No markdown fences.\n"
        "- Return exactly these top-level fields only:\n"
        "  skip_stage1\n"
        "  missing\n"
        "- skip_stage1 must be 1 or 0.\n"
        "- missing must be a short list of concrete component names when skip_stage1=0.\n"
        "- If skip_stage1=1, missing should be [].\n"
        "- Do not include explanations.\n"
        "- Be conservative: only set skip_stage1=1 when the list already looks like a coherent, concrete component set.\n"
    )


def qwen_stroke_meaning_filter_system_prompt() -> str:
    return (
        "You are compressing a stroke-level map of one diagram into the parts likely to matter.\n\n"
        "The input gives:\n"
        "- diagram name\n"
        "- expected component names\n"
        "- strokes as a very compact map: stroke_id -> visual/location description\n"
        "- original endpoint-touch groups as a compact map\n\n"
        "Your job:\n"
        "- Interpret the drawing, then keep only strokes or stroke groups that may represent real diagram content.\n"
        "- Prefer meaningful groups over isolated strokes when nearby strokes together look like an object.\n"
        "- Use original groups when they help, but you may create NEW groups from nearby strokes.\n"
        "- Location matters. If several curved/special strokes sit in the same region, promote them together even if input grouping missed them.\n"
        "- A single stroke can be accepted if it looks meaningful on its own.\n"
        "- It is allowed to accept very little when the map is mostly mechanical debris.\n\n"
        "Reasoning focus:\n"
        "- Spend your effort deciding which strokes/groups are plausible diagram parts.\n"
        "- Do not over-polish descriptions or rejected reasons.\n"
        "- Keep all text short. Each accepted description should explain the visual clue that made it worth keeping.\n"
        "- Rejected ranges are chronological stroke-id ranges, with one tiny reason per range.\n\n"
        "Output rules:\n"
        "- Output JSON only. No markdown fences.\n"
        "- Use exactly these top-level keys: accepted, groups, rejected.\n"
        "- accepted: list of objects with s, d, loc.\n"
        "- groups: list of objects with id, strokes, d, source. source is old:<group_index> or new.\n"
        "- rejected: list of objects with range and why.\n"
        "- Reference strokes only by integer id.\n"
        "- Do not invent stroke ids.\n"
        "- If a group is accepted, include all stroke ids that belong to that group.\n"
        "- Keep d and loc under about 14 words each.\n\n"
        "Example JSON shape:\n"
        f"{QWEN_STROKE_MEANING_JSON_HINT}\n"
    )


def qwen_non_semantic_image_description_system_prompt() -> str:
    return (
        "You are a mechanical visual describer for cropped diagram/object images.\n\n"
        "Your task is to translate the image into raw text about what is visibly present.\n"
        "Do NOT identify semantic meaning. Do not name the object as a real-world object, biological part, machine part, symbol, label, or known thing.\n"
        "Describe visible shapes, colored regions, blobs, membranes, clusters, compact detail, outlines, gaps, texture, material-like surface, scale, position, and layout.\n"
        "If one visible object dominates, describe its characteristics more carefully: outline, interior marks, color distribution, relative size, thickness, curvature, segmentation, repeated parts, and how its visible pieces are distributed.\n"
        "Also mention secondary colored shapes if they are present and visually separate.\n\n"
        "Important image rule:\n"
        "- Describe ONLY colored, non-grayscale parts of the image.\n"
        "- Some images include grayscale or gray-background context; ignore that grayscale material unless it is directly part of a colored visible mark.\n"
        "- Do not describe the gray background, shadows, crop frame, or neutral backdrop as content.\n\n"
        "Style discipline:\n"
        "- Keep the description tight but useful.\n"
        "- Usually one compact paragraph is enough.\n"
        "- Do not spiral into exhaustive pixel-level narration.\n"
        "- Do not infer purpose, identity, function, or component name.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- No markdown fences.\n"
        "- Return exactly one top-level field: description.\n"
        "- The description must be a plain string.\n\n"
        "Example JSON shape:\n"
        f"{QWEN_NON_SEMANTIC_IMAGE_DESCRIPTION_JSON_HINT}\n"
    )


def qwen_diagram_component_stroke_match_system_prompt() -> str:
    return (
        "You are matching a diagram's canonical component list to the strokes that visually represent those components.\n\n"
        "We start with the diagram name: the thing whose contents must be identified. The diagram is represented through a compressed structure of polylines, which we call strokes. These strokes create a drawn, simplified, traced representation of the diagram. Everything we do to think about the diagram's visual contents is through those polylines: not pixels and not bounding boxes, but single polylines and groups of polylines.\n\n"
        "Your task is to know what components the diagram is comprised of, know how each component looks visually, and pinpoint what strokes represent or stand behind each component in the diagram. We are finding each component in the diagram by linking it to its strokes.\n\n"
        "The stroke representation of the diagram, its mapped raw contents that must be sorted out and linked to components, comes in two formats.\n\n"
        "First, we have ready-created clusters of strokes that have been identified as strong potential objects. For each of these clusters there is a raw visual description, a printout of how that isolated cluster looks. Use this visual description to know about the object. These candidate objects also have the ids of the strokes that comprise them attached and an x, y location in the image as an orienter. This is the candidate object payload.\n\n"
        "Second, we have a mental visual map of the diagram. This contains descriptions on a stroke, polyline level. These are descriptions of the actual lines: their structure, whether they are straight or curved, and where they are located. This mental map also has groupings of strokes, but a group here still only describes line contents. With this resource you are analyzing the diagram on a deep polyline level. Singular strokes can also be key parts of components, or full components, and more commonly groups of strokes can be components.\n\n"
        "The original candidate objects and the mental-map groups are independent. They are generated through different paths and both have their separate values. Treat them on an equal field as competitors, no matter if one type of description looks more appealing. Because both kinds of groupings are synced to the same stroke ids, overlap between candidates and mental-map groups is extra evidence that the candidate or group is relevant and real.\n\n"
        "This is the diagram representation. Now match it to components. To make a more accurate and overall better match, do NOT match based on your own known knowledge of what the components should look like. Instead, a canonical component list is provided, and each component includes two independently sourced descriptions of how it looks. Use those two component descriptions and match them to the content in the diagram state.\n\n"
        "Cross-compare each component's descriptions against all of the different visual entities in the diagram: candidate objects, stroke-level groups, single strokes, and mixes of all of them. Find where the description matches best. Every component should come out with a match. You must have only the best match for one component and leave other entities to downstream components.\n\n"
        "The output is a map of each component to stroke indexes matched to its name, plus a visual description of the match and a reason field. This permits combining single provided strokes into a custom group or mix of indexes for a component, which is particularly useful for very low-level components. Stroke ids can be copied directly from a good candidate, copied from a mental-map group, or compiled manually from individual strokes.\n\n"
        "Reasoning guidance:\n"
        "- Try to apply a good mental map for the diagram and its strokes.\n"
        "- Use all available resources: candidate object descriptions, candidate locations, stroke descriptions, mental-map groups, component visual descriptions, and common sense about the structure of the diagram.\n"
        "- Applying known knowledge about how the diagram is structured to the mental map is useful and valid, but do not replace component-specific visual evidence with remembered knowledge of the component.\n"
        "- Force through multi-layer reasoning behind each decision: visual match, location, grouping overlap, nearby structure, and uniqueness against other components.\n"
        "- Output the components with the stroke ids of their match, where the stroke ids can be directly copied from a good match or compiled manually.\n\n"
        "Matching constraints:\n"
        "- The full input component list must be output.\n"
        "- A match must be attempted for every component.\n"
        "- Matches cannot be the exact same for different components.\n"
        "- Do not give two components groups that are about 90 percent the same strokes.\n"
        "- Choose the closest thing available even when uncertain.\n"
        "- Only in a horrifying catastrophe should stroke_ids be null. If that happens, still include visual_description_of_match and reason explaining the failure.\n\n"
        "Output rules:\n"
        "- Output JSON only.\n"
        "- No markdown fences.\n"
        "- Return exactly one top-level field: components.\n"
        "- The components object must contain every component name from the input exactly once as a key.\n"
        "- Each component value must contain exactly: stroke_ids, visual_description_of_match, reason.\n"
        "- stroke_ids must be a list of integer stroke ids, or null only for catastrophic no-match cases.\n"
        "- visual_description_of_match should mirror what you found visually in the chosen strokes.\n"
        "- reason should explain how that visual match fits the component's two visual descriptions and why it wins over nearby alternatives.\n\n"
        "Example JSON shape:\n"
        f"{QWEN_DIAGRAM_COMPONENT_STROKE_MATCH_JSON_HINT}\n"
    )
