from __future__ import annotations

from typing import Any, Dict


LOGICAL_TIMELINE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "chapters": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "chapter_id": {"type": "string", "minLength": 2},
                    "title": {"type": "string", "minLength": 1},
                    "steps": {
                        "type": "object",
                        "minProperties": 1,
                        "additionalProperties": {"type": "string", "minLength": 1},
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
            "minProperties": 1,
            "additionalProperties": {"type": "string", "minLength": 1},
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
                    "name": {"type": "string", "minLength": 1},
                    "diagram": {"type": "integer", "enum": [0, 1]},
                    "required_objects": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "relevant_steps": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
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
                    "name": {"type": "string", "minLength": 1},
                    "text_style_description": {"type": "string", "minLength": 1},
                    "relevant_steps": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
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
