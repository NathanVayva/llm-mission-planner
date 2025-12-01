# mission_planner/planner.py
"""Mission planner: convert natural-language instructions into validated MissionPlan objects.

This module is LLM-agnostic: provide any object that implements BaseLLM.chat(messages) -> LLMResponse.
It expects Pydantic v2 models for MissionPlan (schemas.py).
"""


from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from pydantic import ValidationError

# imports from your package (adjust paths if needed)
from .model_interface import LLMMessage, LLMResponse
from .schemas import MissionPlan
from .LLM import BaseLLM

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SYSTEM_PROMPT= """
    
    You are a mission planning assistant for a robotic rover.

Your output MUST be exactly ONE JSON object. No markdown, no prose, no explanations.  
Never output multiple JSON objects.

The JSON MUST follow this schema:

{
    "mission_name": "<string>",
    "actions": [
        {
            "action": "<string>",
            "parameters": {
                "<key>": "<string value>"
            }
        }
    ]
}

RULES:

1. The plan MUST contain:
    - exactly one "mission_name"
    - exactly one "actions" array
    - all actions described by the user, in strict chronological order.

2. Every action MUST have a "parameters" object.
   All values MUST be strings. No numbers, no lists, no booleans.

3. COORDINATES:
   Whenever the user gives coordinates (e.g., "(400, 250)"),
   you MUST place them in "parameters" using EXACTLY:
       "target_x": "<string>"
       "target_y": "<string>"
   Never use any other key names.
   Never place coordinates outside "parameters".

4. ALLOWED ACTIONS (you may NOT output any other action):
    - "move_to" with parameters: target_x, target_y, speed (optional)
    - "take_photo" with parameters: resolution, zoom (optional)
    - "wait" with parameters: duration
    - "pick_up" with parameters ALWAYS equal to {}
    - "drop_off" with parameters ALWAYS equal to {}

5.  - A "pick_up" action can only occur if the robot is currently not carrying a sample.
    - A "drop_off" action can only occur if the robot is carrying a sample.
    - Do not generate multiple "pick_up" or "drop_off" actions for the same sample unless explicitly instructed.
    - The order of actions must strictly follow the mission steps described, without inventing extra actions.


6. Whenever you output "pick_up" or "drop_off", the parameters MUST be exactly:
        "parameters": {}
7. You MUST translate user instructions using ONLY the allowed actions.
   If the user describes an action that is not in this list, convert it to the closest allowed one.
   You MUST NOT invent new action names.

8. Never output more than ONE JSON object. No text before or after.

9. If the mission cannot be generated, output only:
{"error": "Cannot generate mission"}


    """







class MissionPlanner:
    """
    High-level planner class.

    - llm: instance implementing BaseLLM
    - model_name: optional label used for logs/prompts
    """

    def __init__(self, llm: BaseLLM, model_name: Optional[str] = None):
        self.llm = llm
        self.model_name = model_name or getattr(llm, "model_name", "unknown-llm")

    # ---------------------------
    # Prompt construction
    # ---------------------------
    def _build_prompt(self, instruction: str, constraints: Optional[str] = None) -> List[LLMMessage]:
        """
        Return a chat-style message list (system + user) to send to LLM.
        Keep system prompt focused on JSON output and the mission schema.
        """
        user_text = "Instruction:\n" + instruction.strip()
        if constraints:
            user_text += "\n\nConstraints:\n" + constraints.strip()

        messages = [
            LLMMessage(role="system", content=SYSTEM_PROMPT),
            LLMMessage(role="user", content=user_text),
        ]
        LOGGER.debug("Built prompt messages: %s", messages)
        return messages

    # ---------------------------
    # Public API
    # ---------------------------
    def generate_mission_plan(
        self,
        instruction: str,
        constraints: Optional[str] = None,
        max_attempts: int = 2,
    ) -> MissionPlan:
        """
        Generate a MissionPlan from a natural-language instruction.

        Workflow:
         1. Build prompt
         2. Send to LLM
         3. Extract JSON from LLM text
         4. Validate/parse into MissionPlan (Pydantic)
         5. (Optionally) retry once if parsing fails, asking LLM to return strict JSON
        """
        messages = self._build_prompt(instruction, constraints)

        for attempt in range(1, max_attempts + 1):
            LOGGER.info("LLM call attempt %d/%d using %s", attempt, max_attempts, self.model_name)
            llm_response: LLMResponse = self.llm.generate(messages)
            text = llm_response.content
            LOGGER.debug("LLM raw content: %s", text)

            # try to recover JSON from the LLM text
            json_text = self._extract_json_block(text)
            if json_text is None:
                LOGGER.warning("No JSON block found in LLM response on attempt %d", attempt)
                # ask for a strict JSON-only reply on next attempt
                messages.append(
                    LLMMessage(
                        role="system",
                        content=(
                            "Previous response did not contain parseable JSON. "
                            "Please reply with a single JSON object (or array) only, matching the schema."
                        ),
                    )
                )
                continue

            # try parse+validate via Pydantic (v2)
            try:
                # MissionPlan.model_validate_json is Pydantic v2 API
                plan = MissionPlan.model_validate_json(json_text)
                LOGGER.info("MissionPlan validated successfully.")
                return plan
            except ValidationError as e:
                LOGGER.warning("ValidationError parsing MissionPlan on attempt %d: %s", attempt, e)
                # If validation fails, request correction from the LLM with details
                messages.append(
                    LLMMessage(
                        role="system",
                        content=(
                            "The JSON you returned could not be validated against the mission schema. "
                            f"Validation error: {e}. Please return corrected JSON only."
                        ),
                    )
                )
                continue
            except json.JSONDecodeError as e:
                LOGGER.warning("JSON decode error on attempt %d: %s", attempt, e)
                messages.append(
                    LLMMessage(
                        role="system",
                        content="Your response could not be decoded as JSON. Please reply with strict JSON only."
                    )
                )
                continue

        # If we arrive here, attempts exhausted
        raise RuntimeError("Failed to generate a valid MissionPlan after multiple attempts.")

    # ---------------------------
    # Helpers
    # ---------------------------
    @staticmethod
    def _extract_json_block(text: str) -> Optional[str]:
        """
        Heuristically find the first JSON object or array in text and return its string.
        Looks for balanced {} or [] block starting at first brace/bracket.
        Returns None if none found/decodable.
        """
        if not text:
            return None

        # find first opening brace or bracket
        start_match = re.search(r"[\{\[]", text)
        if not start_match:
            return None
        start = start_match.start()
        open_char = text[start]
        close_char = "}" if open_char == "{" else "]"

        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    # validate with json.loads to ensure it's actually JSON
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        # try continue searching (in case of nested noise)
                        continue
        return None
