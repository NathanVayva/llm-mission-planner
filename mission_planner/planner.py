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


SYSTEM_PROMPT=(
"You are a mission planning assistant for a robotic rover."
""
"Your task is to output EXACTLY ONE and only ONE JSON object. Never output multiple JSON objects, markdown, explanations, or any text before or after the JSON."
""
"The JSON MUST follow this schema:"
""
"{"
"  \"mission_name\": \"<string>\","
"  \"actions\": ["
"    {"
"      \"action\": \"<string>\","
"      \"parameters\": {"
"        \"<key>\": \"<string value>\""
"      }"
"    }"
"  ]"
"}"
""
"Rules:"
""
"1. The output MUST contain exactly:"
"   - one \"mission_name\" (string),"
"   - one \"actions\" array,"
"   - each action must contain:"
"     - \"action\": a short verb-like string (e.g., \"move_to\", \"take_photo\", \"analyse\")"
"     - \"parameters\": an object with string keys AND string values only. No lists, no numbers, no booleans."
""
"2. All parameter values MUST be strings (e.g., \"2m/s\", \"high\", \"A1\")."
"   - If multiple values are needed, merge them into a single string (e.g., \"A1,A2,A3\")."
""
"3. The mission plan MUST include ALL instructions given by the user, even if the user describes several tasks, goals, or sub-missions."
"   - You MUST merge everything into ONE single mission plan."
"   - Produce ONE \"mission_name\"."
"   - Produce ONE \"actions\" array containing ALL actions in chronological order."
""
"4. Never omit an action mentioned in the user instruction."
"   Never invent actions that are not described."
""
"5. Never output more than one JSON object."
"   Never output trailing text or comments."
"   Even if the model overflows with content, the final output MUST be trimmed to ONE valid JSON."
""
"6. If you cannot generate the mission, output ONLY:"
"{\"error\": \"Cannot generate mission\"}"

)




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
