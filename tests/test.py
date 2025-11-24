

import os
from openai import OpenAI

from mission_planner.planner import MissionPlanner
from mission_planner.schemas import MissionPlan
from mission_planner.LLM import OpenAILLM , OllamaLLM



chat=OllamaLLM(model_name="llama3.2")

planner = MissionPlanner(chat, model_name="llama3.2")

mission = "Survey Zone Beta: Move to waypoints B1, B2, and B3 at 2 m/s, take high-resolution photos at each point, then return to base."
print("mission:", mission)
plan = planner.generate_mission_plan(mission)

print("Extracted JSON",plan.model_dump_json(indent=2))
