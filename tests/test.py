

import os
from openai import OpenAI

from mission_planner.planner import MissionPlanner
from mission_planner.schemas import MissionPlan
from mission_planner.LLM import OpenAILLM , OllamaLLM



chat=OllamaLLM(model_name="llama3.2")

planner = MissionPlanner(chat, model_name="llama3.2")



plan = planner.generate_mission_plan("Move to waypoint A1 and take a high-resolution photo.")

print(plan.model_dump_json(indent=2))
