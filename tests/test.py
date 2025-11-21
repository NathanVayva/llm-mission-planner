import os
from openai import OpenAI

from mission_planner.planner import MissionPlanner
from mission_planner.schemas import MissionPlan
from mission_planner.LLM import OpenAILLM , OllamaLLM


chat=OllamaLLM(model_name="llama3.2")

planner = MissionPlanner(chat, model_name="llama3.2")

# Texte simulé venant du LLM
mock_response = """
Here is the mission plan:
{
    "mission_name": "Survey Area A",
    "actions": [
        {"action": "move_to", "parameters": {"target": "zone_1"}},
        {"action": "take_photo", "parameters": {"target": "zone_1"}}
    ]
}
"""



plan = planner.generate_mission_plan(mock_response)

print(plan.model_dump_json(indent=2))
