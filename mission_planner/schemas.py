from pydantic import BaseModel, Field
from typing import List, Dict

class Action(BaseModel):
    action: str
    parameters: Dict[str, str] = Field(default_factory=dict)

class MissionPlan(BaseModel):
    mission_name: str
    actions: List[Action] = Field(default_factory=list)

# create plan from dict (Pydantic convertira automatiquement)
data = {
    "mission_name": "Inspect B",
    "actions": [
        {"action": "move_to", "parameters": {"target": "B1"}},
        {"action": "scan_area", "parameters": {"resolution": "high"}}
    ]
}

plan = MissionPlan(**data)  # ou MissionPlan.parse_obj(data) en v1
print(plan.model_dump_json(indent=2))
