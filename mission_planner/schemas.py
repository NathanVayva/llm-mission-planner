from pydantic import BaseModel, Field
from typing import List, Dict, Any

class Action(BaseModel):
    action: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class MissionPlan(BaseModel):
    mission_name: str
    actions: List[Action] = Field(default_factory=list)

