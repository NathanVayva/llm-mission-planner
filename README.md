# LLM Mission Planner

LLM Mission Planner is a lightweight Python tool that uses a Large Language Model (LLM) 
(OpenAI, LLaMA, etc.) to convert natural-language mission instructions into structured, 
machine-readable robotic mission plans in JSON format.

This project was developed in the context of an internship application at the European Space 
Agency (ESA), showcasing skills in robotics, AI, mission planning, and clean software engineering.

---

## Features

- Natural-language → JSON mission planning
- Modular architecture (planner, schema, validator, LLM interface)
- Support for multiple LLM providers
- JSON validation through Pydantic
- Command-line interface (CLI)
- Fully open-source and easy to extend

---

## Example

**Input:**

> "Inspect area A, avoid obstacles, and keep a 5-meter safety distance from slopes."

**Output:**

```json
[
    {"action": "move_to", "target": "area_A", "speed": 0.5},
    {"action": "scan_area", "parameters": {"resolution": "high"}},
    {"action": "avoid_obstacles", "distance": 1.0},
    {"action": "maintain_safety_distance", "distance": 5}
]
