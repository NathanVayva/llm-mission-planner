# LLM Mission Planner

A lightweight Python tool that uses a Large Language Model (Ollama or OpenAI) to turn
natural-language mission instructions into **structured, schema-validated JSON mission plans**
for a robot. The design keeps the LLM strictly in the *planning* role and hands execution to
deterministic code.

Personal project, built around a robotics / mission-planning use case.

---

## Features

- Natural-language → JSON mission planning
- Modular architecture (planner · schema · validator · LLM interface)
- Pluggable LLM backends: **Ollama** (local, default) or **OpenAI**
- Output validated against a **Pydantic** schema (rejects malformed plans)
- Command-line interface
- Small **pygame** simulation to visualize a plan being executed
- MIT-licensed

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
```

---

## Architecture

```
mission_planner/
  planner.py           # orchestrates: prompt -> LLM -> parse -> validate
  LLM.py               # BaseLLM (ABC) + OllamaLLM / OpenAILLM backends
  model_interface.py   # LLMMessage / LLMResponse dataclasses
  schemas.py           # Pydantic MissionPlan / Action schema (validation)
run_planner.py         # CLI entry point
simulation/            # pygame visualization of a generated plan
tests/                 # basic tests
```

The planner asks the LLM for a plan, extracts the JSON, and validates it against the Pydantic
schema before anything downstream can use it — so an ill-formed or hallucinated plan is caught
rather than executed.

---

## Installation

```bash
pip install pydantic openai requests
# for the local backend, install and run Ollama (https://ollama.com) and pull a model:
ollama pull llama3:instruct
# the pygame simulation additionally needs:
pip install pygame
```

## Usage

```bash
# local model (Ollama, default engine)
python run_planner.py "Survey zone alpha and return to base"

# choose engine / model
python run_planner.py "Inspect area A and keep 5 m from slopes" --engine ollama --model llama3:instruct
python run_planner.py "Inspect area A" --engine openai   # requires OPENAI_API_KEY
```

Run the visualization:

```bash
python simulation/pygame_simulation.py
```

---

## License

MIT — see [LICENSE](LICENSE).
