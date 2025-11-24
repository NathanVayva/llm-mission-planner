#!/usr/bin/env python3
import argparse
import sys

from mission_planner.planner import MissionPlanner
from mission_planner.LLM import OllamaLLM, OpenAILLM


# CLI USER SCRIPT
def main():
    parser = argparse.ArgumentParser(
        description="Generate a mission plan using the LLM Mission Planner."
    )

    parser.add_argument(
        "instruction",
        type=str,
        nargs="*",
        help="Mission instruction given to the planner (e.g., 'Survey zone alpha')."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="llama3:instruct",
        help="Model name (default: llama3:instruct)"
    )

    parser.add_argument(
        "--engine",
        type=str,
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM engine to use: 'ollama' or 'openai' (default: ollama)"
    )

    args = parser.parse_args()

    
    # Join instruction words into a full sentence
    if not args.instruction:
        print("Error: You must provide a mission instruction.")
        sys.exit(1)

    mission_instruction = " ".join(args.instruction)
    print(f"Mission instruction: {mission_instruction}\n")

    # Select LLM engine
    if args.engine == "ollama":
        llm = OllamaLLM(model_name=args.model)
    else:
        llm = OpenAILLM(model_name=args.model)


    # Load Mission Planner
    planner = MissionPlanner(llm, model_name=args.model)

    # Generate plan
    try:
        plan = planner.generate_mission_plan(mission_instruction)
    except RuntimeError as e:
        print(f"Mission generation failed:\n{e}")
        sys.exit(1)

    # Print final JSON
    print("Mission plan generated:\n")
    print(plan.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
