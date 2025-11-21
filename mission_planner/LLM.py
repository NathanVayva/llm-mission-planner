from mission_planner.model_interface import BaseLLM, LLMMessage, LLMResponse
from openai import OpenAI
from typing import List, Optional

import json
import requests

class OllamaLLM(BaseLLM):
    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name

    def generate(self, messages):
        prompt = "\n".join([f"{m.role}: {m.content}" for m in messages])

        response = requests.post(
            "http://localhost:11434/api/generate",

            json={"model": self.model_name, "prompt": prompt},

            stream=False
        )


        response_cat = ""
        for line in response.text.splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "response" in data:
                    response_cat += data["response"]
            except json.JSONDecodeError:
                # ignorer les lignes qui ne sont pas du JSON
                continue

        print("RAW RESPONSE:")
        print(response_cat)

        return LLMResponse(content=response_cat)


class OpenAILLM(BaseLLM):
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def generate(self, messages: List[LLMMessage]) -> LLMResponse:
        # transforme les messages BaseLLM en format OpenAI
        formatted_messages = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages
        )
        # on récupère le texte généré
        content = response.choices[0].message.content
        return LLMResponse(content=content)
