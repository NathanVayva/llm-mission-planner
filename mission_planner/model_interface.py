from abc import ABC, abstractmethod
from typing import Optional, List

from pydantic import BaseModel, Field
from openai import OpenAI


# ============================================================
# 1. Modèles Pydantic pour structurer les données échangées
# ============================================================

class LLMMessage(BaseModel):
    """
    Représente un message envoyé ou reçu par un modèle de langage.
    """
    role: str = Field(description="Rôle du message : 'user', 'assistant' ou 'system'.")
    content: str = Field(description="Contenu textuel du message.")


class LLMResponse(BaseModel):
    """
    Représente la réponse standardisée d'un modèle de langage.
    """
    content: str = Field(description="Texte généré par le modèle.")
    tokens_used: Optional[int] = Field(default=None, description="Nombre de tokens consommés.")
    raw_response: Optional[dict] = Field(default=None, description="Réponse brute pour debugging.")


# ============================================================
# 2. Classe abstraite : Interface générique
# ============================================================

class BaseLLM(ABC):
    """
    Interface générique que tous les modèles LLM doivent respecter.
    """

    @abstractmethod
    def generate(self, messages: List[LLMMessage]) -> LLMResponse:
        """
        Prend une liste de messages (style ChatGPT)
        et retourne une réponse structurée.
        """
        pass


# ============================================================
# 3. Implémentation concrète : Modèle OpenAI
# ============================================================

class OpenAIModel(BaseLLM):
    """
    Implémentation de l'interface utilisant OpenAI.
    """

    def __init__(self, model_name: str = "gpt-4.1", api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def chat(self, messages: List[LLMMessage]) -> LLMResponse:
        """
        Appelle l'API OpenAI avec la liste de messages fournie.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[msg.model_dump() for msg in messages]
        )

        msg = response.choices[0].message.content
        tokens = response.usage.total_tokens if response.usage else None

        return LLMResponse(
            content=msg,
            tokens_used=tokens,
            raw_response=response.model_dump()
        )
