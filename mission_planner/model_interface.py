
from typing import Optional, List

from pydantic import BaseModel, Field
from openai import OpenAI


# 1. Modèles Pydantic pour structurer les données échangées


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



