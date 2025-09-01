from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class ChatRequest(BaseModel):
    """Modèle de requête pour le chat."""
    user_message: str = Field(..., min_length=1, max_length=2000, description="Message de l'utilisateur")
    user_id: Optional[str] = Field(default="default", description="Identifiant unique de l'utilisateur")
    top_k: Optional[int] = Field(default=3, ge=1, le=10, description="Nombre de contextes similaires à récupérer")

    class Config:
        schema_extra = {
            "example": {
                "user_message": "Je me sens triste aujourd'hui",
                "user_id": "user123",
                "top_k": 3
            }
        }

class ChatResponse(BaseModel):
    """Modèle de réponse du chat."""
    response: str = Field(..., description="Réponse du chatbot")
    detected_emotion: str = Field(..., description="Émotion détectée dans le message")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confiance de la détection d'émotion")
    timestamp: str = Field(..., description="Horodatage de la réponse")
    user_id: str = Field(..., description="Identifiant de l'utilisateur")

    class Config:
        schema_extra = {
            "example": {
                "response": "Je comprends que tu te sentes triste. C'est normal d'avoir des moments difficiles...",
                "detected_emotion": "tristesse",
                "confidence": 0.85,
                "timestamp": "2024-01-15 14:30:25",
                "user_id": "user123"
            }
        }

class StatsResponse(BaseModel):
    """Modèle de réponse pour les statistiques."""
    initialized: bool = Field(..., description="État d'initialisation du service")
    active_conversations: int = Field(..., description="Nombre de conversations actives")
    emotion_classifier_ready: bool = Field(..., description="État du classificateur d'émotions")
    total_embeddings: Optional[int] = Field(None, description="Nombre total d'embeddings en base")
    unique_documents: Optional[int] = Field(None, description="Nombre de documents uniques")

class ConnectionTestResponse(BaseModel):
    """Modèle de réponse pour le test des connexions."""
    ollama: bool = Field(..., description="État de la connexion Ollama")
    database: bool = Field(..., description="État de la connexion base de données")
    overall_status: bool = Field(..., description="État global des connexions")

class ClearHistoryRequest(BaseModel):
    """Modèle de requête pour vider l'historique."""
    user_id: Optional[str] = Field(None, description="ID utilisateur (None pour tous)")

class ErrorResponse(BaseModel):
    """Modèle de réponse d'erreur."""
    detail: str = Field(..., description="Description de l'erreur")
    error_type: str = Field(..., description="Type d'erreur")
    timestamp: str = Field(..., description="Horodatage de l'erreur")
