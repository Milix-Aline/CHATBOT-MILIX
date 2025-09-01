from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class EmotionType(str, Enum):
    """Énumération des types d'émotions supportées."""
    JOIE = "joie"
    TRISTESSE = "tristesse"
    COLERE = "colère"
    PEUR = "peur"
    NEUTRE = "neutre"
    ERREUR = "erreur"

class LanguageType(str, Enum):
    """Énumération des langues supportées."""
    FRENCH = "fr"
    ENGLISH = "en"

class ChatRequest(BaseModel):
    """Modèle de requête pour le chat avec validation avancée."""
    user_message: str = Field(
        ..., 
        min_length=1, 
        max_length=2000, 
        description="Message de l'utilisateur",
        example="Je me sens triste aujourd'hui"
    )
    user_id: Optional[str] = Field(
        default="default", 
        max_length=50,
        description="Identifiant unique de l'utilisateur",
        example="user123"
    )
    top_k: Optional[int] = Field(
        default=3, 
        ge=1, 
        le=10, 
        description="Nombre de contextes similaires à récupérer",
        example=3
    )
    language: Optional[LanguageType] = Field(
        default=None,
        description="Langue forcée pour la réponse (auto-détection si None)"
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Identifiant de session pour le suivi de conversation"
    )

    @validator('user_message')
    def validate_message_content(cls, v):
        """Validation du contenu du message."""
        if not v or v.isspace():
            raise ValueError("Le message ne peut pas être vide ou contenir uniquement des espaces")
        
        # Supprimer les caractères de contrôle dangereux
        cleaned = ''.join(char for char in v if ord(char) >= 32 or char in '\n\t')
        if len(cleaned) != len(v):
            raise ValueError("Le message contient des caractères non autorisés")
        
        return cleaned.strip()

    @validator('user_id')
    def validate_user_id(cls, v):
        """Validation de l'ID utilisateur."""
        if v and not v.replace('-', '').replace('_', '').isalnum():
            raise ValueError("L'ID utilisateur ne peut contenir que des caractères alphanumériques, _ et -")
        return v

class ChatResponse(BaseModel):
    """Modèle de réponse du chat avec métriques enrichies."""
    response: str = Field(
        ..., 
        description="Réponse du chatbot",
        example="Je comprends que tu te sentes triste. C'est normal d'avoir des moments difficiles..."
    )
    detected_emotion: EmotionType = Field(
        ..., 
        description="Émotion détectée dans le message",
        example=EmotionType.TRISTESSE
    )
    confidence: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=1.0, 
        description="Confiance de la détection d'émotion",
        example=0.85
    )
    detected_language: LanguageType = Field(
        ...,
        description="Langue détectée du message utilisateur",
        example=LanguageType.FRENCH
    )
    response_language: LanguageType = Field(
        ...,
        description="Langue de la réponse générée",
        example=LanguageType.FRENCH
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Horodatage de la réponse"
    )
    user_id: str = Field(
        ..., 
        description="Identifiant de l'utilisateur",
        example="user123"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Identifiant de session"
    )
    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Temps de traitement en millisecondes"
    )
    context_used: Optional[List[str]] = Field(
        default=None,
        description="Contextes RAG utilisés pour générer la réponse"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class StatsResponse(BaseModel):
    """Modèle de réponse pour les statistiques enrichies."""
    initialized: bool = Field(
        ..., 
        description="État d'initialisation du service"
    )
    active_conversations: int = Field(
        ..., 
        ge=0,
        description="Nombre de conversations actives"
    )
    emotion_classifier_ready: bool = Field(
        ..., 
        description="État du classificateur d'émotions"
    )
    total_embeddings: Optional[int] = Field(
        None, 
        ge=0,
        description="Nombre total d'embeddings en base"
    )
    unique_documents: Optional[int] = Field(
        None, 
        ge=0,
        description="Nombre de documents uniques"
    )
    supported_languages: List[LanguageType] = Field(
        default_factory=lambda: [LanguageType.FRENCH, LanguageType.ENGLISH],
        description="Langues supportées par le système"
    )
    current_language: LanguageType = Field(
        default=LanguageType.FRENCH,
        description="Langue actuellement active"
    )
    uptime_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Temps de fonctionnement en secondes"
    )
    memory_usage_mb: Optional[float] = Field(
        default=None,
        ge=0,
        description="Utilisation mémoire en MB"
    )
    average_response_time_ms: Optional[float] = Field(
        default=None,
        ge=0,
        description="Temps de réponse moyen en millisecondes"
    )

class ConnectionTestResponse(BaseModel):
    """Modèle de réponse pour le test des connexions avec détails."""
    ollama: bool = Field(
        ..., 
        description="État de la connexion Ollama"
    )
    database: bool = Field(
        ..., 
        description="État de la connexion base de données"
    )
    overall_status: bool = Field(
        ..., 
        description="État global des connexions"
    )
    ollama_models: Optional[List[str]] = Field(
        default=None,
        description="Liste des modèles Ollama disponibles"
    )
    database_type: Optional[str] = Field(
        default=None,
        description="Type de base de données utilisée"
    )
    response_times: Optional[Dict[str, float]] = Field(
        default=None,
        description="Temps de réponse des services en ms"
    )
    error_details: Optional[Dict[str, str]] = Field(
        default=None,
        description="Détails des erreurs de connexion"
    )

class ClearHistoryRequest(BaseModel):
    """Modèle de requête pour vider l'historique avec options."""
    user_id: Optional[str] = Field(
        None, 
        max_length=50,
        description="ID utilisateur (None pour tous)"
    )
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        description="ID de session spécifique à nettoyer"
    )
    before_date: Optional[datetime] = Field(
        None,
        description="Supprimer l'historique avant cette date"
    )

class ClearHistoryResponse(BaseModel):
    """Modèle de réponse pour la suppression d'historique."""
    status: str = Field(..., description="Statut de l'opération")
    message: str = Field(..., description="Message descriptif")
    deleted_conversations: int = Field(
        default=0, 
        ge=0,
        description="Nombre de conversations supprimées"
    )
    deleted_messages: int = Field(
        default=0,
        ge=0, 
        description="Nombre de messages supprimés"
    )

class ErrorResponse(BaseModel):
    """Modèle de réponse d'erreur enrichi."""
    detail: str = Field(
        ..., 
        description="Description de l'erreur"
    )
    error_type: str = Field(
        ..., 
        description="Type d'erreur"
    )
    error_code: Optional[str] = Field(
        default=None,
        description="Code d'erreur spécifique"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Horodatage de l'erreur"
    )
    user_id: Optional[str] = Field(
        default=None,
        description="ID de l'utilisateur concerné"
    )
    request_id: Optional[str] = Field(
        default=None,
        description="ID de la requête pour le débogage"
    )
    suggestions: Optional[List[str]] = Field(
        default=None,
        description="Suggestions pour résoudre l'erreur"
    )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class HealthCheckResponse(BaseModel):
    """Modèle de réponse pour le health check."""
    status: str = Field(..., description="Statut général du service")
    timestamp: datetime = Field(default_factory=datetime.now)
    service_initialized: bool = Field(..., description="Service initialisé")
    components: Dict[str, bool] = Field(
        default_factory=dict,
        description="État des composants individuels"
    )
    version: str = Field(default="1.0.0", description="Version de l'API")
    uptime_seconds: Optional[float] = Field(default=None, ge=0)

class EmotionAnalysisRequest(BaseModel):
    """Modèle pour l'analyse d'émotion seule."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Texte à analyser"
    )
    return_all_scores: bool = Field(
        default=False,
        description="Retourner tous les scores d'émotion"
    )

class EmotionAnalysisResponse(BaseModel):
    """Modèle de réponse pour l'analyse d'émotion."""
    detected_emotion: EmotionType = Field(..., description="Émotion principale détectée")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confiance de la prédiction")
    all_scores: Optional[Dict[EmotionType, float]] = Field(
        default=None,
        description="Scores pour toutes les émotions"
    )
    detected_language: LanguageType = Field(..., description="Langue détectée")

class ConversationExportRequest(BaseModel):
    """Modèle pour l'export de conversation."""
    user_id: str = Field(..., max_length=50, description="ID utilisateur")
    format: str = Field(
        default="json",
        regex="^(json|csv|txt)$",
        description="Format d'export"
    )
    include_emotions: bool = Field(
        default=True,
        description="Inclure les données d'émotion"
    )
    date_from: Optional[datetime] = Field(
        default=None,
        description="Date de début pour l'export"
    )
    date_to: Optional[datetime] = Field(
        default=None,
        description="Date de fin pour l'export"
    )