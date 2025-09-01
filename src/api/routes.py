from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from contextlib import asynccontextmanager

from .models import (
    ChatRequest, ChatResponse, StatsResponse, 
    ConnectionTestResponse, ClearHistoryRequest, ErrorResponse
)
from ..services.chatbot_service import ChatbotService

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Instance globale du service
chatbot_service = ChatbotService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion du cycle de vie de l'application."""
    # Startup
    logger.info("Démarrage de l'API MILIX...")
    success = chatbot_service.initialize(setup_rag=True, train_emotions=False)
    if success:
        logger.info("Service chatbot initialisé avec succès")
    else:
        logger.error("Échec de l'initialisation du service chatbot")
    
    yield
    
    # Shutdown
    logger.info("Arrêt de l'API MILIX")

# Création de l'app FastAPI
app = FastAPI(
    title="MILIX API",
    description="API pour chatbot émotionnel avec RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_chatbot_service() -> ChatbotService:
    """Dependency injection pour le service chatbot."""
    if not chatbot_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Le service chatbot n'est pas encore initialisé"
        )
    return chatbot_service

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Gestionnaire d'exceptions global."""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Une erreur interne s'est produite",
            error_type=type(exc).__name__,
            timestamp=datetime.now().isoformat()
        ).dict()
    )

@app.get("/", tags=["Root"])
async def root():
    """Endpoint racine."""
    return {
        "message": "API MILIX - Chatbot Émotionnel",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Vérification de santé de l'API."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service_initialized": chatbot_service.is_initialized
    }

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    request: ChatRequest,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Endpoint principal pour le chat."""
    try:
        response, emotion = service.get_response(
            user_message=request.user_message,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        return ChatResponse(
            response=response,
            detected_emotion=emotion,
            confidence=None,  # TODO: récupérer la confiance
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            user_id=request.user_id
        )
        
    except Exception as e:
        logger.error(f"Erreur chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse: {str(e)}"
        )

@app.get("/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats(service: ChatbotService = Depends(get_chatbot_service)):
    """Récupère les statistiques du service."""
    try:
        stats = service.get_stats()
        rag_stats = stats.get('rag_stats', {})
        
        return StatsResponse(
            initialized=stats['initialized'],
            active_conversations=stats['active_conversations'],
            emotion_classifier_ready=stats['emotion_classifier_ready'],
            total_embeddings=rag_stats.get('total_embeddings'),
            unique_documents=rag_stats.get('unique_documents')
        )
        
    except Exception as e:
        logger.error(f"Erreur stats endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )

@app.post("/clear", tags=["History"])
async def clear_history(
    request: ClearHistoryRequest,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Vide l'historique de conversation."""
    try:
        service.clear_history(request.user_id)
        
        if request.user_id:
            message = f"Historique vidé pour l'utilisateur {request.user_id}"
        else:
            message = "Tous les historiques ont été vidés"
            
        return {"status": "success", "message": message}
        
    except Exception as e:
        logger.error(f"Erreur clear endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la suppression de l'historique: {str(e)}"
        )

@app.get("/test", response_model=ConnectionTestResponse, tags=["Test"])
async def test_connections(service: ChatbotService = Depends(get_chatbot_service)):
    """Test des connexions externes."""
    try:
        results = service.test_connections()
        
        return ConnectionTestResponse(
            ollama=results['ollama'],
            database=results['database'],
            overall_status=results['ollama'] and results['database']
        )
        
    except Exception as e:
        logger.error(f"Erreur test endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du test des connexions: {str(e)}"
        )
