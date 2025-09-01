from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging
import time
import psutil
import asyncio
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import json
from io import StringIO

from .models import (
    ChatRequest, ChatResponse, StatsResponse, 
    ConnectionTestResponse, ClearHistoryRequest, ClearHistoryResponse,
    ErrorResponse, HealthCheckResponse, EmotionAnalysisRequest,
    EmotionAnalysisResponse, ConversationExportRequest
)
from ..services.chatbot_service import ChatbotService

# Configuration du logging avec format enrichi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Instance globale du service
chatbot_service = ChatbotService()
startup_time = time.time()
request_metrics: Dict[str, Any] = {
    "total_requests": 0,
    "response_times": [],
    "error_count": 0
}

# Sécurité optionnelle
security = HTTPBearer(auto_error=False)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion avancée du cycle de vie de l'application."""
    # Startup
    logger.info("🚀 Démarrage de l'API MILIX...")
    start_time = time.time()
    
    try:
        success = chatbot_service.initialize(setup_rag=True, train_emotions=False)
        init_time = time.time() - start_time
        
        if success:
            logger.info(f"✅ Service chatbot initialisé avec succès en {init_time:.2f}s")
            # Test des connexions au démarrage
            connections = chatbot_service.test_connections()
            for service_name, status in connections.items():
                status_icon = "✅" if status else "❌"
                logger.info(f"   {status_icon} {service_name}")
        else:
            logger.error("❌ Échec de l'initialisation du service chatbot")
    except Exception as e:
        logger.error(f"💥 Erreur critique lors de l'initialisation: {e}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Arrêt gracieux de l'API MILIX")
    # Sauvegarde des métriques si nécessaire
    if request_metrics["total_requests"] > 0:
        avg_response_time = sum(request_metrics["response_times"]) / len(request_metrics["response_times"])
        logger.info(f"📊 Statistiques finales: {request_metrics['total_requests']} requêtes, "
                   f"temps moyen: {avg_response_time:.2f}ms, erreurs: {request_metrics['error_count']}")

# Création de l'app FastAPI avec métadonnées enrichies
app = FastAPI(
    title="MILIX API",
    description="""
    ## API pour chatbot émotionnel multilingue avec RAG
    
    MILIX est un assistant conversationnel empathique qui :
    - **Détecte les émotions** dans les messages utilisateurs
    - **Supporte le multilinguisme** (français et anglais)
    - **Utilise RAG** pour des réponses contextualisées
    - **Maintient l'historique** des conversations
    
    ### Fonctionnalités principales:
    - Chat empathique avec détection d'émotion
    - Analyse de sentiment en temps réel
    - Export des conversations
    - Métriques et monitoring
    """,
    version="1.2.0",
    lifespan=lifespan,
    contact={
        "name": "Équipe MILIX",
        "email": "contact@milix.ai"
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
)

# Middleware CORS amélioré
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8080", "https://milix.ai"],  # Spécifier les domaines en production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"]
)

# Middleware pour les métriques et logging
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Ajouter l'ID de requête au contexte
    request.state.request_id = request_id
    
    # Log de la requête entrante
    logger.info(f"📨 [{request_id}] {request.method} {request.url.path} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    # Calculer le temps de traitement
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = request_id
    
    # Mise à jour des métriques
    request_metrics["total_requests"] += 1
    request_metrics["response_times"].append(process_time)
    
    # Garder seulement les 1000 derniers temps de réponse
    if len(request_metrics["response_times"]) > 1000:
        request_metrics["response_times"] = request_metrics["response_times"][-1000:]
    
    # Log de la réponse
    status_icon = "✅" if response.status_code < 400 else "❌"
    logger.info(f"📤 [{request_id}] {status_icon} {response.status_code} - {process_time:.1f}ms")
    
    return response

def get_chatbot_service() -> ChatbotService:
    """Dependency injection pour le service chatbot avec vérification."""
    if not chatbot_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service non disponible",
                "message": "Le service chatbot n'est pas encore initialisé",
                "retry_after": 30
            }
        )
    return chatbot_service

def get_request_id(request: Request) -> str:
    """Récupère l'ID de requête."""
    return getattr(request.state, 'request_id', 'unknown')

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Gestionnaire d'exceptions HTTP personnalisé."""
    request_id = get_request_id(request)
    request_metrics["error_count"] += 1
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail),
            error_type="HTTPException",
            error_code=f"HTTP_{exc.status_code}",
            request_id=request_id,
            suggestions=_get_error_suggestions(exc.status_code)
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Gestionnaire d'exceptions global amélioré."""
    request_id = get_request_id(request)
    request_metrics["error_count"] += 1
    
    logger.error(f"💥 [{request_id}] Erreur non gérée: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Une erreur interne s'est produite",
            error_type=type(exc).__name__,
            error_code="INTERNAL_ERROR",
            request_id=request_id,
            suggestions=[
                "Vérifiez que tous les services externes sont disponibles",
                "Réessayez dans quelques instants",
                "Contactez le support si le problème persiste"
            ]
        ).dict()
    )

def _get_error_suggestions(status_code: int) -> list:
    """Fournit des suggestions basées sur le code d'erreur."""
    suggestions = {
        400: ["Vérifiez le format de votre requête", "Assurez-vous que tous les champs requis sont présents"],
        401: ["Vérifiez vos credentials d'authentification"],
        403: ["Vous n'avez pas les permissions nécessaires pour cette action"],
        404: ["Vérifiez l'URL de votre requête", "Cette ressource n'existe peut-être plus"],
        429: ["Trop de requêtes, attendez avant de réessayer"],
        503: ["Le service est temporairement indisponible", "Réessayez dans quelques instants"]
    }
    return suggestions.get(status_code, ["Contactez le support pour plus d'assistance"])

# === ENDPOINTS ===

@app.get("/", tags=["Root"], summary="Page d'accueil de l'API")
async def root():
    """Endpoint racine avec informations sur l'API."""
    uptime = time.time() - startup_time
    return {
        "message": "🤖 API MILIX - Chatbot Émotionnel Multilingue",
        "version": "1.2.0",
        "status": "running",
        "uptime_seconds": round(uptime, 2),
        "features": [
            "Détection d'émotions",
            "Support multilingue (FR/EN)",
            "Système RAG",
            "Historique de conversation"
        ],
        "endpoints": {
            "chat": "/chat",
            "health": "/health",
            "stats": "/stats",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"], 
         summary="Vérification de santé détaillée")
async def health_check():
    """Vérification de santé complète de l'API."""
    uptime = time.time() - startup_time
    
    # Test des composants
    components = {}
    try:
        components["chatbot_service"] = chatbot_service.is_initialized
        components["emotion_classifier"] = (
            chatbot_service.emotion_classifier.is_ready() 
            if chatbot_service.emotion_classifier else False
        )
        components["rag_system"] = chatbot_service.rag_system is not None
        
        # Test des connexions externes
        connections = chatbot_service.test_connections()
        components.update(connections)
        
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        components["health_check_error"] = False
    
    overall_status = "healthy" if all(components.values()) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        service_initialized=chatbot_service.is_initialized,
        components=components,
        uptime_seconds=uptime
    )

@app.post("/chat", response_model=ChatResponse, tags=["Chat"],
          summary="Conversation avec le chatbot")
async def chat_endpoint(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    service: ChatbotService = Depends(get_chatbot_service),
    request_obj: Request = None
):
    """Endpoint principal pour la conversation avec métriques avancées."""
    start_time = time.time()
    request_id = get_request_id(request_obj)
    
    try:
        logger.info(f"💬 [{request_id}] Chat request from user: {request.user_id}")
        
        response, emotion = service.get_response(
            user_message=request.user_message,
            user_id=request.user_id,
            top_k=request.top_k
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Récupération du contexte utilisé (si disponible)
        context_used = service.get_last_context_used() if hasattr(service, 'get_last_context_used') else None
        
        chat_response = ChatResponse(
            response=response,
            detected_emotion=emotion,
            confidence=service.get_last_emotion_confidence() if hasattr(service, 'get_last_emotion_confidence') else None,
            detected_language=service.current_language,
            response_language=service.current_language,
            user_id=request.user_id,
            session_id=request.session_id,
            processing_time_ms=processing_time,
            context_used=context_used
        )
        
        # Tâche en arrière-plan pour logging/analytics
        background_tasks.add_task(
            log_chat_interaction,
            request.user_id,
            request.user_message[:100],  # Premier 100 caractères pour la log
            emotion,
            processing_time
        )
        
        return chat_response
        
    except Exception as e:
        logger.error(f"❌ [{request_id}] Erreur chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la génération de la réponse: {str(e)}"
        )

async def log_chat_interaction(user_id: str, message_preview: str, emotion: str, processing_time: float):
    """Log asynchrone des interactions de chat."""
    logger.info(f"📊 Interaction logged - User: {user_id}, Emotion: {emotion}, "
               f"Time: {processing_time:.1f}ms, Preview: {message_preview}...")

@app.post("/analyze-emotion", response_model=EmotionAnalysisResponse, tags=["Analysis"],
          summary="Analyse d'émotion seule")
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Analyse l'émotion d'un texte sans générer de réponse."""
    try:
        if not service.emotion_classifier or not service.emotion_classifier.is_ready():
            raise HTTPException(
                status_code=503,
                detail="Le classificateur d'émotions n'est pas disponible"
            )
        
        emotions, confidences = service.emotion_classifier.predict([request.text])
        detected_language = service.detect_language(request.text)
        
        response = EmotionAnalysisResponse(
            detected_emotion=emotions[0] if emotions else "neutre",
            confidence=confidences[0] if confidences else 0.5,
            detected_language=detected_language
        )
        
        if request.return_all_scores:
            # Récupérer tous les scores si demandé
            all_scores = service.emotion_classifier.get_all_scores([request.text])
            if all_scores:
                response.all_scores = all_scores[0]
        
        return response
        
    except Exception as e:
        logger.error(f"Erreur analyse émotion: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'analyse d'émotion: {str(e)}"
        )

@app.get("/stats", response_model=StatsResponse, tags=["Stats"],
         summary="Statistiques complètes du service")
async def get_stats(service: ChatbotService = Depends(get_chatbot_service)):
    """Récupère les statistiques détaillées du service."""
    try:
        stats = service.get_stats()
        rag_stats = stats.get('rag_stats', {})
        uptime = time.time() - startup_time
        
        # Calcul des métriques de performance
        avg_response_time = None
        if request_metrics["response_times"]:
            avg_response_time = sum(request_metrics["response_times"]) / len(request_metrics["response_times"])
        
        # Utilisation mémoire
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        return StatsResponse(
            initialized=stats['initialized'],
            active_conversations=stats['active_conversations'],
            emotion_classifier_ready=stats['emotion_classifier_ready'],
            total_embeddings=rag_stats.get('total_embeddings'),
            unique_documents=rag_stats.get('unique_documents'),
            current_language=stats.get('current_language', 'fr'),
            uptime_seconds=uptime,
            memory_usage_mb=memory_usage,
            average_response_time_ms=avg_response_time
        )
        
    except Exception as e:
        logger.error(f"Erreur stats endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération des statistiques: {str(e)}"
        )

@app.post("/clear", response_model=ClearHistoryResponse, tags=["History"],
          summary="Nettoyage de l'historique")
async def clear_history(
    request: ClearHistoryRequest,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Vide l'historique de conversation avec options avancées."""
    try:
        # Comptage avant suppression
        deleted_conversations = 0
        deleted_messages = 0
        
        if request.user_id:
            if request.user_id in service.conversation_history:
                deleted_messages = len(service.conversation_history[request.user_id])
                deleted_conversations = 1
                service.clear_history(request.user_id)
                message = f"Historique vidé pour l'utilisateur {request.user_id}"
            else:
                message = f"Aucun historique trouvé pour l'utilisateur {request.user_id}"
        else:
            # Nettoyage global
            for user_id, history in service.conversation_history.items():
                deleted_messages += len(history)
                deleted_conversations += 1
            service.clear_history()
            message = "Tous les historiques ont été vidés"
            
        return ClearHistoryResponse(
            status="success",
            message=message,
            deleted_conversations=deleted_conversations,
            deleted_messages=deleted_messages
        )
        
    except Exception as e:
        logger.error(f"Erreur clear endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la suppression de l'historique: {str(e)}"
        )

@app.get("/test", response_model=ConnectionTestResponse, tags=["Test"],
         summary="Test des connexions externes")
async def test_connections(service: ChatbotService = Depends(get_chatbot_service)):
    """Test des connexions avec métriques de performance."""
    try:
        start_time = time.time()
        results = service.test_connections()
        
        # Mesurer les temps de réponse individuels
        response_times = {}
        error_details = {}
        
        # Test Ollama avec temps de réponse
        ollama_start = time.time()
        try:
            import requests
            response = requests.get(f"{service.ollama_base_url}/api/tags", timeout=5)
            response_times["ollama"] = (time.time() - ollama_start) * 1000
            results["ollama"] = response.status_code == 200
        except Exception as e:
            response_times["ollama"] = (time.time() - ollama_start) * 1000
            error_details["ollama"] = str(e)
            results["ollama"] = False
        
        # Test base de données
        db_start = time.time()
        try:
            # Test simple de la base
            conn = service.rag_system._get_connection() if service.rag_system else None
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                conn.close()
                results["database"] = True
            else:
                results["database"] = False
                error_details["database"] = "RAG system not initialized"
        except Exception as e:
            error_details["database"] = str(e)
            results["database"] = False
        finally:
            response_times["database"] = (time.time() - db_start) * 1000
        
        # Récupérer les modèles Ollama disponibles
        ollama_models = []
        if results["ollama"]:
            try:
                import requests
                response = requests.get(f"{service.ollama_base_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    ollama_models = [model["name"] for model in data.get("models", [])]
            except:
                pass
        
        return ConnectionTestResponse(
            ollama=results["ollama"],
            database=results["database"],
            overall_status=results["ollama"] and results["database"],
            ollama_models=ollama_models if ollama_models else None,
            database_type=service.rag_system.db_type if service.rag_system else None,
            response_times=response_times,
            error_details=error_details if error_details else None
        )
        
    except Exception as e:
        logger.error(f"Erreur test endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du test des connexions: {str(e)}"
        )

@app.get("/conversations/{user_id}", tags=["History"], 
         summary="Récupérer l'historique d'un utilisateur")
async def get_conversation_history(
    user_id: str,
    limit: int = 50,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Récupère l'historique de conversation d'un utilisateur."""
    try:
        history = service.conversation_history.get(user_id, [])
        
        # Limiter le nombre de messages retournés
        if limit > 0:
            history = history[-limit:]
        
        return {
            "user_id": user_id,
            "message_count": len(history),
            "messages": history,
            "last_activity": history[-1].get("timestamp") if history else None
        }
        
    except Exception as e:
        logger.error(f"Erreur récupération historique: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la récupération de l'historique: {str(e)}"
        )

@app.post("/export-conversation", tags=["Export"],
          summary="Export des conversations")
async def export_conversation(
    request: ConversationExportRequest,
    service: ChatbotService = Depends(get_chatbot_service)
):
    """Exporte les conversations dans différents formats."""
    try:
        history = service.conversation_history.get(request.user_id, [])
        
        if not history:
            raise HTTPException(
                status_code=404,
                detail=f"Aucune conversation trouvée pour l'utilisateur {request.user_id}"
            )
        
        # Filtrage par date si spécifié
        if request.date_from or request.date_to:
            filtered_history = []
            for msg in history:
                msg_date = datetime.fromisoformat(msg.get("timestamp", "1970-01-01"))
                if request.date_from and msg_date < request.date_from:
                    continue
                if request.date_to and msg_date > request.date_to:
                    continue
                filtered_history.append(msg)
            history = filtered_history
        
        if request.format == "json":
            content = json.dumps({
                "user_id": request.user_id,
                "export_date": datetime.now().isoformat(),
                "message_count": len(history),
                "messages": history
            }, indent=2)
            media_type = "application/json"
            filename = f"conversation_{request.user_id}.json"
            
        elif request.format == "csv":
            output = StringIO()
            import csv
            writer = csv.writer(output)
            
            headers = ["timestamp", "role", "content"]
            if request.include_emotions:
                headers.extend(["emotion", "confidence"])
            writer.writerow(headers)
            
            for msg in history:
                row = [
                    msg.get("timestamp", ""),
                    msg.get("role", ""),
                    msg.get("content", "")
                ]
                if request.include_emotions:
                    row.extend([
                        msg.get("emotion", ""),
                        msg.get("confidence", "")
                    ])
                writer.writerow(row)
            
            content = output.getvalue()
            media_type = "text/csv"
            filename = f"conversation_{request.user_id}.csv"
            
        elif request.format == "txt":
            lines = [f"Conversation Export - User: {request.user_id}"]
            lines.append(f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("=" * 50)
            
            for msg in history:
                timestamp = msg.get("timestamp", "Unknown")
                role = "User" if msg.get("role") == "user" else "Assistant"
                content = msg.get("content", "")
                
                lines.append(f"\n[{timestamp}] {role}:")
                lines.append(content)
                
                if request.include_emotions and msg.get("emotion"):
                    emotion = msg.get("emotion", "")
                    confidence = msg.get("confidence", "")
                    lines.append(f"(Emotion: {emotion}, Confidence: {confidence})")
            
            content = "\n".join(lines)
            media_type = "text/plain"
            filename = f"conversation_{request.user_id}.txt"
        
        # Retourner le fichier en streaming
        return StreamingResponse(
            iter([content.encode()]),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur export conversation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de l'export: {str(e)}"
        )

@app.get("/metrics", tags=["Monitoring"], summary="Métriques système")
async def get_metrics():
    """Retourne les métriques détaillées du système."""
    try:
        uptime = time.time() - startup_time
        memory = psutil.Process().memory_info()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        return {
            "uptime_seconds": uptime,
            "total_requests": request_metrics["total_requests"],
            "error_count": request_metrics["error_count"],
            "error_rate": request_metrics["error_count"] / max(request_metrics["total_requests"], 1),
            "average_response_time_ms": (
                sum(request_metrics["response_times"]) / len(request_metrics["response_times"])
                if request_metrics["response_times"] else 0
            ),
            "memory_usage": {
                "rss_mb": memory.rss / 1024 / 1024,
                "vms_mb": memory.vms / 1024 / 1024
            },
            "cpu_percent": cpu_percent,
            "active_conversations": len(chatbot_service.conversation_history) if chatbot_service.is_initialized else 0
        }
    except Exception as e:
        logger.error(f"Erreur métriques: {e}")
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des métriques")

# Endpoint de debugging (à désactiver en production)
@app.get("/debug/logs", tags=["Debug"], summary="Logs récents (dev only)")
async def get_recent_logs():
    """Retourne les logs récents pour le debugging."""
    # En production, cet endpoint devrait être désactivé ou protégé
    import os
    if os.getenv("ENVIRONMENT", "dev") == "production":
        raise HTTPException(status_code=404, detail="Endpoint non disponible en production")
    
    try:
        # Lire les dernières lignes du fichier de log si configuré
        return {"message": "Endpoint de debugging - logs récents", "logs": []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))