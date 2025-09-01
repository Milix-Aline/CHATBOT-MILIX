import time
import logging
from typing import List, Dict, Optional, Tuple
import requests

from src.models.rag_model import RAGSystem
from src.models.emotion_model import EmotionClassifier
from src.config.settings import (
    DB_CONNECTION_STR, DATA_DIR, OLLAMA_BASE_URL, 
    OLLAMA_CHAT_MODEL, REQUEST_TIMEOUT
)
from src.utils.language_utils import language_detector

logger = logging.getLogger(__name__)

class ChatbotService:
    """Service principal du chatbot émotionnel multilingue."""
    
    def __init__(self):
        self.rag_system: Optional[RAGSystem] = None
        self.emotion_classifier: Optional[EmotionClassifier] = None
        self.conversation_history: Dict[str, List[Dict]] = {}
        self.is_initialized = False
        self.current_language = "fr"

    def initialize(self, setup_rag: bool = True, train_emotions: bool = True) -> bool:
        """Initialise les composants du chatbot."""
        try:
            logger.info("Initialisation du service chatbot...")
            
            # Système RAG
            if setup_rag:
                self.rag_system = RAGSystem(
                    db_connection_str=DB_CONNECTION_STR,
                    data_path=str(DATA_DIR)
                )
                # Remplir la base RAG
                self.rag_system.insert_documents()
                logger.info("Système RAG initialisé et rempli")
            
            # Classificateur d'émotions
            self.emotion_classifier = EmotionClassifier()
            if train_emotions or not self.emotion_classifier.load():
                logger.info("Entraînement du classificateur d'émotions...")
                self.emotion_classifier.train()
            
            self.is_initialized = True
            logger.info("Service chatbot initialisé avec succès")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            self.is_initialized = False
            return False

    def get_response(self, user_message: str, user_id: str = "default", top_k: int = 3) -> Tuple[str, str]:
        """Génère une réponse du chatbot dans la langue de l'utilisateur."""
        if not self.is_initialized:
            return "Le chatbot n'est pas encore initialisé.", "erreur"
        
        try:
            # Détection de langue
            self.current_language = language_detector.detect(user_message)
            
            # Détection d'émotion
            emotion, confidence = self._detect_emotion(user_message)
            
            # Recherche de contexte similaire
            context = self._get_similar_context(user_message, top_k)
            
            # Construction du prompt adapté à la langue
            prompt = self._build_prompt(user_message, context, emotion, confidence, user_id)
            
            # Génération de la réponse
            response = self._generate_ollama_response(prompt)
            
            # Post-traitement de la réponse
            response = self._post_process_response(response)
            
            # Mise à jour de l'historique
            self._update_history(user_id, user_message, response)
            
            return response, emotion
            
        except Exception as e:
            logger.error(f"Erreur génération réponse: {e}")
            return self._get_fallback_response(), "erreur"

    def _detect_emotion(self, text: str) -> Tuple[str, float]:
        """Détecte l'émotion dans le texte."""
        if not self.emotion_classifier or not self.emotion_classifier.is_ready():
            return "neutre", 0.5
        
        try:
            emotions, confidences = self.emotion_classifier.predict([text])
            if emotions and confidences:
                return emotions[0], confidences[0]
            return "neutre", 0.5
        except Exception as e:
            logger.error(f"Erreur détection émotion: {e}")
            return "neutre", 0.5

    def _get_similar_context(self, query: str, top_k: int) -> List[str]:
        """Recherche du contexte similaire via RAG."""
        if not self.rag_system:
            return []
        
        try:
            result = self.rag_system.retrieve_similar_corpus(query, top_k)
            if result:
                _, document, _ = result
                return [document]
            return []
        except Exception as e:
            logger.error(f"Erreur recherche contexte: {e}")
            return []

    def _build_prompt(self, user_message: str, context: List[str], emotion: str, 
                     confidence: float, user_id: str) -> str:
        """Construit le prompt adapté à la langue."""
        # Récupération de l'historique récent
        history = self.conversation_history.get(user_id, [])
        recent_history = history[-4:] if history else []
        
        # Construction du contexte historique
        history_text = ""
        if recent_history:
            history_lines = []
            for msg in recent_history:
                role = "User" if msg['role'] == "user" else "Assistant"
                history_lines.append(f"{role}: {msg['content'][:100]}...")
            history_text = "\n".join(history_lines)
        
        # Contexte similaire
        context_text = "\n".join(context[:2]) if context else "No similar context found."
        
        # Guidance émotionnelle adaptée à la langue
        emotion_guidance = language_detector.get_emotion_guidance(emotion, confidence, self.current_language)
        language_prompt = language_detector.get_language_prompt(self.current_language)
        
        prompt = f"""You are MILIX, an empathetic emotional assistant specialized in mental health support.

STYLE GUIDE:
- Use natural, conversational language
- Avoid generic phrases like "Hello! *briefly validates*"
- Adapt your tone to the detected emotion
- {language_prompt}
- Be concise (2-3 sentences maximum)

EMOTION CONTEXT:
- Detected emotion: {emotion} (confidence: {confidence:.0%})
- Guidance: {emotion_guidance}

SIMILAR CONTEXT:
{context_text}

RECENT HISTORY:
{history_text}

USER MESSAGE: {user_message}

YOUR RESPONSE (natural, empathetic, concise):"""
        
        return prompt.strip()

    def _generate_ollama_response(self, prompt: str) -> str:
        """Génère une réponse via Ollama avec paramètres optimisés."""
        url = f"{OLLAMA_BASE_URL}/api/chat"
        payload = {
            "model": OLLAMA_CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 256,
                "top_p": 0.9,
                "repeat_penalty": 1.2
            }
        }
        
        try:
            response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            
            data = response.json()
            if isinstance(data, dict) and "message" in data:
                content = data["message"].get("content", "").strip()
                if content:
                    return content
            
            return self._get_fallback_response()
            
        except requests.exceptions.Timeout:
            return "Sorry, the response is taking too long to generate."
        except requests.exceptions.ConnectionError:
            return "Connection error. Please check if Ollama is running."
        except Exception as e:
            logger.error(f"Erreur Ollama: {e}")
            return self._get_fallback_response()

    def _post_process_response(self, response: str) -> str:
        """Nettoie et améliore la réponse générée."""
        if not response:
            return self._get_fallback_response()
        
        unwanted_prefixes = [
            "En tant qu'assistant", "Je suis un assistant", "Assistant:", 
            "Réponse:", "Voici ma réponse:", "As an assistant", "I am an assistant"
        ]
        
        for prefix in unwanted_prefixes:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        if response:
            response = response[0].upper() + response[1:]
        
        return response

    def _get_fallback_response(self) -> str:
        """Retourne une réponse de secours adaptée à la langue."""
        fallbacks = {
            'fr': [
                "Je comprends que vous traversez un moment difficile. Souhaitez-vous en parler ?",
                "Je suis là pour vous écouter. Pouvez-vous m'en dire plus ?",
                "Je perçois que quelque chose vous touche. Comment puis-je vous aider ?"
            ],
            'en': [
                "I understand you're going through a difficult time. Would you like to talk about it?",
                "I'm here to listen. Can you tell me more about what's going on?",
                "I sense that something is affecting you. How can I best support you?"
            ]
        }
        
        responses = fallbacks.get(self.current_language, fallbacks['fr'])
        return responses[len(self.current_language) % len(responses)]

    def _update_history(self, user_id: str, user_message: str, bot_response: str):
        """Met à jour l'historique de conversation."""
        if user_id not in self.conversation_history:
            self.conversation_history[user_id] = []
        
        self.conversation_history[user_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": bot_response}
        ])
        
        max_history = 10
        if len(self.conversation_history[user_id]) > max_history:
            self.conversation_history[user_id] = self.conversation_history[user_id][-max_history:]

    def clear_history(self, user_id: Optional[str] = None):
        """Vide l'historique."""
        if user_id:
            self.conversation_history.pop(user_id, None)
        else:
            self.conversation_history.clear()

    def get_stats(self) -> dict:
        """Statistiques du service."""
        rag_stats = self.rag_system.get_stats() if self.rag_system else {}
        
        return {
            "initialized": self.is_initialized,
            "active_conversations": len(self.conversation_history),
            "emotion_classifier_ready": self.emotion_classifier.is_ready() if self.emotion_classifier else False,
            "current_language": self.current_language,
            "rag_stats": rag_stats
        }

    def test_connections(self) -> dict:
        """Teste les connexions externes."""
        results = {
            "ollama": False,
            "database": True
        }
        
        try:
            response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
            results["ollama"] = response.status_code == 200
        except:
            pass
        
        return results
