import re
from typing import Dict, List, Tuple
from ..config.settings import detect_language, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE

class LanguageDetector:
    """Détecteur et gestionnaire de langue amélioré."""
    
    def __init__(self):
        self.supported_langs = SUPPORTED_LANGUAGES
        self.default_lang = DEFAULT_LANGUAGE
        self.current_language = DEFAULT_LANGUAGE
        
        self.patterns = {
            'fr': re.compile(r'\b(le|la|les|un|une|des|je|tu|il|nous|vous|ils|est|sont|mais|ou|et|donc|car|ni|or)\b', re.I),
            'en': re.compile(r'\b(the|a|an|i|you|he|she|we|they|is|are|am|but|or|and|so|because|nor|for)\b', re.I)
        }
        
        self.common_words = {
            'fr': {'bonjour', 'merci', 'salut', 'au revoir', 's\'il vous plaît', 'excusez-moi'},
            'en': {'hello', 'thank you', 'hi', 'goodbye', 'please', 'excuse me'}
        }

    def detect(self, text: str) -> str:
        if not text or len(text) < 3:
            return self.current_language
        
        scores = {lang: 0 for lang in self.supported_langs}
        
        for lang in self.supported_langs:
            matches = self.patterns[lang].findall(text.lower())
            scores[lang] = len(matches)
        
        text_lower = text.lower()
        for lang in self.supported_langs:
            for word in self.common_words[lang]:
                if word in text_lower:
                    scores[lang] += 2
        
        best_lang = max(scores.items(), key=lambda x: x[1])[0]
        
        if scores[best_lang] >= 2:
            self.current_language = best_lang
            return best_lang
        
        return self.current_language

    def get_language_prompt(self, language: str) -> str:
        prompts = {
            'fr': 'Réponds UNIQUEMENT en français. Utilise un ton naturel et conversationnel.',
            'en': 'Respond ONLY in English. Use a natural and conversational tone.'
        }
        return prompts.get(language, prompts[self.default_lang])

    def get_emotion_guidance(self, emotion: str, confidence: float, language: str) -> str:
        guidance = {
            'fr': {
                'tristesse': "Valide la douleur, propose du réconfort et de l'écoute",
                'colère': "Reconnais la frustration, aide à canaliser positivement",
                'peur': "Rassure avec empathie, propose des stratégies d'apaisement",
                'joie': "Partage l'enthousiasme avec mesure et authenticité",
                'neutre': "Reste ouvert et disponible pour explorer les besoins",
                'default': "Reste empathique et à l'écoute"
            },
            'en': {
                'tristesse': "Validate the pain, offer comfort and listening",
                'colère': "Acknowledge the frustration, help channel positively", 
                'peur': "Reassure with empathy, propose calming strategies",
                'joie': "Share enthusiasm with measure and authenticity",
                'neutre': "Stay open and available to explore needs",
                'default': "Stay empathetic and attentive"
            }
        }
        
        lang_guidance = guidance.get(language, guidance['fr'])
        emotion_key = emotion.lower() if emotion.lower() in lang_guidance else 'default'
        
        if confidence < 0.6:
            return lang_guidance.get('uncertain', "Stay neutral and benevolent")
        
        return lang_guidance.get(emotion_key, lang_guidance['default'])

    def should_switch_language(self, text: str) -> bool:
        detected = self.detect(text)
        return detected != self.current_language

language_detector = LanguageDetector()
