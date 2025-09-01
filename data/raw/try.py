import os
import re
import json
import time
import string
import joblib
import numpy as np
import pandas as pd
import psycopg
import faiss
import requests
import warnings
import logging

from typing import List, Tuple, Optional, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppression des warnings non critiques
warnings.filterwarnings("ignore", category=UserWarning)

# ===============================
# 0. CONFIG
# ===============================
CSV1 = "train (2).csv"
CSV2 = "train.csv"
CSV3 = "emotion_dataset.csv"

EXPORT_DIR = "./artifacts"
os.makedirs(EXPORT_DIR, exist_ok=True)
CLF_PATH = os.path.join(EXPORT_DIR, "emotion_clf.joblib")       # classifieur TF-IDF + LinearSVC
LABELS_PATH = os.path.join(EXPORT_DIR, "emotion_labels.json")   # mapping d'étiquettes
EMB_MODEL_NAME = "all-MiniLM-L6-v2"                             # pour RAG
OLLAMA_URL = "http://localhost:11434/api/chat"                  # ✅ CORRECTION: API correcte d'Ollama
OLLAMA_MODEL = "llama2:latest"                                # ou "mistral"
FORCE_FRENCH = True                                             # force les réponses en FR

# Connexion PostgreSQL
PG_CONN_STR = "dbname=chatbot_milix user=postgres password=1234 host=localhost port=5432"

# Paramètres de performance
BATCH_SIZE = 64
MAX_HISTORY = 10
REQUEST_TIMEOUT = 120
DEFAULT_TEMPERATURE = 0.4

# ===============================
# 1. UTILITAIRES AMÉLIORÉS
# ===============================
_punct_table = str.maketrans("", "", string.punctuation)

def clean_text(s: str) -> str:
    """Nettoie et normalise le texte d'entrée."""
    if not isinstance(s, str) or not s.strip():
        return ""
    
    s = s.lower().strip()
    # Suppression des caractères de contrôle
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', s)
    # Suppression de la ponctuation
    s = s.translate(_punct_table)
    # Normalisation des espaces
    s = re.sub(r'\s+', ' ', s, flags=re.M).strip()
    
    return s

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """Valide la structure du DataFrame."""
    if df.empty:
        logger.warning("DataFrame vide détecté")
        return False
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        logger.error(f"Colonnes manquantes: {missing_cols}")
        return False
    
    return True

def softmax_like(margins: np.ndarray) -> np.ndarray:
    """Approximation de probabilité à partir des marges SVM."""
    if margins.ndim == 1:
        margins = margins.reshape(1, -1)
    
    # Stabilité numérique
    m = margins - margins.max(axis=1, keepdims=True)
    e = np.exp(np.clip(m, -500, 500))  # Éviter l'overflow
    return e / (e.sum(axis=1, keepdims=True) + 1e-8)  # Éviter division par zéro

def print_confusion(y_true, y_pred, labels):
    """Affiche la matrice de confusion de manière formatée."""
    try:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        logger.info("\n=== Matrice de confusion ===")
        logger.info(f"\n{df_cm}")
    except Exception as e:
        logger.error(f"Erreur lors de l'affichage de la matrice de confusion: {e}")

def check_file_exists(filepath: str) -> bool:
    """Vérifie l'existence d'un fichier avec gestion d'erreur."""
    return os.path.exists(filepath) and os.path.getsize(filepath) > 0

# ===============================
# 2. CHARGER & FUSIONNER CSV AMÉLIORÉ
# ===============================
def load_and_merge_csvs() -> pd.DataFrame:
    """Charge et fusionne les fichiers CSV avec gestion d'erreurs robuste."""
    dataframes = []
    csv_configs = [
        (CSV1, {"situation": "texte", "emotion": "emotion", "sys_response": "response"}),
        (CSV2, {"situation": "texte", "emotion": "emotion"}),
        (CSV3, {"Text": "texte", "Emotion": "emotion"})
    ]
    
    for csv_file, column_mapping in csv_configs:
        try:
            if not check_file_exists(csv_file):
                logger.warning(f"Fichier non trouvé ou vide: {csv_file}")
                continue
                
            # Lecture avec gestion d'encodage
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                logger.error(f"Impossible de lire {csv_file} avec les encodages testés")
                continue
            
            # Vérification des colonnes requises
            required_cols = list(column_mapping.keys())
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Colonnes manquantes dans {csv_file}: {set(required_cols) - set(df.columns)}")
                continue
            
            # Renommage et sélection des colonnes
            df_clean = df.rename(columns=column_mapping)
            
            # Ajout de la colonne response si manquante
            if "response" not in df_clean.columns:
                df_clean["response"] = None
                
            df_clean = df_clean[["texte", "emotion", "response"]]
            dataframes.append(df_clean)
            logger.info(f"Chargé {len(df_clean)} lignes depuis {csv_file}")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {csv_file}: {e}")
            continue
    
    if not dataframes:
        raise ValueError("Aucun fichier CSV n'a pu être chargé correctement")
    
    # Fusion des DataFrames
    df = pd.concat(dataframes, ignore_index=True)
    
    # Nettoyage amélioré
    df["texte"] = df["texte"].astype(str).apply(clean_text)
    df["emotion"] = df["emotion"].astype(str).apply(lambda x: clean_text(x) if pd.notna(x) else "")
    
    # Filtrage des données invalides
    df = df[
        (df["texte"].str.len() > 2) &  # Texte minimum
        (df["emotion"].str.len() > 1)   # Émotion minimum
    ].copy()
    
    # Suppression des doublons
    initial_len = len(df)
    df = df.drop_duplicates(subset=["texte", "emotion"], keep="first").reset_index(drop=True)
    logger.info(f"Supprimé {initial_len - len(df)} doublons")
    
    if not validate_dataframe(df, ["texte", "emotion", "response"]):
        raise ValueError("DataFrame final invalide")
    
    logger.info(f"Dataset final: {len(df)} lignes après nettoyage et fusion")
    return df

# ===============================
# 3. ENTRAÎNER UN CLASSIFIEUR D'ÉMOTIONS AMÉLIORÉ
# ===============================
def train_emotion_classifier(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """Entraîne un classifieur d'émotions avec validation améliorée."""
    if not validate_dataframe(df, ["texte", "emotion"]):
        raise ValueError("DataFrame invalide pour l'entraînement")
    
    X = df["texte"].tolist()
    y = df["emotion"].tolist()
    
    # Vérification de la distribution des classes
    class_counts = pd.Series(y).value_counts()
    logger.info(f"Distribution des classes:\n{class_counts}")
    
    # Filtrer les classes avec trop peu d'exemples
    min_samples = max(2, len(set(y)) // 20)  # Au moins 2, ou 5% du total
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    
    if len(valid_classes) < 2:
        raise ValueError("Pas assez de classes valides pour l'entraînement")
    
    # Filtrer les données
    mask = pd.Series(y).isin(valid_classes)
    X = [x for x, m in zip(X, mask) if m]
    y = [label for label, m in zip(y, mask) if m]
    
    logger.info(f"Classes retenues: {len(valid_classes)} sur {len(class_counts)}")
    
    # Split stratifié
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.18, random_state=42, stratify=y
        )
    except ValueError as e:
        logger.warning(f"Stratification impossible: {e}. Split aléatoire utilisé.")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.18, random_state=42
        )
    
    # Pipeline optimisé
    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.9,
            max_features=10000,  # Limitation pour éviter l'overfitting
            sublinear_tf=True,
            norm="l2",
            stop_words=None  # Pas de stop words pour conserver l'expressivité émotionnelle
        )),
        ("svm", LinearSVC(
            C=1.0,
            class_weight='balanced',  # Gestion du déséquilibre des classes
            max_iter=2000,
            random_state=42
        ))
    ])
    
    # Entraînement avec gestion d'erreurs
    try:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        logger.info("\n=== Rapport de classification (validation) ===")
        logger.info(f"\n{classification_report(y_val, y_pred, digits=3)}")
        
        labels_sorted = sorted(valid_classes)
        print_confusion(y_val, y_pred, labels_sorted)
        
        # Export sécurisé
        joblib.dump(clf, CLF_PATH)
        with open(LABELS_PATH, "w", encoding="utf-8") as f:
            json.dump(labels_sorted, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Classifieur exporté → {CLF_PATH}")
        logger.info(f"✅ Labels exportés → {LABELS_PATH}")
        
        return clf, labels_sorted
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement: {e}")
        raise

def load_clf() -> Tuple[Pipeline, List[str]]:
    """Charge le classifieur et les labels avec validation."""
    try:
        if not check_file_exists(CLF_PATH) or not check_file_exists(LABELS_PATH):
            raise FileNotFoundError("Fichiers du classifieur manquants")
        
        clf = joblib.load(CLF_PATH)
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        
        if not labels:
            raise ValueError("Liste de labels vide")
            
        logger.info(f"Classifieur chargé avec {len(labels)} classes")
        return clf, labels
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement du classifieur: {e}")
        raise

def predict_emotion(clf: Pipeline, texts: List[str]) -> Tuple[List[str], List[float]]:
    """Prédit l'émotion avec gestion d'erreurs améliorée."""
    if not texts or not all(isinstance(t, str) for t in texts):
        logger.warning("Textes invalides pour la prédiction")
        return [], []
    
    try:
        # Nettoyage des textes
        clean_texts = [clean_text(t) for t in texts]
        
        # Prédictions
        preds = clf.predict(clean_texts)
        
        # Approximation des probabilités
        if hasattr(clf.named_steps["svm"], "decision_function"):
            margins = clf.decision_function(clean_texts)
            if margins.ndim == 1:
                margins = np.vstack([margins, -margins]).T
            probs = softmax_like(margins).max(axis=1).tolist()
        else:
            probs = [0.8] * len(preds)  # Confiance par défaut plus réaliste
            
        return preds.tolist(), probs
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return ["neutre"] * len(texts), [0.5] * len(texts)

# ===============================
# 4. RAG : EMBEDDINGS + POSTGRES + FAISS AMÉLIORÉ
# ===============================
def ensure_pg_table():
    """Crée la table PostgreSQL avec gestion d'erreurs."""
    try:
        with psycopg.connect(PG_CONN_STR) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id SERIAL PRIMARY KEY,
                        texte TEXT NOT NULL,
                        emotion TEXT NOT NULL,
                        response TEXT,
                        embedding FLOAT8[] NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                
                # Index pour améliorer les performances
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_embeddings_emotion ON embeddings(emotion);
                """)
                
                conn.commit()
                logger.info("Table PostgreSQL prête")
                
    except Exception as e:
        logger.error(f"Erreur lors de la création de la table: {e}")
        raise

def insert_rows_with_embeddings(df: pd.DataFrame, emb_model: SentenceTransformer):
    """Insère les données avec embeddings par batch pour optimiser les performances."""
    if not validate_dataframe(df, ["texte", "emotion"]):
        raise ValueError("DataFrame invalide pour l'insertion")
    
    texts = df["texte"].tolist()
    
    try:
        logger.info(f"Génération des embeddings pour {len(texts)} textes...")
        embs = emb_model.encode(
            texts, 
            show_progress_bar=True, 
            batch_size=BATCH_SIZE,
            convert_to_numpy=True
        )
        
        # Insertion par batch pour améliorer les performances
        batch_size = 100
        total_inserted = 0
        
        with psycopg.connect(PG_CONN_STR) as conn:
            with conn.cursor() as cur:
                for i in range(0, len(df), batch_size):
                    batch_df = df.iloc[i:i+batch_size]
                    batch_embs = embs[i:i+batch_size]
                    
                    batch_data = [
                        (row["texte"], row["emotion"], row["response"], 
                         emb.astype(float).tolist())
                        for (_, row), emb in zip(batch_df.iterrows(), batch_embs)
                    ]
                    
                    cur.executemany("""
                        INSERT INTO embeddings (texte, emotion, response, embedding)
                        VALUES (%s, %s, %s, %s)
                    """, batch_data)
                    
                    total_inserted += len(batch_data)
                    if i % (batch_size * 5) == 0:  # Log tous les 500 éléments
                        logger.info(f"Inséré {total_inserted}/{len(df)} entrées")
                
                conn.commit()
                logger.info(f"✅ Toutes les embeddings insérées: {total_inserted} entrées")
                
    except Exception as e:
        logger.error(f"Erreur lors de l'insertion des embeddings: {e}")
        raise

def load_from_pg() -> Tuple[List[str], List[str], List[Optional[str]], np.ndarray]:
    """Charge les données depuis PostgreSQL avec validation."""
    try:
        with psycopg.connect(PG_CONN_STR) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM embeddings;")
                count = cur.fetchone()[0]
                
                if count == 0:
                    logger.warning("Aucune donnée trouvée dans la table embeddings")
                    return [], [], [], np.array([])
                
                logger.info(f"Chargement de {count} entrées depuis PostgreSQL...")
                cur.execute("SELECT texte, emotion, response, embedding FROM embeddings;")
                rows = cur.fetchall()
        
        if not rows:
            return [], [], [], np.array([])
        
        documents = [r[0] for r in rows]
        doc_emotions = [r[1] for r in rows]
        doc_responses = [r[2] for r in rows]
        doc_embeddings = np.array([r[3] for r in rows], dtype=np.float32)
        
        logger.info(f"Chargé {len(documents)} documents avec embeddings")
        return documents, doc_emotions, doc_responses, doc_embeddings
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement depuis PostgreSQL: {e}")
        raise

def build_faiss_index(doc_embeddings: np.ndarray) -> faiss.Index:
    """Construit l'index FAISS avec optimisations."""
    if doc_embeddings.size == 0:
        raise ValueError("Pas d'embeddings pour construire l'index")
    
    try:
        # Normalisation pour stabiliser L2
        doc_embeddings = normalize(doc_embeddings, norm="l2").astype(np.float32)
        dim = doc_embeddings.shape[1]
        
        # Index adaptatif selon la taille des données
        if len(doc_embeddings) > 10000:
            # Index IVF pour de grandes collections
            nlist = min(100, len(doc_embeddings) // 100)
            quantizer = faiss.IndexFlatL2(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            index.train(doc_embeddings)
        else:
            # Index plat pour de petites collections
            index = faiss.IndexFlatL2(dim)
        
        index.add(doc_embeddings)
        logger.info(f"Index FAISS construit: {index.ntotal} vecteurs, dimension {dim}")
        return index
        
    except Exception as e:
        logger.error(f"Erreur lors de la construction de l'index FAISS: {e}")
        raise

def search(index: faiss.Index, embedder: SentenceTransformer, query: str, 
          documents: List[str], doc_emotions: List[str], doc_responses: List[Optional[str]], 
          top_k: int = 3) -> List[Tuple[str, str, Optional[str]]]:
    """Effectue une recherche sémantique avec gestion d'erreurs."""
    if not query.strip():
        logger.warning("Requête vide pour la recherche")
        return []
    
    try:
        q_emb = embedder.encode([clean_text(query)], convert_to_numpy=True)
        q_emb = normalize(np.array(q_emb, dtype=np.float32), norm="l2")
        
        # Ajustement du top_k si nécessaire
        search_k = min(top_k, index.ntotal)
        
        D, I = index.search(q_emb, search_k)
        
        results = []
        for idx in I[0]:
            if idx < len(documents):  # Vérification de sécurité
                results.append((
                    documents[idx], 
                    doc_emotions[idx], 
                    doc_responses[idx]
                ))
        
        return results
        
    except Exception as e:
        logger.error(f"Erreur lors de la recherche: {e}")
        return []

# ===============================
# 5. OLLAMA CHAT AMÉLIORÉ (✅ CORRIGÉ)
# ===============================
def ollama_chat(messages: List[dict], model: str = OLLAMA_MODEL, 
               url: str = OLLAMA_URL, temperature: float = DEFAULT_TEMPERATURE,
               max_retries: int = 3) -> str:
    """Interface Ollama avec le bon endpoint et format."""
    if not messages:
        return "[Erreur] Pas de messages à envoyer"
    
    # Format correct pour l'API Ollama
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,  # Important: désactiver le streaming
        "options": {
            "temperature": temperature,
            "num_predict": 512,
            "top_p": 0.9
        }
    }
    
    for attempt in range(max_retries):
        try:
            logger.debug(f"Tentative {attempt + 1} d'appel à Ollama")
            response = requests.post(
                url, 
                json=payload, 
                timeout=REQUEST_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            data = response.json()
            logger.debug(f"Réponse Ollama: {type(data)} - {list(data.keys()) if isinstance(data, dict) else 'non-dict'}")
            
            # Format de réponse Ollama standard
            if isinstance(data, dict) and "message" in data:
                content = data["message"].get("content", "").strip()
                if content:
                    return content
            
            # Fallback pour d'autres formats
            if isinstance(data, dict) and "response" in data:
                content = data["response"].strip()
                if content:
                    return content
            
            logger.warning(f"Format de réponse Ollama inattendu: {data}")
            return "[Erreur] Format de réponse inattendu d'Ollama"
            
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout Ollama (tentative {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return "[Erreur] Timeout - Ollama met trop de temps à répondre"
            time.sleep(2 ** attempt)
            
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connexion échouée (tentative {attempt + 1}/{max_retries})")
            if attempt == max_retries - 1:
                return "[Erreur] Impossible de se connecter à Ollama. Vérifiez qu'il est démarré."
            time.sleep(2 ** attempt)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur requête Ollama: {e}")
            if attempt == max_retries - 1:
                return f"[Erreur Ollama] {str(e)[:100]}..."
            time.sleep(1)
    
    return "[Erreur] Échec de communication avec Ollama après plusieurs tentatives"

# ===============================
# 6. PIPELINE COMPLET AMÉLIORÉ
# ===============================
class EmotionalRAGBot:
    """Chatbot émotionnel avec RAG amélioré."""
    
    def __init__(self, force_french: bool = True):
        self.force_french = force_french
        self.conversation_history: List[dict] = []
        self.embedder: Optional[SentenceTransformer] = None
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.doc_emotions: List[str] = []
        self.doc_responses: List[Optional[str]] = []
        self.clf: Optional[Pipeline] = None
        self.labels: List[str] = []
        self.is_ready = False

    def setup(self, retrain_clf: bool = True, refresh_store: bool = True) -> bool:
        """Configuration du bot avec gestion d'erreurs complète."""
        try:
            # 1) Chargement des CSV
            logger.info("📦 Chargement des fichiers CSV...")
            df = load_and_merge_csvs()
            
            if df.empty:
                logger.error("Aucune donnée utilisable trouvée dans les CSV")
                return False
            
            # 2) Classifieur d'émotions
            clf_needs_training = retrain_clf or not (
                check_file_exists(CLF_PATH) and check_file_exists(LABELS_PATH)
            )
            
            if clf_needs_training:
                logger.info("🧠 Entraînement du classifieur d'émotions...")
                self.clf, self.labels = train_emotion_classifier(df)
            else:
                logger.info("🧠 Chargement du classifieur exporté...")
                self.clf, self.labels = load_clf()
            
            # 3) RAG Setup
            logger.info("🔎 Initialisation du système RAG...")
            self.embedder = SentenceTransformer(EMB_MODEL_NAME)
            
            # Configuration PostgreSQL
            ensure_pg_table()
            
            if refresh_store:
                logger.info("💾 Reconstruction du store d'embeddings...")
                # Nettoyage de la table
                with psycopg.connect(PG_CONN_STR) as conn:
                    with conn.cursor() as cur:
                        cur.execute("TRUNCATE TABLE embeddings RESTART IDENTITY;")
                        conn.commit()
                
                insert_rows_with_embeddings(df, self.embedder)
            
            # 4) Chargement des données et construction de l'index
            logger.info("📥 Chargement des embeddings depuis PostgreSQL...")
            self.documents, self.doc_emotions, self.doc_responses, doc_embeddings = load_from_pg()
            
            if len(self.documents) == 0:
                logger.error("Aucune donnée chargée depuis PostgreSQL")
                return False
            
            logger.info("🏗️ Construction de l'index FAISS...")
            self.index = build_faiss_index(doc_embeddings)
            
            self.is_ready = True
            logger.info("✅ Chatbot prêt à fonctionner!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la configuration: {e}")
            self.is_ready = False
            return False

    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """Détecte l'émotion avec validation."""
        if not self.clf or not text.strip():
            return "neutre", 0.5
        
        try:
            preds, probs = predict_emotion(self.clf, [text])
            if preds and probs:
                return preds[0], float(probs[0])
            else:
                return "neutre", 0.5
        except Exception as e:
            logger.error(f"Erreur détection émotion: {e}")
            return "neutre", 0.5

    def build_prompt(self, query: str, context: List[Tuple[str, str, Optional[str]]], 
                    sentiment: str, score: float) -> str:
        """Construit le prompt avec contexte enrichi."""
        # Contexte des exemples similaires
        ctx_lines = []
        for i, (text, emotion, response) in enumerate(context[:3], 1):
            response_part = f" → {response}" if response else ""
            ctx_lines.append(f"{i}. [{emotion}] {text[:100]}...{response_part}")
        
        context_text = "\n".join(ctx_lines) if ctx_lines else "Aucun exemple similaire trouvé."
        
        # Historique de conversation (limité)
        history_text = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-6:]  # 3 échanges maximum
            history_lines = []
            for msg in recent_history:
                role_symbol = "👤" if msg['role'] == "user" else "🤖"
                history_lines.append(f"{role_symbol} {msg['content'][:80]}...")
            history_text = "\n".join(history_lines)
        
        # Adaptation du ton selon l'émotion détectée
        emotion_guidance = self._get_emotion_guidance(sentiment, score)
        
        lang_instruction = "Réponds UNIQUEMENT en français." if self.force_french else "Réponds dans la langue de l'utilisateur."
        
        prompt = f"""Tu es un assistant empathique spécialisé en soutien émotionnel.
{lang_instruction}

CONTEXTE ÉMOTIONNEL:
- Émotion détectée: {sentiment} (confiance: {score:.0%})
- Guidance: {emotion_guidance}

EXEMPLES SIMILAIRES:
{context_text}

HISTORIQUE RÉCENT:
{history_text}

MESSAGE UTILISATEUR: {query}

INSTRUCTIONS:
- Commence par valider brièvement l'émotion exprimée
- Propose 1-2 pistes d'aide concrètes et bienveillantes
- Reste concis (3-5 phrases maximum)
- Évite tout jugement ou moralisation
- Adapte ton ton à l'émotion détectée
"""
        return prompt.strip()

    def _get_emotion_guidance(self, emotion: str, score: float) -> str:
        """Fournit des conseils adaptés selon l'émotion détectée."""
        emotion_lower = emotion.lower()
        
        if score < 0.6:
            return "Émotion incertaine - reste neutre et bienveillant"
        
        guidance_map = {
            'tristesse': "Valide la douleur, propose du réconfort et des ressources d'aide",
            'colère': "Reconnais la frustration, aide à canaliser l'énergie positivement", 
            'peur': "Rassure avec empathie, propose des stratégies d'apaisement",
            'anxiété': "Normalise l'inquiétude, suggère des techniques de relaxation",
            'joie': "Partage l'enthousiasme tout en restant mesuré",
            'surprise': "Accompagne la découverte avec curiosité bienveillante",
            'dégoût': "Respecte le ressenti sans le juger, aide à clarifier",
            'neutre': "Reste ouvert et disponible pour creuser les besoins"
        }
        
        for key, guidance in guidance_map.items():
            if key in emotion_lower:
                return guidance
        
        return "Reste empathique et à l'écoute"

    def reply(self, user_text: str, top_k: int = 3) -> str:
        """Génère une réponse avec gestion d'erreurs complète."""
        if not self.is_ready:
            return "❌ Le chatbot n'est pas encore prêt. Veuillez patienter."
        
        if not user_text.strip():
            return "Je n'ai pas bien saisi votre message. Pouvez-vous reformuler ?"
        
        try:
            # 1) Détection d'émotion
            start_time = time.time()
            sentiment, score = self.detect_emotion(user_text)
            
            # 2) Recherche de contexte
            context = search(
                self.index, self.embedder, user_text,
                self.documents, self.doc_emotions, self.doc_responses, 
                top_k=top_k
            )
            
            # 3) Construction du prompt
            prompt = self.build_prompt(user_text, context, sentiment, score)
            
            # 4) Génération de la réponse
            messages = [{"role": "user", "content": prompt}]
            answer = ollama_chat(messages, model=OLLAMA_MODEL)
            
            # 5) Post-traitement de la réponse
            answer = self._post_process_response(answer, sentiment)
            
            # 6) Mise à jour de l'historique
            self._update_conversation_history(user_text, answer)
            
            response_time = time.time() - start_time
            logger.debug(f"Réponse générée en {response_time:.2f}s (émotion: {sentiment}, confiance: {score:.0%})")
            
            return answer
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de réponse: {e}")
            return self._get_fallback_response(user_text)

    def _post_process_response(self, response: str, detected_emotion: str) -> str:
        """Post-traite la réponse pour améliorer la qualité."""
        if not response or response.startswith("[Erreur"):
            return self._get_fallback_response("")
        
        # Nettoyage basique
        response = response.strip()
        
        # Suppression des préfixes indésirables
        prefixes_to_remove = [
            "En tant qu'assistant", "Je suis un assistant", "Assistant:", 
            "Réponse:", "Voici ma réponse:"
        ]
        
        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()
        
        # Limitation de longueur si nécessaire
        if len(response) > 500:
            sentences = response.split('.')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) > 400:
                    break
                truncated.append(sentence)
                current_length += len(sentence)
            
            if truncated:
                response = '.'.join(truncated) + '.'
        
        return response

    def _get_fallback_response(self, user_text: str) -> str:
        """Fournit une réponse de secours en cas d'erreur."""
        fallback_responses = [
            "Je comprends que vous traversez un moment difficile. Voulez-vous me parler de ce qui vous préoccupe ?",
            "Je suis là pour vous écouter. Pouvez-vous m'en dire un peu plus sur votre situation ?",
            "Il semble que quelque chose vous touche. Je suis disponible si vous souhaitez partager.",
            "Je perçois que c'est important pour vous. Comment puis-je vous aider au mieux ?"
        ]
        
        # Sélection basée sur un hash simple du texte pour consistance
        index = len(user_text) % len(fallback_responses)
        return fallback_responses[index]

    def _update_conversation_history(self, user_text: str, bot_response: str):
        """Met à jour l'historique avec limitation de taille."""
        self.conversation_history.append({"role": "user", "content": user_text})
        self.conversation_history.append({"role": "assistant", "content": bot_response})
        
        # Limitation de l'historique pour éviter les prompts trop longs
        if len(self.conversation_history) > MAX_HISTORY * 2:  # *2 car user + assistant
            self.conversation_history = self.conversation_history[-MAX_HISTORY * 2:]

    def get_stats(self) -> dict:
        """Retourne des statistiques sur l'état du bot."""
        return {
            "ready": self.is_ready,
            "documents_count": len(self.documents),
            "emotions_supported": len(self.labels) if self.labels else 0,
            "conversation_length": len(self.conversation_history),
            "model_loaded": self.clf is not None,
            "embedder_loaded": self.embedder is not None,
            "index_ready": self.index is not None
        }

    def clear_history(self):
        """Vide l'historique de conversation."""
        self.conversation_history.clear()
        logger.info("Historique de conversation vidé")

    def test_connection(self) -> bool:
        """Teste la connexion aux services externes."""
        try:
            # Test Ollama
            test_messages = [{"role": "user", "content": "Test de connexion"}]
            response = ollama_chat(test_messages, max_retries=1)
            
            if response.startswith("[Erreur"):
                logger.error("Connexion Ollama échouée")
                return False
            
            # Test PostgreSQL
            with psycopg.connect(PG_CONN_STR) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1;")
                    cur.fetchone()
            
            logger.info("✅ Connexions externes OK")
            return True
            
        except Exception as e:
            logger.error(f"❌ Test de connexion échoué: {e}")
            return False

# ===============================
# 7. FONCTION DE TEST RAPIDE (OPTIONNEL)
# ===============================
def debug_ollama():
    """Debug Ollama étape par étape - fonction utilitaire."""
    import requests
    
    print("🔍 Test de connexion Ollama...")
    
    # Test 1: Service disponible
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"✅ Ollama service: {r.status_code}")
        if r.status_code == 200:
            models = r.json().get('models', [])
            print(f"   Modèles disponibles: {[m['name'] for m in models]}")
    except Exception as e:
        print(f"❌ Ollama service non accessible: {e}")
        return False
    
    # Test 2: Chat endpoint
    try:
        payload = {
            "model": "llama2:latest",
            "messages": [{"role": "user", "content": "Test simple"}],
            "stream": False
        }
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
        print(f"✅ Chat endpoint: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"   Structure réponse: {list(data.keys())}")
            if "message" in data:
                content = data["message"].get("content", "")
                print(f"   Contenu: {content[:50]}...")
                return True
    except Exception as e:
        print(f"❌ Chat endpoint: {e}")
        return False
    
    return False

# ===============================
# 8. MAIN — EXÉCUTION AMÉLIORÉE
# ===============================
def main():
    """Fonction principale avec interface utilisateur améliorée."""
    print("🤖 Initialisation du Chatbot Émotionnel...")
    print("=" * 50)
    
    # Configuration du bot
    bot = EmotionalRAGBot(force_french=FORCE_FRENCH)
    
    # Test des connexions
    if not bot.test_connection():
        print("❌ Erreur: Services externes non disponibles")
        print("Vérifiez qu'Ollama est démarré et que PostgreSQL est accessible")
        print("\n Pour débugger Ollama, vous pouvez aussi lancer:")
        print("   python -c \"from try import debug_ollama; debug_ollama()\"")
        return
    
    # Configuration (première fois: retrain_clf=True, refresh_store=True)
    print("⚙️  Configuration du système...")
    success = bot.setup(retrain_clf=True, refresh_store=True)
    
    if not success:
        print("❌ Échec de l'initialisation du chatbot")
        return
    
    # Affichage des statistiques
    stats = bot.get_stats()
    print(f"""
✅ Chatbot prêt !
📊 Statistiques:
   - Documents indexés: {stats['documents_count']:,}
   - Émotions supportées: {stats['emotions_supported']}
   - Modèle: {OLLAMA_MODEL}

💬 Commandes disponibles:
   - 'quit'/'exit' : Quitter
   - 'stats' : Afficher les statistiques
   - 'clear' : Vider l'historique
   - 'test' : Tester les connexions
   - 'debug' : Debug Ollama détaillé

Tapez votre message et appuyez sur Entrée...
""")
    
    # Boucle de conversation
    while True:
        try:
            user_input = input("\n👤 Vous: ").strip()
            
            if not user_input:
                continue
                
            # Commandes spéciales
            if user_input.lower() in {'quit', 'exit', 'stop', 'bye'}:
                print(" Au revoir ! Prenez soin de vous.")
                break
                
            elif user_input.lower() == 'stats':
                stats = bot.get_stats()
                print(f"""
📊 Statistiques actuelles:
   - Prêt: {'✅' if stats['ready'] else '❌'}
   - Documents: {stats['documents_count']:,}
   - Émotions: {stats['emotions_supported']}
   - Historique: {stats['conversation_length']//2} échanges
""")
                continue
                
            elif user_input.lower() == 'clear':
                bot.clear_history()
                print("🧹Historique vidé")
                continue
                
            elif user_input.lower() == 'test':
                if bot.test_connection():
                    print("✅ Toutes les connexions sont OK")
                else:
                    print("❌ Problème de connexion détecté")
                continue
            
            elif user_input.lower() == 'debug':
                debug_ollama()
                continue
            
            # Génération de la réponse
            start_time = time.time()
            response = bot.reply(user_input)
            response_time = time.time() - start_time
            
            print(f"\n🤖 Assistant ({response_time:.1f}s): {response}")
            
        except KeyboardInterrupt:
            print("\n\n Interruption détectée. Au revoir !")
            break
            
        except EOFError:
            print("\n Session terminée. Au revoir !")
            break
            
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
            print("❌ Une erreur inattendue s'est produite. Veuillez réessayer.")

if __name__ == "__main__":
    main()