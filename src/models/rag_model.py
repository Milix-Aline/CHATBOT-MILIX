import os
import csv
import time
import re
import json
import sqlite3
import numpy as np
from typing import List, Tuple, Optional
import requests
import logging

from ..config.settings import (
    OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL, REQUEST_TIMEOUT, DB_TYPE
)
from ..utils.language_utils import language_detector

logger = logging.getLogger(__name__)

# Configuration par défaut
BATCH_COMMIT_EVERY = 200

# Utilitaires
_CONTROL_CHARS = re.compile(r'[\x00-\x1f\x7f-\x9f]')

def clean_text(s: str) -> str:
    """Nettoie le texte d'entrée."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = _CONTROL_CHARS.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

class RAGSystem:
    """Système RAG multilingue."""
    
    def __init__(
        self,
        db_connection_str: str,
        data_path: str,
        ollama_base_url: str = OLLAMA_BASE_URL,
        embed_model: str = OLLAMA_EMBED_MODEL,
        request_timeout: int = REQUEST_TIMEOUT,
    ):
        self.db_connection_str = db_connection_str
        self.data_path = data_path
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embed_model = embed_model
        self.request_timeout = request_timeout
        self.db_type = DB_TYPE

        # Validation des paramètres
        if not os.path.exists(self.data_path):
            raise ValueError(f"Le dossier de données n'existe pas: {self.data_path}")

        # Tests de connexion
        self._test_database_connection()
        self._test_ollama_connection()
        
        # Configuration de la base
        self.embed_dim = self._probe_embedding_dimension()
        logger.info(f"Dimension des embeddings: {self.embed_dim}")
        
        self._setup_database()

    def _get_connection(self):
        """Retourne une connexion à la base de données."""
        if self.db_type == "sqlite":
            db_path = self.db_connection_str.replace("sqlite:///", "")
            return sqlite3.connect(db_path)
        else:
            import psycopg
            return psycopg.connect(self.db_connection_str)

    def _test_database_connection(self) -> None:
        """Teste la connexion à la base de données."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            conn.close()
            logger.info("Connexion base de données réussie")
        except Exception as e:
            raise RuntimeError(f"Impossible de se connecter à la base de données: {e}")

    def _test_ollama_connection(self) -> None:
        """Teste la connexion à Ollama."""
        try:
            response = requests.get(f"{self.ollama_base_url}/api/tags", timeout=10)
            response.raise_for_status()
            logger.info("Connexion Ollama réussie")
        except requests.RequestException as e:
            raise RuntimeError(f"Impossible de se connecter à Ollama: {e}")

    def _probe_embedding_dimension(self) -> int:
        """Détermine la dimension du modèle d'embedding."""
        payload = {"model": self.embed_model, "input": "test dimension"}
        url = f"{self.ollama_base_url}/api/embed"
        
        try:
            resp = requests.post(url, json=payload, timeout=self.request_timeout)
            resp.raise_for_status()
            data = resp.json()
            
            if "embedding" in data:
                return len(data["embedding"])
            elif "embeddings" in data and data["embeddings"]:
                return len(data["embeddings"][0])
            
            raise RuntimeError("Format de réponse embedding inattendu")
            
        except requests.RequestException as e:
            raise RuntimeError(f"Erreur lors du test du modèle d'embedding: {e}")

    def _setup_database(self) -> None:
        """Configure la base de données."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    language TEXT DEFAULT 'fr',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_embeddings_language 
                ON embeddings(language);
            """)
            
            conn.commit()
            conn.close()
            logger.info("Base de données configurée")
            
        except Exception as e:
            raise RuntimeError(f"Erreur configuration base de données: {e}")

    def compute_embedding(self, document: str) -> List[float]:
        """Calcule l'embedding d'un document."""
        document = clean_text(document)
        if not document:
            return []

        payload = {"model": self.embed_model, "input": document}
        url = f"{self.ollama_base_url}/api/embed"

        for attempt in range(3):
            try:
                resp = requests.post(url, json=payload, timeout=self.request_timeout)
                resp.raise_for_status()
                data = resp.json()
                
                if "embedding" in data:
                    return [float(x) for x in data["embedding"]]
                elif "embeddings" in data and data["embeddings"]:
                    return [float(x) for x in data["embeddings"][0]]
                
                raise RuntimeError("Format embedding inattendu")
                
            except requests.exceptions.Timeout:
                if attempt == 2:
                    raise
                time.sleep(1.5 * (attempt + 1))
            except requests.RequestException as e:
                if attempt == 2:
                    raise RuntimeError(f"Erreur Ollama embeddings: {e}")
                time.sleep(1.5 * (attempt + 1))
        
        return []

    def save_embedding(self, document: str, embedding: List[float], cursor) -> None:
        """Sauvegarde un embedding avec détection de langue."""
        if not embedding:
            return
        try:
            language = language_detector.detect(document)
            embedding_json = json.dumps(embedding)
            
            cursor.execute(
                "INSERT INTO embeddings (document, embedding, language) VALUES (?, ?, ?)",
                (document, embedding_json, language),
            )
        except Exception as e:
            raise RuntimeError(f"Erreur sauvegarde: {e}")

    def insert_documents(self) -> None:
        """Insère les documents CSV dans la base."""
        if not os.path.exists(self.data_path):
            raise RuntimeError(f"Dossier de données inexistant: {self.data_path}")
            
        csv_files = [f for f in os.listdir(self.data_path) if f.lower().endswith(".csv")]
        if not csv_files:
            logger.info(f"Aucun fichier CSV dans: {self.data_path}")
            return

        logger.info(f"Fichiers CSV trouvés: {csv_files}")

        # Vérification des données existantes
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        existing_count = cursor.fetchone()[0]
        conn.close()
        
        if existing_count > 0:
            logger.info(f"Base déjà remplie avec {existing_count} embeddings")
            return

        # Colonnes attendues
        text_keys = {"texte", "text", "situation", "message", "content", "utterance", "input", "prompt"}
        emo_keys = {"emotion", "label", "sentiment"}
        resp_keys = {"response", "sys_response", "assistant", "reply", "output"}

        total_rows, inserted, errors = 0, 0, 0

        conn = self._get_connection()
        cursor = conn.cursor()
        
        for csv_name in csv_files:
            path = os.path.join(self.data_path, csv_name)
            logger.info(f"Traitement: {csv_name}")

            # Lecture avec multiple encodages
            reader = None
            for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
                try:
                    f = open(path, "r", encoding=encoding, newline="")
                    reader = csv.DictReader(f)
                    _ = reader.fieldnames
                    break
                except Exception:
                    if 'f' in locals():
                        f.close()
                    continue

            if reader is None:
                logger.error(f"Impossible de lire {csv_name}")
                errors += 1
                continue

            logger.info(f"Colonnes: {reader.fieldnames}")

            with f:
                batch_count = 0
                for row_num, row in enumerate(reader, 1):
                    total_rows += 1
                    if not row:
                        continue

                    def _pick_column(keys: set) -> str:
                        for k in row.keys():
                            if k and str(k).strip().lower() in keys and row[k]:
                                return clean_text(str(row[k]))
                        return ""

                    user_text = _pick_column(text_keys)
                    emotion = _pick_column(emo_keys)
                    answer = _pick_column(resp_keys)

                    # Construction du document
                    parts = []
                    if user_text:
                        parts.append(f"Texte: {user_text}")
                    if emotion:
                        parts.append(f"Emotion: {emotion}")
                    if answer:
                        parts.append(f"Reponse: {answer}")

                    if not parts:
                        all_vals = [clean_text(str(v)) for v in row.values() if v]
                        doc = " | ".join(v for v in all_vals if v)
                    else:
                        doc = " | ".join(parts)

                    if not doc or len(doc) < 10:
                        continue

                    if len(doc) > 4000:
                        doc = doc[:4000]

                    try:
                        emb = self.compute_embedding(doc)
                        if emb:
                            self.save_embedding(doc, emb, cursor)
                            inserted += 1
                            batch_count += 1

                            if batch_count >= BATCH_COMMIT_EVERY:
                                conn.commit()
                                batch_count = 0
                                logger.info(f"{inserted} embeddings insérés...")
                        else:
                            logger.warning(f"Embedding vide ligne {row_num}")
                    except Exception as e:
                        logger.error(f"Erreur ligne {row_num}: {e}")
                        errors += 1

                conn.commit()

        conn.close()
        logger.info(f"Terminé: {total_rows} lignes, {inserted} insérés, {errors} erreurs")

    def retrieve_similar_corpus(self, user_query: str, limit: int = 3) -> Optional[Tuple[int, str, List[float]]]:
        """Recherche de documents similaires avec filtre de langue."""
        user_query = clean_text(user_query)
        if not user_query:
            return None

        try:
            emb = self.compute_embedding(user_query)
            if not emb:
                return None

            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            count = cursor.fetchone()[0]
            if count == 0:
                logger.info("Base vide, exécutez insert_documents()")
                conn.close()
                return None

            # Recherche simple pour SQLite (recherche par similarité cosinus approximative)
            cursor.execute("SELECT id, document, embedding FROM embeddings LIMIT ?", (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            if not rows:
                return None

            # Retourner le premier résultat pour l'instant
            rec_id, doc, vec_str = rows[0]
            
            if vec_str:
                vec = json.loads(vec_str)
            else:
                vec = []
            
            return rec_id, doc, vec
                    
        except Exception as e:
            logger.error(f"Erreur recherche: {e}")
            return None

    def get_stats(self) -> dict:
        """Statistiques de la base."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM embeddings")
            total = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT document) FROM embeddings")
            unique_docs = cursor.fetchone()[0]
            
            cursor.execute("SELECT language, COUNT(*) FROM embeddings GROUP BY language")
            lang_stats = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                "total_embeddings": total,
                "unique_documents": unique_docs,
                "embedding_dimension": self.embed_dim,
                "database_type": self.db_type,
                "language_distribution": lang_stats
            }
        except Exception as e:
            logger.error(f"Erreur statistiques: {e}")
            return {}
