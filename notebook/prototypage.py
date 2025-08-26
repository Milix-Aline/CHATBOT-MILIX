import pandas as pd
import numpy as np
import psycopg
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import ollama

# ===============================
# 1. Charger et fusionner les CSV
# ===============================
df1 = pd.read_csv("train (2).csv")
df2 = pd.read_csv("train.csv")
df3 = pd.read_csv("emotion_dataset.csv")

# Normaliser les colonnes
df1_clean = df1.rename(columns={"situation": "texte", "emotion": "emotion", "sys_response": "response"})[["texte", "emotion", "response"]]
df2_clean = df2.rename(columns={"situation": "texte", "emotion": "emotion"})
df2_clean["response"] = None
df2_clean = df2_clean[["texte", "emotion", "response"]]
df3_clean = df3.rename(columns={"Text": "texte", "Emotion": "emotion"})
df3_clean["response"] = None
df3_clean = df3_clean[["texte", "emotion", "response"]]

# Fusionner
df_final = pd.concat([df1_clean, df2_clean, df3_clean], ignore_index=True)

# ===============================
# 2. Générer embeddings
# ===============================
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(df_final["texte"].tolist(), show_progress_bar=True)

# ===============================
# 3. Sauvegarder dans PostgreSQL
# ===============================
with psycopg.connect("dbname=chatbot_milix user=postgres password=1234 host=localhost port=5432") as conn:
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                texte TEXT,
                emotion TEXT,
                response TEXT,
                embedding FLOAT8[]
            );
        """)
        conn.commit()

        # Insérer les données (éviter doublons avec ON CONFLICT si besoin)
        for i, row in df_final.iterrows():
            cur.execute("""
                INSERT INTO embeddings (texte, emotion, response, embedding)
                VALUES (%s, %s, %s, %s)
            """, (row["texte"], row["emotion"], row["response"], embeddings[i].tolist()))
        conn.commit()

print("✅ Données insérées dans PostgreSQL avec succès !")

# ===============================
# 4. Charger depuis PostgreSQL
# ===============================
with psycopg.connect("dbname=chatbot_milix user=postgres password=1234 host=localhost port=5432") as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT texte, emotion, response, embedding FROM embeddings;")
        rows = cur.fetchall()

documents = [row[0] for row in rows]
doc_emotions = [row[1] for row in rows]
doc_responses = [row[2] for row in rows]
doc_embeddings = np.array([row[3] for row in rows], dtype=np.float32)

# ===============================
# 5. Construire index FAISS
# ===============================
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

def search(query, top_k=3):
    query_emb = embedder.encode([query])
    D, I = index.search(np.array(query_emb, dtype=np.float32), top_k)
    return [(documents[i], doc_emotions[i], doc_responses[i]) for i in I[0]]

# ===============================
# 6. Détection de sentiment
# ===============================
sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def detect_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return result["label"], result["score"]

# ===============================
# 7. RAG + Réponse affective avec mémoire
# ===============================
conversation_history = []  # stocke les échanges {role, content}

def emotional_rag(query):
    # Détecter sentiment
    sentiment, score = detect_sentiment(query)

    # Chercher contexte
    context = search(query, top_k=3)
    context_text = "\n".join([f"- Texte: {c[0]} | Emotion: {c[1]} | Réponse: {c[2]}" for c in context])

    # Construire prompt avec mémoire de conversation
    history_text = "\n".join([f"{h['role']}: {h['content']}" for h in conversation_history])

    prompt = f"""
Tu es un assistant empathique.
Voici des exemples provenant de la base de données :
{context_text}

Historique de conversation :
{history_text}

Message utilisateur : {query}
Sentiment détecté : {sentiment} (confiance {score:.2f})

Réponds avec empathie et de manière adaptée à l’émotion de l’utilisateur.
"""

    response = ollama.chat(
        model="llama2",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # Sauvegarder dans l’historique
    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer

# ===============================
# Boucle interactive
# ===============================
print(" Chatbot émotionnel prêt ! (tape 'quit' pour arrêter)")

while True:
    user_input = input("Utilisateur : ")
    if user_input.lower() in ["quit", "exit", "stop"]:
        print(" Fin de la conversation.")
        break
    bot_reply = emotional_rag(user_input)
    print("Milix :", bot_reply)
