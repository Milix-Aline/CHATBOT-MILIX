# CHATBOT-MILIX
Ce projet à pour but Implémenter un chatbot à l'aide de l'architecture RAG


# Chatbot Émotionnel RAG — Milix

Un chatbot intelligent et empathique capable de détecter les émotions dans les messages utilisateurs, de retrouver du contexte pertinent grâce au mécanisme de RAG (Retrieval-Augmented Generation) et de générer des réponses adaptées en s’appuyant sur un modèle de langage (LLM) exécuté localement via Ollama.

---

## Fonctionnalités

- Détection des émotions via un classifieur de Machine Learning (TF-IDF + SVM).  
- Mécanisme RAG basé sur SentenceTransformers, PostgreSQL et FAISS.  
- Génération de réponses par un modèle LLM (Ollama avec LLaMA2 ou Mistral).  
- Prise en compte de l’historique conversationnel pour une interaction cohérente.  
- Mécanisme de repli (fallback) si Ollama ou PostgreSQL sont inaccessibles.  
- Journalisation des étapes principales et des erreurs éventuelles.  

---

## Prérequis

- Python 3.10 ou supérieur  
- PostgreSQL installé et en cours d’exécution  
- Ollama installé et configuré (https://ollama.ai/)  
- Une machine avec au moins 6 Go de mémoire RAM si vous souhaitez utiliser LLaMA2 (Mistral est plus léger et adapté aux machines avec moins de mémoire)  

---

## Installation

1. Cloner le dépôt  
```bash
git clone https://github.com/Milix-Aline/CHATBOT-MILIX.git
cd chatbot-milix
```

2. Créer et activer un environnement virtuel  
```bash
python -m venv venv
venv\Scripts\activate      # Windows
```

3. Installer les dépendances  
```bash
pip install -r requirements.txt
```

4. Configurer PostgreSQL  
Créer une base de données nommée `chatbot_milix` et ajuster la chaîne de connexion dans le code si nécessaire :  
```python
PG_CONN_STR = "dbname=chatbot_milix user=postgres password=1234 host=localhost port=5432"
```

5. Démarrer Ollama avec un modèle supporté  
```bash
ollama run llama2
# ou
ollama run mistral
```

---

## Utilisation

Lancer le chatbot :  
```bash
python try.py
```

Au démarrage, le chatbot initialise ses composants, charge les données et indique s’il est prêt.  

Commandes spéciales disponibles :  
- `quit` ou `exit` : quitter la session  
- `stats` : afficher les statistiques internes  
- `clear` : réinitialiser l’historique  
- `test` : vérifier la connexion aux services externes  

---

## Architecture du pipeline

1. Prétraitement des données :  
   - Chargement et fusion des fichiers CSV  
   - Nettoyage et uniformisation des colonnes  
   - Suppression des doublons  

2. Entraînement du modèle de détection des émotions :  
   - Vectorisation TF-IDF  
   - Entraînement avec un SVM  
   - Évaluation sur un jeu de test  

3. Indexation et stockage :  
   - Génération d’embeddings avec SentenceTransformer  
   - Stockage dans PostgreSQL  
   - Indexation avec FAISS pour la recherche sémantique rapide  

4. Interaction utilisateur :  
   - Détection de l’émotion dans le message utilisateur  
   - Recherche de contexte dans la base  
   - Construction d’un prompt enrichi avec l’historique et le contexte  
   - Génération de la réponse via Ollama  

---

## Exemple de conversation

```
Utilisateur : Je me sens très stressé en ce moment.
Assistant   : Je comprends que tu traverses une période difficile et que la pression est forte. 
Prendre quelques minutes pour respirer ou faire une courte pause peut aider. 
Souhaites-tu que je partage des méthodes de relaxation ?
```

---

## Technologies utilisées

- Python (scikit-learn, pandas, numpy)  
- SentenceTransformers (all-MiniLM-L6-v2)  
- PostgreSQL (stockage des textes et embeddings)  
- FAISS (recherche sémantique rapide)  
- Ollama avec LLaMA2 ou Mistral (génération de texte)  

---

## Auteur

Projet développé par **Aline Miliguia** dans le cadre du projet Chatbot Milix.  
