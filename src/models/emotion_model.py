# src/models/emotion_model.py

from typing import List, Tuple
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import os

logger = logging.getLogger(__name__)

# Dataset simple pour entraînement/évaluation
class EmotionTextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_len: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EmotionClassifier:
    """
    Classificateur d'émotions utilisant BERT (ou modèle Hugging Face compatible).
    Prédit une émotion parmi un set défini.
    """

    EMOTION_LABELS = ['joie', 'tristesse', 'colère', 'peur', 'neutre']

    def __init__(self, model_name: str = "camembert-base", device: str = None, model_path: str = "models/emotion_bert"):
        self.model_name = model_name
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.ready = False

        if os.path.exists(model_path):
            self.load()

    def train(self, texts: List[str] = None, labels: List[int] = None, epochs: int = 2, batch_size: int = 16):
        """
        Entraîne le modèle sur les textes et labels fournis.
        Si texts/labels non fournis, utilise un dataset dummy (simulé).
        """
        try:
            logger.info("Préparation de l'entraînement EmotionClassifier...")

            # Dataset simulé si non fourni
            if texts is None or labels is None:
                texts = [
                    "Je suis très heureux aujourd'hui",
                    "Je me sens triste et seul",
                    "Je suis en colère contre cette situation",
                    "J'ai peur de l'avenir",
                    "Rien de spécial, journée normale"
                ]
                labels = [0, 1, 2, 3, 4]  # indices des EMOTION_LABELS

            dataset = EmotionTextDataset(texts, labels, self.tokenizer)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=len(self.EMOTION_LABELS)
            ).to(self.device)

            # Arguments d'entraînement
            training_args = TrainingArguments(
                output_dir="./models/tmp",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                logging_dir="./logs",
                logging_steps=10,
                save_strategy="no"
            )

            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset
            )

            trainer.train()
            self.ready = True

            # Sauvegarde du modèle
            os.makedirs(self.model_path, exist_ok=True)
            self.model.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            logger.info(f"Modèle EmotionClassifier entraîné et sauvegardé dans {self.model_path}")

        except Exception as e:
            logger.error(f"Erreur lors de l'entraînement: {e}")
            self.ready = False

    def load(self) -> bool:
        """Charge le modèle depuis le disque."""
        try:
            if os.path.exists(self.model_path):
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.ready = True
                logger.info(f"Modèle EmotionClassifier chargé depuis {self.model_path}")
                return True
            else:
                logger.warning("Aucun modèle EmotionClassifier trouvé sur le disque")
                return False
        except Exception as e:
            logger.error(f"Erreur lors du chargement: {e}")
            self.ready = False
            return False

    def predict(self, texts: List[str]) -> Tuple[List[str], List[float]]:
        """Prédit les émotions pour une liste de textes."""
        if not self.is_ready():
            logger.warning("EmotionClassifier non prêt. Retourne neutre.")
            return ["neutre" for _ in texts], [0.5 for _ in texts]

        self.model.eval()
        emotions = []
        confidences = []

        try:
            for text in texts:
                encoding = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**encoding)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    conf, pred_idx = torch.max(probs, dim=1)
                    emotions.append(self.EMOTION_LABELS[pred_idx.item()])
                    confidences.append(conf.item())
        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {e}")
            return ["neutre" for _ in texts], [0.5 for _ in texts]

        return emotions, confidences

    def is_ready(self) -> bool:
        """Indique si le modèle est prêt."""
        return self.ready
