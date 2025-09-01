#!/usr/bin/env python3
"""
Point d'entrée principal pour CHATBOT-MILIX
Usage:
    python main.py              # Lance l'interface en ligne de commande
    python main.py --api        # Lance l'API FastAPI
    python main.py --setup      # Configuration initiale
"""

import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.services.chatbot_service import ChatbotService
from src.config.settings import API_HOST, API_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_cli():
    print("🤖 MILIX - Chatbot Émotionnel")
    print("=" * 50)
    
    service = ChatbotService()
    print("Initialisation en cours...")
    
    if not service.initialize():
        print("❌ Échec de l'initialisation")
        return
    
    print("✅ Chatbot prêt !")
    print("\nCommandes: 'quit', 'stats', 'clear', 'test'")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 Vous: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Au revoir !")
                break
                
            elif user_input.lower() == 'stats':
                stats = service.get_stats()
                print(f"📊 Stats: {stats}")
                continue
                
            elif user_input.lower() == 'clear':
                service.clear_history()
                print("🧹 Historique vidé")
                continue
                
            elif user_input.lower() == 'test':
                results = service.test_connections()
                print(f"🔗 Tests: {results}")
                continue
            
            response, emotion = service.get_response(user_input)
            print(f"\n🤖 MILIX ({emotion}): {response}")
            
        except KeyboardInterrupt:
            print("\n\nAu revoir !")
            break
        except Exception as e:
            logger.error(f"Erreur: {e}")
            print("❌ Une erreur s'est produite")

def run_api():
    try:
        import uvicorn
        uvicorn.run(
            "src.api.routes:app",
            host=API_HOST,
            port=API_PORT,
            reload=True,
            log_level="info"
        )
    except ImportError:
        print("❌ uvicorn non installé. Installez-le avec: pip install uvicorn")
    except Exception as e:
        logger.error(f"Erreur démarrage API: {e}")

def setup():
    print("🔧 Configuration initiale de MILIX")
    
    from src.config.settings import PROJECT_ROOT, DATA_DIR, ARTIFACTS_DIR
    
    print(f"📁 Projet: {PROJECT_ROOT}")
    print(f"📁 Données: {DATA_DIR}")
    print(f"📁 Artifacts: {ARTIFACTS_DIR}")
    
    csv_files = list(DATA_DIR.glob("*.csv"))
    print(f"📄 Fichiers CSV trouvés: {len(csv_files)}")
    for csv_file in csv_files:
        print(f"   - {csv_file.name}")
    
    if not csv_files:
        print("⚠️  Aucun fichier CSV trouvé dans data/raw/")
        print("Ajoutez vos fichiers CSV dans ce dossier avant de continuer.")
        return
    
    print("\n🔗 Test des connexions...")
    service = ChatbotService()
    connections = service.test_connections()
    
    for service_name, status in connections.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {service_name}")
    
    if not all(connections.values()):
        print("\n⚠️  Certains services ne sont pas disponibles.")
        print("Vérifiez que PostgreSQL et Ollama sont démarrés.")
        return
    
    print("\n🚀 Initialisation complète...")
    if service.initialize(setup_rag=True, train_emotions=True):
        print("✅ Configuration terminée avec succès !")
    else:
        print("❌ Échec de la configuration")

def main():
    parser = argparse.ArgumentParser(description="CHATBOT-MILIX")
    parser.add_argument("--api", action="store_true", help="Lance l'API FastAPI")
    parser.add_argument("--setup", action="store_true", help="Configuration initiale")
    parser.add_argument("--cli", action="store_true", help="Interface ligne de commande (par défaut)")
    
    args = parser.parse_args()
    
    if args.setup:
        setup()
    elif args.api:
        run_api()
    else:
        run_cli()

if __name__ == "__main__":
    main()
