# Image de base légère et officielle Python
FROM python:3.12-slim

# Variables d'environnement pour optimiser Python en conteneur
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Installation des dépendances en premier pour profiter du cache Docker
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copie du code source et des ressources nécessaires
COPY app/ ./app/
COPY prompts/ ./prompts/
COPY data/ ./data/

# Création d'un utilisateur non-root pour la sécurité en production
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Vérification automatique de santé via l'endpoint /health
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
