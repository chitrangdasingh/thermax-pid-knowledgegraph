FROM python:3.12-slim

ARG INSTALL_VISION=false
ARG INSTALL_GRAPHRAG=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements*.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && if [ "$INSTALL_VISION" = "true" ]; then python -m pip install --no-cache-dir -r requirements-vision.txt; fi \
    && if [ "$INSTALL_GRAPHRAG" = "true" ]; then python -m pip install --no-cache-dir -r requirements-graphrag.txt; fi

COPY . .
RUN mkdir -p /app/runtime

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["python", "-m", "streamlit", "run", "app.py"]
