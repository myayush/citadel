FROM python:3.11-slim

# HF Docker Spaces run the container as user 1000; anything it writes must be owned by it.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence-transformers

USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image so cold starts do not download 90MB.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY --chown=user src/ src/
COPY --chown=user static/ static/

EXPOSE 7860
CMD python -m uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-7860}