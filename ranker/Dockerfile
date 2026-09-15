# Hugging Face Spaces, Docker SDK.
#
# Two decisions worth reading before you change anything here:
#
#  1. torch comes from the CPU wheel index. The default wheel bundles CUDA and
#     is roughly 2GB larger, for hardware a free Space does not have.
#  2. the model is downloaded at build time, not at runtime. A free Space
#     sleeps when idle, so the first request after a nap already pays a cold
#     start; making it also download 130MB of weights is what turns that into
#     a timeout.
FROM python:3.11-slim

# Spaces runs the container as uid 1000. Creating that user ourselves keeps the
# pip cache and the model cache writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Bake the weights into the image.
ARG RANKER_MODEL=BAAI/bge-small-en-v1.5
ENV RANKER_MODEL=${RANKER_MODEL}
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${RANKER_MODEL}')"

COPY --chown=user . .

# Spaces expects 7860 unless app_port says otherwise in README.md.
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
