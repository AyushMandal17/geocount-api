FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# CPU-only torch keeps the image small
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

ARG MODEL_WEIGHTS=yolo26m-obb.pt
ENV MODEL_WEIGHTS=${MODEL_WEIGHTS} YOLO_CONFIG_DIR=/tmp/Ultralytics
# bake weights into the image so the first request is not a download
RUN python -c "import os; from ultralytics import YOLO; YOLO(os.environ['MODEL_WEIGHTS'])"

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
