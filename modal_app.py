import modal

WEIGHTS = "yolo26m-obb.pt"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements("requirements.txt")
    .run_commands(
        f"mkdir -p /models && cd /models && python -c \"from ultralytics import YOLO; YOLO('{WEIGHTS}')\""
    )
    .env({
        "MODEL_WEIGHTS": f"/models/{WEIGHTS}",
        "ALLOWED_ORIGINS": "*",
        "YOLO_CONFIG_DIR": "/tmp/Ultralytics",
    })
    .add_local_python_source("app")
)

app = modal.App("geocount-api", image=image)


@app.function(cpu=4, memory=4096, timeout=300, scaledown_window=300)
@modal.asgi_app()
def web():
    from app.main import app as fastapi_app

    return fastapi_app
