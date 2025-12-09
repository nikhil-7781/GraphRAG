"""
GraphLLM - Modal Deployment
Serverless ML deployment with auto-scaling
"""
import modal

# Create Modal app
app = modal.App("graphllm")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("tesseract-ocr", "ghostscript", "gcc", "g++")
    .pip_install_from_requirements("requirements.txt")
)

# Create persistent volume for data storage
volume = modal.Volume.from_name("graphllm-data", create_if_missing=True)

# Mount FastAPI app
@app.function(
    image=image,
    gpu=None,  # Use CPU (cheaper)
    memory=4096,  # 4GB RAM
    timeout=600,  # 10 min timeout
    volumes={"/app/data": volume},
    secrets=[modal.Secret.from_name("graphllm-secrets")],  # GEMINI_API_KEY
)
@modal.asgi_app()
def fastapi_app():
    """
    Mount the FastAPI application
    """
    import sys
    sys.path.insert(0, "/root")

    # Import main FastAPI app
    from main import app as fastapi_app

    return fastapi_app


# Local testing endpoint
@app.local_entrypoint()
def main():
    """
    Test the deployment locally
    """
    print("GraphLLM deployed to Modal!")
    print("Access your app at: https://YOUR_USERNAME--graphllm-fastapi-app.modal.run")
