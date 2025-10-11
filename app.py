import runpod
from transformers import pipeline


def load_model():
    return pipeline(
        "text-generation", model="google/gemma-2b", use_auth_token=os.getenv("HF_TOKEN")
    )


def handler(event):
    global model

    if "model" not in globals():
        model = load_model()

    text = event["input"].get("text")

    if not text:
        return {"error": "No text provided."}

    result = model(text)[0]

    return {"answer": result}


runpod.serverless.start({"handler": handler})