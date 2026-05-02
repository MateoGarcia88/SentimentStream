import os
import sys
import socket
import joblib
from datetime import datetime, timezone

from pymongo import MongoClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from preprocessing import clean_text

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "model", "sentiment_model.pkl"
)
HOST = os.getenv("INGEST_HOST", "localhost")
PORT = int(os.getenv("INGEST_PORT", 9999))
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = "sentimentstream"
MONGO_COL = "predictions"


def predict_stream():
    artifact = joblib.load(MODEL_PATH)
    pipeline = artifact["pipeline"]
    label_names = artifact["label_names"]

    mongo = MongoClient(MONGO_URI)
    col = mongo[MONGO_DB][MONGO_COL]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print(f"[Predict] Connected to ingest at {HOST}:{PORT}")

    buffer = ""
    try:
        while True:
            chunk = sock.recv(4096).decode("utf-8")
            if not chunk:
                break
            buffer += chunk
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) != 4:
                    continue
                row_id, texto, sentimiento_real, fecha = parts

                limpio = clean_text(texto)
                proba = pipeline.predict_proba([limpio])[0]
                idx = int(proba.argmax())
                prediction = label_names[idx]
                confidence = float(proba[idx])

                doc = {
                    "id": int(row_id),
                    "texto": texto,
                    "prediccion": prediction,
                    "sentimiento_real": sentimiento_real,
                    "confianza": round(confidence, 4),
                    "timestamp": datetime.now(timezone.utc),
                    "fecha": fecha,
                }
                col.insert_one(doc)
                print(
                    f"[Predict] #{row_id} → {prediction} ({confidence:.1%}) | real: {sentimiento_real}"
                )
    finally:
        sock.close()
        mongo.close()
        print("[Predict] Done.")


if __name__ == "__main__":
    predict_stream()
