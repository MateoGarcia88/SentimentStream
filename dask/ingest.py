import csv
import os
import socket
import sys
import time
from datetime import datetime, timezone

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "dataset_sentimientos_500.csv"
)
HOST = os.getenv("INGEST_HOST", "localhost")
PORT = int(os.getenv("INGEST_PORT", 9999))
DELAY = 0.5


def stream_csv(host=HOST, port=PORT, delay=DELAY):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)
    print(f"[Ingest] Waiting for connection on {host}:{port} ...")

    conn, addr = server.accept()
    print(f"[Ingest] Client connected: {addr}")

    try:
        with open(DATA_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader, start=1):
                fecha = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                line = f"{i}|{row['texto']}|{row['etiqueta']}|{fecha}\n"
                conn.sendall(line.encode("utf-8"))
                print(f"[Ingest] Sent row {i}: {row['texto'][:45]}...")
                time.sleep(delay)
    finally:
        conn.close()
        server.close()
        print("[Ingest] Stream finished.")


if __name__ == "__main__":
    delay = float(sys.argv[1]) if len(sys.argv) > 1 else DELAY
    stream_csv(delay=delay)
