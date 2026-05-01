import os
import re
import joblib
import nltk
from nltk.corpus import stopwords
from datetime import datetime
from flask import Flask, jsonify, request
from pymongo import MongoClient, DESCENDING
from bson import ObjectId

nltk.download('stopwords', quiet=True)

app = Flask(__name__)

MONGO_URI = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'sentiment_model.pkl')

_mongo = MongoClient(MONGO_URI)
_col = _mongo['sentimentstream']['predictions']

STOPWORDS_ES = set(stopwords.words('spanish'))
_model_cache = None


def _get_model():
    global _model_cache
    if _model_cache is None:
        _model_cache = joblib.load(MODEL_PATH)
    return _model_cache


def _clean(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', '', text)
    return ' '.join(t for t in text.split() if t not in STOPWORDS_ES and len(t) > 2)


def _serialize(doc: dict) -> dict:
    doc['_id'] = str(doc['_id'])
    if isinstance(doc.get('timestamp'), datetime):
        doc['timestamp'] = doc['timestamp'].isoformat()
    return doc


@app.route('/sentiments', methods=['GET'])
def sentiments():
    """List predictions with optional filters: ?sentimiento=positivo&fecha=2024-03-01&limit=50"""
    query = {}
    if s := request.args.get('sentimiento'):
        query['prediccion'] = s
    if f := request.args.get('fecha'):
        query['fecha'] = f
    limit = int(request.args.get('limit', 50))
    docs = list(_col.find(query).sort('timestamp', DESCENDING).limit(limit))
    return jsonify([_serialize(d) for d in docs])


@app.route('/stats', methods=['GET'])
def stats():
    """Class distribution, total count, and model accuracy."""
    dist_agg = [
        {'$group': {'_id': '$prediccion', 'count': {'$sum': 1}}},
        {'$sort': {'count': -1}},
    ]
    distribution = {d['_id']: d['count'] for d in _col.aggregate(dist_agg)}
    total = sum(distribution.values())

    acc_agg = [
        {'$match': {'sentimiento_real': {'$ne': None}}},
        {'$project': {'ok': {'$cond': [{'$eq': ['$prediccion', '$sentimiento_real']}, 1, 0]}}},
        {'$group': {'_id': None, 'hits': {'$sum': '$ok'}, 'n': {'$sum': 1}}},
    ]
    acc = list(_col.aggregate(acc_agg))
    accuracy = round(acc[0]['hits'] / acc[0]['n'], 4) if acc else None

    return jsonify({'total': total, 'distribucion': distribution, 'accuracy': accuracy})


@app.route('/predict', methods=['POST'])
def predict():
    """Classify a new text. Body: {"texto": "..."}"""
    body = request.get_json(force=True) or {}
    texto = body.get('texto', '').strip()
    if not texto:
        return jsonify({'error': 'Campo "texto" requerido'}), 400

    artifact = _get_model()
    pipeline = artifact['pipeline']
    label_names = artifact['label_names']

    proba = pipeline.predict_proba([_clean(texto)])[0]
    idx = int(proba.argmax())
    prediction = label_names[idx]
    confidence = float(proba[idx])

    _col.insert_one({
        'texto': texto,
        'prediccion': prediction,
        'sentimiento_real': None,
        'confianza': round(confidence, 4),
        'timestamp': datetime.utcnow(),
        'fecha': datetime.utcnow().strftime('%Y-%m-%d'),
    })

    return jsonify({
        'prediccion': prediction,
        'confianza': round(confidence, 4),
        'probabilidades': {label_names[i]: round(float(p), 4) for i, p in enumerate(proba)},
    })

@app.route('/sentiments/bi', methods=['GET'])
def sentiments_bi():
    """BI-Friendly sentiments endpoint"""
    limit = int(request.args.get('limit', 500))

    cursor = (
        _col.find({},{
            '__id': 0,
            'texto': 1,
            'prediccion': 1,
            'confianza': 1,
            'timestamp': 1
        })
        .sort('timestamp', DESCENDING)
        .limit(limit)
    )
    data = []
    for d in cursor:
        if isinstance(d.get('timestamp'), datetime):
            dt = d['timestamp']
        else:
            continue

        data.append({
            'texto': d.get('texto'),
            'sentimiento': d.get('prediccion'),
            'confianza': d.get('confianza'),
            'datetime': dt.isoformat()
        })
        return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
