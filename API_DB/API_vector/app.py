
import requests
import logging
from flask import Flask, request, jsonify
from functions import deploy, search

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

@app.route("/", methods=["GET"])
def home():
    return "Hello, Flask!"

@app.route("/set_vectordb", methods=["POST"])
def set_vectordb():
    data = request.json  # JSON 요청 받기
    config_chat  = data.get("config_chat", "key value does not exists")
    config_model = data.get("config_model", "key value does not exists")

    result = deploy(config_model, config_chat)

    logging.info(f'set_vertordb is done')
    return jsonify({"response": None})

@app.route("/search_vectordb", methods=["POST"])
def search_vectordb():
    data = request.json  # JSON 요청 받기
    config_chat   = data.get("config_chat", "key value does not exists")
    config_model  = data.get("config_model", "key value does not exists")
    embedded_text = data.get("embedded_text", "key value does not exists")

    search_results = search(embedded_text, config_model, config_chat)

    return jsonify({"response": search_results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5001)
