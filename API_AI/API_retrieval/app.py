import requests
import json
import logging
from openai import OpenAI
from flask import Flask, request, jsonify
from functions import set_retrieval

logger = logging.getLogger("Retireval Logger")
logger.setLevel(logging.INFO)

retriever     = None

# set obj
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "Hello, retrieval!"

@app.route("/retrieve", methods=["POST"])
def retrieval():
    data = request.json  # JSON 요청 받기
    user_message    = data.get("user_message", "user_message value does not exists")
    config_model    = data.get("config_model", "config_model value does not exists") 
    config_chat     = data.get("config_chat", "config_chat value does not exists") 

    retrieved_text = retriever.retrieval(user_message)

    return jsonify({"response": '. '.join(retrieved_text)})

@app.route("/embedding", methods=["POST"])
def embedding():
    data = request.json  # JSON 요청 받기
    texts           = data.get("texts", "chat_history value does not exists")

    embedded_texts   = retriever.get_embedding(texts)

    return jsonify({"response": embedded_texts})

@app.route("/set_retriever", methods=["POST"])
def set_retriever():
    global retriever
    data = request.json  # JSON 요청 받기
    config_model  = data.get("config_model", " config_model does not exists")
    config_chat   = data.get("config_chat", " config_chat does not exists")
    config_common = data.get("config_common", " config_common does not exists")

    retriever, set_results = set_retrieval(config_common, config_chat, config_model)

    return jsonify({"response": set_results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=2234)
