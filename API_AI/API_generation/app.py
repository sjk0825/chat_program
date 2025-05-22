from flask import Flask, request, jsonify
from functions import set_generatation

# set obj
app = Flask(__name__)
generator = None

@app.route("/", methods=["GET"])
def home():
    return "Hello, retrieval!"

@app.route("/set_generator", methods=["POST"])
def request_generator():
    global generator
    data = request.json  # JSON 요청 받기
    config_model  = data.get("config_model", " config_model does not exists")
    config_chat   = data.get("config_chat", " config_chat does not exists")
    config_common = data.get("config_common", " config_common does not exists")

    generator, set_results = set_generatation(config_common, config_chat, config_model)

    return jsonify({"response": set_results})

@app.route("/generation", methods=["POST"])
def generation():
    global generator

    if generator is None:
        raise NotImplementedError("please add request logic to generator setting")

    data = request.json  # JSON 요청 받기
    chat_history    = data.get("chat_history", "chat_history value does not exists")
    user_message    = data.get("user_message", "user_message value does not exists")
    generated_text  = generator.generation(chat_history, user_message)

    return jsonify({"response": generated_text})


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5002)