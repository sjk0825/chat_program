import json
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

config_model    = open('../common/config_model.json','r',encoding='utf-8')
config_model    = json.load(config_model)

config_chat    = open('../common/config_chat.json','r',encoding='utf-8')
config_chat    = json.load(config_chat)

config_common  = open('../common/config_common.json','r',encoding='utf-8')
config_common  = json.load(config_common)

vectordb_url   = config_common['vectordb_url']
retrieval_url  = config_common['retrieval_url']
generation_url = config_common['generation_url']

@app.route("/", methods=["GET"])
def home():
    return "Hello, Flask!"

@app.route("/echo", methods=["POST"])
def echo():
    data = request.json  # JSON 요청 받기
    message = data.get("user_message", "key value does not exists")
    return jsonify({"response": f"Echo: {message}"})

@app.route("/bot", methods=["POST"])
def bot():
    data = request.json  # JSON 요청 받기
    chat_history  = data.get("chat_history", "key value does not exists")
    user_message  = data.get("user_message", "user_message value does not exists")

    ### request to retrieval server # api or private model
    sending_data = {
        "chat_history" : chat_history,
        "user_message" : user_message
    }

    response = requests.post(generation_url+'/generation', json=sending_data)
    message = response.json().get("response")

    return jsonify({"response": f"bot: {message}"})

'''
# TODO
@app.route("/set_generator", methods=["GET"])
def set_generator():
    sending_data = {
        "config_chat" : config_chat,
        "config_model": config_model,
        "config_common": config_common
    }

    response = requests.post(retrieval_url+'/set_generation', json=sending_data)
    response = response.json().get("response")
    return response

# TODO
@app.route("/set_retriever", methods=["GET"])
def set_retriever():

    sending_data = {
        "config_chat" : config_chat,
        "config_model": config_model,
        "config_common": config_common
    }

    response = requests.post(retrieval_url+'/set_retriever', json=sending_data)
    response = response.json().get("response")
    return response
'''

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
