import gradio as gr
import requests
import json
from functions import to_base64

config_common = open('../common/config_common.json','r',encoding='utf-8')
config_common = json.load(config_common)

config_chat  =  open('../common/config_chat.json', 'r',encoding='utf-8')
config_chat  = json.load(config_chat)

config_model = open('../common/config_model.json','r', encoding='utf-8')
config_model = json.load(config_model)

vectordb_url = config_common['vectordb_url']
backend_url  = config_common['backend_url']
generation_url = config_common['generation_url']
retrieval_url = config_common['retrieval_url']

def chat_with_history(user_message, chat_history):

    if chat_history is None:
        chat_history = []
    
    sending_data = {
        "chat_history": chat_history,
        "user_message": user_message,
    }

    response = requests.post(generation_url+'/generation', json=sending_data)
    bot_message = response.json().get("response")

    chat_history.append(("User: " + user_message, bot_message))
    return "", chat_history

def set_generator():
    sending_data = {
        "config_chat" : config_chat,
        "config_model": config_model,
        "config_common": config_common
    }

    response = requests.post(generation_url+'/set_generator', json=sending_data)
    response = response.json().get("response")

    return response


def sef_retriever():
    sending_data = {
        "config_chat" : config_chat,
        "config_model": config_model,
        "config_common": config_common
    }

    response = requests.post(retrieval_url+'/set_retriever', json=sending_data)
    response = response.json().get("response")

    return response

def set_vectordb():
    global config_chat
    sending_data = {
        "config_chat" : config_chat,
        "config_model": config_model
    }

    response = requests.post(vectordb_url+'/set_vectordb', json=sending_data)
    response = response.json().get("response")

    return response

def set_ai():
    print('hello world')
    set_generator()
    sef_retriever()
    return

with gr.Blocks() as demo:
    # setting generator

    # ai setting
    ai_btn = gr.Button("ai setting")
    ai_btn.click(set_ai, inputs=[], outputs=[])

    # db setting
    db_btn = gr.Button("db setting")
    db_btn.click(set_vectordb, inputs=[], outputs=[])

    with gr.Row():
        img_input = gr.Image(type="pil", label="이미지를 업로드하세요")
        b64_output = gr.Textbox(label="Base64 문자열", lines=5)
    img_input.change(to_base64, inputs=img_input, outputs=b64_output) # TODO retriever base64 check

    # chat base
    chatbot = gr.Chatbot(label="Chat with History")
    user_input = gr.Textbox(placeholder="메시지를 입력하세요...", show_label=False)
    send_btn = gr.Button("Send")

    user_input.submit(chat_with_history, inputs=[user_input, chatbot], outputs=[user_input, chatbot])
    send_btn.click(chat_with_history, inputs=[user_input, chatbot], outputs=[user_input, chatbot])


demo.launch(server_name='0.0.0.0',server_port=7860)
