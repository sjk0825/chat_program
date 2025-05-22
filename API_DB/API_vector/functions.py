
import requests
import json
from datasets import load_dataset
from pymilvus import MilvusClient


config_common = open('../../common/config_common.json','r',encoding='utf-8')
retrieval_url = json.load(config_common)['retrieval_url']
# MilvusClient(uri="http://localhost:19530") #

def deploy(config_model, config_chat):
    client = MilvusClient('milvus_demo.db')  
    embedder = config_chat['embedder']

    collection_name = "project1_" + embedder.replace('-','_')
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    client.create_collection(
        collection_name=collection_name,
        dimension=config_model[embedder]['dim']
    )

    data = load_data()

    sending_data = {
        "texts" :  data,
        "config_chat":config_chat,
        "config_model": config_model
    }

    response = requests.post(retrieval_url+'/embedding', json=sending_data)
    embeds = response.json().get("response")

    input_data = []
    for i, (text, embed) in enumerate(zip(data, embeds)):
        input_data.append({"id": i, "vector":embed, "text": text})
    
    client.insert(collection_name=collection_name, data=input_data)
    
    return result

def search(embed_texts, config_model, config_chat):
    client = MilvusClient('milvus_demo.db')  
    embedder = config_chat['embedder']
    collection_name = "project1_" + embedder.replace('-','_')

    search_res = client.search(
        collection_name=collection_name,
        data=embed_texts, 
        limit=1,
        search_params={"metric_type": "COSINE", "params": {}}, 
        output_fields=["text"],
    )
    search_res = [item[0]['entity']['text'] for item in search_res]
    return search_res

def load_data():
    ds = load_dataset("Gwanwoo/combined_korean_wiki")

    ds_title = ds['train']['title']
    ds_text  = ds['train']['text']
    
    ds_data = [title + ':' + text for title, text in zip(ds_title, ds_text)]

    return ds_data[:2] # TODO remove