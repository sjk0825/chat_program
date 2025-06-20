
import requests

from openai import OpenAI
from huggingface_hub import InferenceClient


class retrievalManager():
    def __init__(self,config_common, config_chat, config_model):

        # init all model
        embedder = config_chat['embedder']
        api_key  = config_model[embedder]['api_key']

        self.config_common = config_common
        self.config_chat   = config_chat
        self.config_model  = config_model

        self.encoder_url = config_common['encoder_url']
        self.vectordb_url  = config_common['vectordb_url']

        if embedder in ['text-embedding-3-small']:
            client  = OpenAI(api_key=api_key)
            self.platform_manager = openaiManager(client, embedder)
        
        elif embedder in ['intfloat/multilingual-e5-large']:
            client = InferenceClient(model=embedder, token=api_key)
            self.platform_manager = huggingfaceAPIManager(client,api_key)
        else:
            raise NotImplementedError("please set various model logic") 
            
        return
    
    def retrieval(self,items):

        # get embedding from platform model # List
        embedded_usermessage = self.platform_manager.get_embedding(items)

        # search vector db 
        retrieved_text = self.search_vectordb(embedded_usermessage)

        return retrieved_text

    def search_vectordb(self,embedded_item):

        # search
        sending_data = {
            "config_model"  : self.config_model,
            "config_chat"   : self.config_chat,
            "embedded_text" : embedded_item # TODO item
        }

        response = requests.post(self.vectordb_url+'/search_vectordb', json=sending_data)
        retrieved_text = response.json().get("response")

        return retrieved_text # List

    def get_embedding(self, items):
        return self.platform_manager.get_embedding(items)

class openaiManager():
    def __init__(self, client, embedder):
        self.client = client
        self.embedder = embedder
        return
    

    def get_embedding(self,items):
        if isinstance(items, str):
            items = [items]
            
        batch = 50
        embeddings = []

        for _ in range(len(items),batch):
            response = self.client.embeddings.create(input=items, model=self.embedder)
            embeddings.extend([item.embedding for item in response.data])

        return embeddings

def huggingfaceAPIManager():
    def __init__(self, client, embedder):
        self.client = client
        self.embedder = embedder
        return
    
    def get_embedding(self,items):
        # we use only user_messages

        if isinstance(items, str):
            items = [items]
            
        batch = 50  # manually batch, check document for api option
        embeddings = []

        for _ in range(len(items),50):
            response = self.client.feature_extraction(items)
            embeddings.extend([item.embedding for item in response.data])

        return embeddings
