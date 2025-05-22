import http
import json
import requests

from openai import OpenAI
from .PromptManager import promptManager

class generationManager():
    def __init__(self, config_common, config_chat, config_model):
        # init all model
        generation_model = config_chat['generation_model']
        api_key          = config_model[generation_model]['api_key']

        self.config_common = config_common
        self.config_chat   = config_chat
        self.config_model  = config_model

        self.max_context   = config_model[generation_model]['max_context']

        self.retrieval_url = config_common['retrieval_url']
        self.vectordb_url  = config_common['vectordb_url']

        if generation_model in ['gpt-4o-mini']:
            client  = OpenAI(api_key=api_key)
            self.platform_manager = openaiManager(client=client,model=generation_model, tool_sequence=self.tool_sequence,max_context=self.max_context)
        else:
            raise 
            
        return
    
    def generation(self,chat_history, user_message):
        generated_text = self.platform_manager.generation(chat_history, user_message)

        return generated_text


    def tool_sequence(self,user_message):
        results_web = self.get_websearch(user_message)
        results_retrieval = self.get_retrieval(user_message)

        return 'web_info: ' + results_web + 'retrieval_info: ' + results_retrieval

    def get_websearch(self, user_message):

        api_key    = self.config_model['serper']['api_key']

        conn = http.client.HTTPSConnection("google.serper.dev")
        payload = json.dumps({
            "q": user_message,
            "gl": "kr",
            "hl": "ko"
        })
        headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
        }
        conn.request("POST", "/search", payload, headers)
        res  = conn.getresponse()
        data = res.read()
        data = json.loads(data.decode('utf-8'))  
        data = data['organic']                   
        data = [item['snippet'] for item in data]

        return '. '.join(data)

    def get_retrieval(self, user_message):
        sending_data = {
            "config_model": self.config_model,
            "config_chat" : self.config_chat,
            "user_message": user_message
        }

        response       = requests.post(self.retrieval_url+'/retrieve', json=sending_data)
        retrieved_text = response.json().get("response")
        return retrieved_text


class openaiManager():
    def __init__(self, client,model,tool_sequence,max_context):
        self.client = client
        self.model   = model
        self.max_context = max_context
        self.tool_sequence=tool_sequence
        return
    
    def generation(self, chat_history, user_message):
        history_format = []
        for i, item in enumerate(chat_history):
            user = {"role":"user", "content": item[0]}
            bot  = {"role":"assistant", "content": item[1]}
            history_format.append(user)
            history_format.append(bot)

        history_format.append({"role":"developer","content":self.tool_sequence(user_message)[:self.max_context]})
        history_format.append({"role":"user","content":user_message})

        tools = [{
            "type": "function",
            "name": "get_websearch",
            "description": "Get related web search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"}
                },
                "required": ["user_message"],
                "additionalProperties": False
            },
            "strict": True
        }]

        tools = [{
            "type": "function",
            "name": "get_retrieval",
            "description": "Get related information from previously saved context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_message": {"type": "string"}
                },
                "required": ["user_message"],
                "additionalProperties": False
            },
            "strict": True
        }]
        
        # we do not use function calling, forcly run a search function
        response = self.client.responses.create(
            model=self.model,
            #tools=tools,
            #tool_choice={"type":"function", "name":"tool_sequence"},
            input=history_format,
        )

        return  response.output_text
