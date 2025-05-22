
instruction_chat = """You are a chatbot. I will give you some relevant information, so please respond kindly. 

retrieved_text : {retrieved_text} 

web_search : {web_search}

chat_history : {chat_history}

user_message : {user_message}

answer: 
"""

instruction_rerank = """You are a re-ranker. Rank the given information in order of highest relevance to the user_message.

candidate1: {candidate1}
candidate2: {candidate2}
candidate3: {candidate3}
candidate4: {candidate4}
candidate5: {candidate5}

"""

class promptManager():
    def __init__(self):

        return
