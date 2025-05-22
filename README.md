

# window wsl2 setting

# wsl2(ubuntu) docker, container setting
- window <-> wsl2 container portforwaring / container host open
- retrieval, generation, fron/backend, DB, evaluation container

# port
- retrieval: flask(2234:2234)
- embedder (triton): 8000,8001,8002 # do not use
- generation: flask(5002:5002)
- front/backend: gradio(7860:7860), flask(5000:5000)
- DB: postgresql(5432:5432), milvus(2379:2379, 19530:19530), m2ysql(3306:3306), flask(5001:5001)
- evaluation: mlflow()

# account
- huggingface, openai, serper

# deployement

# training

# chatting
