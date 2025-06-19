
from tritonclient.http import InferenceServerClient, InferInput
import numpy as np

# Triton 서버 주소
triton_url = "172.22.245.142:8000"
model_name = "ensemble"

'''
curl -X POST localhost:8000/v2 \

'''

# Triton client 생성
client = InferenceServerClient(url=triton_url)

# 입력 텍스트 (np.object_ 또는 dtype=object 필요)
texts = [["hello triton", "generative ai is powerful"]]
np_texts = np.array(texts, dtype=object)

# Triton 입력 생성 (BYTES 타입)
text_input = InferInput("RAW_INPUT", np_texts.shape, "BYTES")
text_input.set_data_from_numpy(np_texts)

# 요청
response = client.infer(model_name=model_name, inputs=[text_input])
print(response.as_numpy('output__0'))