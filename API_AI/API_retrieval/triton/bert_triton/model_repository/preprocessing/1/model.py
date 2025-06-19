import triton_python_backend_utils as pb_utils
import numpy as np
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        print('Initialized...')
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.max_length = 128

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                # 입력 추출
                text_tensor = pb_utils.get_input_tensor_by_name(request, "RAW_INPUT")
                texts = text_tensor.as_numpy().astype(str).tolist()

                # 토크나이징
                encoded = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="np"
                )

                # Triton Tensor로 변환
                input_ids_tensor = pb_utils.Tensor("input_ids", encoded["input_ids"].astype(np.int64))
                attention_mask_tensor = pb_utils.Tensor("attention_mask", encoded["attention_mask"].astype(np.int64))

                # token_type_ids는 없을 수도 있음 → default로 0 배열 생성
                if "token_type_ids" in encoded:
                    token_type_ids_tensor = pb_utils.Tensor("token_type_ids", encoded["token_type_ids"].astype(np.int64))
                else:
                    token_type_ids_tensor = pb_utils.Tensor(
                        "token_type_ids", np.zeros_like(encoded["input_ids"], dtype=np.int64)
                    )

                # 응답 생성
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[input_ids_tensor, attention_mask_tensor, token_type_ids_tensor]
                )
                responses.append(inference_response)

            except Exception as e:
                # Triton 에러 응답
                error_response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(f"Preprocessing error: {str(e)}")
                )
                responses.append(error_response)

        return responses

    def finalize(self):
        print("Cleaning up...")
