import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        """
        Parameters
        ----------
        args : dict
          - "model_config" : config.pbtxt의 내용을 담은 JSON string
          - "model_instance_kind"
          - "model_instance_device_id"
          - "model_repository" : model repository 경로
          - "model_version"
          - "model_name"
        """
        print('Initialized...')
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    
    def execute(self, requests):
        """
        Parameters
        ----------
        requests : list of pb_utils.InferenceRequest

        Returns
        -------
        list of pb_utils.InferenceResponse. 
        """
        
        responses = []
        for request in requests:
            # 입력 추출 (BYTES type → np.object_ → Python str)
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            texts = text_tensor.as_numpy().tolist()  # list of bytes
            texts = [t.decode("utf-8") if isinstance(t, bytes) else t for t in texts]

            # 토크나이징
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="np"
            )

            # Triton Tensor로 변환
            input_ids_tensor = pb_utils.Tensor(
                "input_ids", encoded["input_ids"].astype(np.int64)
            )
            attention_mask_tensor = pb_utils.Tensor(
                "attention_mask", encoded["attention_mask"].astype(np.int64)
            )

            # 응답 생성
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[input_ids_tensor, attention_mask_tensor]
            )
            responses.append(inference_response)

        return responses
    
    def finalize(self):
        print("Cleaning up...")