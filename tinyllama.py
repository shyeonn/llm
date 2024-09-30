from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# GPU 사용 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 모델 및 토크나이저 로드
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. 모델을 GPU로 이동
model = model.to(device)
token = 0
ex_time = 0
iteration = 20

ds = load_dataset("tatsu-lab/alpaca", split="train")
ds_16 = ds.filter(lambda example: 16 == len(tokenizer(example['instruction'])['input_ids']) and 0 == len(example['input']))

# 3. 입력 텍스트 설정
for i in range(iteration):
    print (f"Iteration {i}")

    input_text = ds_16[i]['instruction']
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    # 4. 모델에 입력 데이터 전달 (GPU에서 실행)
    start = time.time()
    with torch.no_grad():  # 추론 시에는 그래디언트 계산 비활성화
        output = model.generate(input_ids, max_length=2048)
    
    ex_time = ex_time + (time.time() - start)

    token = token + len(output[0])
    print(token)

latency = token / ex_time

print(f"Latnecy(token/s) : {latency}")   
