from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

app = FastAPI(title="Text Generation API", description="AI 텍스트 생성 API")

model_id = "openai/gpt-oss-20b"

# MPS 사용 가능 여부 확인 및 device 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 모델과 토크나이저를 전역으로 로드
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto"
)

class TextRequest(BaseModel):
    prompt: str

class TextResponse(BaseModel):
    generated_text: str

def generate_text(prompt: str):
    try:
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

        # 입력 길이와 컨텍스트 여유 계산 (권장: 여유 32~64 토큰)
        max_ctx = getattr(model.config, "max_position_embeddings", 8192)  # 모델에 따라 다름
        inp_len = inputs.input_ids.shape[1]
        safe_room = 64
        max_new = min(512, max_ctx - inp_len - safe_room)  # 여유 내에서 생성길이 설정
        max_new = max(64, max_new)  # 최소 64 보장

        gen_out = model.generate(
            **inputs,
            max_new_tokens=max_new,        # 충분히 크게
            min_new_tokens=32,             # 너무 짧게 안 끊기게
            do_sample=True,                # 필요 시 False로 바꿔도 됨(결정론적)
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,        # 반복 줄이기(선택)
            repetition_penalty=1.05,       # 반복 줄이기(선택)
            return_dict_in_generate=True,
            output_scores=True,
        )

        # 프롬프트를 제외한 "완성 부분"만 디코드
        generated_ids = gen_out.sequences[0]
        completion_ids = generated_ids[inp_len:]  # 여기서부터가 실제 answer
        answer = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"텍스트 생성 중 오류가 발생했습니다: {str(e)}")

@app.post("/generate", response_model=TextResponse)
async def generate_text_endpoint(request: TextRequest):
    generated_text = generate_text(request.prompt)
    return TextResponse(generated_text=generated_text)

@app.get("/")
async def root():
    return {"message": "Text Generation API에 오신 것을 환영합니다"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
