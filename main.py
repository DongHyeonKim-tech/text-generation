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
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device)

class TextRequest(BaseModel):
    prompt: str

class TextResponse(BaseModel):
    generated_text: str

def generate_text(prompt: str):
    try:
        messages = [
            {"role": "user", "content": prompt},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(**inputs, max_new_tokens=256)
        decoded_outputs = tokenizer.decode(outputs[0])
        print(decoded_outputs)
        generated_text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        skip_special_tokens_response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"skip_special_tokens_response: {skip_special_tokens_response}")
        return generated_text
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
