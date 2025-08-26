from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import torch

app = FastAPI(title="Text Generation API", description="AI 텍스트 생성 API")

model_id = "openai/gpt-oss-20b"

class TextRequest(BaseModel):
    prompt: str

class TextResponse(BaseModel):
    generated_text: str

def generate_text(prompt: str):
    try:
        pipe = pipeline(
            "text-generation",
            model=model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        messages = [
            {"role": "user", "content": prompt},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        return outputs[0]["generated_text"]
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
