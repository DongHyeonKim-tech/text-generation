from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import re
import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI(title="Text Generation API", description="AI 텍스트 생성 API")

class TextRequest(BaseModel):
    prompt: str

class TextResponse(BaseModel):
    generated_text: str

MODEL_ID = "openai/gpt-oss-20b"

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
)

LENGTH_PRESETS = {
    "short":  {"max_new": 128, "min_new": 24,  "max_chars": 350},
    "medium": {"max_new": 384, "min_new": 64,  "min_chars": 200},
    "long":   {"max_new": 768, "min_new": 128, "min_chars": 450},
}

SHORT_HINTS  = ["한줄", "짧게", "간단히", "요약", "tl;dr"]
LONG_HINTS   = ["자세히", "길게", "예시", "코드", "단계별", "튜토리얼", "상세히"]

def _detect_length(prompt: str) -> str:
    p = prompt.lower()
    if any(h in prompt for h in SHORT_HINTS): return "short"
    if any(h in prompt for h in LONG_HINTS):  return "long"
    return "medium"

def _korean_ratio(text: str) -> float:
    # 한글 문자 비율로 대충 감지
    hangul = re.findall(r"[가-힣]", text)
    return len(hangul) / max(1, len(text))

def _shorten_to_sentences(text: str, n: int = 2) -> str:
    # '.' '!' '?' 기준으로 앞 n문장만 반환 (한국어 마침표 포함)
    parts = re.split(r"(?<=[\.!?。！?])\s+", text.strip())
    return " ".join(parts[:n]).strip()

# ─────────────────────────────────────────────────────────────
# 메인 함수
# ─────────────────────────────────────────────────────────────
def get_text_from_pipeline(prompt: str, length: str = "auto", force_korean: bool = True, verbose: bool = False) -> str:
    """
    length: "auto" | "short" | "medium" | "long"
    - auto: 프롬프트 키워드로 길이 추정
    - force_korean: 한국어 비율이 낮으면 1회 재생성하여 한국어로만 다시 요청
    """
    if length == "auto":
        length = _detect_length(prompt)
    preset = LENGTH_PRESETS[length]

    sys = "Always answer in Korean."

    base_prompt = f"{sys}\nquestion: {prompt}\nanswer:"
    if verbose:
        print("[pipeline_prompt]", base_prompt)

    # 1차 생성
    out = gen(
        base_prompt,
        # max_new_tokens=preset["max_new"],
        # min_new_tokens=preset["min_new"],
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        no_repeat_ngram_size=3,
        repetition_penalty=1.05,
        return_full_text=False,  # ← 프롬프트 제외하고 결과만
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    text = out[0]["generated_text"].strip()
    print(f"text: {text}")
    print(f"answer: {text.split('answer:')[1].strip()}")
    # # 길이 보정: short는 너무 길면 2문장으로 컷, medium/long은 너무 짧으면 1회 재시도
    # if length == "short":
    #     if len(text) > preset["max_chars"]:
    #         text = _shorten_to_sentences(text, n=2)

    # else:
    #     min_chars = preset.get("min_chars", 0)
    #     if len(text) < min_chars:
    #         # 더 자세히 요청하여 1회 재생성
    #         regen_prompt = (
    #             f"{sys}\n{lang_rule}\n\n"
    #             f"질문: {prompt}\n"
    #             f"지시사항: 위 질문에 대해 더 자세하고 구체적으로, 예시/항목/단계 등을 포함해 설명하세요. 반복은 피하고 새로운 정보와 구조화를 제공하세요.\n"
    #             f"답변:"
    #         )
    #         if verbose:
    #             print("[regen_prompt: too short]")

    #         out2 = gen(
    #             regen_prompt,
    #             max_new_tokens=min(1024, preset["max_new"] * 2),
    #             min_new_tokens=max(64, preset["min_new"]),
    #             temperature=0.7,
    #             top_p=0.95,
    #             do_sample=True,
    #             no_repeat_ngram_size=3,
    #             repetition_penalty=1.05,
    #             return_full_text=False,
    #             eos_token_id=tok.eos_token_id,
    #             pad_token_id=tok.eos_token_id,
    #         )
    #         text = out2[0]["generated_text"].strip()
    
    # 한국어 강제: 한글 비율이 낮으면 1회 재작성
    if force_korean and _korean_ratio(text) < 0.5:
        print("=========================================한글 비율 낮아서 재생성=========================================")
        fix_prompt = (
            f"{sys}\n\n"
            f"다음 내용을 자연스러운 한국어로만 다시 작성하세요. 불필요한 영어 표기는 제거하고 맥락을 유지하세요.\n\n"
            f"내용:\n{text}\n\n"
            f"한국어 답변:"
        )
        if verbose:
            print("[regen_prompt: force korean]")

        out3 = gen(
            fix_prompt,
            max_new_tokens=min(1024, preset["max_new"] * 2),
            min_new_tokens=64,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.05,
            return_full_text=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
        text = out3[0]["generated_text"].strip()
        print(f"재생성 결과: {text}")

    return text
# @app.post("/generate", response_model=TextResponse)
# async def generate_text_endpoint(request: TextRequest):
#     generated_text = generate_text(request.prompt)
#     return TextResponse(generated_text=generated_text)

@app.post("/generate_pipeline", response_model=TextResponse)
async def generate_text_endpoint(request: TextRequest):
    generated_text = get_text_from_pipeline(request.prompt)
    return TextResponse(generated_text=generated_text)

@app.post("/generate-with-runpod", response_model=TextResponse)
async def generate_text_with_runpod(request: TextRequest):
    # 환경 변수에서 API 키 가져오기, 없으면 기본값 사용
    api_key = os.environ.get("RUNPOD_KEY", "your_runpod_api_key_here")
    
    print(f"api_key: {api_key}")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.runpod.ai/v2/kvu3npoylfiknt/openai/v1",
    )
    
    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[{"role": "user", "content": request.prompt}],
        temperature=0,
        max_tokens=100,
    )

    generated_text = response.choices[0].message.content
    return TextResponse(generated_text=generated_text)

@app.get("/")
async def root():
    return {"message": "Text Generation API에 오신 것을 환영합니다"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
