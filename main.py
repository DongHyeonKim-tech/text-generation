from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import json
import re

app = FastAPI(title="Text Generation API", description="AI 텍스트 생성 API")

# model_id = "openai/gpt-oss-20b"

# # MPS 사용 가능 여부 확인 및 device 설정
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")

# # 모델과 토크나이저를 전역으로 로드
# tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     torch_dtype="auto",
#     device_map="auto"
# )

class TextRequest(BaseModel):
    prompt: str

class TextResponse(BaseModel):
    generated_text: str

# def generate_text(prompt: str):
#     try:
#         inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)

#         # 입력 길이와 컨텍스트 여유 계산 (권장: 여유 32~64 토큰)
#         max_ctx = getattr(model.config, "max_position_embeddings", 8192)  # 모델에 따라 다름
#         inp_len = inputs.input_ids.shape[1]
#         safe_room = 64
#         max_new = min(512, max_ctx - inp_len - safe_room)  # 여유 내에서 생성길이 설정
#         max_new = max(64, max_new)  # 최소 64 보장

#         gen_out = model.generate(
#             **inputs,
#             max_new_tokens=max_new,        # 충분히 크게
#             min_new_tokens=32,             # 너무 짧게 안 끊기게
#             do_sample=True,                # 필요 시 False로 바꿔도 됨(결정론적)
#             temperature=0.7,
#             top_p=0.9,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.eos_token_id,
#             no_repeat_ngram_size=3,        # 반복 줄이기(선택)
#             repetition_penalty=1.05,       # 반복 줄이기(선택)
#             return_dict_in_generate=True,
#             output_scores=True,
#         )

#         # 프롬프트를 제외한 "완성 부분"만 디코드
#         generated_ids = gen_out.sequences[0]
#         completion_ids = generated_ids[inp_len:]  # 여기서부터가 실제 answer
#         answer = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        
#         return answer
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"텍스트 생성 중 오류가 발생했습니다: {str(e)}")

MODEL_ID = "openai/gpt-oss-20b"

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
)

# 🔁 매 호출마다 만들지 말고 전역에서 한 번만 생성
gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tok,
)

# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
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

    sys = "당신은 항상 한국어로만, 사용자가 이해하기 쉽게 답변합니다."
    # 한국어 고정 강화 문장
    lang_rule = "반드시 100% 한국어로만 답변하세요. 영어 용어가 필요하면 괄호 없이 자연스러운 한국어로 풀이하세요."

    base_prompt = f"{sys}\n{lang_rule}\n\n질문: {prompt}\n답변:"
    if verbose:
        print("[pipeline_prompt]", base_prompt)

    # 1차 생성
    out = gen(
        base_prompt,
        max_new_tokens=preset["max_new"],
        min_new_tokens=preset["min_new"],
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

    # 길이 보정: short는 너무 길면 2문장으로 컷, medium/long은 너무 짧으면 1회 재시도
    if length == "short":
        if len(text) > preset["max_chars"]:
            text = _shorten_to_sentences(text, n=2)

    else:
        min_chars = preset.get("min_chars", 0)
        if len(text) < min_chars:
            # 더 자세히 요청하여 1회 재생성
            regen_prompt = (
                f"{sys}\n{lang_rule}\n\n"
                f"질문: {prompt}\n"
                f"지시사항: 위 질문에 대해 더 자세하고 구체적으로, 예시/항목/단계 등을 포함해 설명하세요. 반복은 피하고 새로운 정보와 구조화를 제공하세요.\n"
                f"답변:"
            )
            if verbose:
                print("[regen_prompt: too short]")

            out2 = gen(
                regen_prompt,
                max_new_tokens=min(1024, preset["max_new"] * 2),
                min_new_tokens=max(64, preset["min_new"]),
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.05,
                return_full_text=False,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id,
            )
            text = out2[0]["generated_text"].strip()

    # 한국어 강제: 한글 비율이 낮으면 1회 재작성
    if force_korean and _korean_ratio(text) < 0.5:
        fix_prompt = (
            f"{sys}\n{lang_rule}\n\n"
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

    return text
# @app.post("/generate", response_model=TextResponse)
# async def generate_text_endpoint(request: TextRequest):
#     generated_text = generate_text(request.prompt)
#     return TextResponse(generated_text=generated_text)

@app.post("/generate_pipeline", response_model=TextResponse)
async def generate_text_endpoint(request: TextRequest):
    generated_text = get_text_from_pipeline(request.prompt)
    return TextResponse(generated_text=generated_text)

@app.get("/")
async def root():
    return {"message": "Text Generation API에 오신 것을 환영합니다"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8012)
