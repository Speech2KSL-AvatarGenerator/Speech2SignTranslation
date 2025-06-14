# 한국어-한국수어 변환 

음성 입력을 기반으로 한국어 문장을 수어 문법 구조로 자동 변환하는 Transformer 기반의 번역 모델입니다. 본 저장소는 훈련된 모델 가중치와 토크나이저 설정 파일들을 포함하고 있으며, SKT의 KoBART를 기반으로 fine-tuning 되었습니다.

## 구성 파일
| 파일명                       | 설명                          |
| ------------------------- | --------------------------- |
| `config.json`             | KoBART 모델의 구성 정보       |
| `generation_config.json`  | 텍스트 생성 설정 (e.g., 최대 길이, 반복 억제 등)  |
| `special_tokens_map.json` | 특수 토큰 매핑 정보 (e.g., PAD, BOS, EOS)   |
| `tokenizer.json`          | 학습된 토크나이저의 전체 구조            |
| `tokenizer_config.json`   | 토크나이저 세부 설정                 |


## 사용 기술
- 토크나이저 : KoBARTTokenizer
- 학습 프레임워크 : pytorch, huggingface transformers
- 모델 아키텍쳐 : KoBART (gogamza/kobart-base-v2)
- 데이터셋 : 국립 국어원 한국어-한국수어 데이터

## 모델 설명
- 기반 모델: KoBART (Korean BART)
- 입력 데이터: 텍스트 또는 음성 인식 결과 (예: Whisper를 통해 STT 변환한 문장)
- 출력 데이터: 수어 문법에 맞는 변환된 한국어 문장 (예: SOV 구조)

적용 예시:
```
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("./")
model = AutoModelForSeq2SeqLM.from_pretrained("./")

sentence = "오늘 날씨 어때?"
inputs = tokenizer(sentence, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## 학습 노트
- 훈련 목적: 일반 한국어 문장을 수어 문법 구조로 변환하는 Seq2Seq 모델 학습
- 데이터셋 구성 : 국립국어원의 한국어-한국수어 병렬 말뭉치
- 데이터 증강 방식: GPT 기반 말뭉치 증강
| 구분    | 내용                                                     |
| ----- | ------------------------------------------------------ |
| 목적    | 동일 수어 문장에 대해 다양한 한국어 표현 생성으로 데이터 다양성 확보 및 모델 일반화 성능 향상 |
| 방법    | GPT 모델을 prompt-tunning하여 원문 의미 유지, 자연스럽고 다양한 표현 5개 생성 후 원본 데이터에 추가  | 
| 사용 도구 | OpenAI GPT-3.5-turbo API               |  


- 사전 학습 언어 모델 : gogamza/kobart-base-v2 (한국어에 특화된 BART 구조의 사전학습 모델)
- 토크나이저 : KoBARTTokenizer 사용, <s>, </s>, <pad> 특수 토큰 포함
- 하이퍼파라미터 : 
  - max_length: 128
  - num_train_epochs: 3
  - learning_rate: 5e-5
  - warmup_steps: 500
  - per_device_train_batch_size: 16
  - gradient_accumulation_steps: 2
  - fp16: True
