import openai
import csv
import time

# OpenAI API 키 설정
openai.api_key = "YOUR_API_KEY"

# 입력 및 출력 파일 경로
input_file = "train_file_sum.tsv"
output_file = "train_augmented.tsv"

# GPT 프롬프트 생성 함수
def get_augmented_sentences(text):
    prompt = f"""다음 문장의 의미를 유지하면서 자연스럽고 다양한 표현으로 바꾼 문장을 5개 만들어줘. 문장은 너무 길지 않게 해주고, 일상적인 톤으로 써줘. 각 문장은 줄 바꿈해서 출력해줘.

문장: {text}
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9
        )
        result = response.choices[0].message.content.strip()
        # 줄바꿈 기준으로 문장 분리
        sentences = [line.lstrip("12345. ").strip() for line in result.split('\n') if line.strip()]
        return sentences
    except Exception as e:
        print(f"GPT 호출 오류: {e}")
        return []

# 증강 수행
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8", newline='') as fout:
    reader = csv.DictReader(fin, delimiter="\t")
    fieldnames = ["koreanText", "sign_lang_sntenc"]
    writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()

    for i, row in enumerate(reader):
        original_text = row["koreanText"]
        sign_lang = row["sign_lang_sntenc"]

        # 기존 문장 포함
        writer.writerow({"koreanText": original_text, "sign_lang_sntenc": sign_lang})

        # GPT로 5개 생성
        augmented_texts = get_augmented_sentences(original_text)

        for j, aug_text in enumerate(augmented_texts):
            writer.writerow({"koreanText": aug_text, "sign_lang_sntenc": sign_lang})
            print(f"[{i}] {j+1}개 생성: {aug_text}")

        time.sleep(1.2)  # API 속도 제한 회피용 (gpt-4 사용 시 필수)
