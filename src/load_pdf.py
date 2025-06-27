import os
import json
import fitz
import re
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

PDF_PATH = "/home/rhtm02/workspace/mcp_ipsi/data/2026학년도 고려대학교 서울캠퍼스 수시모집요강(2025.06.17.)(배포용).pdf"
VECTOR_DB_DIR = "faiss_korea_from_gpt"
JSON_OUTPUT_PATH = "korea_admission_parsed.json"
EMBEDDING_MODEL_NAME = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
GPT_MODEL = "gpt-4o"
TEMPERATURE = 0.0
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 200
UNIV = '고려대학교'
YEAR = '2026'


# 1. PDF 블록 기반 텍스트 추출
def extract_pdf_text_blocks(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    for page in doc:
        page_blocks = page.get_text("blocks")
        for block in page_blocks:
            text = block[4].strip()
            if text:
                blocks.append(text)
    doc.close()
    return "\n\n".join(blocks)

# 2. 마크다운 스타일 정리
def convert_to_markdown(text: str) -> str:
    lines = text.splitlines()
    markdown_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            markdown_lines.append("")
            continue

        if re.match(r'^[(<【]?\d+[\).＞】]?\s*', line):
            markdown_lines.append(f"## {line}")
        elif "전형" in line and len(line) < 30:
            markdown_lines.append(f"### {line}")
        elif ":" in line:
            markdown_lines.append(f"**{line}**")
        elif line.startswith(("•", "-")):
            markdown_lines.append(f"- {line.lstrip('•-').strip()}")
        else:
            markdown_lines.append(line)

    return "\n".join(markdown_lines)

# 3. 청크 분할
def split_text_for_gpt(text: str, chunk_size=3000, chunk_overlap=300):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)


def split_text_for_gpt(full_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(full_text)


def build_contextual_prompt(chunk, univ, years, previous_summary=""):
    context = f"이전까지 요약된 전형 정보:\n{previous_summary.strip()}\n\n" if previous_summary else ""
    return f"""
{context}
너는 {univ} {years}학년도 수시모집요강 전문을 구조화하고 있어.
지금까지 일부 전형들을 추출해왔고, 아래는 그 다음 청크야.

이 청크 안에 등장하는 전형에 대해서도 이전과 동일한 방식으로,
**누락 없이, 축약 없이, 전형별/모집단위별 특이사항을 JSON으로 구조화** 해줘.
단, 청크 내에서 전형 정보와 무관한 일반 설명(예: 학과 개요, 캠퍼스 위치, 수시/정시 모집 계획 비교 등)은 제외해줘.

추가로 **"채용조건형 계약학과처럼 특별한 제도나 장학금, 이수 조건 등이 있는 경우 반드시 ‘모집단위특이사항’ 필드에 상세히 포함할 것"**

출력은 아래 포맷을 따라:
[
  {{
    "page_content": "해당 전형에 대한 원문 전체 또는 일부 요약 (되도록 원문 기반으로 구체적으로 작성)",
    "metadata": {{
      "전형명": "학생부종합(일반전형)",
      "전형유형": "학생부종합",
      "전형방식": "서류 100%",
      "평가요소": ["학생부", "자기소개서"],
      "수능최저여부": "적용 안 함",
      "모집단위": ["경영학과", "정치외교학과", "..."],
      "모집단위특이사항": {{
        "의과대학": "수능최저 적용",
        "간호학과": "면접 있음"
        "특수학과": "채용형, 장학금 있음"
      }}
    }}
  }},
  ...
]
응답은 반드시 JSON 배열로만 출력하고, ```json 같은 코드블록 마크다운은 사용하지 마.
청크 내용:
{chunk.strip()}
""".strip()


def query_gpt(prompt_with_chunk):
    chat = ChatOpenAI(model=GPT_MODEL, temperature=TEMPERATURE)
    response = chat([HumanMessage(content=prompt_with_chunk)])
    return response.content


def save_json(parsed_all, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_all, f, ensure_ascii=False, indent=2)


def parse_documents(parsed_json):
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in parsed_json]


def save_faiss_vectorstore(docs, output_dir):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(output_dir)
    return output_dir


if __name__ == "__main__":
    print("🔍 PDF 텍스트 추출 중...")
    extracted_text = extract_pdf_text_blocks(PDF_PATH)
    markdown_text = convert_to_markdown(extracted_text)

    print("✂️ 텍스트 분할 중...")
    chunks = split_text_for_gpt(markdown_text)

    all_parsed = []
    all_summaries = []

    for i, chunk in enumerate(chunks):
        print(f"🤖 GPT 요청 중... chunk {i+1}/{len(chunks)}")
        prev_summary = "\n\n".join([f"[전형명: {p['metadata'].get('전형명', '')}]\n요약 내용: {p.get('page_content', '').strip()}" for p in all_parsed[-3:]])
        prompt = build_contextual_prompt(chunk, univ=UNIV, years=YEAR, previous_summary=prev_summary)
        try:
            response = query_gpt(prompt)
            parsed = json.loads(response)
            all_parsed.extend(parsed)
        except Exception as e:
            print(f"❌ 청크 {i+1} 처리 실패:", e)

    print(f"📄 JSON 저장 중... ({JSON_OUTPUT_PATH})")
    save_json(all_parsed, JSON_OUTPUT_PATH)

    print("📦 Document 객체 변환 중...")
    documents = parse_documents(all_parsed)

    print(f"💾 벡터 DB 저장 중... ({VECTOR_DB_DIR})")
    save_faiss_vectorstore(documents, VECTOR_DB_DIR)

    print(f"✅ 완료: {len(documents)}개의 문서가 저장되었습니다.")
