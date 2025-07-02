from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import json
import re
import torch 
import os

# 한국어 임베딩 모델 불러오기
embedding_model = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)
# 파일 경로
ACCEPTANCE_PATH = '/home/rhtm02/workspace/mcp_ipsi/data/cases.json'

ADMISSION_GUIDE_PATH = '/home/rhtm02/workspace/mcp_ipsi/data/'

ADMISSION_GUIDE_LIST = [path for path in os.listdir(ADMISSION_GUIDE_PATH) if path.endswith('.json') and 'cases' not in path]



# 마침표 기반 문장 분리 함수 (소수점 보호)
def split_into_sentences(text: str):
    return re.split(r'(?<=[.!?])\s+(?=[A-Z가-힣])', text.strip())

def remove_cite_tags(text: str) -> str:
    return re.sub(r"\[cite:.*?\]", "", text)

# JSON 로드 함수
def load_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# 합격 사례 -> 문장 단위 Document로 변환
# def parse_acceptance_documents(data):
#     documents = []
#     for item in data:
#         meta = item["metadata"]
#         sentences = split_into_sentences(item["vector_input_text"])
#         for sentence in sentences:
#             if sentence.strip():
#                 documents.append(Document(page_content=sentence.strip(), metadata=meta))
#         import ipdb;ipdb.set_trace()
#     return documents
def parse_acceptance_documents(data):
    documents = []
    for item in data:
        meta = item["metadata"]

        sentences = remove_cite_tags(item["vector_input_text"])
        sentences = split_into_sentences(sentences)

        merged_text = " ".join([s.strip() for s in sentences if s.strip()])
        documents.append(Document(page_content=merged_text, metadata=meta))

    return documents


# 입시 요강은 문서 단위로 처리 (분리 없음)
# def parse_admission_documents(data):
#     documents = []
#     for item in data:
#         import ipdb;ipdb.set_trace()
#         text = item["page_content"]
#         meta = item["metadata"]
#         documents.append(Document(page_content=text.strip(), metadata=meta))
#     return documents
def parse_admission_documents(datas):
    documents = []
    
    def _preprocess_dict(_data: dict) -> dict:
        for key, _ in _data.items():
            if isinstance(_data[key], str):
                text = remove_cite_tags(_data[key])
                _data[key] = text
            elif isinstance(_data[key], list):
                for item in _data[key]:
                    text = remove_cite_tags(item)
                    _data[key] = text
            elif isinstance(_data[key], dict):
                _data[key] = _preprocess_dict(_data[key])
        return _data

    for data in datas:
        university = remove_cite_tags(data.get("대학명"))
        vision = remove_cite_tags(data.get("인재상", ""))
        for info in data.get("전형정보", []):
            # text = f"[인재상] {vision}\n\n[전형요강] {info['page_content']}"
            text = remove_cite_tags(info['page_content'])
            meta = info.get("metadata", {})
            meta = _preprocess_dict(meta)
            meta["대학명"] = university  # 대학명 추가
            meta["인재상"] = vision  # 인재상 추가
            documents.append(Document(page_content=text.strip(), metadata=meta))
    return documents

# 벡터 DB 구축 함수
def build_vector_db():
    acceptance_data = load_json(ACCEPTANCE_PATH)
    admission_datas = [load_json(ADMISSION_GUIDE_PATH + path) for path in ADMISSION_GUIDE_LIST]

    acceptance_docs = parse_acceptance_documents(acceptance_data)
    admission_docs = parse_admission_documents(admission_datas)

    vectordb_acceptance = FAISS.from_documents(acceptance_docs, embedding_model)
    vectordb_admission = FAISS.from_documents(admission_docs, embedding_model)

    vectordb_acceptance.save_local("./vectorstore_acceptance")
    print(f"✅ 저장 완료! 합격 사례 문서 수: {len(acceptance_docs)}")

    vectordb_admission.save_local("./vectorstore_admission")
    print(f"✅ 저장 완료! 입시 요강 문서 수: {len(admission_docs)}")

build_vector_db()
