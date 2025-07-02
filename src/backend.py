import json
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI # LLM은 그대로 OpenAI 모델을 사용
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

admissions_guidelines_path = 'path/to/your/admissions_guidelines.json'
success_cases_path = 'path/to/your/success_cases.json'

loader_guidelines = JSONLoader(
    file_path=admissions_guidelines_path,
    jq_schema='.[] | {content: .content, metadata: .metadata}',
    text_content=False)
loader_cases = JSONLoader(
    file_path=success_cases_path,
    jq_schema='.[] | {content: .content, metadata: .metadata}',
    text_content=False)

documents_guidelines = loader_guidelines.load()
documents_cases = loader_cases.load()
all_documents = documents_guidelines + documents_cases

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_documents = text_splitter.split_documents(all_documents)

# --- 2. 벡터 DB 구축 (임베딩 모델 변경) ---

# --- 수정된 부분 시작 ---
# 한국어 특화 모델인 'ko-sroberta-multitask'를 사용합니다.
# 이 모델은 로컬 환경에서 실행되므로 별도의 API 키가 필요 없습니다.
model_name = "jhgan/ko-sroberta-multitask"
model_kwargs = {'device': 'cpu'}  # GPU 사용 시 'cuda'로 변경
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
# --- 수정된 부분 끝 ---


# FAISS 벡터 스토어 생성 및 문서 저장
vector_store = FAISS.from_documents(split_documents, embeddings)


# --- 3. 검색기(Retriever) 및 QA 체인 설정 (이전과 동일) ---
retriever = vector_store.as_retriever()
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key="YOUR_OPENAI_API_KEY")

prompt_template = """
당신은 입시 컨설턴트입니다. 제공된 '컨텍스트' 정보를 바탕으로 '질문'에 대해 상세하고 친절하게 답변해주세요.
컨텍스트에서 정보를 찾을 수 없다면, 아는 대로 답변하지 말고 "정보를 찾을 수 없습니다."라고 솔직하게 말해주세요.

컨텍스트:
{context}

질문:
{question}

답변:
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)


# --- 4. 사용자 입력 처리 및 시스템 실행 (이전과 동일) ---
def get_consultation(student_info: dict, question: str) -> str:
    full_question = f"""
    아래는 학생의 정보입니다:
    - 내신 성적: {student_info.get('gpa', 'N/A')}
    - 주요 활동: {student_info.get('activities', 'N/A')}

    이 학생의 상황에 맞춰 다음 질문에 답변해주세요:
    {question}
    """
    result = qa_chain.invoke({"query": full_question})
    return result['result']


if __name__ == '__main__':
    user_student_info = {
        "gpa": "1.5",
        "activities": "AI 동아리 활동, 자율주행차 프로젝트, 관련 논문 탐구"
    }
    user_question = "제 생기부와 성적으로 서울대학교 컴퓨터공학부에 지원할 때 어떤 점을 강조하면 좋을까요?"

    consulting_answer = get_consultation(user_student_info, user_question)

    print("--- 입시 컨설팅 결과 ---")
    print(consulting_answer)