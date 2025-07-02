from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# ✅ Step 1: 임베딩 모델 불러오기 (vector DB 생성 시 사용한 것과 동일하게!)
embedding_model = HuggingFaceEmbeddings(
    model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS",
    model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
)

# ✅ Step 2: 저장된 벡터 DB 로드
vectordb_acceptance = FAISS.load_local(
    "vectorstore_acceptance",
    embedding_model,
    allow_dangerous_deserialization=True  # 🔥 반드시 신뢰된 파일만 허용!
)

vectordb_admission = FAISS.load_local(
    "vectorstore_admission",
    embedding_model,
    allow_dangerous_deserialization=True  # 🔥 반드시 신뢰된 파일만 허용!
)
# ✅ Step 3: 테스트용 쿼리 작성
query = """
내신은 평균 2.35이며, 주요 교과에서는 꾸준히 상위권을 유지했습니다. 진로는 심리학자 또는 상담교사로 설정했고, 이에 따라 3년간 심리학 관련 독서 활동과 심리 동아리 활동에 적극적으로 참여했습니다. '심리탐구반'이라는 자율 동아리에서 부장으로 활동하며 '인지심리학의 실제 적용 사례'에 대한 발표와 토론을 진행했고, 학급 내 또래상담 활동을 통해 실질적인 상담 실습을 경험했습니다.

또한 사회문제에 대한 관심을 바탕으로 '청소년 자살 예방 캠페인'을 기획 및 진행했고, 교육청 주최 심리 관련 논문 공모전에도 참가하여 '청소년의 SNS 중독과 자존감 관계 분석'이라는 주제로 입상하였습니다. 수능 모의 성적은 국어 3, 수학 4, 영어 2, 탐구(생활과 윤리/사회문화) 3, 2등급을 기록했습니다.

자기주도학습 태도도 우수하여 독서 활동으로는 『나는 내가 죽었다고 생각했습니다』, 『그로부터 어떻게 살아남았는가』 등을 읽고 생명존중과 심리 회복에 대한 에세이를 작성해 학교 독서상도 수상하였습니다. 교내 생명존중 프로젝트 참여, 상담실 프로그램 협력 등 전공 연계 활동이 매우 탄탄하게 구성되어 있으며, 면접 대비 과정에서도 심리적 자원 회복과 공감 능력에 대한 질문에 명확한 답변이 가능할 정도로 진로의식이 성숙합니다.

"""

# ✅ Step 4: 유사도 검색 실행
results_acceptance = vectordb_acceptance.similarity_search(query, k=5)  # 상위 5개 유사 문서
results_admission = vectordb_admission.similarity_search(query, k=5)  # 상위 5개 유사 문서

# ✅ Step 5: 결과 출력
print(f"\n[🔍 입력 쿼리] {query}\n")
print("🔎 유사도 검색 결과 (상위 5건):\n")

for i, doc in enumerate(results_acceptance, 1):
    print(f"[{i}]")
    print("📄 문장 내용:", doc.page_content)
    print("📎 메타데이터:", doc.metadata)
    print("-" * 70)
print("=" * 100)
for i, doc in enumerate(results_admission, 1):
    print(f"[{i}]")
    print("📄 문장 내용:", doc.page_content)
    print("📎 메타데이터:", doc.metadata)
    print("-" * 70)

'''
추가로 구현해야할점: 사례 유사도 기반 대학 전형중 유사도를 찾는게 중요해 보임, 여기서 대학별 vertorDB를 구축해 놓는것도 좋아보임
'''