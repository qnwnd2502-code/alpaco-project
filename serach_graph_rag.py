# search_graph_rag.py

import numpy as np
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# ------------------------
# 경로 & 설정
# ------------------------
CSV_PATH = "/home/alpaco/kyj/Rnn/mergeds.csv"
EMB_PATH = "/home/alpaco/kyj/Rnn/new_question_embs.npy"

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "Neo4jPass123!"

# ------------------------
# 로딩
# ------------------------
print("CSV 로드 중...")
df = pd.read_csv(CSV_PATH)
df = df[["category", "question", "summary"]].fillna("")

print("임베딩 로드 중...")
q_embs = np.load(EMB_PATH)

device = "cpu"
sbert = SentenceTransformer("jhgan/ko-sroberta-multitask", device=device)

driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))


# ------------------------
# SIMILAR 이웃 가져오기
# ------------------------
def get_similar_from_graph(idx: int, limit: int = 5):
    with driver.session() as session:
        result = session.run(
            """
            MATCH (q:Question {idx: $idx})-[r:SIMILAR]->(other:Question)
            RETURN other.idx AS idx, other.text AS text, other.category AS category, r.score AS score
            ORDER BY r.score DESC
            LIMIT $limit
            """,
            idx=int(idx),
            limit=int(limit),
        )
        return list(result)


# ------------------------
# 메인 검색 함수
# ------------------------
def search_questions_graph_rag(
    raw_keywords,
    selected_category=None,
    topk_pool=20,
    final_topk=5,
    dup_threshold=0.90
):
    """
    raw_keywords : ["협업", "갈등"] 같은 키워드 리스트
    selected_category : "ICT" 등
    """

    # 1. 키워드 → 쿼리 문장
    query_sentence = ", ".join(raw_keywords) + " 에 대한 면접 질문"

    # 2. 쿼리 임베딩
    q_vec = sbert.encode(
        [query_sentence],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    # 3. 전체 임베딩과 cosine similarity
    sims = q_embs @ q_vec

    # 4. 카테고리 필터
    if selected_category is not None:
        mask = (df["category"].astype(str) == selected_category).to_numpy()
        sims = np.where(mask, sims, -1.0)

    # 5. 상위 pool 뽑기
    pool_idx = np.argsort(-sims)[:topk_pool]

    # 6. 중복 제거 → 최종 top-k
    selected = []
    for idx in pool_idx:
        if len(selected) >= final_topk:
            break

        if not selected:
            selected.append(idx)
            continue

        too_similar = False
        for s in selected:
            sim_ss = float(np.dot(q_embs[idx], q_embs[s]))
            if sim_ss >= dup_threshold:
                too_similar = True
                break

        if too_similar:
            continue
        selected.append(idx)

    # 7. 그래프 RAG: SIMILAR 이웃 확장
    result = []
    for idx in selected:
        sim_list = get_similar_from_graph(int(idx), limit=5)

        result.append({
            "idx": int(idx),
            "question": df.loc[idx, "question"],
            "summary": df.loc[idx, "summary"],
            "category": df.loc[idx, "category"],
            "similar_neighbors": sim_list,
            "score": float(sims[idx]),
        })

    return result

if __name__ == "__main__":
    # 예시 테스트
    hits = search_questions_graph_rag(
        raw_keywords=["프로젝트","경험"],
        selected_category="ICT",   # 없으면 None
        topk_pool=20,
        final_topk=5,
    )

    for h in hits:
        print("\n==============================")
        print("메인 질문:", h["question"])
        print("카테고리:", h["category"])
        print("유사도:", f"{h['score']:.4f}")
        print("요약:", h["summary"][:80], "...")

        print("\n  >> Neo4j SIMILAR 이웃들")
        for nei in h["similar_neighbors"]:
            print("   -", nei["text"])
