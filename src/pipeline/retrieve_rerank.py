import os
import json
import numpy as np

from src.models.embedding import BGEEmbedder
from src.models.reranker import BGEReranker
from src.index.ann_index import ANNIndex


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_corpus(corpus_path):
    """
    Build the corpus.
    """
    docs = load_jsonl(corpus_path)
    id2doc = {}
    texts = []
    ids = []
    for i, d in enumerate(docs):
        ids.append(i)
        id2doc[i] = d["doc_id"]
        texts.append(d["text"])
    return ids, texts, id2doc


def build_qrels(qrels_path, split="test"):
    """
    Build the qrels.
    """
    rows = load_jsonl(qrels_path)
    qrels = {}
    for r in rows:
        if r.get("split", "test") != split:
            continue
        qid = r["query_id"]
        did = r["doc_id"]
        score = int(r["score"])
        qrels.setdefault(qid, {})[did] = score
    return qrels


def build_queries(queries_path):
    """
    Build the queries.
    """
    rows = load_jsonl(queries_path)
    return {r["query_id"]: r["text"] for r in rows}


def run_pipeline(cfg, dataset_name, base_dir="data/beir"):
    """
    Run the pipeline.
    """
    dpath = os.path.join(base_dir, dataset_name)
    corpus_path = os.path.join(dpath, "corpus.jsonl")
    queries_path = os.path.join(dpath, "queries.jsonl")
    qrels_path = os.path.join(dpath, "qrels.jsonl")

    ids, passages, int2docid = build_corpus(corpus_path)
    qrels = build_qrels(qrels_path, split="test")
    queries = build_queries(queries_path)

    # build embeddings
    emb = BGEEmbedder(
        model_name=cfg["embedding_model"],
        device=cfg["runtime"]["device"],
        max_length=cfg["embedding_max_length"],
        normalize=cfg["normalize_embeddings"],
        query_instruction=cfg.get("bge_query_instruction", ""),
    )
    passage_embs = emb.encode_passages(passages, batch_size=cfg["embedding_batch_size"])

    # build index
    index = ANNIndex(
        dim=passage_embs.shape[1],
        kind=cfg["index"]["kind"],
        M=cfg["index"]["M"],
        ef_construction=cfg["index"]["ef_construction"],
        ef_search=cfg["index"]["ef_search"],
    )
    index.add(passage_embs, list(range(len(passages))))

    # build reranker
    reranker = (
        BGEReranker(
            cfg["rerank"]["model"],
            device=cfg["runtime"]["device"],
            max_length=cfg["embedding_max_length"],
        )
        if cfg["rerank"]["enabled"]
        else None
    )

    # build query embeddings
    test_query_ids = [qid for qid in qrels.keys() if qid in queries]
    query_texts = [queries[qid] for qid in test_query_ids]
    query_embs = emb.encode_queries(query_texts, batch_size=cfg["embedding_batch_size"])

    # retrieve
    top_k = cfg["retrieve"]["top_k"]
    retrieved = index.search(query_embs, top_k=top_k)

    # rerank
    results = {}
    for qi, qid in enumerate(test_query_ids):
        cand = retrieved[qi]
        rows = [(int2docid[ii], passages[ii], sim) for ii, sim in cand]
        if reranker:
            Kp = min(cfg["rerank"]["top_k"], len(rows))
            subset = rows[:Kp]
            scores = reranker.score(
                queries[qid], [t for _, t, _ in subset], batch_size=cfg["rerank"]["batch_size"]
            )
            order = np.argsort(-scores)
            rows = [subset[i] for i in order] + rows[Kp:]
        results[qid] = [doc_id for doc_id, _, _ in rows]

    return results, qrels


