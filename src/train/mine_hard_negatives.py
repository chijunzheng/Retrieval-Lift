import os, json, argparse, numpy as np
from tqdm import tqdm
from src.models.embedding import BGEEmbedder
from src.index.ann_index import ANNIndex

def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def main(cfg_path):
    import yaml
    cfg = yaml.safe_load(open(cfg_path))

    base = cfg["paths"]["out_dir"]
    corpus = list(load_jsonl(os.path.join(base, "corpus.jsonl")))
    doc_texts = [d["text"] for d in corpus]
    doc_ids = [d["doc_id"] for d in corpus]
    id2doc = {i: doc_ids[i] for i in range(len(doc_ids))}
    docid2i = {doc_ids[i]: i for i in range(len(doc_ids))}

    # Embed corpus
    emb = BGEEmbedder(
        cfg["embedding_model"],
        device=cfg["train"]["device"],
        max_length=cfg["embedding_max_length"],
        normalize=cfg["normalize_embeddings"],
        query_instruction=cfg.get("bge_query_instruction",""),
    )
    X = emb.encode_passages(doc_texts, batch_size=64)

    # Build ANN
    ann = ANNIndex(
        dim=X.shape[1],
        kind=cfg["index"]["kind"],
        M=cfg["index"]["M"],
        ef_construction=cfg["index"]["ef_construction"],
        ef_search=cfg["index"]["ef_search"],
    )
    ann.add(X, list(range(len(doc_texts))))

    # Load train pairs (query + positives)
    train_pairs = list(load_jsonl(os.path.join(base, "train_pairs.jsonl")))

    out_triples = []
    for row in tqdm(train_pairs, desc="Mining HN"):
        q = row["query"]
        pos_ids = set(row["positives"])
        q_emb = emb.encode_queries([q], batch_size=1)
        retrieved = ann.search(q_emb, top_k=cfg["mining"]["retrieve_top_k"])[0]

        # keep non-positives as candidates, sorted by similarity
        candidates = []
        for int_id, sim in retrieved:
            did = id2doc[int_id]
            if did in pos_ids:
                continue
            candidates.append(did)
            if len(candidates) >= cfg["mining"]["hard_neg_per_query"]:
                break

        if not candidates:
            continue

        out_triples.append({
            "query": q,
            "positives": list(pos_ids)[:1],   # keep one pos (can expand)
            "negatives": candidates
        })

    os.makedirs(os.path.join(base, "mined"), exist_ok=True)
    out_path = os.path.join(base, "mined", "triples.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in out_triples:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote mined triples to {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_msmarco.yaml")
    args = ap.parse_args()
    main(args.config)