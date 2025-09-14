import argparse
import yaml
from tabulate import tabulate
from statistics import mean

from src.pipeline.retrieve_rerank import run_pipeline
from src.eval.metrics import ndcg_at_k, recall_at_k, mrr_at_k


def evaluate_dataset(cfg, name):
    results, qrels = run_pipeline(cfg, name)
    ks = cfg["metrics"]["ks"]
    m_ndcg = {k: [] for k in ks}
    m_recall = {k: [] for k in ks}
    m_mrr10 = []

    for qid, ranked in results.items():
        rels = qrels.get(qid, {})
        m_mrr10.append(mrr_at_k(ranked, rels, k=10))
        for k in ks:
            m_ndcg[k].append(ndcg_at_k(ranked, rels, k=k))
            m_recall[k].append(recall_at_k(ranked, rels, k=k))

    out = {
        "dataset": name,
        "mrr@10": mean(m_mrr10) if m_mrr10 else 0.0,
        **{f"ndcg@{k}": mean(m_ndcg[k]) if m_ndcg[k] else 0.0 for k in ks},
        **{f"recall@{k}": mean(m_recall[k]) if m_recall[k] else 0.0 for k in ks},
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    table = []
    headers = [
        "dataset",
        "mrr@10",
        *[f"ndcg@{k}" for k in cfg["metrics"]["ks"]],
        *[f"recall@{k}" for k in cfg["metrics"]["ks"]],
    ]
    for d in cfg["datasets"]:
        name = d["name"]
        out = evaluate_dataset(cfg, name)
        table.append(
            [
                out["dataset"],
                f"{out['mrr@10']:.4f}",
                *[f"{out[f'ndcg@{k}']:.4f}" for k in cfg["metrics"]["ks"]],
                *[f"{out[f'recall@{k}']:.4f}" for k in cfg["metrics"]["ks"]],
            ]
        )

    print(tabulate(table, headers=headers, tablefmt="github"))


