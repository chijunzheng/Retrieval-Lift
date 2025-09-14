import math

# Discounted Cumulative Gain
def dcg(rels):
    return sum(((2**r - 1) / math.log2(i + 2)) for i, r in enumerate(rels))

# Normalized Discounted Cumulative Gain
def ndcg_at_k(sorted_docids, qrels, k=10):
    gains = [1 if doc in qrels and qrels[doc] > 0 else 0 for doc in sorted_docids[:k]]
    ideal_gains = sorted([1] * sum(1 for v in qrels.values() if v > 0), reverse=True)[:k]
    denom = dcg(ideal_gains)
    return (dcg(gains) / denom) if denom > 0 else 0.0

# Recall at k
def recall_at_k(sorted_docids, qrels, k=10):
    gold = {d for d, r in qrels.items() if r > 0}
    if not gold:
        return 0.0
    hit = len(gold.intersection(set(sorted_docids[:k])))
    return hit / len(gold)

# Mean Reciprocal Rank at k
def mrr_at_k(sorted_docids, qrels, k=10):
    gold = {d for d, r in qrels.items() if r > 0}
    for i, d in enumerate(sorted_docids[:k], start=1):
        if d in gold:
            return 1.0 / i
    return 0.0


