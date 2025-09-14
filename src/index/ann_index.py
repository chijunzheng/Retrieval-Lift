import numpy as np


class ANNIndex:
    def __init__(
        self,
        dim: int,
        kind: str = "auto",
        M: int = 32,
        ef_construction: int = 200,
        ef_search: int = 96,
    ):
        self.dim = dim
        self.kind = kind
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self._impl = None
        self._build_impl()

    def _build_impl(self):
        """
        Build the implementation of the ANNIndex.
        """
        use_faiss = False
        if self.kind in ("faiss", "auto"):
            try:
                import faiss

                self.faiss = faiss
                index = faiss.IndexHNSWFlat(self.dim, self.M)
                index.hnsw.efConstruction = self.ef_construction
                index.hnsw.efSearch = self.ef_search
                self._impl = ("faiss", index)
                use_faiss = True
            except Exception:
                use_faiss = False

        if not use_faiss:
            import hnswlib

            index = hnswlib.Index(space="cosine", dim=self.dim)
            index.init_index(
                max_elements=1, ef_construction=self.ef_construction, M=self.M
            )
            index.set_ef(self.ef_search)
            self._impl = ("hnsw", index)

    def add(self, embeddings: np.ndarray, ids: list):
        """
        Add embeddings to the ANNIndex.
        """
        kind, index = self._impl
        if kind == "faiss":
            # faiss doesn't track ids; maintain our own mapping
            if not hasattr(self, "_ids"):
                self._ids = []
            self._ids.extend(ids)
            index.add(embeddings.astype(np.float32))
        else:
            # hnswlib requires init with capacity; re-init accordingly
            import hnswlib

            if isinstance(index, hnswlib.Index) and index.get_max_elements() < len(ids):
                # Re-init with proper capacity
                index.init_index(
                    max_elements=len(ids),
                    ef_construction=self.ef_construction,
                    M=self.M,
                )
                index.set_ef(self.ef_search)
            index.add_items(embeddings.astype(np.float32), ids)

    def search(self, query_embs: np.ndarray, top_k: int):
        """
        Search the ANNIndex.
        """
        kind, index = self._impl
        if kind == "faiss":
            D, I = index.search(query_embs.astype(np.float32), top_k)
            # map to original ids
            return [
                [
                    (int(self._ids[j]), float(1 - D[i, k]))
                    for k, j in enumerate(I[i])
                    if j != -1
                ]
                for i in range(query_embs.shape[0])
            ]
        else:
            labels, distances = index.knn_query(query_embs.astype(np.float32), k=top_k)
            return [
                [
                    (int(labels[i][k]), float(1 - distances[i][k]))
                    for k in range(len(labels[i]))
                ]
                for i in range(query_embs.shape[0])
            ]


