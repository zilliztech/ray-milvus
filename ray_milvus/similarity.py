"""
Vector similarity calculation for Ray Data map_batches operations.

This module provides a callable class for calculating vector similarity
that can be passed as an argument to Ray Data's map_batches method.
"""

from typing import List, Union

import numpy as np
import pyarrow as pa


class VectorSimilarityCalculator:
    """
    Callable class to calculate vector similarity for use with Ray Data map_batches.

    This class computes similarity scores between a query vector and vectors in batches,
    supporting different similarity metrics (COSINE, L2, IP) and filtering by threshold.

    Args:
        query_vector: The query vector to compare against (numpy array or list)
        k: Number of top similar results to return (default: 100)
        metric: Similarity metric to use - "COSINE", "L2", or "IP" (inner product) (default: "COSINE")
        threshold: Minimum similarity score threshold for filtering (default: 0.8)
        vector_col: Name of the column containing vectors (default: "vector")
        output_score_col: Name of the output column for similarity scores (default: "similarity")

    Example:
        >>> import ray
        >>> import numpy as np
        >>> from ray_milvus.similarity import VectorSimilarityCalculator
        >>>
        >>> # Create a dataset with vectors
        >>> data = {"id": [1, 2, 3], "vector": [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]}
        >>> ds = ray.data.from_items([data])
        >>>
        >>> # Calculate similarity
        >>> query_vec = np.array([1, 0])
        >>> calculator = VectorSimilarityCalculator(
        ...     query_vector=query_vec,
        ...     k=10,
        ...     metric="COSINE",
        ...     threshold=0.5,
        ...     vector_col="vector",
        ...     output_score_col="similarity"
        ... )
        >>>
        >>> # Use with map_batches
        >>> result_ds = ds.map_batches(calculator, batch_format="pyarrow")
    """

    def __init__(
        self,
        query_vector: Union[np.ndarray, List[float]],
        k: int = 100,
        metric: str = "COSINE",
        threshold: float = 0.8,
        vector_col: str = "vector",
        output_score_col: str = "similarity",
    ):
        """Initialize the vector similarity calculator."""
        # Convert query vector to numpy array if it's a list
        if isinstance(query_vector, list):
            self.query_vector = np.array(query_vector, dtype=np.float32)
        else:
            self.query_vector = query_vector.astype(np.float32)

        # Normalize query vector if using cosine similarity
        if metric.upper() == "COSINE":
            query_norm = np.linalg.norm(self.query_vector)
            if query_norm > 0:
                self.query_vector = self.query_vector / query_norm

        self.k = k
        self.metric = metric.upper()
        self.threshold = threshold
        self.vector_col = vector_col
        self.output_score_col = output_score_col

        # Validate metric
        if self.metric not in ["COSINE", "L2", "IP"]:
            raise ValueError(f"Unsupported metric: {metric}. Must be one of: COSINE, L2, IP")

    def _calculate_similarity(self, vectors: np.ndarray) -> np.ndarray:
        """
        Calculate similarity scores between query vector and batch vectors.

        Args:
            vectors: Batch of vectors as numpy array (shape: [batch_size, vector_dim])

        Returns:
            Array of similarity scores
        """
        if self.metric == "COSINE":
            # Normalize batch vectors
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            normalized_vectors = vectors / norms

            # Compute cosine similarity (dot product of normalized vectors)
            scores = np.dot(normalized_vectors, self.query_vector)

        elif self.metric == "IP":
            # Inner product (dot product)
            scores = np.dot(vectors, self.query_vector)

        elif self.metric == "L2":
            # L2 distance (Euclidean distance) - convert to similarity (inverse)
            distances = np.linalg.norm(vectors - self.query_vector, axis=1)
            # Convert distance to similarity score (smaller distance = higher similarity)
            scores = 1.0 / (1.0 + distances)

        return scores

    def __call__(self, batch: pa.Table) -> pa.Table:
        """
        Process a batch of data and calculate similarity scores.

        Args:
            batch: PyArrow Table containing vector data

        Returns:
            PyArrow Table with similarity scores and filtered by threshold and top-k
        """
        # Convert to pandas for easier manipulation
        df = batch.to_pandas()

        if self.vector_col not in df.columns:
            raise ValueError(f"Vector column '{self.vector_col}' not found in batch")

        # Convert vectors to numpy array
        vectors = np.stack(df[self.vector_col].values)

        # Calculate similarity scores
        scores = self._calculate_similarity(vectors)

        # Add scores to dataframe
        df[self.output_score_col] = scores

        # Filter by threshold
        df = df[df[self.output_score_col] >= self.threshold]

        # Sort by similarity score (descending) and take top-k
        df = df.nlargest(min(self.k, len(df)), self.output_score_col)

        # Convert back to PyArrow Table
        return pa.Table.from_pandas(df)
