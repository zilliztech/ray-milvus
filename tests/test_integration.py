"""
Integration tests for Milvus Ray datasource and datasink.
"""

import os
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet
import pytest


def test_import():
    """Test that the package can be imported."""
    from ray_milvus import MilvusDatasink, MilvusDatasource, read_milvus, write_milvus

    assert MilvusDatasource is not None
    assert MilvusDatasink is not None
    assert read_milvus is not None
    assert write_milvus is not None


def test_version():
    """Test that package version is defined."""
    import ray_milvus

    assert hasattr(ray_milvus, "__version__")
    assert ray_milvus.__version__ == "0.1.0"


class TestMilvusDatasource:
    """Tests for MilvusDatasource class."""

    def test_init(self):
        """Test MilvusDatasource initialization."""
        from ray_milvus import MilvusDatasource

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32())),
            ]
        )
        column_groups = ['{"segments": [{"path": "/tmp/test"}]}']

        datasource = MilvusDatasource(
            column_groups=column_groups,
            schema=schema,
            columns=["id"],
            predicate="id > 0",
            properties={"fs.storage_type": "local"},
        )

        assert datasource._column_groups == column_groups
        assert datasource._schema == schema
        assert datasource._columns == ["id"]
        assert datasource._predicate == "id > 0"
        assert datasource._properties == {"fs.storage_type": "local"}

    def test_estimate_inmemory_data_size(self):
        """Test that estimate_inmemory_data_size returns None."""
        from ray_milvus import MilvusDatasource

        schema = pa.schema([pa.field("id", pa.int64())])
        datasource = MilvusDatasource(column_groups=[], schema=schema)

        assert datasource.estimate_inmemory_data_size() is None

    def test_get_read_tasks_empty(self):
        """Test get_read_tasks with empty column groups."""
        from ray_milvus import MilvusDatasource

        schema = pa.schema([pa.field("id", pa.int64())])
        datasource = MilvusDatasource(column_groups=[], schema=schema)

        tasks = datasource.get_read_tasks(parallelism=4)
        assert tasks == []

    def test_get_read_tasks_parallelism(self):
        """Test get_read_tasks respects parallelism."""
        from ray_milvus import MilvusDatasource

        schema = pa.schema([pa.field("id", pa.int64())])
        column_groups = [f'{{"segments": [{{"path": "/tmp/cg{i}"}}]}}' for i in range(10)]

        datasource = MilvusDatasource(column_groups=column_groups, schema=schema)

        # Test with parallelism less than column groups
        tasks = datasource.get_read_tasks(parallelism=3)
        assert len(tasks) == 3

        # Test with parallelism greater than column groups
        tasks = datasource.get_read_tasks(parallelism=20)
        assert len(tasks) == 10


class TestMilvusDatasink:
    """Tests for MilvusDatasink class."""

    def test_init(self):
        """Test MilvusDatasink initialization."""
        from ray_milvus import MilvusDatasink

        schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("vector", pa.list_(pa.float32())),
            ]
        )

        datasink = MilvusDatasink(
            path="/tmp/test_output",
            schema=schema,
            properties={"fs.storage_type": "local"},
        )

        assert datasink._path == "/tmp/test_output"
        assert datasink._schema == schema
        assert datasink._properties == {"fs.storage_type": "local"}

    def test_init_default_properties(self):
        """Test MilvusDatasink initialization with default properties."""
        from ray_milvus import MilvusDatasink

        schema = pa.schema([pa.field("id", pa.int64())])
        datasink = MilvusDatasink(path="/tmp/test", schema=schema)

        assert datasink._properties == {}

    def test_get_name(self):
        """Test MilvusDatasink get_name."""
        from ray_milvus import MilvusDatasink

        schema = pa.schema([pa.field("id", pa.int64())])
        datasink = MilvusDatasink(path="/tmp/test", schema=schema)

        assert datasink.get_name() == "Milvus"


class TestReadMilvusFunction:
    """Tests for read_milvus convenience function."""

    def test_string_to_list_conversion(self):
        """Test that single string column_groups is converted to list."""
        from ray_milvus.datasource import MilvusDatasource

        schema = pa.schema([pa.field("id", pa.int64())])
        single_cg = '{"segments": [{"path": "/tmp/test"}]}'

        # Create datasource manually to verify the behavior
        datasource = MilvusDatasource(
            column_groups=[single_cg],  # Should be list
            schema=schema,
        )

        assert datasource._column_groups == [single_cg]


class TestVectorSimilarityCalculator:
    """Tests for VectorSimilarityCalculator class."""

    def test_init_with_list(self):
        """Test initialization with list query vector."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[1.0, 0.0, 0.0],
            k=10,
            metric="COSINE",
        )

        assert isinstance(calc.query_vector, np.ndarray)
        assert calc.k == 10
        assert calc.metric == "COSINE"

    def test_init_with_numpy(self):
        """Test initialization with numpy query vector."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        query = np.array([1.0, 0.0, 0.0])
        calc = VectorSimilarityCalculator(query_vector=query)

        assert isinstance(calc.query_vector, np.ndarray)

    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        with pytest.raises(ValueError, match="Unsupported metric"):
            VectorSimilarityCalculator(
                query_vector=[1.0, 0.0],
                metric="INVALID",
            )

    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[1.0, 0.0],
            metric="COSINE",
            threshold=0.0,
        )

        # Create test batch
        data = {
            "id": [1, 2, 3],
            "vector": [
                np.array([1.0, 0.0]),  # Same direction, similarity = 1.0
                np.array([0.0, 1.0]),  # Perpendicular, similarity = 0.0
                np.array([1.0, 1.0]),  # 45 degrees, similarity ~ 0.707
            ],
        }
        batch = pa.Table.from_pydict(data)

        result = calc(batch)
        result_df = result.to_pandas()

        # All should pass threshold of 0.0
        assert len(result_df) == 3

        # Check similarity scores
        scores = result_df.set_index("id")["similarity"]
        assert abs(scores[1] - 1.0) < 0.01
        assert abs(scores[2] - 0.0) < 0.01
        assert abs(scores[3] - 0.707) < 0.01

    def test_l2_similarity(self):
        """Test L2 distance similarity calculation."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[0.0, 0.0],
            metric="L2",
            threshold=0.0,
        )

        data = {
            "id": [1, 2],
            "vector": [
                np.array([0.0, 0.0]),  # Distance = 0, similarity = 1.0
                np.array([1.0, 0.0]),  # Distance = 1, similarity = 0.5
            ],
        }
        batch = pa.Table.from_pydict(data)

        result = calc(batch)
        result_df = result.to_pandas()

        scores = result_df.set_index("id")["similarity"]
        assert abs(scores[1] - 1.0) < 0.01
        assert abs(scores[2] - 0.5) < 0.01

    def test_ip_similarity(self):
        """Test inner product similarity calculation."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[1.0, 2.0],
            metric="IP",
            threshold=-100.0,
        )

        data = {
            "id": [1, 2],
            "vector": [
                np.array([1.0, 1.0]),  # IP = 1*1 + 2*1 = 3
                np.array([2.0, 0.0]),  # IP = 1*2 + 2*0 = 2
            ],
        }
        batch = pa.Table.from_pydict(data)

        result = calc(batch)
        result_df = result.to_pandas()

        scores = result_df.set_index("id")["similarity"]
        assert abs(scores[1] - 3.0) < 0.01
        assert abs(scores[2] - 2.0) < 0.01

    def test_threshold_filtering(self):
        """Test that results are filtered by threshold."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[1.0, 0.0],
            metric="COSINE",
            threshold=0.5,
        )

        data = {
            "id": [1, 2, 3],
            "vector": [
                np.array([1.0, 0.0]),  # similarity = 1.0, passes
                np.array([0.0, 1.0]),  # similarity = 0.0, filtered
                np.array([1.0, 1.0]),  # similarity ~ 0.707, passes
            ],
        }
        batch = pa.Table.from_pydict(data)

        result = calc(batch)
        result_df = result.to_pandas()

        # Only 2 results should pass threshold
        assert len(result_df) == 2
        assert 2 not in result_df["id"].values

    def test_top_k(self):
        """Test that results are limited to top-k."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[1.0, 0.0],
            metric="COSINE",
            threshold=0.0,
            k=2,
        )

        data = {
            "id": [1, 2, 3, 4],
            "vector": [
                np.array([1.0, 0.0]),
                np.array([0.9, 0.1]),
                np.array([0.5, 0.5]),
                np.array([0.0, 1.0]),
            ],
        }
        batch = pa.Table.from_pydict(data)

        result = calc(batch)
        result_df = result.to_pandas()

        # Only top 2 should be returned
        assert len(result_df) == 2

    def test_missing_vector_column(self):
        """Test that missing vector column raises ValueError."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[1.0, 0.0],
            vector_col="nonexistent",
        )

        data = {"id": [1], "vector": [np.array([1.0, 0.0])]}
        batch = pa.Table.from_pydict(data)

        with pytest.raises(ValueError, match="not found in batch"):
            calc(batch)

    def test_custom_column_names(self):
        """Test custom vector and output column names."""
        from ray_milvus.similarity import VectorSimilarityCalculator

        calc = VectorSimilarityCalculator(
            query_vector=[1.0, 0.0],
            vector_col="embedding",
            output_score_col="score",
            threshold=0.0,
        )

        data = {
            "id": [1],
            "embedding": [np.array([1.0, 0.0])],
        }
        batch = pa.Table.from_pydict(data)

        result = calc(batch)
        result_df = result.to_pandas()

        assert "score" in result_df.columns
        assert "similarity" not in result_df.columns


class TestWriteReadCycle:
    """End-to-end tests for write and read operations."""

    def test_write_to_milvus_storage(self):
        """Test writing data to Milvus storage."""
        import ray

        from ray_milvus import MilvusDatasink

        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Define schema with fixed_size_list for vectors
            schema = pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), 4)),
                ]
            )

            # Create PyArrow table with correct types
            table = pa.table(
                {
                    "id": pa.array([1, 2, 3], type=pa.int64()),
                    "text": pa.array(["hello", "world", "test"], type=pa.string()),
                    "vector": pa.array(
                        [
                            [0.1, 0.2, 0.3, 0.4],
                            [0.5, 0.6, 0.7, 0.8],
                            [0.9, 1.0, 1.1, 1.2],
                        ],
                        type=pa.list_(pa.float32(), 4),
                    ),
                }
            )

            # Create Ray dataset from Arrow table
            ds = ray.data.from_arrow(table)

            # Write to Milvus storage
            # Note: path is relative to fs.root_path
            output_path = "test_data"
            properties = {
                "fs.storage_type": "local",
                "fs.root_path": tmpdir,
            }

            datasink = MilvusDatasink(
                path=output_path,
                schema=schema,
                properties=properties,
            )

            ds.write_datasink(datasink)

            # Check that data directory was created
            # The actual path is fs.root_path + path + _data
            data_dir = os.path.join(tmpdir, output_path, "_data")
            assert os.path.exists(data_dir), f"Data directory not created at {data_dir}"

            # Check that parquet files were created
            parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
            assert len(parquet_files) > 0, "No parquet files created"

            # Read the parquet file to verify content
            parquet_path = os.path.join(data_dir, parquet_files[0])
            result_table = pa.parquet.read_table(parquet_path)

            # Verify row count
            assert result_table.num_rows == 3

            # Verify column names
            assert set(result_table.column_names) == {"id", "text", "vector"}

            # Verify data
            ids = sorted(result_table.column("id").to_pylist())
            assert ids == [1, 2, 3]

            texts = sorted(result_table.column("text").to_pylist())
            assert texts == ["hello", "test", "world"]


if __name__ == "__main__":
    test_import()
    print("Import test passed!")
