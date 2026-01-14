"""
Milvus datasource implementation for Ray Data.
"""

from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import pyarrow as pa
from milvus_storage import Reader
from ray.data.block import Block, BlockMetadata
from ray.data.datasource import Datasource, ReadTask


class MilvusDatasource(Datasource):
    """
    Ray Data datasource for reading from Milvus Storage.

    This datasource allows Ray Data to read data stored in Milvus Storage format,
    which uses Apache Arrow and Parquet as the underlying storage format.

    Example:
        >>> import ray
        >>> from ray_milvus import MilvusDatasource
        >>>
        >>> # Read column groups
        >>> column_groups = ['{"segments": [{"path": "/tmp/cg1"}]}']
        >>>
        >>> # Create Ray dataset from Milvus storage
        >>> ds = ray.data.read_datasource(
        ...     MilvusDatasource(
        ...         column_groups=column_groups,
        ...         schema=schema,
        ...         properties={"fs.storage_type": "local", "fs.root_path": "/tmp/"}
        ...     )
        ... )
    """

    def __init__(
        self,
        column_groups: List[str],
        schema: pa.Schema,
        columns: Optional[List[str]] = None,
        predicate: Optional[str] = None,
        properties: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Milvus datasource.

        Args:
            column_groups: List of JSON strings, each containing a column group manifest
            schema: PyArrow schema for the dataset
            columns: Optional list of column names to read
            predicate: Optional filter expression
            properties: Optional configuration properties for Milvus storage
        """
        super().__init__()
        self._column_groups = column_groups
        self._schema = schema
        self._columns = columns
        self._predicate = predicate
        self._properties = properties

    def estimate_inmemory_data_size(self) -> Optional[int]:
        """
        Estimate the in-memory data size in bytes.

        Returns:
            None since we cannot estimate the size without reading the data.
            Ray will handle size estimation during execution.
        """
        return None

    def get_read_tasks(
        self,
        parallelism: int,
        per_task_row_limit: Optional[int] = None,
    ) -> List[ReadTask]:
        """
        Get read tasks for parallel reading from Milvus Storage.

        Args:
            parallelism: Number of parallel read tasks to create
            per_task_row_limit: Optional row limit per task (not used currently)

        Returns:
            List of ReadTask objects for parallel execution
        """

        def _read_fn(
            column_groups: List[str],
            schema: pa.Schema,
            columns: Optional[List[str]],
            predicate: Optional[str],
            properties: Optional[Dict[str, str]],
        ) -> Iterator[Block]:
            """Read function executed by each task."""
            # Each task reads multiple column groups
            for column_group in column_groups:
                reader = Reader(
                    column_groups=column_group,
                    schema=schema,
                    columns=columns,
                    properties=properties,
                )

                try:
                    # Scan and yield batches
                    for batch in reader.scan(predicate=predicate):
                        table = pa.Table.from_batches([batch])
                        yield table
                finally:
                    reader.close()

        # Create read tasks
        read_tasks = []

        if not self._column_groups:
            return read_tasks

        num_column_groups = len(self._column_groups)

        # Use the minimum of parallelism and number of column groups
        num_tasks = min(parallelism, num_column_groups) if parallelism > 0 else num_column_groups

        # Split column groups across tasks using numpy
        column_group_splits = np.array_split(self._column_groups, num_tasks)

        for task_column_groups in column_group_splits:
            # Convert numpy array back to list
            task_column_groups_list = task_column_groups.tolist()

            metadata = BlockMetadata(
                num_rows=None,  # Unknown until read
                size_bytes=None,  # Unknown until read
                input_files=None,
                exec_stats=None,
            )

            read_task = ReadTask(
                read_fn=lambda cgs=task_column_groups_list: _read_fn(
                    cgs, self._schema, self._columns, self._predicate, self._properties
                ),
                metadata=metadata,
            )
            read_tasks.append(read_task)

        return read_tasks


def read_milvus(
    column_groups: Union[str, List[str]],
    schema: pa.Schema,
    columns: Optional[List[str]] = None,
    predicate: Optional[str] = None,
    properties: Optional[Dict[str, str]] = None,
    parallelism: int = -1,
    **read_args,
):
    """
    Convenience function to read Milvus Storage into a Ray Dataset.

    Args:
        column_groups: List of JSON strings, each containing a column group manifest,
                      or a single JSON string for backward compatibility
        schema: PyArrow schema for the dataset
        columns: Optional list of column names to read
        predicate: Optional filter expression
        properties: Optional configuration properties for Milvus storage
        parallelism: Number of parallel read tasks (-1 for default)
        **read_args: Additional read arguments

    Returns:
        Ray Dataset containing the data from Milvus Storage

    Example:
        >>> import ray
        >>> import pyarrow as pa
        >>> from ray_milvus import read_milvus
        >>>
        >>> schema = pa.schema([
        ...     pa.field("id", pa.int64()),
        ...     pa.field("vector", pa.list_(pa.float32())),
        ...     pa.field("text", pa.string())
        ... ])
        >>>
        >>> # Read from multiple column groups
        >>> ds = read_milvus(
        ...     column_groups=[
        ...         '{"segments": [{"path": "/tmp/cg1"}]}',
        ...         '{"segments": [{"path": "/tmp/cg2"}]}'
        ...     ],
        ...     schema=schema,
        ...     properties={"fs.storage_type": "local", "fs.root_path": "/tmp/"}
        ... )
        >>>
        >>> # Backward compatible: single column group as string
        >>> with open("/tmp/manifest.json", "r") as f:
        ...     column_group = f.read()
        >>>
        >>> ds = read_milvus(
        ...     column_groups=column_group,
        ...     schema=schema,
        ...     properties={"fs.storage_type": "local", "fs.root_path": "/tmp/"}
        ... )
    """
    import ray

    # Convert single string to list for backward compatibility
    if isinstance(column_groups, str):
        column_groups = [column_groups]

    datasource = MilvusDatasource(
        column_groups=column_groups,
        schema=schema,
        columns=columns,
        predicate=predicate,
        properties=properties,
    )

    return ray.data.read_datasource(
        datasource,
        parallelism=parallelism,
        **read_args,
    )
