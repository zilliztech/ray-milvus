"""
Milvus datasink implementation for Ray Data.
"""

from typing import Any, Dict, Iterable, Optional

import pyarrow as pa
from milvus_storage import Writer
from ray.data._internal.execution.interfaces import TaskContext
from ray.data.datasource import Datasink


class MilvusDatasink(Datasink):
    """
    Ray Data datasink for writing to Milvus Storage.

    This datasink allows Ray Data to write datasets to Milvus Storage format,
    which uses Apache Arrow and Parquet as the underlying storage format.

    Example:
        >>> import ray
        >>> import pyarrow as pa
        >>> from ray_milvus import MilvusDatasink
        >>>
        >>> # Create a Ray dataset
        >>> ds = ray.data.range(100)
        >>>
        >>> # Define schema
        >>> schema = pa.schema([pa.field("id", pa.int64())])
        >>>
        >>> # Write to Milvus storage
        >>> ds.write_datasink(
        ...     MilvusDatasink(
        ...         path="/tmp/my_dataset",
        ...         schema=schema,
        ...         properties={"fs.storage_type": "local", "fs.root_path": "/tmp/"}
        ...     )
        ... )
    """

    def __init__(
        self,
        path: str,
        schema: pa.Schema,
        properties: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize the Milvus datasink.

        Args:
            path: Base path where data will be written
            schema: PyArrow schema for the dataset
            properties: Optional configuration properties for Milvus storage
        """
        super().__init__()
        self._path = path
        self._schema = schema
        self._properties = properties or {}

    def get_name(self) -> str:
        """Return a human-readable name for this datasink."""
        return "Milvus"

    def on_write_start(self) -> None:
        """Callback invoked when a write job begins."""
        # Writer will be initialized in write() to avoid serialization issues
        pass

    def write(
        self,
        blocks: Iterable[pa.Table],
        ctx: TaskContext,
    ) -> Dict[str, Any]:
        """
         Write blocks to Milvus Storage.

         Args:
             blocks: Iterable of data blocks (Arrow tables) to write
             ctx: Ray task context
        Returns:
             Dictionary with metadata about the write operation
        """
        # Initialize writer here to avoid serialization issues with CFFI objects
        # Each Ray worker will create its own writer instance
        writer = Writer(path=self._path, schema=self._schema, properties=self._properties)

        num_rows = 0
        num_bytes = 0

        try:
            for block in blocks:
                # blocks should be PyArrow Tables
                if isinstance(block, pa.Table):
                    # Write each batch from the table
                    for batch in block.to_batches():
                        writer.write(batch)
                        num_rows += batch.num_rows
                        num_bytes += batch.nbytes
                else:
                    raise TypeError(f"Unsupported block type: {type(block)}")
        finally:
            writer.flush()
            # Close the writer to ensure resources are cleaned up
            writer.close()

        return {
            "num_rows": num_rows,
            "num_bytes": num_bytes,
        }

    def on_write_complete(self, write_results) -> None:
        """
        Callback triggered upon successful write job completion.

        Args:
            write_results: WriteResult object containing aggregated write statistics
        """
        total_rows = write_results.num_rows
        total_bytes = write_results.size_bytes

        print(f"Write completed: {total_rows} rows, {total_bytes} bytes written")

    def on_write_failed(self, error: Exception) -> None:
        """
        Callback executed if a write job encounters failure.

        Args:
            error: The exception that caused the failure
        """
        # Writer is closed in the write() method's finally block
        print(f"Write failed: {error}")


def write_milvus(
    dataset,
    path: str,
    schema: pa.Schema,
    properties: Optional[Dict[str, str]] = None,
    **write_args,
) -> None:
    """
    Convenience function to write a Ray Dataset to Milvus Storage.

    Args:
        dataset: Ray Dataset to write
        path: Base path where data will be written
        schema: PyArrow schema for the dataset
        properties: Optional configuration properties for Milvus storage
        **write_args: Additional write arguments

    Example:
        >>> import ray
        >>> import pyarrow as pa
        >>> from ray_milvus import write_milvus
        >>>
        >>> # Create dataset
        >>> ds = ray.data.range(1000)
        >>>
        >>> # Define schema
        >>> schema = pa.schema([pa.field("id", pa.int64())])
        >>>
        >>> # Write to Milvus storage
        >>> write_milvus(
        ...     ds,
        ...     path="/tmp/my_dataset",
        ...     schema=schema,
        ...     properties={"fs.storage_type": "local", "fs.root_path": "/tmp/"}
        ... )
    """
    datasink = MilvusDatasink(
        path=path,
        schema=schema,
        properties=properties,
    )

    return dataset.write_datasink(
        datasink,
        **write_args,
    )
