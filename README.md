# ray-milvus

Ray Data integration for Milvus Storage, providing efficient datasource and datasink implementations for reading from and writing to Milvus Storage format.

## Overview

`ray-milvus` enables seamless integration between [Ray Data](https://docs.ray.io/en/latest/data/data.html) and [Milvus Storage](https://github.com/milvus-io/milvus-storage), allowing you to:

- Read data from Milvus Storage into Ray Datasets for distributed processing
- Write Ray Datasets to Milvus Storage format (Apache Arrow/Parquet)
- Leverage Ray's parallel processing capabilities with Milvus Storage
- Build scalable data pipelines for vector data and machine learning workloads

## Installation

```bash
# Using pip
pip install ray-milvus

# Using uv
uv add ray-milvus

# For development
git clone https://github.com/your-repo/ray-milvus.git
cd ray-milvus
uv sync
```

## Requirements

- Python >= 3.10
- ray[data] >= 2.51.1
- pyarrow >= 21.0.0
- numpy >= 1.24.0
- milvus-storage

## Quick Start

### Writing Data to Milvus Storage

```python
import ray
import pyarrow as pa
from ray_milvus import write_milvus

# Initialize Ray
ray.init()

# Create a Ray dataset
ds = ray.data.range(1000)

# Define schema for Milvus storage
schema = pa.schema([
    pa.field("id", pa.int64(), nullable=True, metadata={"PARQUET:field_id": "1"})
])

# Write to Milvus storage
write_milvus(
    ds,
    path="/tmp/my_dataset",
    schema=schema,
    properties={"fs.storage_type": "local", "fs.root_path": "/tmp/"}
)
```

### Reading Data from Milvus Storage

```python
from ray_milvus import read_milvus

# Define column groups (JSON manifest)
column_groups = '''
{
  "column_groups": [
    {
      "columns": ["id", "vector", "text"],
      "format": "parquet",
      "paths": ["/tmp/my_dataset/column_group_0.parquet"]
    }
  ]
}'''

# Read from Milvus storage
ds = read_milvus(
    column_groups=column_groups,
    schema=schema,
    properties={"fs.storage_type": "local", "fs.root_path": "/tmp/"}
)

# Process with Ray Data
filtered = ds.filter(lambda row: row["id"] > 500)
print(f"Filtered dataset: {filtered.count()} rows")
```

## Features

### MilvusDatasource

Ray Data datasource for reading from Milvus Storage.

**Key Features:**
- Parallel reading with configurable parallelism
- Column projection support (read only specific columns)
- Predicate pushdown for efficient filtering
- Automatic batch processing

**Usage:**
```python
import ray
from ray_milvus import MilvusDatasource

datasource = MilvusDatasource(
    column_groups=[column_groups_json],
    schema=schema,
    columns=["id", "vector"],  # Optional: read specific columns
    predicate="id > 100",      # Optional: filter expression
    properties={"fs.storage_type": "local"}
)

ds = ray.data.read_datasource(datasource, parallelism=4)
```

### MilvusDatasink

Ray Data datasink for writing to Milvus Storage.

**Key Features:**
- Parallel writing with Ray workers
- Automatic schema conversion
- Progress tracking and statistics
- Resource cleanup and error handling

**Usage:**
```python
from ray_milvus import MilvusDatasink

datasink = MilvusDatasink(
    path="/tmp/output",
    schema=schema,
    properties={"fs.storage_type": "local"}
)

ds.write_datasink(datasink)
```

## Advanced Usage

### Multiple Column Groups

```python
# Read from multiple column groups in parallel
column_groups = [
    '{"segments": [{"path": "/tmp/cg1"}]}',
    '{"segments": [{"path": "/tmp/cg2"}]}',
    '{"segments": [{"path": "/tmp/cg3"}]}'
]

ds = read_milvus(
    column_groups=column_groups,
    schema=schema,
    properties=properties,
    parallelism=8  # Control parallel tasks
)
```

### Column Projection

```python
# Read only specific columns for better performance
ds = read_milvus(
    column_groups=column_groups,
    schema=schema,
    columns=["id", "vector"],  # Only read these columns
    properties=properties
)
```

### Data Processing Pipeline

```python
import ray
import numpy as np
from ray_milvus import read_milvus, write_milvus

# Read data
ds = read_milvus(column_groups, schema, properties)

# Transform with Ray
def normalize_vector(row):
    vector = np.array(row["vector"])
    norm = np.linalg.norm(vector)
    row["vector"] = (vector / norm).tolist() if norm > 0 else vector.tolist()
    return row

processed_ds = ds.map(normalize_vector)

# Filter
filtered_ds = processed_ds.filter(lambda row: row["id"] > 1000)

# Write back
write_milvus(filtered_ds, "/tmp/processed", schema, properties)
```

### Distributed Processing

```python
# Group and aggregate
def compute_stats(batch):
    import pyarrow as pa
    df = batch.to_pandas()
    vectors = np.array(df["vector"].tolist())
    return {
        "label": df["label"].iloc[0],
        "count": len(df),
        "mean_norm": float(np.linalg.norm(vectors, axis=1).mean())
    }

stats = ds.groupby("label").map_groups(
    compute_stats,
    batch_format="pyarrow"
)

for stat in stats.take_all():
    print(stat)
```

## Configuration

### Storage Properties

Common properties for Milvus Storage:

```python
properties = {
    "fs.storage_type": "local",           # or "s3", "azure", etc.
    "fs.root_path": "/tmp/",              # Base path for storage

    # S3 configuration (if using S3)
    # "fs.s3.endpoint": "s3.amazonaws.com",
    # "fs.s3.access_key": "your-access-key",
    # "fs.s3.secret_key": "your-secret-key",
    # "fs.s3.region": "us-west-2",
}
```

### Schema Definition

Define schemas with PARQUET field IDs for Milvus Storage:

```python
schema = pa.schema([
    pa.field("id", pa.int64(), nullable=True,
             metadata={"PARQUET:field_id": "1"}),
    pa.field("vector", pa.list_(pa.float32()), nullable=True,
             metadata={"PARQUET:field_id": "2"}),
    pa.field("text", pa.string(), nullable=True,
             metadata={"PARQUET:field_id": "3"}),
])
```

## Examples

The `examples/` directory contains a complete working example:

- `basic_example.py` - Basic read/write operations and data processing

Run the example:
```bash
python examples/basic_example.py
```

## API Reference

### read_milvus()

```python
def read_milvus(
    column_groups: Union[str, List[str]],
    schema: pa.Schema,
    columns: Optional[List[str]] = None,
    predicate: Optional[str] = None,
    properties: Optional[Dict[str, str]] = None,
    parallelism: int = -1,
    **read_args
) -> ray.data.Dataset
```

**Parameters:**
- `column_groups`: JSON string(s) with column group manifests
- `schema`: PyArrow schema for the dataset
- `columns`: Optional list of columns to read
- `predicate`: Optional filter expression
- `properties`: Storage configuration properties
- `parallelism`: Number of parallel read tasks (-1 for default)
- `**read_args`: Additional Ray read arguments

**Returns:** Ray Dataset

### write_milvus()

```python
def write_milvus(
    dataset: ray.data.Dataset,
    path: str,
    schema: pa.Schema,
    properties: Optional[Dict[str, str]] = None,
    **write_args
) -> None
```

**Parameters:**
- `dataset`: Ray Dataset to write
- `path`: Base path for output
- `schema`: PyArrow schema for the dataset
- `properties`: Storage configuration properties
- `**write_args`: Additional Ray write arguments

## Development

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/your-repo/ray-milvus.git
cd ray-milvus

# Install with dev dependencies
uv sync

# Run tests
pytest tests/
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=ray_milvus

# Run specific test file
pytest tests/test_integration.py
```

## Performance Tips

1. **Parallelism**: Tune the `parallelism` parameter based on your cluster size and data distribution
2. **Column Projection**: Only read columns you need to reduce I/O
3. **Batch Size**: Adjust Ray's batch size for your workload
4. **Memory**: Monitor memory usage when processing large datasets

## Troubleshooting

### Common Issues

**Import Error: Cannot import milvus_storage**
```bash
pip install milvus-storage
```

**Schema Mismatch Error**
Ensure your PyArrow schema matches the data types in your Milvus Storage files.

**Memory Issues**
Reduce parallelism or use streaming operations:
```python
ds.map_batches(fn, batch_size=100).write_milvus(...)
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the **Server Side Public License v1 (SSPLv1)** and the **GNU Affero General Public License v3 (AGPLv3)**.

## Links

- [Ray Data Documentation](https://docs.ray.io/en/latest/data/data.html)
- [Milvus Storage](https://github.com/milvus-io/milvus-storage)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)

## Changelog

### v0.1.0
- Initial release
- MilvusDatasource implementation
- MilvusDatasink implementation
- Support for parallel reading and writing
- Column projection and predicate pushdown
- Examples and documentation
