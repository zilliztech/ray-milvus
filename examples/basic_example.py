#!/usr/bin/env python
"""
Basic example of using Ray Data with Milvus Storage.

This example demonstrates:
1. Writing data to Milvus Storage using Ray Data
2. Reading data from Milvus Storage using Ray Data
3. Processing data with Ray Data transformations
"""

import ray
import pyarrow as pa
import pandas as pd
from ray_milvus import read_milvus, write_milvus


def main():
    """Run the basic example."""
    # Initialize Ray
    ray.init(ignore_reinit_error=True)

    print("=== Writing Data to Milvus Storage with Ray ===\n")

    # Create sample data
    data = {
        "id": list(range(100)),
        "vector": [f"vector" for i in range(100)],
        "text": [f"item_{i}" for i in range(100)],
        "score": [float(i) * 0.1 for i in range(100)],
    }

    # Create Ray dataset from pandas
    df = pd.DataFrame(data)
    ds = ray.data.from_pandas(df)

    print(f"Created Ray dataset with {ds.count()} rows")
    print(f"Schema: {ds.schema()}\n")

    # Define Milvus storage schema
    # Note: Ray's schema uses different types, so we define the target schema
    milvus_schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=True, metadata={"PARQUET:field_id": "1"}),
            pa.field(
                "vector",
                pa.binary(6),
                nullable=True,
                metadata={"PARQUET:field_id": "2"},
            ),
            pa.field("text", pa.string(), nullable=True, metadata={"PARQUET:field_id": "3"}),
            pa.field(
                "score", pa.float64(), nullable=True, metadata={"PARQUET:field_id": "4"}
            ),
        ]
    )

    # Storage configuration
    storage_path = "/tmp/ray_milvus_example"
    properties = {
        "fs.storage_type": "local",
        "fs.root_path": "/tmp/",
    }

    # Write to Milvus storage
    print(f"Writing to {storage_path}...")

    # Convert to Arrow Table with the correct schema before writing
    arrow_table = pa.Table.from_pandas(df, schema=milvus_schema)
    ds_with_schema = ray.data.from_arrow(arrow_table)

    write_milvus(
        ds_with_schema, path=storage_path, schema=milvus_schema, properties=properties
    )

    print(f"Data written to {storage_path}")
    print(f"Manifest will be created by Milvus Storage\n")

    print("=== Reading Data from Milvus Storage with Ray ===\n")


    column_groups_content = '''
    {
      "column_groups": [
        {
          "columns": [
            "id",
            "vector",
            "text",
            "score"
          ],
          "format": "parquet",
          "paths": [
            "/tmp/ray_milvus_example/column_group_0.parquet"
          ]
        }
      ]
    }'''
    # Read back the data
    ds_read = read_milvus(
        column_groups=column_groups_content, schema=milvus_schema, properties=properties
    )

    print(f"Read Ray dataset with {ds_read.count()} rows")
    print(f"Schema: {ds_read.schema()}\n")

    # Show first few rows
    print("First 5 rows:")
    print(ds_read.take(5))
    print()

    print("=== Processing Data with Ray ===\n")

    # Example: Filter data
    filtered_ds = ds_read.filter(lambda row: row["id"] >= 50)
    print(f"Filtered dataset (id >= 50): {filtered_ds.count()} rows")

    # Example: Transform data
    def compute_magnitude(row):
        import math

        vector = row["vector"]
        magnitude = math.sqrt(sum(x * x for x in vector))
        row["magnitude"] = magnitude
        return row

    transformed_ds = filtered_ds.map(compute_magnitude)
    print("Added magnitude computation")
    print("Sample row with magnitude:")
    print(transformed_ds.take(1))
    print()

    print("=== Reading Specific Columns ===\n")

    # Read only specific columns
    ds_projected = read_milvus(
        column_groups=column_groups_content,
        schema=milvus_schema,
        columns=["id", "text"],
        properties=properties,
    )

    print(f"Projected dataset (id, text only): {ds_projected.count()} rows")
    print("First 3 rows:")
    print(ds_projected.take(3))
    print()

    print("=== Example Complete ===")

    # Cleanup
    ray.shutdown()


if __name__ == "__main__":
    main()
