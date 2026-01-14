"""
Integration tests for Milvus Ray datasource and datasink.
"""

import pytest


def test_import():
    """Test that the package can be imported."""
    from ray_milvus import MilvusDatasink, MilvusDatasource, read_milvus, write_milvus

    assert MilvusDatasource is not None
    assert MilvusDatasink is not None
    assert read_milvus is not None
    assert write_milvus is not None


def test_write_read_cycle():
    """Test basic write and read cycle."""
    # This is a placeholder test that would require milvus-storage to be installed
    # and properly configured
    pytest.skip("Requires milvus-storage installation and configuration")


if __name__ == "__main__":
    test_import()
    print("Import test passed!")
