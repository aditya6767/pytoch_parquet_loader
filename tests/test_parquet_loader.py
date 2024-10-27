import os
import pytest
from pathlib import Path
import torch
from pytorch_parquet_loader import load_parquet_as_dataloader

PARQUET_FILE_PATH = Path(__file__).parent / "data" / "flights_1m.parquet"


@pytest.mark.skipif(not os.path.exists(str(PARQUET_FILE_PATH)), reason="Parquet file not found.")
def test_dask_parquet_dataset():
    batch_size = 32
    dataloader = load_parquet_as_dataloader(file_path=PARQUET_FILE_PATH, batch_size=batch_size)

    # Iterate through the dataloader and check the shape of each batch
    for i, batch in enumerate(dataloader):
        assert isinstance(batch, torch.Tensor), f"Batch {i} is not a torch.Tensor"
        assert batch.shape[0] <= batch_size, f"Batch {i} should have at most {batch_size} rows"

        # Limit the number of batches tested to avoid excessive output
        if i >= 3:
            break
