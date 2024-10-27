# Parquet Loader

`parquet_loader` is a Python library for loading large Parquet files as PyTorch `DataLoader` objects, optimized for memory efficiency using Dask.

## Features

- Efficiently loads large Parquet files into PyTorch DataLoader.
- Automatically handles `object` (string) and `date32[day]` data types by encoding them to numeric values.
- Compatible with both numeric and categorical data in a format compatible with PyTorch.

## Installation

### Prerequisites
Ensure you have Python 3.9 or higher installed.

### Install the Library

To install the package from PyPI:

```bash
pip install pytorch_parquet_loader
```

## Testing

To run the test, you can the below script:
```python
from parquet_loader import load_parquet_as_dataloader

# Path to your Parquet file
file_path = "path/to/your_large_file.parquet"

# Load Parquet file as a DataLoader
dataloader = load_parquet_as_dataloader(file_path, batch_size=32, num_workers=0)

# Iterate over the DataLoader
for batch in dataloader:
    print(batch)  # Each batch is a PyTorch tensor
```
