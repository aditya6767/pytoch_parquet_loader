from torch.utils.data import DataLoader

from .dataset import ParquetDataset


def load_parquet_as_dataloader(file_path, batch_size=32, columns=None, meta=None, num_workers=0):
    """
    Load a Parquet file as a PyTorch DataLoader.

    Args:
        file_path (str): Path to the Parquet file.
        batch_size (int): Number of rows per batch.
        columns (list): List of columns to read from the Parquet file.
        meta (dict): Metadata dictionary specifying column types.
        num_workers (int): Number of worker processes for DataLoader.

    Returns:
        DataLoader: A PyTorch DataLoader instance.
    """
    # Instantiate the dataset
    dataset = ParquetDataset(file_path, batch_size=batch_size, columns=columns)
    # Wrap in DataLoader and return
    return DataLoader(dataset, batch_size=None, num_workers=num_workers)


def create_dataloader(file_path, batch_size=32, columns=None, num_workers=0):
    dataset = ParquetDataset(file_path, batch_size=batch_size, columns=columns)
    return DataLoader(dataset, batch_size=None, num_workers=num_workers)