import dask.dataframe as dd
import torch
import pandas as pd

from torch.utils.data import IterableDataset
from torch.utils.data import get_worker_info
from sklearn.preprocessing import LabelEncoder


class ParquetDataset(IterableDataset):
    def __init__(self, file_path, batch_size=32, columns=None):
        """
        Args:
            file_path (str): Path to the Parquet file.
            batch_size (int): Number of rows per batch.
            columns (list): List of columns to read from the Parquet file.
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.columns = columns
        # self.dataframe = dd.read_parquet(file_path, columns=columns, engine='pyarrow')
        # self.dataframe = self.dataframe.apply(self._convert_to_numeric, axis=1)
        self.dataframe = dd.read_parquet(file_path, columns=columns, engine='pyarrow')
        self.dataframe = self.dataframe.map_partitions(self._filter_and_convert_columns)
        self.num_rows = len(self.dataframe)
        self.label_encoders = {}

    @staticmethod
    def _convert_to_numeric(row):
        """Convert non-numeric columns to NaN and ensure numeric types."""
        return row.apply(pd.to_numeric, errors='coerce').fillna(0)

    def _filter_and_convert_columns(self, df):
        """Keep only numeric, date32[day], and object columns, converting as needed."""
        for col in df.columns:
            if str(df[col].dtype) == "date32[day]":
                # Convert date32[day] to datetime64[ns] and then to integer timestamps in seconds
                df[col] = pd.to_datetime(df[col], errors='coerce').astype("int64") // 10**9
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Retain numeric columns as-is
                continue
            elif df[col].dtype == 'object':
                # Apply label encoding to object columns and store the encoder
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))  # Convert all entries to strings before encoding
                self.label_encoders[col] = le  # Store the encoder for later access
            else:
                # Drop columns that are neither numeric, date32[day], nor object
                df = df.drop(columns=[col])
        return df

    def get_partition_indices(self, worker_id, num_workers):
        """
        Divides the dataset into partitions for each worker based on worker_id.
        """
        # Number of rows per worker
        partition_size = self.num_rows // num_workers
        start_idx = worker_id * partition_size
        end_idx = start_idx + partition_size
        if worker_id == num_workers - 1:  # Last worker takes any remaining rows
            end_idx = self.num_rows
        return start_idx, end_idx

    def __iter__(self):

        worker_info = get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            start_idx, end_idx = self.get_partition_indices(worker_id, num_workers)
        else:
            # If single worker, use the entire dataset
            start_idx, end_idx = 0, self.num_rows

        # Stream data in batches within the assigned partition
        for batch_start in range(start_idx, end_idx, self.batch_size):
            batch_end = min(batch_start + self.batch_size, end_idx)
            # Load the batch as a Pandas DataFrame and convert it to a tensor
            batch_df = self.dataframe.loc[batch_start:batch_end - 1].compute()
            batch_tensor = torch.tensor(batch_df.values, dtype=torch.float32)
            yield batch_tensor

    def get_label_encoder(self, column_name):
        """
        Get the LabelEncoder for a specific column.

        Args:
            column_name (str): The name of the column.

        Returns:
            LabelEncoder: The LabelEncoder used for the specified column.
        """
        return self.label_encoders.get(column_name, None)
