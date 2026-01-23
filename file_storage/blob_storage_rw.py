import fnmatch
import io
import logging
from typing import Dict, Optional

import pandas as pd
from azure.storage.blob import BlobServiceClient

from file_storage.file_storage_interface import FileStorageInterface


class BlobStorageRW(FileStorageInterface):
    def __init__(self, connection_string: str, container_name: str, folder_path: str = ""):
        self.connection_string = connection_string
        self.container_name = container_name
        self.folder_path = folder_path.rstrip("/") if folder_path else ""
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_client = self.blob_service_client.get_container_client(container_name)
        logging.info(f"Initialized BlobStorageRW: container='{container_name}, folder='{folder_path}'")

    def _get_blob_path(self, filename: str) -> str:
        """Construct full blob path including folder."""
        if self.folder_path:
            return f"{self.folder_path}/{filename}"
        return filename

    def read_csv(self, filename: str) -> pd.DataFrame:
        """Read CSV file from blob storage."""
        blob_path = self._get_blob_path(filename)
        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            stream = io.BytesIO()
            blob_client.download_blob().readinto(stream)
            stream.seek(0)

            df = pd.read_csv(stream)
            logging.info(f"Successfully read {blob_path} ({len(df)} rows, {len(df.columns)} columns)")
            return df
        except Exception as e:
            logging.error(f"Error reading {blob_path} from blob storage: {str(e)}")
            raise FileNotFoundError("Could not read {blob_path} from container {self.container_name}") from e

    def write_excel(
        self, filename: str, sheets: Optional[Dict[str, pd.DataFrame]] = None, **kwargs: pd.DataFrame
    ) -> None:
        """
        Write DataFrame(s) to Excel file in blob storage.

        Supports multiple sheets via dict or kwargs.
        """
        blob_path = self._get_blob_path(filename)

        # Merge all sheets
        all_sheets = {}
        if sheets:
            all_sheets.update(sheets)
        if kwargs:
            all_sheets.update(kwargs)

        # Validation
        if not all_sheets:
            logging.error("No DataFrames provided. Use either 'sheets' parameter or keyword arguments")
            raise ValueError("No DataFrames provided. Use either 'sheets' parameter or keyword arguments")

        # Validate sheet names
        for sheet_name in all_sheets.keys():
            if len(sheet_name) > 31:
                raise ValueError(f"Sheet name '{sheet_name}' exceeds Excel's 31 character limit")
            if any(char in sheet_name for char in [":", "\\", "/", "?", "*", "[", "]"]):
                raise ValueError(f"Sheet name '{sheet_name}' contains invalid characters: ': \\ / ? * [ ]'")

        try:
            # Convert DataFrame(s) to Excel in memory
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for sheet_name, df in all_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logging.debug(f"Added sheet '{sheet_name}': {len(df)} rows, {len(df.columns)} columns")
            output.seek(0)

            # Upload to blob storage
            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(output, overwrite=True)

            sheet_summary = ", ".join(f"'{name}' ({len(df)} rows)" for name, df in all_sheets.items())
            logging.info(f"Successfully wrote {blob_path} with {len(all_sheets)} sheets(s): {sheet_summary}")
        except Exception as e:
            logging.error(f"Error writing {blob_path} to blob storage: {str(e)}")
            raise Exception(f"Could not write {blob_path} to container {self.container_name}") from e

    def list_files(self, pattern: Optional[str] = None) -> list[str]:
        """List files in the blob container/folder."""
        try:
            prefix = f"{self.folder_path}/" if self.folder_path else ""
            blobs = self.container_client.list_blobs(name_starts_with=prefix)

            filenames = []
            for blob in blobs:
                # Remove folder prefix from blob name
                name = blob.name
                if self.folder_path and name.startswith(f"{self.folder_path}/"):
                    name = name[len(self.folder_path) + 1 :]

                # Apply pattern filter if provided
                if pattern is None or self._matches_pattern(name, pattern):
                    filenames.append(name)

            logging.info(f"Found {len(filenames)} files matching pattern '{pattern}'")
            return filenames

        except Exception as e:
            logging.error(f"Error listing files: {str(e)}")
            return []

    def file_exists(self, filename: str) -> bool:
        """Check if file exists in blob storage."""
        blob_path = self._get_blob_path(filename)

        try:
            blob_client = self.container_client.get_blob_client(blob_path)
            return blob_client.exists()
        except Exception as e:
            logging.error(f"Error checking if {blob_path} exists: {str(e)}")
            return False

    @staticmethod
    def _matches_pattern(filename: str, pattern: str) -> bool:
        """Simple wildcard pattern matching."""
        return fnmatch.fnmatch(filename, pattern)
