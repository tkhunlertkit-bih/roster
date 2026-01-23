import fnmatch
import io
import logging
from typing import Dict, Optional

import pandas as pd
from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient

from file_storage.file_storage_interface import FileStorageInterface


class SharePointRW(FileStorageInterface):
    """SharePoint implementation of FileStorageInterface."""

    def __init__(self, tenant_id: str, client_id: str, client_secret: str, site_id: str, folder_path: str = ""):
        """
        Initialize SharePoint connection.

        Args:
            tenant_id: Azure AD tenant ID
            client_id: Application (client) ID
            client_secret: Client secret value
            site_id: SharePoint site ID
            folder_path: Folder path within SharePoint (e.g., "/Shared Documents/Data")
        """
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.site_id = site_id
        self.folder_path = folder_path.rstrip("/") if folder_path else ""

        logging.info(f"Initialized SharePointRW: site_id='{site_id}', " f"folder='{folder_path}'")

    def _get_graph_client(self):
        """Get authenticated Microsoft Graph client."""

        credential = ClientSecretCredential(
            tenant_id=self.tenant_id, client_id=self.client_id, client_secret=self.client_secret
        )

        return GraphServiceClient(credentials=credential, scopes=["https://graph.microsoft.com/.default"])

    def _get_file_path(self, filename: str) -> str:
        """Construct full file path including folder."""
        if self.folder_path:
            return f"{self.folder_path}/{filename}"
        return filename

    def read_csv(self, filename: str) -> pd.DataFrame:
        """Read CSV file from SharePoint."""

        file_path = self._get_file_path(filename)

        try:
            client = self._get_graph_client()

            # Get file content
            file_content = client.sites.by_site_id(self.site_id).drive.root.item_with_path(file_path).content.get()

            # Read into DataFrame
            df = pd.read_csv(io.BytesIO(file_content))
            logging.info(f"Successfully read {file_path} " f"({len(df)} rows, {len(df.columns)} columns)")
            return df

        except Exception as e:
            logging.error(f"Error reading {file_path} from SharePoint: {str(e)}")
            raise FileNotFoundError(f"Could not read {file_path} from SharePoint site {self.site_id}") from e

    def write_excel(
        self, filename: str, sheets: Optional[Dict[str, pd.DataFrame]] = None, **kwargs: pd.DataFrame
    ) -> None:
        """
        Write DataFrame(s) to Excel file in SharePoint.

        Supports multiple sheets via dict or kwargs.
        """

        file_path = self._get_file_path(filename)

        # Merge sheets dict and kwargs (kwargs take precedence)
        all_sheets = {}
        if sheets:
            all_sheets.update(sheets)
        if kwargs:
            all_sheets.update(kwargs)

        # Validation
        if not all_sheets:
            raise ValueError("No DataFrames provided. Use either 'sheets' parameter or " "keyword arguments")

        # Validate sheet names
        for sheet_name in all_sheets.keys():
            if len(sheet_name) > 31:
                raise ValueError(f"Sheet name '{sheet_name}' exceeds Excel's 31 character limit")
            if any(char in sheet_name for char in [":", "\\", "/", "?", "*", "[", "]"]):
                raise ValueError(f"Sheet name '{sheet_name}' contains invalid characters")

        try:
            client = self._get_graph_client()

            # Convert DataFrame(s) to Excel in memory
            output = io.BytesIO()

            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                for sheet_name, df in all_sheets.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logging.debug(f"Added sheet '{sheet_name}': " f"{len(df)} rows, {len(df.columns)} columns")

            output.seek(0)

            # Upload to SharePoint
            (client.sites.by_site_id(self.site_id).drive.root.item_with_path(file_path).content.put(output.read()))

            sheet_summary = ", ".join(f"'{name}' ({len(df)} rows)" for name, df in all_sheets.items())
            logging.info(f"Successfully wrote {file_path} with {len(all_sheets)} sheet(s): " f"{sheet_summary}")

        except Exception as e:
            logging.error(f"Error writing {file_path} to SharePoint: {str(e)}")
            raise Exception(f"Could not write {file_path} to SharePoint site {self.site_id}") from e

    def list_files(self, pattern: Optional[str] = None) -> list[str]:
        """List files in SharePoint folder."""
        try:
            client = self._get_graph_client()

            # List items in folder
            items = client.sites.by_site_id(self.site_id).drive.root.item_with_path(self.folder_path).children.get()

            filenames = []
            for item in items.value:
                if not item.folder:  # Only files, not folders
                    filename = item.name
                    if pattern is None or self._matches_pattern(filename, pattern):
                        filenames.append(filename)

            logging.info(f"Found {len(filenames)} files matching pattern '{pattern}'")
            return filenames

        except Exception as e:
            logging.error(f"Error listing files: {str(e)}")
            return []

    def file_exists(self, filename: str) -> bool:
        """Check if file exists in SharePoint."""
        file_path = self._get_file_path(filename)

        try:
            client = self._get_graph_client()
            client.sites.by_site_id(self.site_id).drive.root.item_with_path(file_path).get()
            return True
        except Exception:
            return False

    @staticmethod
    def _matches_pattern(filename: str, pattern: str) -> bool:
        """Simple wildcard pattern matching."""
        return fnmatch.fnmatch(filename, pattern)
