from abc import ABC, abstractmethod
from typing import Dict, Optional

import pandas as pd


class FileStorageInterface(ABC):
    """Abstract base class for file storage operations."""

    @abstractmethod
    def read_csv(self, filename: str) -> pd.DataFrame:
        """
        Read CSV file and return as DataFrame.

        Args:
            filename: Name of the csv file to read

        Returns:
            pandas DataFrame containing the file data

        Raises:
            FileNotFountError: If the file does not exist
            Exception: For other storage-specific errors
        """
        pass

    @abstractmethod
    def write_excel(
        self, filename: str, sheets: Optional[Dict[str, pd.DataFrame]] = None, **kwargs: pd.DataFrame
    ) -> None:
        """
        Write DataFrame(s) to Excel file with support for multiple sheets.

        Args:
            filename: Name or path of the output file
            sheets: Dictionary mapping sheet names to DataFrames
            **kwargs: Alternative way to specify sheets as keyword arguments
                     (e.g., roster=df1, summary=df2)

        Examples:
            # Single sheet (default name 'Sheet1')
            storage.write_excel("output.xlsx", sheets={"Data": df})

            # Multiple sheets using dict
            storage.write_excel("output.xlsx", sheets={
                "Roster": roster_df,
                "Summary": summary_df,
                "Stats": stats_df
            })

            # Multiple sheets using kwargs
            storage.write_excel("output.xlsx",
                               Roster=roster_df,
                               Summary=summary_df,
                               Statistics=stats_df)

            # Mix both (kwargs take precedence)
            storage.write_excel("output.xlsx",
                               sheets={"Data": df1},
                               Summary=df2)

        Raises:
            ValueError: If no DataFrames provided or invalid sheet names
            Exception: For storage-specific errors
        """
        pass

    @abstractmethod
    def list_files(self, pattern: Optional[str] = None) -> list[str]:
        """
        List files in storage.

        Args:
            pattern: Optional filter pattern (e.g., "*.csv")

        Returns:
            List of filenames matching the pattern
        """
        pass

    @abstractmethod
    def file_exists(self, filename: str) -> bool:
        """
        Check if file exists.

        Args:
            filename: Name or path of the file to check

        Returns:
            True if file exists, False otherwise
        """
        pass
