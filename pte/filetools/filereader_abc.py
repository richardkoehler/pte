"""Define abstract base classes to construct FileReader classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import os
from typing import List, Optional

import mne_bids

from .. import settings


@dataclass
class FileReader(ABC):
    """Basic representation of class for finding and filtering files."""

    directory: str = field(init=False)
    files: list = field(init=False)

    @abstractmethod
    def find_files(
        self,
        directory: str,
        keywords: list = None,
        extensions: list = None,
        verbose: bool = True,
    ) -> None:
        """Find files in directory with optional
        keywords and extensions."""

    @abstractmethod
    def filter_files(
        self,
        keywords: list = None,
        hemisphere: str = None,
        stimulation: str = None,
        medication: str = None,
        exclude: str = None,
        verbose: bool = True,
    ) -> None:
        """Filter filepaths for given parameters."""

    @staticmethod
    def _keyword_search(files, keywords):
        if not keywords:
            return files
        filtered_files = []
        for file in files:
            if any([kword.lower() in file.lower() for kword in keywords]):
                filtered_files.append(file)
        return filtered_files

    def _print_files(self, files) -> None:
        if not files:
            print("No corresponding files found.")
        else:
            print("Corresponding files found:")
            for idx, file in enumerate(files):
                print(idx, ":", os.path.basename(file))

    def _find_files(
        self,
        directory: str,
        keywords: list = None,
        extensions: list = None,
        verbose: bool = True,
    ) -> List[str]:
        """Find all files in directory with optional
        keywords and extensions.

        Args:
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """
        if not os.path.isdir(directory):
            raise DirectoryNotFoundError(directory)
        self.directory = directory

        files = []
        for _, _, fnames in os.walk(directory):
            fnames = self._keyword_search(fnames, keywords)
            fnames = self._keyword_search(fnames, extensions)
            files.extend(fnames)
        if files:
            files = [os.path.join(directory, file) for file in files]

        if verbose:
            self._print_files(files)
        return files

    def _filter_files(
        self,
        files: List[str],
        keywords: Optional[list] = None,
        hemisphere: Optional[str] = None,
        stimulation: Optional[str] = None,
        medication: Optional[str] = None,
        exclude: Optional[str] = None,
        verbose: bool = True,
    ) -> List[str]:
        """Filter list of filepaths for given parameters and return filtered list."""
        filtered_files = files
        if keywords:
            if isinstance(keywords, str):
                keywords = [keywords]
            filtered_files = [
                file
                for file in filtered_files
                if any([key in file for key in keywords])
            ]
        if stimulation:
            if stimulation.lower() in "stimon":
                stim = "StimOn"
            elif stimulation.lower() in "stimoff":
                stim = "StimOff"
            else:
                raise ValueError("Keyword for stimulation not valid.")
            filtered_files = [file for file in filtered_files if stim in file]
        if medication:
            if medication.lower() in "medon":
                med = "MedOn"
            elif medication.lower() in "medoff":
                med = "MedOff"
            else:
                raise ValueError("Keyword for medication not valid.")
            filtered_files = [file for file in filtered_files if med in file]
        if hemisphere:
            matching_files = []
            for file in filtered_files:
                entities = mne_bids.get_entities_from_fname(file)
                hem = settings.ECOG_HEMISPHERES[entities["subject"]] + "_"
                if hemisphere.lower() in "ipsilateral" and hem in file:
                    matching_files.append(file)
                if hemisphere.lower() in "contralateral" and hem not in file:
                    matching_files.append(file)
            filtered_files = matching_files
        if exclude:
            filtered_files = [
                file
                for file in filtered_files
                if not any(item in file for item in exclude)
            ]
        if verbose:
            self._print_files(filtered_files)
        return filtered_files


class DirectoryNotFoundError(Exception):
    """Exception raised when invalid Reader is passed.

    Attributes:
        directory -- input directory which caused the error
    """

    def __init__(
        self, directory, message="Input directory was not found.",
    ):
        self.directory = directory
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{{self.message}} Got: {self.directory}."
