"""Define abstract base classes to construct FileFinder classes."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import mne_bids

from .. import settings


@dataclass
class FileFinder(ABC):
    """Basic representation of class for finding and filtering files."""

    directory: Union[Path, str] = field(init=False)
    files: list = field(init=False, default_factory=list)

    def __str__(self):
        if not self.files:
            return "No corresponding files found."
        headers = ["Index", "Filename"]
        col_width = max(len(os.path.basename(file)) for file in self.files)
        format_row = f"{{:>{len(headers[0]) + 2}}}{{:>{col_width + 2}}}"
        return "\n".join(
            (
                "Corresponding files found:",
                "".join(
                    f"{{:>{len(header) + 2}}}".format(header)
                    for header in headers
                ),
                "\u2500" * os.get_terminal_size().columns,
                *(
                    format_row.format(idx, os.path.basename(file))
                    for idx, file in enumerate(self.files)
                ),
            )
        )

    def __len__(self) -> int:
        if not self.files:
            return 0
        return len(self.files)

    @abstractmethod
    def find_files(
        self,
        directory: Union[str, Path],
        keywords: Optional[Union[list, str]] = None,
        extensions: Optional[Union[list, str]] = None,
        verbose: bool = False,
    ) -> None:
        """Find files in directory with optional
        keywords and extensions."""

    @abstractmethod
    def filter_files(
        self,
        keywords: Optional[Union[str, list]] = None,
        hemisphere: Optional[str] = None,
        stimulation: Optional[str] = None,
        medication: Optional[str] = None,
        exclude: Optional[Union[str, list]] = None,
        verbose: bool = False,
    ) -> None:
        """Filter list of filepaths for given parameters."""

    @staticmethod
    def _keyword_search(
        files: list[str], keywords: Optional[Union[str, list]]
    ) -> list:
        if not keywords:
            return files
        if not isinstance(keywords, list):
            keywords = [keywords]
        filtered_files = [
            file for file in files if any(key in file for key in keywords)
        ]
        return filtered_files

    def _find_files(
        self,
        directory: Union[Path, str],
        keywords: Optional[Union[list, str]] = None,
        extensions: Optional[Union[list, str]] = None,
    ) -> list[str]:
        """Find all files in directory with optional
        keywords and extensions.

        Args:
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """

        files = []
        for root, _, fnames in os.walk(directory):
            fnames = [os.path.join(root, file) for file in fnames]
            fnames = self._keyword_search(fnames, keywords)
            fnames = self._keyword_search(fnames, extensions)
            if fnames:
                files.extend(fnames)

        return files

    def _filter_files(
        self,
        keywords: Optional[Union[str, list[str]]] = None,
        hemisphere: Optional[str] = None,
        stimulation: Optional[str] = None,
        medication: Optional[str] = None,
        exclude: Optional[Union[str, list[str]]] = None,
    ) -> list[str]:
        """Filter filepaths for given parameters and return filtered list."""
        filtered_files = self.files
        if exclude:
            if not isinstance(exclude, list):
                exclude = [exclude]
            filtered_files = [
                file
                for file in filtered_files
                if not any(item in file for item in exclude)
            ]
        if keywords:
            if not isinstance(keywords, list):
                keywords = [keywords]
            filtered_files = self._keyword_search(filtered_files, keywords)
        if stimulation:
            if stimulation.lower() in "stimon":
                stim = "StimOn"
            elif stimulation.lower() in "stimoff":
                stim = "StimOff"
            else:
                raise ValueError("Keyword for stimulation not valid.")
            filtered_files = self._keyword_search(filtered_files, [stim])
        if medication:
            if medication.lower() in "medon":
                med = "MedOn"
            elif medication.lower() in "medoff":
                med = "MedOff"
            else:
                raise ValueError("Keyword for medication not valid.")
            filtered_files = self._keyword_search(filtered_files, [med])
        if hemisphere:
            matching_files = []
            for file in filtered_files:
                subject = mne_bids.get_entities_from_fname(file)["subject"]
                if (
                    subject not in settings.ECOG_HEMISPHERES
                    or settings.ECOG_HEMISPHERES[subject] is None
                ):
                    raise HemisphereNotSpecifiedError(
                        subject, settings.ECOG_HEMISPHERES
                    )
                hem = settings.ECOG_HEMISPHERES[subject] + "_"
                if hemisphere.lower() in "ipsilateral" and hem in file:
                    matching_files.append(file)
                if hemisphere.lower() in "contralateral" and hem not in file:
                    matching_files.append(file)
            filtered_files = matching_files
        self.files = filtered_files
        return filtered_files


class DirectoryNotFoundError(Exception):
    """Exception raised when invalid Reader is passed.

    Attributes:
        directory -- input directory which caused the error
    """

    def __init__(
        self,
        directory: Union[Path, str],
        message="Input directory was not found.",
    ):
        self.directory = directory
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} Got: {self.directory}."


class HemisphereNotSpecifiedError(Exception):
    """Exception raised when electrode hemisphere is not specified in settings.

    Attributes:
        subject -- input subject which caused the error
        hemisphere -- specified hemispheres
        message -- explanation of the error
    """

    def __init__(
        self,
        subject,
        hemispheres,
        message=(
            "Input ECOG hemisphere is not specified in `settings.py` for"
            " given subject."
        ),
    ) -> None:
        self.subject = subject
        self.hemispheres = hemispheres
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return (
            f"{self.message} Specified hemispheres: {self.hemispheres}."
            f"Unspecified subject: {self.subject}."
        )
