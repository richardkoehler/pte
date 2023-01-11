"""Define abstract base classes to construct FileFinder classes."""

import os
import shutil
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mne_bids


@dataclass
class FileFinder(ABC):
    """Basic representation of class for finding and filtering files."""

    hemispheres: dict[str, str] | None = field(default_factory=dict)
    directory: Path | str = field(init=False)
    files: list[str] = field(init=False, default_factory=list)

    def __str__(self) -> str:
        if not self.files:
            return "No corresponding files found."
        headers = ["Index", "Filename"]
        col_width = max(len(os.path.basename(file)) for file in self.files)
        format_row = f"{{:>{len(headers[0]) + 2}}}{{:>{col_width + 2}}}"
        terminal_size = "\u2500" * shutil.get_terminal_size().columns
        return "\n".join(
            (
                "Corresponding files found:",
                "".join(
                    f"{{:>{len(header) + 2}}}".format(header)
                    for header in headers
                ),
                terminal_size,
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
        directory: Path | str,
        extensions: str | Sequence | None = None,
        keywords: str | Sequence[str] | None = None,
        hemisphere: str | None = None,
        stimulation: str | None = None,
        medication: str | None = None,
        exclude: str | Sequence[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Find files in directory with optional
        keywords and extensions."""

    @abstractmethod
    def filter_files(
        self,
        keywords: str | Sequence[str] | None = None,
        hemisphere: str | None = None,
        stimulation: str | None = None,
        medication: str | None = None,
        exclude: str | Sequence[str] | None = None,
        verbose: bool = False,
    ) -> None:
        """Filter list of filepaths for given parameters."""

    @staticmethod
    def _keyword_search(
        files: list[str], keywords: str | Sequence[str] | None
    ) -> list:
        if not keywords:
            return files
        if isinstance(keywords, str):
            keywords = [keywords]
        filtered_files = [
            file for file in files if any(key in file for key in keywords)
        ]
        return filtered_files

    def _find_files(
        self,
        directory: Path | str,
        extensions: str | Sequence | None = None,
    ) -> None:
        """Find files in directory with optional extensions.

        Args:
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """

        files = []
        for root, _, fnames in os.walk(directory):
            fnames = [os.path.join(root, file) for file in fnames]
            fnames = self._keyword_search(fnames, extensions)
            if fnames:
                files.extend(fnames)
        self.files = files

    def _filter_files(
        self,
        keywords: str | Sequence[str] | None = None,
        hemisphere: str | None = None,
        stimulation: str | None = None,
        medication: str | None = None,
        exclude: str | Sequence[str] | None = None,
    ) -> None:
        """Filter filepaths for given parameters."""
        filtered_files = self.files
        if exclude:
            if isinstance(exclude, str):
                exclude = [exclude]
            filtered_files = [
                file
                for file in filtered_files
                if not any(item in file for item in exclude)
            ]
        if keywords:
            if isinstance(keywords, str):
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
        if hemisphere is not None:
            matching_files = []
            for file in filtered_files:
                subject = mne_bids.get_entities_from_fname(file)["subject"]
                if (
                    subject not in self.hemispheres
                    or self.hemispheres[subject] is None
                ):
                    raise HemisphereNotSpecifiedError(
                        subject, self.hemispheres
                    )
                hem = self.hemispheres[subject] + "_"
                if hemisphere.lower() in "ipsilateral" and hem in file:
                    matching_files.append(file)
                if hemisphere.lower() in "contralateral" and hem not in file:
                    matching_files.append(file)
            filtered_files = matching_files
        self.files = filtered_files


class DirectoryNotFoundError(Exception):
    """Exception raised when invalid Reader is passed.

    Attributes:
        directory -- input directory which caused the error
    """

    def __init__(
        self,
        directory: Path | str,
        message: str = "Input directory was not found.",
    ) -> None:
        self.directory = directory
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return "\n".join((f"{self.message}", f"Got: {self.directory}."))


class HemisphereNotSpecifiedError(Exception):
    """Exception raised when electrode hemisphere is not specified in settings.

    Attributes:
        subject -- input subject which caused the error
        hemisphere -- specified hemispheres
        message -- explanation of the error
    """

    def __init__(
        self,
        subject: str,
        hemispheres: dict[str, str],
        message: str = (
            "Input ECOG hemisphere is not specified in"
            " `filefinder_settings.py` for given subject."
        ),
    ) -> None:
        self.subject = subject
        self.hemispheres = hemispheres
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return "\n".join(
            (
                self.message,
                f"Unspecified subject: {self.subject}.",
                f"Specified hemispheres: {self.hemispheres}.",
            )
        )
