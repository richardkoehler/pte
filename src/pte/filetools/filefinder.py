"""Find and filter files. Supports BIDSPath objects from `mne-bids`."""
from abc import ABC, abstractmethod
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import mne_bids


@dataclass
class FileFinder(ABC):
    """Basic representation of class for finding and filtering files."""

    hemispheres: dict[str, str] | None = field(default_factory=dict)
    directory: Path | str = field(init=False)

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
                    f"{{:>{len(header) + 2}}}".format(header) for header in headers
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
    def _keyword_search(files: list[str], keywords: str | Sequence[str] | None) -> list:
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
            if hemisphere.lower() not in ("ipsilateral", "contralateral"):
                raise ValueError(
                    "Keyword for hemisphere not valid. Must be one of either"
                    f"'ipsilateral' or 'contralateral'. Got: {hemisphere}."
                )
            hemisphere = hemisphere.lower()
            matching_files = []
            for file in filtered_files:
                entities = mne_bids.get_entities_from_fname(file)
                subject = entities["subject"]
                task: str = entities["task"]
                assert (
                    self.hemispheres is not None
                ), "self.hemispheres must be specified."
                if subject not in self.hemispheres or self.hemispheres[subject] is None:
                    raise HemisphereNotSpecifiedError(subject, self.hemispheres)
                hem_sub: str = self.hemispheres[subject]
                if hemisphere == "ipsilateral" and task.endswith(hem_sub):
                    matching_files.append(file)
                if hemisphere == "contralateral" and not task.endswith(hem_sub):
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


@dataclass
class DefaultFinder(FileFinder):
    """Class for finding and handling any type of file."""

    files: list[str] = field(init=False, default_factory=list)

    def __iter__(self):
        self._n = 0
        return self

    def __next__(self) -> Any:
        if self._n == len(self.files):
            raise StopIteration
        file = self.files[self._n]
        self._n += 1
        return file

    def find_files(
        self,
        directory: Path | str,
        extensions: Sequence | str | None = None,
        keywords: list[str] | str | None = None,
        hemisphere: str | None = None,
        stimulation: str | None = None,
        medication: str | None = None,
        exclude: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Find files in directory with optional
        keywords and extensions.

        Args:
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise DirectoryNotFoundError(self.directory)
        self._find_files(self.directory, extensions)
        self._filter_files(
            keywords=keywords,
            hemisphere=hemisphere,
            stimulation=stimulation,
            medication=medication,
            exclude=exclude,
        )
        if verbose:
            print(self)

    def filter_files(
        self,
        keywords: list | None = None,
        hemisphere: str | None = None,
        stimulation: str | None = None,
        medication: str | None = None,
        exclude: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Filter filepaths for given parameters and return filtered list."""
        self._filter_files(
            keywords=keywords,
            hemisphere=hemisphere,
            stimulation=stimulation,
            medication=medication,
            exclude=exclude,
        )
        if verbose:
            print(self)


@dataclass
class BIDSFinder(FileFinder):
    """Class for finding and handling data files in BIDS-compliant format."""

    bids_root: str = field(init=False)
    files: list[mne_bids.BIDSPath] = field(init=False, default_factory=list)

    def find_files(
        self,
        directory: Path | str,
        extensions: Sequence | str | None = (".vhdr", ".edf"),
        keywords: list | None = None,
        hemisphere: str | None = None,
        stimulation: str | None = None,
        medication: str | None = None,
        exclude: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Find files in directory with optional keywords and extensions.


        Parameters
        ----------
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """
        self.directory = directory
        self._find_files(self.directory, extensions)
        self._filter_files(
            keywords=keywords,
            hemisphere=hemisphere,
            stimulation=stimulation,
            medication=medication,
            exclude=exclude,
        )
        self.files = self._make_bids_paths(self.files)
        if verbose:
            print(self)

    def filter_files(
        self,
        keywords: list | None = None,
        hemisphere: str | None = None,
        stimulation: str | None = None,
        medication: str | None = None,
        exclude: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Filter list of filepaths for given parameters."""
        self.files = [str(file.fpath.resolve()) for file in self.files]
        self._filter_files(
            keywords=keywords,
            hemisphere=hemisphere,
            stimulation=stimulation,
            medication=medication,
            exclude=exclude,
        )
        self.files = self._make_bids_paths(self.files)
        if verbose:
            print(self)

    def _make_bids_paths(self, filepaths: list[str]) -> list[mne_bids.BIDSPath]:
        """Create list of mne-bids BIDSPath objects from list of filepaths."""
        bids_paths = []
        for filepath in filepaths:
            # entities = mne_bids.get_entities_from_fname(filepath)
            try:
                bids_path = mne_bids.get_bids_path_from_fname(
                    fname=filepath, verbose=False
                )
                bids_path.update(root=self.directory)
            except ValueError as err:
                print(
                    f"ValueError while creating BIDS_Path object for file "
                    f"{filepath}: {err}"
                )
            else:
                bids_paths.append(bids_path)
        return bids_paths


class FinderNotFoundError(Exception):
    """Exception raised when invalid Finder is passed.

    Attributes:
        datatype -- input datatype which caused the error
        finders -- allowed datatypes
        message -- explanation of the error
    """

    def __init__(
        self,
        datatype,
        finders,
        message="Input `datatype` is not an allowed value.",
    ) -> None:
        self.datatype = datatype
        self.finders = tuple(val for val in finders)
        self.message = message
        super().__init__(self.message)

    def __str__(self) -> str:
        return (
            f"{self.message} Allowed values: {self.finders}." f" Got: {self.datatype}."
        )


def get_filefinder(
    datatype: Literal["any", "bids"], hemispheres: dict | None = None, **kwargs
) -> DefaultFinder | BIDSFinder:
    """Create and return FileFinder of desired type.

    Parameters
    ----------
    datatype : str
        Allowed values for `datatype`: ["any", "bids"].

    Returns
    -------
    FileFinder
        Instance of FileFinder for reading given `datatype`.
    """
    finders = {
        "any": DefaultFinder,
        "bids": BIDSFinder,
    }
    datatype = datatype.lower()
    if datatype not in finders:
        raise FinderNotFoundError(datatype, finders)

    return finders[datatype](hemispheres=hemispheres, **kwargs)
