"""Find and filter files in given directory. Supports BIDSPath objects from `mne-bids`."""

from dataclasses import dataclass, field
from typing import List

import mne_bids

from .filereader_abc import FileReader


@dataclass
class DefaultReader(FileReader):
    """Class for finding and handling any type of file."""

    def find_files(
        self,
        directory: str,
        keywords: list = None,
        extensions: list = None,
        verbose: bool = True,
    ) -> None:
        """Find files in directory with optional
        keywords and extensions.

        Args:
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """
        self.directory = directory
        self.files = self._find_files(directory, keywords, extensions, verbose)

    def filter_files(
        self,
        keywords: list = None,
        hemisphere: str = None,
        stimulation: str = None,
        medication: str = None,
        exclude: str = None,
        verbose: bool = True,
    ) -> None:
        """Filter list of filepaths for given parameters and return filtered list."""
        self.files = self._filter_files(
            self.files,
            keywords=keywords,
            hemisphere=hemisphere,
            stimulation=stimulation,
            medication=medication,
            exclude=exclude,
            verbose=verbose,
        )


@dataclass
class BIDSReader(FileReader):
    """Class for finding and handling data files in BIDS-compliant format."""

    bids_root: str = field(init=False)

    def find_files(
        self,
        directory: str,
        keywords: list = None,
        extensions: list = None,
        verbose: bool = True,
    ):
        """Find files in directory with optional keywords and extensions.


        Args:
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """
        self.directory = directory
        files = self._find_files(self.directory, keywords, extensions, verbose)
        self.files = self._make_bids_paths(files, ".vhdr")

    def filter_files(
        self,
        keywords: list = None,
        hemisphere: str = None,
        stimulation: str = None,
        medication: str = None,
        exclude: str = None,
        verbose: bool = True,
    ) -> None:
        files = self._filter_files(
            keywords, hemisphere, stimulation, medication, exclude, verbose
        )
        self.files = self._make_bids_paths(files, ".vhdr")

    def _make_bids_paths(
        self, filepaths: List[str], extension: str = ".vhdr",
    ) -> List[mne_bids.BIDSPath]:

        """Create list of mne-bids BIDSPath objects from list of filepaths."""
        bids_paths = []
        for filepath in filepaths:
            entities = mne_bids.get_entities_from_fname(filepath)
            try:
                bids_path = mne_bids.BIDSPath(
                    subject=entities["subject"],
                    session=entities["session"],
                    task=entities["task"],
                    run=entities["run"],
                    acquisition=entities["acquisition"],
                    suffix=entities["suffix"],
                    extension=extension,
                    root=self.directory,
                )
            except ValueError as err:
                print(
                    f"ValueError while creating BIDS_Path object for file "
                    f"{filepath}: {err}"
                )
            else:
                bids_paths.append(bids_path)
        return bids_paths


def get_filereader(datatype: str) -> FileReader:
    """Create and return FileReader of desired type.

    Parameters
    ----------
    datatype : str
        Allowed values for `datatype`: ["any", "bids"].

    Returns
    -------
    FileReader
        Instance of FileReader for reading given `datatype`.
    """
    readers = {
        "any": DefaultReader,
        "bids": BIDSReader,
    }
    datatype = datatype.lower()
    if datatype in readers:
        return readers[datatype]()
    raise ReaderNotFoundError(datatype, readers)


class ReaderNotFoundError(Exception):
    """Exception raised when invalid Reader is passed.

    Attributes:
        datatype -- input datatype which caused the error
        readers -- allowed datatypes
        message -- explanation of the error
    """

    def __init__(
        self,
        datatype,
        readers,
        message="Input datatype is not an allowed value.",
    ) -> None:
        self.datatype = datatype
        self.readers = readers.values
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f"{{self.message}} Allowed values: {self.readers}. Got: {self.datatype}."
