"""Find and filter files. Supports BIDSPath objects from `mne-bids`."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import mne_bids

from .filefinder_abc import DirectoryNotFoundError, FileFinder


@dataclass
class DefaultFinder(FileFinder):
    """Class for finding and handling any type of file."""

    def find_files(
        self,
        directory: Union[Path, str],
        keywords: Optional[Union[list[str], str]] = None,
        extensions: Optional[Union[list[str], str]] = None,
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
        self.files = self._find_files(self.directory, keywords, extensions)
        if verbose:
            print(self)

    def filter_files(
        self,
        keywords: list = None,
        hemisphere: str = None,
        stimulation: str = None,
        medication: str = None,
        exclude: str = None,
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

    def find_files(
        self,
        directory: str,
        keywords: list = None,
        extensions: list = None,
        verbose: bool = False,
    ):
        """Find files in directory with optional keywords and extensions.


        Parameters
        ----------
            directory (string)
            keywords (list): e.g. ["SelfpacedRota", "ButtonPress] (optional)
            extensions (list): e.g. [".json" or "tsv"] (optional)
            verbose (bool): verbosity level (optional, default=True)
        """
        self.directory = directory
        files = self._find_files(self.directory, keywords, extensions)
        self.files = self._make_bids_paths(files)

    def filter_files(
        self,
        keywords: list = None,
        hemisphere: str = None,
        stimulation: str = None,
        medication: str = None,
        exclude: str = None,
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

    def _make_bids_paths(
        self, filepaths: list[str]
    ) -> list[mne_bids.BIDSPath]:

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


def get_filefinder(datatype: str) -> FileFinder:
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
    if datatype in finders:
        return finders[datatype]()
    raise FinderNotFoundError(datatype, finders)


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
        message="Input ``datatype`` is not an allowed value.",
    ) -> None:
        self.datatype = datatype
        self.finders = finders.values
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return (
            f"{{self.message}} Allowed values: {self.finders}."
            f" Got: {self.datatype}."
        )
