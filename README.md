[![Python Versions][python-shield]][python-url]
[![PyPi][pypi-shield]][pypi-url]
[![License][license-shield]][license-url]
[![Contributors][contributors-shield]][contributors-url]
[![Code Style][codestyle-shield]][codestyle-url]
[![Homepage][homepage-shield]][homepage-url]


# PTE - Python tools for electrophysiology

PTE is an open-source software package for working with electrophysiological data

PTE builds upon the packages [MNE](https://mne.tools/stable/index.html) and [MNE-BIDS](https://mne.tools/mne-bids/stable/index.html).

## Installing PTE

### Stable release

To install the latest stable release, simply type:

```bash
$ pip install pte
```

### Development version

To install the latest delevopment version, first clone this repository using [git](https://git-scm.com/):

```bash
$ git clone https://github.com/richardkoehler/pte
```

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to set up a new working environment. To do so navigate to the PTE root directory in your terminal and type:

```bash
$ conda env create -f env.yml
```

This will set up a new conda environment called ``pte``.

To activate the environment then type:

```bash
$ conda activate pte
```

If you want to install pte-stats into an existing environment, type:

```bash
$ pip install -e .
```

## Usage

```python
import pte

# Examples
```

## Contributing
Please feel free to contribute yourselves or to open an **issue** when you encounter a bug or would like to add a new feature.

For any minor additions or bugfixes, you may simply create a **pull request**. 

For any major changes, make sure to open an **issue** first. When you then create a pull request, be sure to **link the pull request** to the open issue in order to close the issue automatically after merging.

To contribute yourselves, consider installing the full conda development environment to include such tools as black, pylint and isort:

```bash
$ conda env create -f env_dev.yml
$ conda activate pte-dev
```

Continuous Integration (CI) including automated testing are set up.

## License
PTE is licensed under the [MIT license](license-url).

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[python-shield]: https://img.shields.io/static/v1?label=Python&message=3.10&logoColor=black&labelColor=grey&color=blue
[python-url]: https://pypi.org/project/pte-stats/
[homepage-shield]: https://img.shields.io/static/v1?label=Homepage&message=ICN&logoColor=black&labelColor=grey&color=9cf
[homepage-url]: https://www.icneuromodulation.org/
[contributors-shield]: https://img.shields.io/github/contributors/richardkoehler/pte.svg?
[contributors-url]: https://github.com/richardkoehler/pte/graphs/contributors
[license-shield]: https://img.shields.io/static/v1?label=License&message=MIT&logoColor=black&labelColor=grey&color=yellow
[license-url]: https://github.com/richardkoehler/pte/blob/main/LICENSE/
[codestyle-shield]: https://img.shields.io/static/v1?label=CodeStyle&message=black&logoColor=black&labelColor=grey&color=black
[codestyle-url]: https://github.com/psf/black
[pypi-shield]: https://img.shields.io/static/v1?label=PyPi&message=v0.1.0&logoColor=black&labelColor=grey&color=blue
[pypi-url]: https://pypi.org/project/pte/