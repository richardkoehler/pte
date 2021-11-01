# PTE - Python tools for electrophysiology

PTE is an open-source non-comprehensive software package that provides functionality for working with electrophysiological data.

## Installing PTE

First, get the current development version of PTE using [git](https://git-scm.com/). Then type the following command into a terminal:

```bash
git clone https://github.com/richardkoehler/pte
```

Use the package manager [conda](https://docs.conda.io/projects/conda/en/latest/index.html) to set up a new working environment. To do so, use ``cd`` in your terminal to navigate to the PTE root directory and type:

```bash
conda env create -f environment.yml
```

This will set up a new conda environment called ``pte``.

To activate the environment then type:

```bash
conda activate pte
```

Finally, to install PTE in an editable development version inside your conda enviroment type the following inside the PTE root directory:

```bash
conda develop .
```

Done! 

## Usage

```python
import pte

# Examples
```

## Contributing
Please feel free to contribute. 

For any minor additions or bugfixes, you may simply create a **pull request**. 

For any major changes, make sure to open an **issue** first. When you then create a pull request, be sure to **link the pull request** to the open issue in order to close the issue automatically after merging.

Continuous Integration (CI) including automated testing are set up.

## License
To be added