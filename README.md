<div align="center">
  <img src="logo.png" width="400"/>
</div>
<br>


# app-tools

A collection of tools for working with Apprentice, a program for tuning MC event generators (DOI: 10.1051/epjconf/202125103060, GitHub repository https://github.com/HEPonHPC/apprentice).
These additional tools provide functionalities such as improved grid generation, merging multiple newscan directories, and performing chi-squared analyses.

Several of the tools are specifically tailored to streamline the tuning of multiple event types within Apprentice, e.g. a combined tune on Drell-Yan and dijet data.

## Prerequisites

**Before installing this package, you must install:**

- **YODA >= 1.8.0**
- **Rivet >= 3.0.0**

You can follow these instructions: https://gitlab.com/hepcedar/rivetbootstrap

## Installation

```bash
git clone https://github.com/MoritzP2602/app-tools.git
cd app-tools
pip install .
```

## Requirements

This package automatically installs:
- Python 3.7+
- numpy >= 1.19.0
- matplotlib >= 3.3.0
- tabulate >= 0.8.0

**Manual installation required:**
- YODA >= 1.8.0
- Rivet >= 3.0.0

## Tools

### Python Scripts:
- `app-tools-compute_chi2`: Compute chi-squared statistics and write results to `chi2.json`
- `app-tools-plot_chi2`: Plot chi-squared results from `chi2.json` and generate an HTML report
- `app-tools-combine_weights`: Scale and combine weight files
- `app-tools-create_grid`: Generate parameter grids (enhanced version of app-sample)
- `app-tools-merge_surrogates`: Merge multiple surrogate JSON files
- `app-tools-split_reweighting`: Split variations in YODA files in separate files
- `app-tools-split_weights`: Split weight files for parallel processing
- `app-tools-write_weights`: Extract observables from YODA files and write weights file

## Usage

Run the tools using the command line interface. To see all available options, examples and usage intructions for each tool, use the `--help` flag.

## Troubleshooting

### Verifying YODA and Rivet Installation

Before using this package, verify that YODA and Rivet are properly installed:

```bash
# Check YODA installation and version
python -c "import yoda; print(f'YODA version: {yoda.__version__}')"

# Check Rivet installation and version  
python -c "import rivet; print(f'Rivet version: {rivet.version()}')"
```

## License

MIT License
