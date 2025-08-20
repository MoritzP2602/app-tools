# App Tools

A collection of tools for working with apprentice, a program for tuning MC event generators (DOI: 10.1051/epjconf/202125103060, GitHub repository).
These additional tools provide functionalities such as improved grid generation, merging multiple newscan directories, and performing chi-squared analyses.

Several of the tools are specifically tailored to streamline the tuning of multiple event types within apprentice, e.g. a combined tune on Drell-Yan and dijet data.

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
- `app-tools-chi_squared`: Compute chi-squared statistics from YODA files
- `app-tools-combine_weights`: Scale and combine weight files
- `app-tools-create_grid`: Generate parameter grids (enhanced version of app-sample)
- `app-tools-split_build_process`: Split the build process (used in combination with app-build)
- `app-tools-write_weights`: Extract observables from YODA files and write weights file

### Shell Scripts:
- `app-tools-prepare_run_directories.sh`: Create and manage run directories
- `app-tools-yodamerge.sh`: Merge YODA files in directories
- `app-tools-yodamerge_directories.sh`: Merge multiple (newscan) directories

## Usage

Chi-squared analysis: 
```bash
# get chi-squared values for all yoda files in a directory and all subdirectories (e.g. newscan/), use --tag to filter .yoda files (e.g. git hash)
app-tools-chi_squared weights.txt directory/ [--tag nametag]
# get chi-squared values for all yoda files, --plots creates plots for each analysis, --default creates additional ratio plots
app-tools-chi_squared weights.txt data1.yoda [data2.yoda ...] [--labels label1 label2 ...] [--plots] [--default default.yoda] [--default_label default_label]
```
Combine weight files:
```bash
# combine and scale multiple weight files
app-tools-combine_weights weight_file1.txt factor1 [weight_file2.txt factor2 ...] -o combined.txt
```
Create parameter grid:
```bash
# create grid with n points, with parameters sampled randomly within specified intervalls (same function as app-sample)
app-tools-create_grid parameter.json template.yaml npoints [--seed s] [--table] [-o outdir] (--mode random)
# copy grid from another directory
app-tools-create_grid directory/ template.yaml [--table] [-o outdir]
# create grid with n points, with parameters sampled uniformly within the specified intervalls
app-tools-create_grid parameter.json template.yaml n --mode uniform [--table] [-o outdir]
# create grid with tuned parameters, loaded from tune directories (created with app-tune2) within specified directory
app-tools-create_grid scan_directory/ template.yaml --mode tune [--default default.json] [--tune_tag foldertag] [-o outdir]
# create grid with min/max values of the parameters, used to ensure a suitable parameter range
app-tools-create_grid directory template.yaml --default default.json --mode minmax [-o outdir]
```
Prepare run directories:
```bash
# creates n subfolders in each subfolder of a specified directory and writes path to new subfolders in run_directories.txt file (skips subfolders, that already have subfolders)
app-tools-prepare_run_directory directory/ n
```
Merge YODA files:
```bash
# uses yodamerge to merge all .yoda (and .yoda.gz) files in every subfolder of a specified directory
app-tools-yodamerge directory/ [nproc]
```
Merge different directories:
```bash
# uses yodamerge to merge the .yoda (and .yoda.gz) files in the subfolders with the same name (and params.dat file) from all input directories (e.g. to merge multiple newscan directories)
app-tools-yodamerge_directories inputdir1/ inputdir2/ [inputdir3/ ...] outdir/ [nproc]
```
Extract observables and write weights:
```bash
# creates a new weights file from specified .yoda file (this file can be used for app-build)
app-tools-write_weights file.yoda -o weights.txt
```
Split the build process (used in combination with app-build):
```bash
# splits a weight file in n files saved in the directory weight_files and writes the path to each file in weight_files.txt
app-tools-split_build_process weights.txt n
# merges n .json files in a specified input directory in the file directory.json
app-tools-split_build_process directory/ n
```

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
