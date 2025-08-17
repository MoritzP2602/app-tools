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

## Usage Examples

```bash
# Chi-squared analysis
app-tools-chi_squared weights.txt data1.yoda data2.yoda --plots --default default.yoda

# Combine weight files
app-tools-combine_weights weight_file1.txt 1.0 weight_file2.txt 0.5 -o combined.txt

# Create parameter grid
app-tools-create_grid parameter.json template.yaml 500

# Prepare run directories
app-tools-prepare_run_directories newscan/ 10

# Merge YODA files
app-tools-yodamerge newscan/

# Merge different newscan directories
app-tools-yodamerge_directories newscan1/ newscan2/ outdir/

# Extract observables from YODA files and write weights
app-tools-write_weights data.yoda -o data_weights.txt

# Split the build process (used in combination with app-build)
app-tools-split_build_process weights.txt 20
app-tools-split_build_process app_5_0 20
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
