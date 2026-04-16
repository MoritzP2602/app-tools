
import os
import re
import sys
import argparse
import shutil
from pathlib import Path
from collections import defaultdict

try:
    import yoda
except ImportError:
    print("Error: YODA module not found. Please install it before running this script.")
    sys.exit(1)


def parse_variations_file(variations_path):
    variations = {}
    
    if not variations_path.exists():
        return variations
    with open(variations_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            match = re.match(r'(\S+)\s+\[(.*)\]', line)
            if match:
                param_name = match.group(1)
                values_str = match.group(2)
                values = [float(v.strip()) for v in values_str.split(',')]
                variations[param_name] = values
    return variations


def copy_and_extend_params_file(src_params_path, dst_params_path, variation_params):
    params_lines = []
    if src_params_path.exists():
        with open(src_params_path, 'r') as f:
            params_lines = f.readlines()
    with open(dst_params_path, 'w') as f:
        for line in params_lines:
            f.write(line)
        for param_name in sorted(variation_params.keys()):
            param_value = variation_params[param_name]
            f.write(f"{param_name} {param_value:.6e}\n")
    return


def parse_yoda_file(yoda_path, variation_pattern):
    yd = yoda.read(str(yoda_path))
    
    variations      = defaultdict(list)
    escaped_pattern = re.escape(variation_pattern)
    regex_pattern   = rf'\[{escaped_pattern}(\d+)\]'
    
    for path, obj in yd.items():
        variation_match = re.search(regex_pattern, path)
        if variation_match:
            variation = f'v{variation_match.group(1)}'
            clean_path = re.sub(regex_pattern, '', path)
            obj.setPath(clean_path)
            variations[variation].append(obj)
    return variations


def get_variation_numbers(variations_dict):
    var_numbers = []
    for key in variations_dict.keys():
        if key.startswith('v') and key[1:].isdigit():
            var_num = int(key[1:])
            if var_num == 0:
                continue
            var_numbers.append(var_num)
    return sorted(var_numbers)


def write_yoda_file(output_path, analysis_objects):
    yoda.write(analysis_objects, str(output_path))
    return


def find_yoda_files(directory):
    yoda_files = []
    for pattern in ("*.yoda", "*.yoda.gz"):
        yoda_files.extend(directory.glob(pattern))
    return sorted(yoda_files, key=lambda path: path.name)


def split_yodas(input_dir, variation_pattern, output_dir, variations_file=None, equal_variations=False):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    variation_params_map = {}
    n_expected_variations = None
    
    if variations_file:
        variations_file_path = Path(variations_file)
        if not variations_file_path.exists():
            print(f"Error: Variations file '{variations_file}' not found!")
            sys.exit(1)
            
        variation_params_map = parse_variations_file(variations_file_path)
        
        if variation_params_map:
            print(f"Loaded variation parameters from {variations_file}:")
            for param_name, values in variation_params_map.items():
                print(f"  {param_name}: {values}")
                if n_expected_variations is None:
                    n_expected_variations = len(values)
                elif n_expected_variations != len(values):
                    print(f"Error: All parameters in variations file must have the same number of values!")
                    sys.exit(1)
    
    if input_path.is_file():
        yoda_files_to_process = [(None, input_path)]
        print(f"Processing single YODA file: {input_path.name}")
    elif input_path.is_dir():
        direct_yoda_files = find_yoda_files(input_path)
        subdirs = sorted([d for d in input_path.iterdir() if d.is_dir()])
        
        if direct_yoda_files and not subdirs:
            if len(direct_yoda_files) > 1:
                print(f"Warning: Multiple YODA files found in flat directory, using first one: {direct_yoda_files[0].name}")
            yoda_files_to_process = [(None, direct_yoda_files[0])]
            print(f"Processing flat directory with YODA file: {direct_yoda_files[0].name}")
        elif subdirs:
            yoda_files_to_process = []
            for subdir in sorted(subdirs, key=lambda d: d.name):
                yoda_files = find_yoda_files(subdir)
                if yoda_files:
                    yoda_files_to_process.append((subdir, yoda_files[0]))
            
            if not yoda_files_to_process:
                print(f"Error: No YODA files found in subdirectories of {input_dir}!")
                sys.exit(1)
            
            print(f"Found {len(yoda_files_to_process)} subdirectories with YODA files in {input_dir}")
        else:
            print(f"Error: No YODA files or subdirectories found in {input_dir}!")
            sys.exit(1)
    else:
        print(f"Error: Input '{input_dir}' is neither a file nor a directory!")
        sys.exit(1)
    
    parsed_yoda_entries = []
    for subdir, yoda_file in yoda_files_to_process:
        try:
            variations = parse_yoda_file(yoda_file, variation_pattern)
        except Exception as e:
            print(f"Error: Failed to parse YODA file '{yoda_file}': {e}")
            sys.exit(1)

        var_numbers = get_variation_numbers(variations)
        if not var_numbers:
            print(f"Error: No variations matching pattern '{variation_pattern}' found in YODA file '{yoda_file}'!")
            sys.exit(1)
        parsed_yoda_entries.append((subdir, yoda_file, variations, var_numbers))

    first_var_numbers = parsed_yoda_entries[0][3]
    n_variations_first = len(first_var_numbers)
    total_detected_variations = sum(len(entry[3]) for entry in parsed_yoda_entries)
    N = len(parsed_yoda_entries)

    if equal_variations:
        for subdir, yoda_file, _, var_numbers in parsed_yoda_entries:
            if len(var_numbers) != n_variations_first:
                print(
                    f"Error: --equal-variations requires all YODA files to have the same number of variations. "
                    f"Expected {n_variations_first} from '{parsed_yoda_entries[0][1].name}', "
                    f"but '{yoda_file.name}' has {len(var_numbers)}."
                )
                sys.exit(1)

        print(f"Detected {n_variations_first} variations per YODA file (excluding v0): {['v' + str(v) for v in first_var_numbers]}")
    else:
        print(f"Detected {total_detected_variations} total variations across {N} YODA files (excluding v0)")

    if n_expected_variations is not None:
        if equal_variations:
            if n_expected_variations != n_variations_first:
                print(
                    f"Error: Variations file specifies {n_expected_variations} variations, "
                    f"but --equal-variations mode expects {n_variations_first} (per YODA file, excluding v0)!"
                )
                sys.exit(1)
        else:
            if n_expected_variations != total_detected_variations:
                print(
                    f"Error: Variations file specifies {n_expected_variations} variations, "
                    f"but YODA files contain {total_detected_variations} total variations (excluding v0)!"
                )
                sys.exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    if equal_variations:
        total_output_dirs = N * n_variations_first
    else:
        total_output_dirs = total_detected_variations

    num_digits = len(str(total_output_dirs - 1)) + 1

    param_offset = 0
    sequential_output_idx = 0

    for idx, (subdir, yoda_file, variations, var_numbers) in enumerate(parsed_yoda_entries):
        if subdir is not None:
            subdir_name = subdir.name
            print(f"Processing {subdir_name}...")
            yaml_files = list(subdir.glob("*.yaml")) + list(subdir.glob("*.yml"))
            yaml_file = yaml_files[0] if yaml_files else None
            params_file = subdir / "params.dat"
        else:
            subdir_name = yoda_file.stem
            print(f"Processing {yoda_file.name}...")
            yaml_file = None
            params_file = input_path.parent / "params.dat" if input_path.is_file() else input_path / "params.dat"
            if not params_file.exists():
                params_file = None

        for var_idx, var_num in enumerate(var_numbers):
            variation_key = f'v{var_num}'

            if equal_variations:
                new_idx = idx + var_idx * N
                params_idx = var_idx
            else:
                new_idx = sequential_output_idx
                params_idx = param_offset + var_idx
                sequential_output_idx += 1

            new_subdir_name = f"{new_idx:0{num_digits}d}"
            new_subdir_path = output_path / new_subdir_name
            new_subdir_path.mkdir(parents=True, exist_ok=True)
            
            yd = variations.get(variation_key, [])
            if not yd:
                print(f"  Warning: No analysis objects found for variation v{var_num}")
                continue
            
            new_yoda_name = f"{new_idx:0{num_digits}d}.yoda"
            new_yoda_path = new_subdir_path / new_yoda_name
            write_yoda_file(new_yoda_path, yd)
            
            if yaml_file and yaml_file.exists():
                new_yaml_path = new_subdir_path / yaml_file.name
                shutil.copy2(yaml_file, new_yaml_path)
            
            if params_file and params_file.exists() or variation_params_map:
                new_params_path = new_subdir_path / "params.dat"
                var_params = {}
                for param_name, param_values in variation_params_map.items():
                    if params_idx < len(param_values):
                        var_params[param_name] = param_values[params_idx]
                
                src_params = params_file if params_file and params_file.exists() else Path("/dev/null")
                copy_and_extend_params_file(src_params if src_params.exists() else Path(), new_params_path, var_params)

            if subdir is not None:
                source_str = f"{subdir.name}/{yoda_file.name}:v{var_num}"
            else:
                source_str = f"{yoda_file.name}:v{var_num}"

            if variation_params_map:
                point_values = []
                for param_name in variation_params_map.keys():
                    param_values = variation_params_map[param_name]
                    if params_idx < len(param_values):
                        point_values.append(param_values[params_idx])
                point_str = ", ".join(f"{v:.6g}" for v in point_values)
                print(f"  Created {new_subdir_name}/{new_yoda_name} with {len(yd)} observables "
                      f"from variation v{var_num} (point: [{point_str}], source: {source_str})")
            else:
                print(f"  Created {new_subdir_name}/{new_yoda_name} with {len(yd)} observables "
                      f"from variation v{var_num} (source: {source_str})")
        if not equal_variations:
            param_offset += len(var_numbers)

    print(f"\nDone! Created {total_output_dirs} subdirectories in {output_dir}\n")
    return


def main():
    parser = argparse.ArgumentParser(
        description='Split variations in YODA files in separate files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  split_reweighted_runs.py newscan "MPI.v" -o output_newscan
  split_reweighted_runs.py newscan "MPI.v" -o output_newscan --variations variations.dat
  split_reweighted_runs.py reweighted.yoda "MPI.v" -o output_newscan
  
This will process YODA file(s) in 'newscan/', split each by variations 
(e.g., [MPI.v0], [MPI.v1], [MPI.v2]), and create a new directory structure 
in 'output_newscan/' with N * N_variations subdirectories.

Supports three input structures:
  1. Single YODA file: newscan.yoda
  2. Flat directory: newscan/ containing newscan.yoda
  3. Nested directory: newscan/ containing 00/, 01/, etc. with YODA files

If --variations is provided, params.dat files will be extended with the
corresponding variation parameter values.
        """
    )
    
    parser.add_argument('input', type=str, help='Input YODA file or directory containing YODA files')
    parser.add_argument('pattern', type=str, help='Variation pattern to match (e.g., "MPI.v" for [MPI.v0], [MPI.v1], etc.)')
    parser.add_argument('-o', '--outdir', type=str, default=None, help='Output directory for split YODA files (default: input + ".split")')
    parser.add_argument('-v', '--variations', type=str, default=None, help='Path to variations.dat file containing parameter variations')
    parser.add_argument('--equal-variations', action='store_true', help='Assume each input YODA file has the same variation list and start variations.dat indexing from 0 for each file')
    parser.add_argument('--overwrite', action='store_true', help='Skip overwrite confirmation if output directory exists')
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input '{args.input}' does not exist!")
        sys.exit(1)

    if args.outdir is None:
        if os.path.isfile(args.input):
            args.outdir = Path(args.input).stem + '.split'
        else:
            args.outdir = args.input.rstrip('/') + '.split'
    
    if args.variations and not os.path.isfile(args.variations):
        print(f"Error: Variations file '{args.variations}' does not exist!")
        sys.exit(1)
    
    if os.path.exists(args.outdir):
        if not args.overwrite:
            response = input(f"Warning: Output directory '{args.outdir}' already exists. Continue and overwrite? [y/N] ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(1)
        if os.path.isdir(args.outdir):
            shutil.rmtree(args.outdir)
        else:
            os.remove(args.outdir)
    
    split_yodas(args.input, args.pattern, args.outdir, args.variations, args.equal_variations)
    return 0


if __name__ == '__main__':
    exit(main())
