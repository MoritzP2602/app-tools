
import argparse
import glob
import math
import os
import re


def read_weights(path, scale):
    weights = {}
    order = []
    with open(path) as f:
        for line in f:
            if line.strip() == "" or line.strip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            key = parts[0]
            try:
                value = float(parts[1]) * scale
            except ValueError:
                continue
            comment = " ".join(parts[2:]) if len(parts) > 2 else ""
            if key not in weights:
                order.append(key)
            weights[key] = (value, comment)
    return weights, order


def read_objective_value_from_tune_dir(tune_dir):
    minimum_files = sorted(glob.glob(os.path.join(tune_dir, "minimum_*.txt")))
    if not minimum_files:
        raise FileNotFoundError(f"No file matching 'minimum_*.txt' found in directory: {tune_dir}")

    minimum_file = minimum_files[0]
    with open(minimum_file) as f:
        for line in f:
            match = re.search(r"Objective value at best fit point:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
            if match: return float(match.group(1)), minimum_file

    raise ValueError(f"Could not find 'Objective value at best fit point' in file: {minimum_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Scale and combine weights files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  combine_weights.py weights1.txt 1.0 weights2.txt 2.5 -o combined.txt
  combine_weights.py w1.txt 0.5 w2.txt 1.5 w3.txt 2.0
  combine_weights.py weights1.txt tune_dir1 weights2.txt tune_dir2

This script reads multiple weight files, scales each, and combines them into
a single output file. Each weight line format is:
  observable_name weight_value [optional_comment]

Input mode A (manual scales):
    file1 scale1 [file2 scale2 ...]

Input mode B (auto scales from tune directories):
    file1 tune_dir1 [file2 tune_dir2 ...]
    For each tune directory, the script reads minimum_*.txt and extracts:
        Objective value at best fit point: X
    It then computes scale factors as:
        sqrt(min(X_i) / X_i)
    so the smallest objective value gets scale 1.0 automatically.

Scaling example:
  Input file 1:
    /ATLAS_Analysis/d01-x01-y01 1.0  # bins: 20
    /ATLAS_Analysis/d02-x01-y01 1.0  # bins: 15
    ...
  Input file 2:
    /CMS_Analysis/d07-x01-y01 1.0  # bins: 12
    /CMS_Analysis/d08-x01-y01 1.0  # bins: 18
    ...
  
  Command: combine_weights.py file1.txt 2.0 file2.txt 0.5 -o out.txt
  
  Output:
    /ATLAS_Analysis/d01-x01-y01 2.0  # bins: 20
    /ATLAS_Analysis/d02-x01-y01 2.0  # bins: 15
    ...
    /CMS_Analysis/d07-x01-y01 0.5  # bins: 12
    /CMS_Analysis/d08-x01-y01 0.5  # bins: 18
    ...
        """
    )
    parser.add_argument("files_and_scales", nargs="+", help="Alternating file and scale pairs: file1 scale1 [file2 scale2 ...]")
    parser.add_argument("-o", "--output", default="weights.txt", help="Output file name (default: weights.txt)")
    
    args = parser.parse_args()
    
    if len(args.files_and_scales) % 2 != 0:
        parser.error("Arguments must be provided in pairs: file1 scale1 [file2 scale2 ...]")

    manual_pairs = []
    tune_pairs   = []
    for i in range(0, len(args.files_and_scales), 2):
        file_path  = args.files_and_scales[i]
        second_arg = args.files_and_scales[i + 1]
        try:
            scale = float(second_arg)
            manual_pairs.append((file_path, scale))
        except ValueError:
            try:
                objective_value, minimum_file = read_objective_value_from_tune_dir(second_arg)
            except (FileNotFoundError, ValueError) as err:
                parser.error(str(err))
            tune_pairs.append((file_path, second_arg, objective_value, minimum_file))
    if manual_pairs and tune_pairs:
        parser.error("Do not mix input modes. Use either file+scale pairs or file+tune_dir pairs.")
    if tune_pairs:
        min_objective = min(obj for _, _, obj, _ in tune_pairs)
        file_scale_pairs = [(file_path, math.sqrt(min_objective / objective_value))
                            for file_path, _, objective_value, _ in tune_pairs]
        print("Scale factors from tune directories:")
        for (file_path, _, objective_value, minimum_file), (_, scale) in zip(tune_pairs, file_scale_pairs):
            print(f"  {file_path}: objective = {objective_value:.6f}, "
                  f"scale = {scale:.6f} (from {minimum_file})")
    else: file_scale_pairs = manual_pairs
    
    all_seen_keys = set()
    with open(args.output, "w") as out:
        for file_path, scale in file_scale_pairs:
            weights, order = read_weights(file_path, scale)
            for key in order:
                if key in all_seen_keys:
                    print(f"Warning: Weight '{key}' appears in multiple files.")
                else: all_seen_keys.add(key)
                value, comment = weights[key]
                out.write(f"{key} {value:.3f} {comment}\n")
    print(f"\nOutput written to: {args.output}.\n")


if __name__ == "__main__":
    main()