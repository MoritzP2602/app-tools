
import sys
import os
import argparse


def is_bin_specific(line):
    return '#' in line.split()[0]


def split_weight_file(input_path, n, out_dir=None):
    if out_dir is None:
        out_dir = "weight_files"
    if os.path.exists(out_dir):
        print(f"Error: Output directory '{out_dir}' already exists, please remove it or choose a different name!")
        sys.exit(1)
    os.makedirs(out_dir)

    with open(input_path) as f:
        lines = [line for line in f if line.strip() and not is_bin_specific(line)]

    lines.sort()

    if not lines:
        print("Error: No matching lines found!")
        sys.exit(1)

    total = len(lines)
    base = total // n
    rem = total % n
    chunks = []
    start = 0
    for i in range(n):
        end = start + base + (1 if i < rem else 0)
        chunks.append(lines[start:end])
        start = end

    out_paths = []
    num_digits = len(str(n - 1))
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(out_dir, f"{i:0{num_digits}d}")
        with open(out_path, "w") as f:
            f.writelines(chunk)
        out_paths.append(out_path)

    weight_file = out_dir + ".txt"
    with open(weight_file, "w") as f:
        for path in out_paths:
            f.write(path + "\n")

    print(f"Done. Wrote {n} files to {out_dir} and listed them in {weight_file}.")


def main():
    parser = argparse.ArgumentParser(description="Split weight files into multiple files for parallel processing.")
    parser.add_argument("input_file", help="Path to the weight file")
    parser.add_argument("n", type=int, help="Number of output files")
    parser.add_argument("-o", "--output", help="Name for output directory (default: weight_files)")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: File '{args.input_file}' does not exist!")
        sys.exit(1)

    if os.path.isdir(args.input_file):
        print(f"Error: '{args.input_file}' is a directory, not a file!")
        sys.exit(1)

    if args.n <= 0:
        print("Error: n must be a positive integer!")
        sys.exit(1)

    split_weight_file(args.input_file, args.n, args.output)


if __name__ == "__main__":
    main()
