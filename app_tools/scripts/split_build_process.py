
import sys
import os
import json
import shutil
import argparse


def is_bin_specific(line):
    return '#' in line.split()[0]


def merge_jsons_in_dir(json_dir, N):
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    files = [f for f in files if f[:-5].isdigit() and 0 <= int(f[:-5]) < N]
    files.sort(key=lambda x: int(x[:-5]))
    
    if len(files) != N:
        print(f"Error: Expected {N} json files, found {len(files)} in {json_dir}.")
        sys.exit(1)

    merged = {}
    for fname in files:
        with open(os.path.join(json_dir, fname)) as f:
            data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in merged:
                        if isinstance(merged[k], list) and isinstance(v, list):
                            merged[k].extend(v)
                        else:
                            merged[k] = v
                    else:
                        merged[k] = v
            elif isinstance(data, list):
                merged.setdefault("__list__", []).extend(data)
            else:
                print(f"Warning: Unexpected JSON structure in {fname}")

    xmin = merged.pop("__xmin", None)
    xmax = merged.pop("__xmax", None)

    if xmin is not None:
        merged["__xmin"] = xmin
    if xmax is not None:
        merged["__xmax"] = xmax

    outname = os.path.basename(os.path.normpath(json_dir)) + ".json"
    with open(outname, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {N} json files into {outname}")

    if os.path.exists(json_dir):
        shutil.rmtree(json_dir)
        print(f"Deleted directory {json_dir}")


def split_weight_file(input_path, N):
    out_dir = "weight_files"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
        print(f"Deleted existing directory '{out_dir}'.")
    os.makedirs(out_dir)

    with open(input_path) as f:
        lines = [line for line in f if line.strip() and not is_bin_specific(line)]

    lines.sort()

    if not lines:
        print("No matching lines found.")
        sys.exit(1)

    total = len(lines)
    base = total // N
    rem = total % N
    chunks = []
    start = 0
    for i in range(N):
        end = start + base + (1 if i < rem else 0)
        chunks.append(lines[start:end])
        start = end

    out_paths = []
    num_digits = len(str(N - 1))
    for i, chunk in enumerate(chunks):
        out_path = os.path.join(out_dir, f"{i:0{num_digits}d}")
        with open(out_path, "w") as f:
            f.writelines(chunk)
        out_paths.append(out_path)

    with open("weight_files.txt", "w") as f:
        for path in out_paths:
            f.write(path + "\n")

    print(f"Done. Wrote {N} files to {out_dir} and listed them in weight_files.txt.")


def main():
    parser = argparse.ArgumentParser(
        description="Split weight files or merge JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Split weight file:   app-tools-split_build_process weights.txt 10
  Merge JSON files:    app-tools-split_build_process json_dir/ 10

If input_path is a file, splits the weight file into N alphabetically sorted files (ignoring bin-specific weights).
If input_path is a directory, merges N JSON files (named 00.json, 01.json, ..., N-1.json) in that directory into one JSON named <directory>.json.
        """
    )
    parser.add_argument("input_path", help="Path to the weight file OR directory of JSONs")
    parser.add_argument("N", type=int, help="Number of output files (for splitting) or number of JSONs to merge (for merging)")
    
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Path '{args.input_path}' does not exist")
        sys.exit(1)

    if args.N <= 0:
        print("Error: N must be a positive integer")
        sys.exit(1)

    if os.path.isdir(args.input_path):
        merge_jsons_in_dir(args.input_path, args.N)
    else:
        split_weight_file(args.input_path, args.N)


if __name__ == "__main__":
    main()
