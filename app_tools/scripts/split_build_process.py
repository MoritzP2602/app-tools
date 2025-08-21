
import sys
import os
import json
import shutil
import argparse


def is_bin_specific(line):
    return '#' in line.split()[0]


def merge_jsons_in_dir(json_dir, n, stop_rm=False, out_file=None):
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    files = [f for f in files if f[:-5].isdigit() and 0 <= int(f[:-5]) < n]
    files.sort(key=lambda x: int(x[:-5]))

    if len(files) != n:
        print(f"Error: Expected {n} json files, found {len(files)} in {json_dir}.")
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

    if out_file is None:
        out_file = os.path.basename(os.path.normpath(json_dir)) + ".json"
    else:
        if not out_file.endswith('.json'):
            out_file += ".json"

    with open(out_file, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {n} json files into {out_file}")

    if os.path.exists(json_dir) and not stop_rm:
        shutil.rmtree(json_dir)
        print(f"Deleted directory {json_dir}")


def split_weight_file(input_path, n, out_dir=None):
    if out_dir is None:
        out_dir = "weight_files"
    if os.path.exists(out_dir):
        print(f"Error: Output directory '{out_dir}' already exists, please remove it or choose a different name")
        sys.exit(1)
    os.makedirs(out_dir)

    with open(input_path) as f:
        lines = [line for line in f if line.strip() and not is_bin_specific(line)]

    lines.sort()

    if not lines:
        print("No matching lines found.")
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
            rel = os.path.relpath(path, out_dir)
            f.write(rel + "\n")

    print(f"Done. Wrote {n} files to {out_dir} and listed them in {weight_file}.")


def main():
    parser = argparse.ArgumentParser(description="Split weight files or merge JSON files.")
    parser.add_argument("input_path", help="Path to the weight file OR directory of JSONs")
    parser.add_argument("n", type=int, help="Number of output files (for splitting) or number of JSONs to merge (for merging)")
    parser.add_argument("--mode", choices=["split", "merge"], help="Mode of operation (default: depends on input)")
    parser.add_argument("--stop_rm", action="store_true", help="Don't remove the JSON directory after merging")
    parser.add_argument("-o", "--output", help="Name for output directory/.txt file")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"Error: Path '{args.input_path}' does not exist")
        sys.exit(1)

    if args.n <= 0:
        print("Error: n must be a positive integer")
        sys.exit(1)

    if os.path.isdir(args.input_path):
        if args.mode == "split":
            print("Error: Splitting requires weight file as input")
            sys.exit(1)
        merge_jsons_in_dir(args.input_path, args.n, args.stop_rm, args.output)
    else:
        if args.mode == "merge":
            print(f"Error: Merging requires a directory of JSON files as input")
            sys.exit(1)
        if args.stop_rm:
            print(f"Warning: --stop_rm ignored in splitting mode")
        split_weight_file(args.input_path, args.n, args.output)


if __name__ == "__main__":
    main()
