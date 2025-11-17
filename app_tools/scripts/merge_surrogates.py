
import sys
import os
import json
import shutil
import argparse


def merge_jsons_in_dir(json_dir, keep_dir=False, out_file=None):
    files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    files = [f for f in files if f[:-5].isdigit()]
    
    if len(files) == 0:
        print(f"Error: No numbered JSON files found in {json_dir}!")
        sys.exit(1)
    
    files.sort(key=lambda x: int(x[:-5]))
    n = len(files)

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
                print(f"Warning: Unexpected JSON structure in {fname}.")

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
    print(f"Merged {n} json files into {out_file}.")

    if os.path.exists(json_dir) and not keep_dir:
        shutil.rmtree(json_dir)
        print(f"Deleted directory {json_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Merge multiple surrogate JSON files into a single file.")
    parser.add_argument("input_dir", help="Directory containing the JSON files to merge")
    parser.add_argument("--keep_dir", action="store_true", help="Keep the input JSON directory after merging")
    parser.add_argument("-o", "--output", help="Name for output file (default: <directory_name>.json)")
    args = parser.parse_args()

    if not os.path.exists(args.input_dir):
        print(f"Error: Directory '{args.input_dir}' does not exist!")
        sys.exit(1)

    if not os.path.isdir(args.input_dir):
        print(f"Error: '{args.input_dir}' is not a directory!")
        sys.exit(1)

    merge_jsons_in_dir(args.input_dir, args.keep_dir, args.output)


if __name__ == "__main__":
    main()
