
import argparse


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


def main():
    parser = argparse.ArgumentParser(description="Scale and combine weights files")
    parser.add_argument("files_and_scales", nargs="+", help="Alternating file and scale pairs: file1 scale1 [file2 scale2 ...]")
    parser.add_argument("-o", "--output", default="weights.txt", help="Output file name (default: weights.txt)")
    
    args = parser.parse_args()
    
    if len(args.files_and_scales) % 2 != 0:
        parser.error("Arguments must be provided in pairs: file1 scale1 [file2 scale2 ...]")
    
    file_scale_pairs = []
    for i in range(0, len(args.files_and_scales), 2):
        file_path = args.files_and_scales[i]
        try:
            scale = float(args.files_and_scales[i + 1])
        except ValueError:
            parser.error(f"Scale factor '{args.files_and_scales[i + 1]}' is not a valid number")
        file_scale_pairs.append((file_path, scale))
    
    all_seen_keys = set()
    with open(args.output, "w") as out:
        for file_path, scale in file_scale_pairs:
            weights, order = read_weights(file_path, scale)
            for key in order:
                if key in all_seen_keys:
                    print(f"Warning: Weight '{key}' appears in multiple files")
                else:
                    all_seen_keys.add(key)
                value, comment = weights[key]
                out.write(f"{key} {value:.6f} {comment}\n")


if __name__ == "__main__":
    main()