
import argparse


def extract_observables(yoda_file):
    observables = []
    current_path = None
    bin_count = 0

    with open(yoda_file, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BEGIN"):
                current_path = None
                bin_count = 0
            elif line.startswith("Path:"):
                current_path = line.split(":", 1)[1].strip()
            elif line.startswith("# xlow"):
                bin_count = 0
            elif current_path and line and not line.startswith("#") and not line.startswith("END"):
                bin_count += 1
            elif line.startswith("END") and current_path:
                observables.append((current_path, bin_count))
                current_path = None

    return observables


def main():
    parser = argparse.ArgumentParser(description="Extract observables from YODA file and write weights")
    parser.add_argument("yoda_file", help="YODA file to process")
    parser.add_argument("-o", "--output", default="weights.txt", help="Output weights file name (default: weights.txt)")
    
    args = parser.parse_args()

    all_observables = extract_observables(args.yoda_file)
    
    excluded_patterns = ["/REF", "/RAW", "/MC", "/tmp", "/sum", "/total", "[", "/_"]
    observables = []
    ignored_count = 0
    
    for path, bins in all_observables:
        if any(pattern in path for pattern in excluded_patterns):
            ignored_count += 1
        else:
            observables.append((path, bins))
    
    if ignored_count > 0:
        print(f"Ignored {ignored_count} observables containing: {', '.join(excluded_patterns)}")

    total_bins = sum(bins for _, bins in observables)

    with open(args.output, "w") as out:
        for path, bins in observables:
            out.write(f"{path} 1.0  # bins: {bins}\n")

    print(f"Total number of bins: {total_bins}")


if __name__ == "__main__":
    main()
