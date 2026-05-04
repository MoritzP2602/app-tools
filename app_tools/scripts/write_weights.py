
import argparse
import os
import yoda
import rivet


class yodaLoader:
    paths = rivet.getAnalysisRefPaths()
    yodas = {}

    def load(self, analysis_name):
        if analysis_name in self.yodas:
            return self.yodas[analysis_name]
        for base_path in self.paths:
            gz_path = f"{base_path}/{analysis_name}.yoda.gz"
            yoda_path = f"{base_path}/{analysis_name}.yoda"
            if os.path.exists(gz_path):
                ref_yoda = yoda.readYODA(gz_path)
                self.yodas[analysis_name] = ref_yoda
                return ref_yoda
            elif os.path.exists(yoda_path):
                ref_yoda = yoda.readYODA(yoda_path)
                self.yodas[analysis_name] = ref_yoda
                return ref_yoda
        raise FileNotFoundError(f"No reference YODA file found for {analysis_name} in any Rivet path")


def main():
    parser = argparse.ArgumentParser(
        description="Extract observables from YODA file and write weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  write_weights.py output.yoda -o weights.txt
  write_weights.py output.yoda --bins -o weights_by_bin.txt

This script reads a YODA file and extracts all observables, writing a weight
file with one weight entry per observable (or per bin if --bins is used).

Output format (default):
  /ATLAS_Analysis/d01-x01-y01 1.0  # bins: 20
  /ATLAS_Analysis/d02-x01-y01 1.0  # bins: 15
  ...

Output format (with --bins):
  /ATLAS_Analysis/d01-x01-y01#0 1.0  # bin 1
  /ATLAS_Analysis/d01-x01-y01#1 1.0  # bin 2
  /ATLAS_Analysis/d01-x01-y01#2 1.0  # bin 3
  ...

Warnings are printed for observables that cannot be found in Rivet references.
        """
    )
    parser.add_argument("yoda_file", help="YODA file to process")
    parser.add_argument("-o", "--output", default="weights.txt", help="Output weights file name (default: weights.txt)")
    parser.add_argument("--bins", action="store_true", help="List all bins of all observables instead of just observables")
    
    args = parser.parse_args()

    loader = yodaLoader()
    
    try:
        yd = yoda.readYODA(args.yoda_file)
    except Exception as e:
        print(f"Error: Could not read YODA file {args.yoda_file}: {e}!")
        return
    
    observables = []
    ignored_count = 0
    
    for obs in yd:
        try:
            analysis_name = obs.split("/")[1].split(":")[0]
        except (IndexError, AttributeError):
            ignored_count += 1
            continue
        
        try:
            ref_yoda = loader.load(analysis_name)
            obs_ref = f"/REF/{analysis_name}/{obs.split('/')[-1]}"
            ref_obj = ref_yoda.get(obs_ref)
            
            if ref_obj is None:
                ignored_count += 1
                continue
            
            if hasattr(ref_obj, 'bins') and callable(ref_obj.bins):
                bin_count = len(ref_obj.bins())
            elif hasattr(ref_obj, 'points') and callable(ref_obj.points):
                bin_count = len(ref_obj.points())
            elif hasattr(ref_obj, 'numBins') and callable(ref_obj.numBins):
                bin_count = ref_obj.numBins()
            else:
                bin_count = 1
            
            observables.append((obs, bin_count))
            
        except (FileNotFoundError, KeyError):
            ignored_count += 1
            continue
        except Exception as e:
            ignored_count += 1
            continue
    
    if ignored_count > 0:
        print(f"Warning: Ignored {ignored_count} observables (no Rivet reference available).")

    total_bins = sum(bins for _, bins in observables)

    with open(args.output, "w") as out:
        if args.bins:
            for path, bins in observables:
                for bin_idx in range(bins):
                    out.write(f"{path}#{bin_idx} 1.0  # bin {bin_idx + 1}\n")
        else:
            for path, bins in observables:
                out.write(f"{path} 1.0  # bins: {bins}\n")

    print(f"Total number of observables: {len(observables)}")
    print(f"Total number of bins: {total_bins}")
    print(f"\nOutput written to: {args.output}\n")


if __name__ == "__main__":
    main()
