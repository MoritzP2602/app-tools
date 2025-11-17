
import argparse
import os
import yoda
import rivet


class yodaLoader:
    paths = rivet.getAnalysisRefPaths()
    yodas = {}

    def load(self, analysis_name):
        base_path = self.paths[0]
        gz_path = f"{base_path}/{analysis_name}.yoda.gz"
        yoda_path = f"{base_path}/{analysis_name}.yoda"
        if analysis_name not in self.yodas.keys():
            if os.path.exists(gz_path):
                ref_yoda = yoda.readYODA(gz_path)
            elif os.path.exists(yoda_path):
                ref_yoda = yoda.readYODA(yoda_path)
            else:
                raise FileNotFoundError(f"Neither {gz_path} nor {yoda_path} found")
            self.yodas[analysis_name] = ref_yoda
            return ref_yoda
        else:
            return self.yodas[analysis_name]


def main():
    parser = argparse.ArgumentParser(description="Extract observables from YODA file and write weights")
    parser.add_argument("yoda_file", help="YODA file to process")
    parser.add_argument("-o", "--output", default="weights.txt", help="Output weights file name (default: weights.txt)")
    
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
        for path, bins in observables:
            out.write(f"{path} 1.0  # bins: {bins}\n")

    print(f"Total number of observables: {len(observables)}")
    print(f"Total number of bins: {total_bins}")


if __name__ == "__main__":
    main()
