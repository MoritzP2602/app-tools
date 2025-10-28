
import os
import yoda
import rivet
import argparse
import datetime
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import matplotlib as mpl
import shutil
from pathlib import Path
from tabulate import tabulate

if tuple(map(int, yoda.__version__.split("."))) < (2, 1, 0):
    print("[WARNING] This script is optimized for YODA 2.1.0. You are using YODA", yoda.__version__, "... please double-check your results!\n")


def read_weights(wfile):
    """
    Read weights from a file.

    :param wfile: Path to the weights file. Format should be: OBSERVABLE_NAME#BIN_NAME weight
    :return: Dictionary of weights where keys are observable/bin names and values are their weights.
    """
    weights = {}
    with open(wfile, "r") as f:
        lines = f.readlines()
    for l in lines:
        if l.startswith("#"):
            continue
        else:
            ls = l.split()
            weights[ls[0]] = float(ls[1])
    return weights


def read_analyses(afile):
    """
    Read list of analyses from a file.

    :param afile: Path to the analyses file. Each line should contain one analysis name.
    :return: Set of analysis names to include.
    """
    analyses = set()
    with open(afile, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#"):
            analyses.add(line)
    return analyses


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

    def extract_bins_scatter(self, obj):
        """
        Extract bin names, y values, and y errors for Histo1D and Scatter objects using mkScatter.
        Returns: (bin_names, y_values, y_errors)
        """
        ynew = obj.mkScatter()
        path = obj.path() if callable(obj.path) else obj.path
        bin_names = []
        y_values = []
        y_errors = []
        for i, p in enumerate(ynew.points()):
            bin_name = f"{path}#{i}"
            if p.dim() == 1:
                y_val = p.x()
                y_err = (p.xErrs()[0] + p.xErrs()[1]) / 2.0
            elif p.dim() == 2:
                y_val = p.y()
                y_err = (p.yErrs()[0] + p.yErrs()[1]) / 2.0
            else:
                continue
            bin_names.append(bin_name)
            y_values.append(y_val)
            y_errors.append(y_err)
        return bin_names, y_values, y_errors

    def extract_bins_profile1d(self, obj):
        """
        Extract bin names, y values, and y errors for Profile1D objects.
        Returns: (bin_names, y_values, y_errors)
        """
        path = obj.path() if callable(obj.path) else obj.path
        bin_names = []
        y_values = []
        y_errors = []
        for i, bin in enumerate(obj.bins()):
            bin_name = f"{path}#{i}"
            y_val = bin.yMean() if hasattr(bin, "yMean") and callable(bin.yMean) else np.nan
            y_err = bin.yStdErr() if hasattr(bin, "yStdErr") and callable(bin.yStdErr) else np.nan
            bin_names.append(bin_name)
            y_values.append(y_val)
            y_errors.append(y_err)
        return bin_names, y_values, y_errors

            ### --- todo --- ###
            # Fix the extraction of y values and errors for Profile1D objects for YODA < 2.0.0
            # some bins contribute significantly to the chi2 by using the scatter method, but have
            # y_err = 0 with the Profile1D method in YODA-2.1.0.
            # IDEA: For YODA < 2, compute mean and error manually (but this still shows mismatch in y_err values!)

            # yoda_version = tuple(map(int, yoda.__version__.split(".")))
            # if yoda_version[0] < 2:
            #     sumw = bin.sumW() if hasattr(bin, "sumW") else bin.sumw
            #     sumwy = bin.sumWY() if hasattr(bin, "sumWY") else bin.sumwy
            #     sumwy2 = bin.sumWY2() if hasattr(bin, "sumWY2") else bin.sumwy2
            #     numEntries = bin.numEntries() if hasattr(bin, "numEntries") else bin.numentries if hasattr(bin, "numentries") else None
            #     # Compute mean
            #     y_val = sumwy / sumw if sumw != 0 else np.nan
            #     if numEntries is None or numEntries < 50:
            #         y_err = np.nan
            #     else:
            #         # Compute error on the mean
            #         if sumw > 0 and numEntries and numEntries > 1:
            #             variance = (sumwy2 / sumw) - (y_val ** 2)
            #             y_err = np.sqrt(variance / numEntries) if variance > 0 else 0.0
            #         else:
            #             y_err = np.nan

            ### ----------- ###

    def extract_bins_estimate(self, obj):
        """
        Extract bin names, y values, and y errors for Estimate1D/BinnedEstimate1D objects.
        Returns: (bin_names, y_values, y_errors)
        """
        path = obj.path() if callable(obj.path) else obj.path
        bin_names = []
        y_values = []
        y_errors = []
        for i, b in enumerate(obj.bins()):
            bin_name = f"{path}#{i}"
            y_val = b.val()
            if hasattr(b, "sources"):
                error_labels = b.sources()
            else:
                error_labels = []
            err_dn = 0.0
            err_up = 0.0
            for label in error_labels:
                try:
                    err_dn += abs(b.errNeg(label))**2
                except Exception:
                    pass
                try:
                    err_up += abs(b.errPos(label))**2
                except Exception:
                    pass
            err_dn = np.sqrt(err_dn)
            err_up = np.sqrt(err_up)
            y_err = (err_dn + err_up) / 2.0
            bin_names.append(bin_name)
            y_values.append(y_val)
            y_errors.append(y_err)
        return bin_names, y_values, y_errors

    def extract_estimate0d(self, obj):
        """
        Extract bin name, value, and error for Estimate0D objects.
        Returns: (bin_names, y_values, y_errors)
        """
        bin_names = [obj.path() if callable(obj.path) else obj.path]
        y_values = [obj.val()]
        try:
            y_err = [obj.errAvg()]
        except Exception:
            y_err = [0.0]
        return bin_names, y_values, y_err

    def get_bin_differences(self, yoda_file, weights=None, analyses=None, include_ref_error=True, debug=False):
        """
        Compute differences between data and reference YODA files.

        :param yoda_file: Path to the YODA file containing data.
        :param weights: Optional weights for the observables.
        :param analyses: Optional set of analysis names to include (if None, all analyses are included).
        :param include_ref_error: Whether to include reference errors in the computation.
        :param debug: Whether to print debug information.
        :return: Tuple of differences, squared errors, bin weights, bin names, and analyses.
        """
        if debug: print(f"Executing get_bin_differences for {yoda_file}")

        yd = yoda.readYODA(yoda_file)
        differences = []
        squared_errors = []
        bin_names = []
        bin_weights = []
        analyses_found = set()
        yoda_version = tuple(map(int, yoda.__version__.split(".")))

        for obs in yd:
            if weights is not None and (obs not in weights and not any(k.startswith(obs + "#") for k in weights)):
                if debug: print(f"  Skipping observable {obs} (not in weights file)")
                continue
            try:
                analysis_name = obs.split("/")[1].split(":")[0]
            except (IndexError, AttributeError):
                if debug: print(f"  Failed to parse analysis name from {obs}")
                continue
            
            if analyses is not None and analysis_name not in analyses:
                if debug: print(f"  Skipping observable {obs} (analysis {analysis_name} not in analyses list)")
                continue
            
            obs_weight = 1.0
            if weights and obs in weights:
                obs_weight = weights[obs]
            
            try:
                ref_yoda = self.load(analysis_name)
                obs_ref = f"/REF/{analysis_name}/{obs.split('/')[-1]}"
                obj = yd[obs]
                ref_obj = ref_yoda.get(obs_ref)
                
                if ref_obj is None:
                    if debug: print(f"  Reference data for {obs} not found, tried loading from {analysis_name}, skipping...")
                    continue
            except (FileNotFoundError, KeyError) as e:
                if debug: print(f"  Error loading reference data from {analysis_name}: {e}")
                continue

            if debug: print(f"  Processing: {obj}, /REF: {ref_obj}")

            if yoda_version[0] < 2:
                data_bins, data_vals, data_errs = self.extract_bins_scatter(obj)
                ref_bins, ref_vals, ref_errs = self.extract_bins_scatter(ref_obj)
            else:
                if isinstance(obj, yoda.Profile1D):
                    data_bins, data_vals, data_errs = self.extract_bins_profile1d(obj)
                elif isinstance(obj, yoda.core.Estimate0D):
                    data_bins, data_vals, data_errs = self.extract_estimate0d(obj)
                elif "Estimate" in type(obj).__name__:
                    data_bins, data_vals, data_errs = self.extract_bins_estimate(obj)
                else:
                    data_bins, data_vals, data_errs = self.extract_bins_scatter(obj)

                if isinstance(ref_obj, yoda.Profile1D):
                    ref_bins, ref_vals, ref_errs = self.extract_bins_profile1d(ref_obj)
                elif isinstance(ref_obj, yoda.core.Estimate0D):
                    ref_bins, ref_vals, ref_errs = self.extract_estimate0d(ref_obj)
                elif "Estimate" in type(ref_obj).__name__:
                    ref_bins, ref_vals, ref_errs = self.extract_bins_estimate(ref_obj)
                else:
                    ref_bins, ref_vals, ref_errs = self.extract_bins_scatter(ref_obj)

            if len(data_bins) != len(ref_bins):
                if debug: print(f"  Data and reference bins do not match in length: {len(data_bins)} vs {len(ref_bins)} for observable {obs}")
                continue
            
            n_bins = min(len(data_bins), len(ref_bins))
            for i in range(n_bins):
                bin_name = data_bins[i]
                if data_errs[i] == 0 or np.isnan(data_errs[i]):
                    if debug: print(f"  Skipping bin {bin_name} due to data error=0 or nan (data_err={data_errs[i]})")
                    continue
                
                diff = data_vals[i] - ref_vals[i]
                if include_ref_error:
                    err = np.sqrt(data_errs[i] ** 2 + ref_errs[i] ** 2)
                else:
                    err = data_errs[i]
                
                bin_weight = obs_weight
                if weights and bin_name in weights:
                    if debug: print(f"  Using weight for bin {bin_name}: {weights[bin_name]}")
                    bin_weight = weights[bin_name]
                
                differences.append(diff)
                squared_errors.append(err ** 2)
                bin_names.append(bin_name)
                bin_weights.append(bin_weight)
            analyses_found.add(analysis_name)
        
        return np.array(differences), np.array(squared_errors), np.array(bin_weights), bin_names, tuple(sorted(analyses_found))
    
    def get_valid_bins(self, up_file, dn_file, weights=None, debug=False):
        """
        Check if reference values are within envelope bounds.
        
        :param up_file: Path to YODA file with maximum values
        :param dn_file: Path to YODA file with minimum values  
        :param weights: Dictionary of weights (optional)
        :param debug: If True, print debug info
        :return: List of valid bin names
        """
        if debug: print(f"Executing get_valid_bins for {up_file}, {dn_file}")

        up_yd = yoda.readYODA(up_file)
        dn_yd = yoda.readYODA(dn_file)
        
        valid_bins = []
        invalid_bins = []
        
        yoda_version = tuple(map(int, yoda.__version__.split(".")))
        
        obs_to_check = weights.keys() if weights else []
        
        for obs_name in obs_to_check:
            if '#' in obs_name:
                continue
                
            try:
                analysis_name = obs_name.split("/")[1].split(":")[0]
                ref_yoda = self.load(analysis_name)
                ref_path = f"/REF/{analysis_name}/{obs_name.split('/')[-1]}"
                
                up_obj = up_yd.get(obs_name)
                dn_obj = dn_yd.get(obs_name)
                ref_obj = ref_yoda.get(ref_path)
                
                if up_obj is None:
                    if debug: print(f"  Up object not found for {obs_name}")
                    continue
                if dn_obj is None:
                    if debug: print(f"  Down object not found for {obs_name}")
                    continue
                if ref_obj is None:
                    if debug: print(f"  Reference object not found for {ref_path}")
                    continue
                
                if yoda_version[0] < 2:
                    up_bins, up_vals, _ = self.extract_bins_scatter(up_obj)
                    dn_bins, dn_vals, _ = self.extract_bins_scatter(dn_obj)
                    ref_bins, ref_vals, _ = self.extract_bins_scatter(ref_obj)
                else:
                    if isinstance(up_obj, yoda.Profile1D):
                        up_bins, up_vals, _ = self.extract_bins_profile1d(up_obj)
                    elif isinstance(up_obj, yoda.core.Estimate0D):
                        up_bins, up_vals, _ = self.extract_estimate0d(up_obj)
                    elif "Estimate" in type(up_obj).__name__:
                        up_bins, up_vals, _ = self.extract_bins_estimate(up_obj)
                    else:
                        up_bins, up_vals, _ = self.extract_bins_scatter(up_obj)
                    
                    if isinstance(dn_obj, yoda.Profile1D):
                        dn_bins, dn_vals, _ = self.extract_bins_profile1d(dn_obj)
                    elif isinstance(dn_obj, yoda.core.Estimate0D):
                        dn_bins, dn_vals, _ = self.extract_estimate0d(dn_obj)
                    elif "Estimate" in type(dn_obj).__name__:
                        dn_bins, dn_vals, _ = self.extract_bins_estimate(dn_obj)
                    else:
                        dn_bins, dn_vals, _ = self.extract_bins_scatter(dn_obj)
                    
                    if isinstance(ref_obj, yoda.Profile1D):
                        ref_bins, ref_vals, _ = self.extract_bins_profile1d(ref_obj)
                    elif isinstance(ref_obj, yoda.core.Estimate0D):
                        ref_bins, ref_vals, _ = self.extract_estimate0d(ref_obj)
                    elif "Estimate" in type(ref_obj).__name__:
                        ref_bins, ref_vals, _ = self.extract_bins_estimate(ref_obj)
                    else:
                        ref_bins, ref_vals, _ = self.extract_bins_scatter(ref_obj)
                
                if len(up_bins) != len(dn_bins) or len(up_bins) != len(ref_bins):
                    if debug: print(f"  Bin count mismatch for {obs_name}: up={len(up_bins)}, dn={len(dn_bins)}, ref={len(ref_bins)}")
                    continue
                
                for i in range(len(up_bins)):
                    bin_name = up_bins[i]
                    up_val = up_vals[i]
                    dn_val = dn_vals[i]
                    ref_val = ref_vals[i]
                    
                    if np.isnan(up_val) or np.isnan(dn_val) or np.isnan(ref_val):
                        if debug: print(f"  NaN value in bin {bin_name}: up={up_val}, dn={dn_val}, ref={ref_val}")
                        continue
                    
                    min_val = min(up_val, dn_val)
                    max_val = max(up_val, dn_val)
                    
                    if min_val <= ref_val <= max_val:
                        valid_bins.append(bin_name)
                    else:
                        invalid_bins.append((bin_name, ref_val, min_val, max_val))
                        
            except (FileNotFoundError, KeyError) as e:
                if debug: print(f"  Error processing {obs_name}: {e}")
                continue
            except Exception as e:
                if debug: print(f"  Unexpected error processing {obs_name}: {e}")
                continue
        
        total_bins = len(valid_bins) + len(invalid_bins)
        if debug: 
            print(f"  Envelope validation: {len(valid_bins)} valid bins, {len(invalid_bins)} invalid bins (total: {total_bins})")
            for invalid_bin in invalid_bins:
                print(f"    Invalid bin: {invalid_bin[0]}, ref: {invalid_bin[1]}, min: {invalid_bin[2]}, max: {invalid_bin[3]}")

        return valid_bins
    

def obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=False, valid_bins=None, debug=False):
    """
    Compute the reduced chi-squared statistic per observable.

    :param differences: Array of bin differences (data - reference).
    :param squared_errors: Array of squared errors for each bin.
    :param bin_weights: Array of weights for each bin.
    :param bin_names: List of bin names (including observable name).
    :param weighted: If True, apply weights to the chi-squared calculation.
    :param valid_bins: If provided, only include bins in this list (envelope filtering).
    :param debug: If True, print debug info.
    :return: Dictionary {observable: reduced chi2}
    """
    if debug: print("Executing obs_chi2")

    obs_bins = {}
    chi2s = {}

    if valid_bins is not None and debug:
        valid_mask = np.array([name in valid_bins for name in bin_names])
        if debug:
            filtered_count = len([bin_names[i] for i in range(len(bin_names)) if valid_mask[i]])
            total_count = len(valid_mask)
            print(f"  Envelope filtering: using {filtered_count}/{total_count} bins")
    
    for idx, name in enumerate(bin_names):
        obs = name.split("#")[0]
        if valid_bins is not None and name not in valid_bins:
            continue
        if obs not in obs_bins:
            obs_bins[obs] = []
        obs_bins[obs].append(idx)

    for obs, indices in obs_bins.items():
        diffs = differences[indices]
        errs = squared_errors[indices]
        ws = bin_weights[indices]

        if weighted:
            if np.any(ws > 1):
                raise ValueError("  Error: Bin weights > 1 encountered in weighted mode")
            valid_mask = errs != 0
            chi2_series = (ws[valid_mask]**2) * (diffs[valid_mask]**2) / errs[valid_mask]
            ndf_series = ws[valid_mask]**2
        else:
            valid_mask = errs != 0
            chi2_series = (diffs[valid_mask]**2) / errs[valid_mask]
            ndf_series = np.ones_like(chi2_series)
        
        final_mask = (ws[valid_mask] > 0) & (~np.isnan(chi2_series)) & (~np.isinf(chi2_series))
        ndf = np.sum(ndf_series[final_mask])
        
        if ndf > 0:
            chi2s[obs] = np.sum(chi2_series[final_mask]) / ndf
        else:
            chi2s[obs] = -1
        
        if debug: print(f"  Observable: {obs}, chi2: {chi2s[obs]}, ndf: {ndf}")
    
    return chi2s


def global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=False, valid_bins=None, debug=False):
    """
    Compute the reduced chi-squared statistic over all bins.

    :param differences: Array of bin differences (data - reference).
    :param squared_errors: Array of squared errors for each bin.
    :param bin_weights: Array of weights for each bin.
    :param bin_names: List of bin names (including observable name).
    :param weighted: If True, apply weights to the chi-squared calculation.
    :param valid_bins: If provided, only include bins in this list (envelope filtering).
    :param debug: If True, print debug info.
    :return: Chi squared value and ndf.
    """
    if debug: print("Executing global_chi2")

    if bin_weights is None:
        bin_weights = np.ones_like(differences)

    if valid_bins is not None:
        valid_mask = np.array([name in valid_bins for name in bin_names])
        differences = differences[valid_mask]
        squared_errors = squared_errors[valid_mask]
        bin_weights = bin_weights[valid_mask]
        bin_names = [bin_names[i] for i in range(len(bin_names)) if valid_mask[i]]
        if debug:
            filtered_count = len(bin_names)
            total_count = len(valid_mask)
            print(f"  Envelope filtering: using {filtered_count}/{total_count} bins")

    if weighted:
        if np.any(bin_weights > 1):
            raise ValueError("  Error: Bin weights > 1 encountered in weighted mode")
        valid_mask = squared_errors != 0
        chi2_series = (bin_weights[valid_mask]**2) * (differences[valid_mask]**2) / squared_errors[valid_mask]
        ndf_series = bin_weights[valid_mask]**2
    else:
        valid_mask = squared_errors != 0
        chi2_series = (differences[valid_mask]**2) / squared_errors[valid_mask]
        ndf_series = np.ones_like(chi2_series)

    final_mask = (bin_weights[valid_mask] > 0) & (~np.isnan(chi2_series)) & (~np.isinf(chi2_series))
    
    if debug:
        invalid_bins = ~final_mask
        if np.any(invalid_bins):
            for idx in np.where(invalid_bins)[0]:
                original_idx = np.where(valid_mask)[0][idx]
                name = bin_names[original_idx] if bin_names is not None else original_idx
                print(f"  Problematic bin: {name}, chi2_series: {chi2_series[idx]}, difference: {differences[original_idx]}, squared_error: {squared_errors[original_idx]}, weight: {bin_weights[original_idx]}")
    
    chi2 = np.sum(chi2_series[final_mask])
    ndf = np.sum(ndf_series[final_mask])
    return chi2, ndf


def group_yoda_files_by_analyses(file_analyses, debug=False):
    """
    Group YODA files based on their list of analyses.
    
    :param file_analyses: Dictionary mapping YODA file paths to analysis signatures
    :param debug: If True, print debug info
    :return: Dictionary mapping analysis signatures to lists of YODA files
    """
    if debug: print("Grouping YODA files by analyses...")
    
    groups = {}
    for yoda_file, signature in file_analyses.items():
        if not signature:
            continue
        if signature not in groups:
            groups[signature] = []
        groups[signature].append(yoda_file)
    
    if debug:
        print(f"Found {len(groups)} groups:")
        for i, (signature, files) in enumerate(groups.items()):
            print(f"  Group {i+1}: {len(signature)} analyses ({len(files)} files)")
            print(f"    Analyses: {', '.join(signature)}")
            print(f"    Files: {[f.name for f in files]}")
    
    return groups


def group_yoda_files_by_name(yoda_files, debug=False):
    """
    Group YODA files based on their filename suffix (after last '-').
    
    :param yoda_files: List of YODA file paths
    :param debug: If True, print debug info
    :return: Dictionary mapping name suffixes to lists of YODA files
    """
    if debug: print("Grouping YODA files by filename suffix...")
    
    groups = {}
    for yoda_file in yoda_files:
        name = yoda_file.stem
        if name.endswith('.yoda'):
            name = name[:-5]
        if name.endswith('.yoda.gz'):
            name = name[:-8]
        
        if '-' in name:
            suffix = name.split('-')[-1]
        else:
            suffix = name
        
        if suffix not in groups:
            groups[suffix] = []
        groups[suffix].append(yoda_file)
    
    if debug:
        print(f"Found {len(groups)} groups:")
        for i, (suffix, files) in enumerate(groups.items()):
            print(f"  Group {i+1}: {suffix} ({len(files)} files)")
            print(f"    Files: {[f.name for f in files]}")
    
    return groups


def find_matching_default_file(group_signature, group_name, default_file_analyses, group_by_name=False, debug=False):
    """
    Find a matching default file for a group.
    
    :param group_signature: Group signature (tuple of analyses or filename suffix)
    :param group_name: Group name
    :param default_file_analyses: Dictionary mapping default files to their analyses
    :param group_by_name: Whether grouping is by name
    :param debug: Debug flag
    :return: Path to matching default file or None
    """
    if debug: print(f"Looking for default file matching group: {group_name}")
    
    if group_by_name:
        for dfile in default_file_analyses.keys():
            name = dfile.stem
            if name.endswith('.yoda'):
                name = name[:-5]
            if name.endswith('.yoda.gz'):
                name = name[:-8]
            
            if '-' in name:
                suffix = name.split('-')[-1]
            else:
                suffix = name
            
            if suffix == group_signature:
                if debug: print(f"  Found matching default file by name: {dfile.name}")
                return dfile
    else:
        for dfile, analyses in default_file_analyses.items():
            if analyses == group_signature:
                if debug: print(f"  Found matching default file by analyses: {dfile.name}")
                return dfile
    
    if debug: print(f"  No matching default file found for group: {group_name}")
    return None


def get_label_for_file(yfile, tags, labels):
    """
    Get the appropriate label for a YODA file based on tags and labels.
    
    :param yfile: Path object of the YODA file
    :param tags: List of tags to match against
    :param labels: List of labels corresponding to tags (same order)
    :return: Corresponding label if found, otherwise None
    """
    if not tags or not labels or len(tags) != len(labels):
        return None
    
    filename = yfile.name
    for tag, label in zip(tags, labels):
        if tag in filename:
            return label
    return None


def get_color_for_file(yfile, tags, colors):
    """
    Get the appropriate color for a YODA file based on tags and colors.
    
    :param yfile: Path object of the YODA file
    :param tags: List of tags to match against
    :param colors: List of colors corresponding to tags (same order)
    :return: Corresponding color if found, otherwise None
    """
    if not tags or not colors or len(tags) != len(colors):
        return None
    
    filename = yfile.name
    for tag, color in zip(tags, colors):
        if tag in filename:
            return color
    return None


def sort_files_by_tag_order(files, tags):
    """
    Sort files based on the order of tags in the command line.
    Files are ordered by the position of their matching tag in the tags list.
    
    :param files: List of file Path objects
    :param tags: List of tags in command line order
    :return: List of files sorted by tag order
    """
    if not tags:
        return files
    
    def get_tag_index(yfile):
        filename = yfile.name
        for i, tag in enumerate(tags):
            if tag in filename:
                return i
        return len(tags)
    
    return sorted(files, key=get_tag_index)


def create_chi2_plot_from_obs(obs_chi2s):
    """
    Create chi2_plot dictionary from obs_chi2s with automatic bin ID simplification.
    
    :param obs_chi2s: Dictionary mapping observable paths to chi2 values
    :return: Dictionary mapping analysis names to list of (bin_id, chi2_value) tuples
    """
    chi2_plot = {}
    
    for obs, chi2_value in obs_chi2s.items():
        if chi2_value == -1:
            continue
        analysis_name = obs.split("/")[1]
        bin_id = obs.split("/")[2]
        if analysis_name not in chi2_plot:
            chi2_plot[analysis_name] = []
        chi2_plot[analysis_name].append((bin_id, chi2_value))
    
    for analysis_name in chi2_plot:
        bin_tuples = chi2_plot[analysis_name]
        bin_ids = [b[0] for b in bin_tuples]
        
        d_parts = []
        for bin_id in bin_ids:
            if '-' in bin_id:
                d_part = bin_id.split('-')[0]
            else:
                d_part = bin_id
            d_parts.append(d_part)
        
        if len(set(d_parts)) == len(bin_ids):
            simplified_tuples = []
            for (bin_id, chi2_value), d_part in zip(bin_tuples, d_parts):
                simplified_tuples.append((d_part, chi2_value))
            chi2_plot[analysis_name] = simplified_tuples
    
    return chi2_plot


def plot_chi2_per_analysis(all_chi2_plots, labels, colors, chi2_plot_def=None, default_label="default", default_color="black", outdir="chi2_plots", debug=False):
    """
    Plot chi2 per analysis for multiple tunes, optionally comparing to default.
    """
    os.makedirs(outdir, exist_ok=True)
    
    if colors is None:
        plasma = plt.get_cmap('plasma')
        colors = [plasma(0.2 + 0.6 * i / max(1, len(labels)-1)) for i in range(len(labels))]
    elif len(colors) != len(labels):
        plasma = plt.get_cmap('plasma')
        colors = [plasma(0.2 + 0.6 * i / max(1, len(labels)-1)) for i in range(len(labels))]

    all_analyses = set()
    for chi2_plot in all_chi2_plots:
        all_analyses.update(chi2_plot.keys())
    if chi2_plot_def:
        all_analyses.update(chi2_plot_def.keys())

    for analysis_name in sorted(all_analyses):
        tune_bin_data = []
        tune_bin_ids = []
        tune_chi2s = []
        for chi2_plot in all_chi2_plots:
            if analysis_name in chi2_plot:
                bin_ids, chi2_values = zip(*chi2_plot[analysis_name])
                tune_bin_ids.append(bin_ids)
                tune_chi2s.append(chi2_values)
                tune_bin_data.append(dict(zip(bin_ids, chi2_values)))
            else:
                tune_bin_ids.append(())
                tune_chi2s.append(())
                tune_bin_data.append({})

        all_bin_ids = set()
        for bin_ids in tune_bin_ids:
            all_bin_ids.update(bin_ids)
        
        ref_idx = max(range(len(tune_bin_ids)), key=lambda i: len(tune_bin_ids[i])) if any(tune_bin_ids) else 0
        ref_bin_ids = tune_bin_ids[ref_idx] if tune_bin_ids[ref_idx] else []
        
        bin_ids = list(ref_bin_ids)
        for bid in sorted(all_bin_ids - set(ref_bin_ids)):
            bin_ids.append(bid)
        
        bin_ids = sorted(bin_ids)
        n_bins = len(bin_ids)

        chi2_values_def = None
        if chi2_plot_def and analysis_name in chi2_plot_def:
            def_dict = dict(chi2_plot_def[analysis_name])
            chi2_values_def = [def_dict.get(bid, np.nan) for bid in bin_ids]

        chunk_size = 100
        n_chunks = (n_bins + chunk_size - 1) // chunk_size

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end = min((chunk_idx + 1) * chunk_size, n_bins)
            chunk_bin_ids = bin_ids[start:end]
            fig, (ax_top, ax_bottom) = plt.subplots(
                2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})

            for i, (bin_data, label) in enumerate(zip(tune_bin_data, labels)):
                if not bin_data:
                    continue
                chunk_chi2s = [bin_data.get(bid, np.nan) for bid in chunk_bin_ids]
                
                valid_indices = [j for j, val in enumerate(chunk_chi2s) if not np.isnan(val)]
                if not valid_indices:
                    continue
                    
                valid_bin_ids = [chunk_bin_ids[j] for j in valid_indices]
                valid_chi2s = [chunk_chi2s[j] for j in valid_indices]
                
                ax_top.plot(valid_bin_ids, valid_chi2s, marker='o', linestyle='', label=label, color=colors[i])
                if len(valid_chi2s) > 1:
                    ax_top.plot(valid_bin_ids, valid_chi2s, linestyle='-', alpha=0.5, color=colors[i])
                    
            if chi2_values_def is not None:
                chunk_chi2_values_def = chi2_values_def[start:end]
                if default_color == "black":
                    ax_top.plot(chunk_bin_ids, chunk_chi2_values_def, marker='o', color=default_color, linestyle='', label=default_label, alpha=0.6)
                    ax_top.plot(chunk_bin_ids, chunk_chi2_values_def, linestyle='-', color=default_color, alpha=0.3)
                else:
                    ax_top.plot(chunk_bin_ids, chunk_chi2_values_def, marker='o', color=default_color, linestyle='', label=default_label)
                    ax_top.plot(chunk_bin_ids, chunk_chi2_values_def, linestyle='-', color=default_color, alpha=0.5)

            ax_top.set_ylabel(r'$\chi^2 / \mathrm{ndf}$')
            if n_chunks == 1:
                ax_top.set_title(f'{analysis_name}')
            else:
                ax_top.set_title(f'{analysis_name} (part {chunk_idx + 1}/{n_chunks})')
            ax_top.legend()
            ax_top.tick_params(axis='x', which='both', bottom=True, top=False, length=3.5, direction='in', labelbottom=False)

            if chi2_values_def is not None:
                for i, (bin_data, label) in enumerate(zip(tune_bin_data, labels)):
                    if not bin_data:
                        continue
                    chunk_chi2s = np.array([bin_data.get(bid, np.nan) for bid in chunk_bin_ids])
                    chunk_chi2_values_def = np.array(chi2_values_def[start:end])
                    
                    valid_mask = ~(np.isnan(chunk_chi2s) | np.isnan(chunk_chi2_values_def))
                    if not np.any(valid_mask):
                        continue
                        
                    valid_bin_ids = [chunk_bin_ids[j] for j in range(len(chunk_bin_ids)) if valid_mask[j]]
                    valid_ratios = (chunk_chi2s / chunk_chi2_values_def)[valid_mask]
                    
                    ax_bottom.plot(valid_bin_ids, valid_ratios, marker='o', linestyle='', label=f"{label}/{default_label}", color=colors[i])
                    if len(valid_ratios) > 1:
                        ax_bottom.plot(valid_bin_ids, valid_ratios, linestyle='-', alpha=0.5, color=colors[i])

                ax_bottom.axhline(1, color='black', linestyle='--', alpha=0.2)
                ax_bottom.set_ylabel(f'data/{default_label}')
                ax_bottom.set_xlabel('observable')
                ax_bottom.set_ylim(0.0, 2.0)
                ax_bottom.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
                ax_bottom.tick_params(axis='y', which='minor', length=3)

                if len(chunk_bin_ids) > 10:
                    step = max(1, len(chunk_bin_ids) // 15)
                    xticks = list(range(0, len(chunk_bin_ids), step))
                    ax_bottom.set_xticks(xticks)
                    ax_bottom.set_xticklabels([chunk_bin_ids[i] for i in xticks], rotation=90)
                else:
                    ax_bottom.set_xticks(range(len(chunk_bin_ids)))
                    ax_bottom.set_xticklabels(chunk_bin_ids, rotation=90)
            else:
                ax_bottom.set_visible(False)
                ax_top.set_xlabel('observable')
                if len(chunk_bin_ids) > 10:
                    step = max(1, len(chunk_bin_ids) // 15)
                    xticks = list(range(0, len(chunk_bin_ids), step))
                    ax_top.set_xticks(xticks)
                    ax_top.set_xticklabels([chunk_bin_ids[i] for i in xticks], rotation=90)
                else:
                    ax_top.set_xticks(range(len(chunk_bin_ids)))
                    ax_top.set_xticklabels(chunk_bin_ids, rotation=90)
                ax_top.tick_params(axis='x', which='both', bottom=True, top=False, length=3.5, direction='out', labelbottom=True)

            plt.tight_layout()
            plt.subplots_adjust(hspace=0.0)
            if n_chunks == 1:
                fname = f"{outdir}/{analysis_name}_chi2_plot.pdf"
            else:
                fname = f"{outdir}/{analysis_name}_chi2_plot_part{chunk_idx+1}.pdf"
            plt.savefig(fname, dpi=300)
            if fname.endswith('.pdf'):
                plt.savefig(fname.replace('.pdf', '.png'), dpi=150)
            plt.close(fig)

            print(f"Plot for {analysis_name} (part {chunk_idx + 1}/{n_chunks}) saved to {fname}/png" if n_chunks > 1 else f"Plot for {analysis_name} saved to {fname}/png")


def plot_chi2_distribution(all_valid_chi2s, labels, colors, valid_chi2s_def=None, default_label="default", default_color="black", outdir="chi2_plots", debug=False):
    """
    Plot the distribution of chi2 values (log10) for multiple YODA files.
    """
    os.makedirs(outdir, exist_ok=True)
    
    filtered_chi2s = []
    filtered_labels = []
    all_logs = []
    
    for chi2s, label in zip(all_valid_chi2s, labels):
        if len(chi2s) > 0:
            valid_chi2s = [x for x in chi2s if x > 0 and np.isfinite(x)]
            if len(valid_chi2s) > 0:
                filtered_chi2s.append(valid_chi2s)
                filtered_labels.append(label)
                all_logs.extend(np.log10(valid_chi2s))
            else:
                if debug: print(f"Warning: No valid chi2 values for {label}, skipping from distribution plot")
        else:
            if debug: print(f"Warning: Empty chi2 array for {label}, skipping from distribution plot")
    
    valid_chi2s_def_filtered = None
    if valid_chi2s_def is not None and len(valid_chi2s_def) > 0:
        valid_def = [x for x in valid_chi2s_def if x > 0 and np.isfinite(x)]
        if len(valid_def) > 0:
            valid_chi2s_def_filtered = valid_def
            all_logs.extend(np.log10(valid_def))
        else:
            if debug: print(f"Warning: No valid chi2 values for default data, skipping from distribution plot")

    if len(all_logs) == 0:
        if debug: print("Warning: No valid chi2 data found for distribution plot")
        return
    
    bins = np.linspace(np.min(all_logs), np.max(all_logs), 51)

    all_hists = []
    for chi2s in filtered_chi2s:
        hist, _ = np.histogram(np.log10(chi2s), bins=bins, density=True)
        all_hists.append(hist)
    if valid_chi2s_def_filtered is not None:
        hist_def, _ = np.histogram(np.log10(valid_chi2s_def_filtered), bins=bins, density=True)
        all_hists.append(hist_def)
    
    max_hist_val = max([np.max(h) for h in all_hists if len(h) > 0]) if all_hists else 1.0
    global_ylim = (0, max_hist_val * 1.08)

    n_plots = len(filtered_chi2s) + (1 if valid_chi2s_def_filtered is not None else 0)
    if n_plots == 0:
        if debug: print("Warning: No plots to generate for chi2 distribution")
        return
        
    fig, axes = plt.subplots(n_plots, 1, figsize=(4.67, max(3, n_plots)), sharex=True, gridspec_kw={'hspace': 0})

    if n_plots == 1:
        axes = [axes]

    if colors is None:
        plasma = plt.get_cmap('plasma')
        colors = [plasma(0.2 + 0.6 * i / max(1, len(labels)-1)) for i in range(len(labels))]
    elif len(colors) < len(filtered_labels):
        plasma = plt.get_cmap('plasma')
        colors = [plasma(0.2 + 0.6 * i / max(1, len(labels)-1)) for i in range(len(labels))]

    plot_idx = 0
    for chi2s, label, color in zip(filtered_chi2s, filtered_labels, colors):
        hist, bin_edges = np.histogram(np.log10(chi2s), bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax = axes[plot_idx]
        ax.bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0]), alpha=0.2, color=color, label=None)
        ax.step(bin_edges, np.append(hist, hist[-1]), where='post', color=color, alpha=0.7, label=label)
        ax.set_ylabel('Density')
        ax.set_ylim(global_ylim)
        ax.legend(loc='upper right')
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))
        plot_idx += 1

    if valid_chi2s_def_filtered is not None:
        hist_def, bin_edges_def = np.histogram(np.log10(valid_chi2s_def_filtered), bins=bins, density=True)
        bin_centers_def = 0.5 * (bin_edges_def[:-1] + bin_edges_def[1:])
        ax = axes[plot_idx]
        if default_color == "black":
            ax.bar(bin_centers_def, hist_def, width=(bin_edges_def[1]-bin_edges_def[0]), alpha=0.2, color=default_color, label=None)
            ax.step(bin_edges_def, np.append(hist_def, hist_def[-1]), where='post', color=default_color, alpha=0.7, label=default_label)
        else:
            ax.bar(bin_centers_def, hist_def, width=(bin_edges_def[1]-bin_edges_def[0]), alpha=0.2, color=default_color, label=None)
            ax.step(bin_edges_def, np.append(hist_def, hist_def[-1]), where='post', color=default_color, alpha=0.7, label=default_label)
        ax.set_ylabel('Density')
        ax.set_ylim(global_ylim)
        ax.legend(loc='upper right')
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))

    axes[-1].set_xlabel(r'$\log_{10}(\chi^2 / \mathrm{ndf})$')
    fname = f"{outdir}/chi2_distribution.pdf"
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    if fname.endswith('.pdf'):
        plt.savefig(fname.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"Plot of Chi2 distribution saved to {outdir}/chi2_distribution.pdf/png")


def create_master_index_html(outdir="chi2_plots", groups=None, default_label=None, group_by_name=False):
    """
    Create a master index.html file that links to all group-specific plot directories.
    
    :param outdir: Output directory containing group subdirectories
    :param groups: Dictionary mapping analysis signatures to YODA file lists
    :param default_label: Label for default comparison if used
    """
    now = datetime.datetime.now().strftime("%A, %d. %B %Y %H:%M")
    
    html = f"""<html>
    <head>
        <title>Chi2 Analysis Groups - Master Index</title>
        <style>
        html {{ font-family: sans-serif; }}
        img {{ border: 0; max-width: 600px; }}
        a {{ text-decoration: none; font-weight: bold; }}
        table {{ border-collapse: collapse; margin-bottom: 2em; }}
        td, th {{ border: 1px solid #ccc; padding: 8px 12px; }}
        th {{ background: #eee; }}
        .group-summary {{ margin: 1em 0; padding: 1em; border: 1px solid #ddd; background: #f9f9f9; }}
        </style>
        <script type="text/x-mathjax-config">
        MathJax.Hub.Config({{
            tex2jax: {{inlineMath: [["$","$"]]}}
        }});
        </script>
        <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
        </script>
    </head>
    <body>
        <h1>Chi2 Analysis Groups</h1>
        <p><a href="../index.html" style="display: inline-block; padding: 8px 16px; text-decoration: none; ">&laquo; Back to ../index.html</a></p>
        <p>Generated from directory-based YODA analysis with automatic grouping.</p>
    """
    
    if groups:
        html += f"<h2>Analysis Groups ({len(groups)} groups found)</h2>\n"
        if group_by_name:
            html += "<p>Files have been automatically grouped by their filename suffix (after last '-'). Each group contains YODA files with the same naming pattern.</p>\n"
        else:
            html += "<p>Files have been automatically grouped by their analysis content. Each group contains YODA files with the same set of analyses.</p>\n"
        
        if group_by_name:
            sorted_groups = sorted(groups.items(), key=lambda x: x[0])
        else:
            sorted_groups = groups.items()
        
        for group_idx, (signature, files) in enumerate(sorted_groups):
            if group_by_name:
                group_name = signature
            else:
                endings = set()
                for yfile in files:
                    name = yfile.stem
                    if name.endswith('.yoda'):
                        name = name[:-5]
                    if name.endswith('.yoda.gz'):
                        name = name[:-8]
                    if '-' in name:
                        ending = name.split('-')[-1]
                        endings.add(ending)
                if len(endings) > 1:
                    group_name = f"group_{group_idx+1}"
                else:
                    group_name = list(endings)[0] if endings else f"group_{group_idx+1}"
            
            group_dir = os.path.join(outdir, group_name)
            
            html += f'<div class="group-summary">\n'
            if group_by_name:
                html += f'<h3><a href="{group_name}/index.html">{group_name}</a></h3>\n'
                html += f'<p><strong>Filename suffix:</strong> {signature}</p>\n'
            else:
                if len(endings) > 1:
                    html += f'<h3><a href="{group_name}/index.html">Group {group_idx+1}</a></h3>\n'
                else:
                    html += f'<h3><a href="{group_name}/index.html">Group {group_idx+1} ({group_name})</a></h3>\n'
                html += f'<p><strong>Analyses ({len(signature)}):</strong> {", ".join(sorted(signature))}</p>\n'
            
            html += f'<p><strong>Files ({len(files)}):</strong> {", ".join([f.name for f in files])}</p>\n'
            
            if os.path.exists(group_dir) and os.path.exists(os.path.join(group_dir, "index.html")):
                html += f'<p><a href="{group_name}/index.html">&raquo; View plots and analysis for this group</a></p>\n'
            else:
                html += f'<p><em>Plots not generated for this group</em></p>\n'
            
            html += '</div>\n'
    else:
        html += "<p>No analysis groups found.</p>\n"
    
    if default_label:
        html += f'<p><strong>Default comparison:</strong> {default_label}</p>\n'
    
    html += f"""
        <p><a href="../index.html" style="display: inline-block; padding: 8px 16px; text-decoration: none; ">&laquo; Back to ../index.html</a></p>
        <footer style="clear:both; margin-top:3em; padding-top:3em">
            <p>Generated at {now}</p>
            <p>Created with command: <pre>{" ".join(os.sys.argv)}</pre></p>
        </footer>
    </body>
    </html>
    """
    
    master_index_path = os.path.join(outdir, "index.html")
    with open(master_index_path, "w") as f:
        f.write(html)
    print(f"Created master index at {master_index_path}")


def create_index_html(outdir="chi2_plots", summaries=None, all_chi2_plots=None):
    """
    Create an index.html file in the output directory to access all plot files.
    """
    plot_files = []
    for fname in sorted(os.listdir(outdir)):
        if fname.endswith((".pdf", ".png", ".jpg", ".jpeg", ".svg")):
            plot_files.append(fname)

    exp_analysis_groups = {}
    for fname in plot_files:
        if fname.startswith("chi2_"):
            continue
        if "_chi2_plot" in fname:
            group = fname.split("_chi2_plot")[0]
        else:
            parts = fname.split('_')
            if len(parts) >= 3:
                group = "_".join(parts[:3])
            elif len(parts) >= 2:
                group = "_".join(parts[:2])
            else:
                group = parts[0]
        if group not in exp_analysis_groups:
            exp_analysis_groups[group] = []
        exp_analysis_groups[group].append(fname)

    now = datetime.datetime.now().strftime("%A, %d. %B %Y %H:%M")
    html = f"""<html>
    <head>
        <title>Chi2 Plots</title>
        <style>
        html {{ font-family: sans-serif; }}
        img {{ border: 0; max-width: 600px; }}
        a {{ text-decoration: none; font-weight: bold; }}
        table {{ border-collapse: collapse; margin-bottom: 2em; }}
        td, th {{ border: 1px solid #ccc; padding: 4px 8px; }}
        th {{ background: #eee; }}
        </style>
        <script type="text/x-mathjax-config">
        MathJax.Hub.Config({{
            tex2jax: {{inlineMath: [["$","$"]]}}
        }});
        </script>
        <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
        </script>
    </head>
    <body>
        <h2>Chi2 Plots</h2>
        <p><a href="../index.html" style="display: inline-block; padding: 8px 16px; text-decoration: none; ">&laquo; Back to ../index.html</a></p>
    """

    if summaries:
        html += "<h3>chi2 summary</h3>\n"
        html += "<table>\n<tr><th>Label</th><th>Global chi2</th><th>Degrees of freedom (ndf)</th><th>Reduced chi2</th><th>Averaged chi2s over all obs.</th></tr>\n"
        for summary in summaries:
            html += f"<tr><td>{summary.get('label','')}</td><td>{summary.get('Global chi2','')}</td><td>{summary.get('Degrees of freedom (ndf)','')}</td><td>{summary.get('Reduced chi2','')}</td><td>{summary.get('Averaged chi2s over all obs.','')}</td></tr>\n"
        html += "</table>\n"

    for group, files in sorted(exp_analysis_groups.items()):
        html += f'    <h3>{group}</h3>\n'
        pngs = [f for f in files if f.endswith('.png')]
        avg_chi2s = []
        for chi2_plot in all_chi2_plots:
            if group in chi2_plot:
                chi2_vals = [v for _, v in chi2_plot[group]]
                if chi2_vals:
                    avg_chi2s.append(sum(chi2_vals) / len(chi2_vals))
                else:
                    avg_chi2s.append(None)
            else:
                avg_chi2s.append(None)
        if any(avg is not None for avg in avg_chi2s):
            html += '<div style="font-size:1.1em; margin-bottom:0.3em;"><b>Average chi2 per YODA:</b><ul style="margin:0.2em 0 0.5em 1em;">'
            for idx, avg in enumerate(avg_chi2s):
                if avg is not None:
                    label = summaries[idx]['label'] if idx < len(summaries) else f'YODA_{idx+1}'
                    html += f'<li>{label}: {avg:.3f}</li>'
            html += '</ul></div>\n'
        for idx, png in enumerate(sorted(pngs)):
            pdf = png.replace('.png', '.pdf')
            html += f'    <div style="margin-bottom:1.5em;">'
            html += f'<img src="{png}" alt="{png}" style="max-width:800px;"><br>'
            if pdf in files:
                html += f'<a href="{pdf}">Download PDF</a>'
            html += '</div>\n'

    if "chi2_distribution.png" in plot_files:
        html += '<h3>Chi2 Distribution</h3>\n'
        html += '<div style="margin-bottom:1.5em;">'
        html += '<img src="chi2_distribution.png" alt="chi2_distribution.png" style="max-width:800px;"><br>'
        if "chi2_distribution.pdf" in plot_files:
            html += '<a href="chi2_distribution.pdf">Download PDF</a>'
        html += '</div>\n'

    html += f"""    <footer style="clear:both; margin-top:3em; padding-top:3em">
        <p><a href="../index.html" style="display: inline-block; padding: 8px 16px; text-decoration: none; ">&laquo; Back to ../index.html</a></p>
        <p>Generated at {now}</p>
        <p>Created with command: <pre>{" ".join(os.sys.argv)}</pre></p>
        </footer>
    </body>
    </html>
    """

    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(html)
    print(f"Created {os.path.join(outdir, 'index.html')}")


def apply_rivet_style(use_tex_preference=False):
    """Apply a Rivet-like Matplotlib style via rcParams.
    This mirrors the provided default.mplstyle as closely as reasonable in-script.

    If LaTeX is unavailable, text.usetex will be disabled to avoid runtime errors.
    """
    usetex = bool(use_tex_preference)
    if use_tex_preference:
        if shutil.which("pdflatex") is None:
            usetex = False

    if usetex:
        mpl.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Palatino"],
            "mathtext.fontset": "custom",
            "mathtext.rm": "Palatino",
            "mathtext.bf": "Palatino:bold",
            "mathtext.it": "Palatino:italic",
            "mathtext.default": "it",
            "text.usetex": True,
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 9,
            "figure.titlesize": 10,
        })
    else:
        mpl.rcParams.update({
            "font.family": "serif",
            "text.usetex": False,
            "font.size": 10,
            "axes.titlesize": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 10,
        })
    mpl.rcParams.update({
        "figure.figsize": (4.67, 4.21),
        "figure.subplot.bottom": 0.092,
        "figure.subplot.top": 0.934,
        "figure.subplot.left": 0.125,
        "figure.subplot.right": 0.968,
        "figure.subplot.hspace": 0.0,
    })
    mpl.rcParams.update({
        "axes.axisbelow": False,
        "axes.titlepad": 7.5,
        "axes.labelpad": 2.0,
        "axes.linewidth": 0.3,
        "xaxis.labellocation": "right",
        "lines.markersize": 1.8,
        "lines.linewidth": 1.0,
        "axes.formatter.min_exponent": 1,
    })
    mpl.rcParams.update({
        "xtick.major.width": 0.3,
        "ytick.major.width": 0.3,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        "xtick.major.size": 9,
        "ytick.major.size": 9,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.top": True,
        "ytick.right": True,
    })
    mpl.rcParams.update({
        "legend.frameon": False,
        "legend.labelspacing": 0.2,
    })


def main():
    parser = argparse.ArgumentParser(description="Compute global chi2 from YODA file and weights.")
    parser.add_argument("yoda_files", nargs="+", help="Path(s) to one or more YODA files or a directory")
    parser.add_argument("-w",  "--weights",             default=None, help="Path to the weights file (optional, if not provided, all weights default to 1.0)")
    parser.add_argument("-a",  "--analyses",            default=None, help="Path to file containing list of analyses to include (optional, if not provided, all analyses are included)")
    parser.add_argument(       "--weighted",            action="store_true", default=False, help="Enable weighted chi2 calculation, i.e. each term is multiplied by its weight squared: chi^2 += w^2 * chi^2_bin and ndf += w^2 (weights > 1 not allowed) (otherwise, all weights > 0 default to 1.0)")
    parser.add_argument("-l",  "--labels",              nargs="+", help="Labels for each YODA file (same order) or labels corresponding to tags when used with --tags")
    parser.add_argument("-c",  "--colors",              nargs="+", help="Colors for each YODA file (same order) or colors corresponding to tags when used with --tags. Can be color names, hex codes, or matplotlib color specifications.")
    parser.add_argument("-p",  "--plots",               action="store_true", default=False, help="Enable plotting of chi2 per analysis")
    parser.add_argument(       "--use_tex",             action="store_true", default=False, help="Use LaTeX for plots")
    parser.add_argument("-d",  "--default",             default=None, help="Path to a YODA file, directory, or tag for comparison of chi2. For directories/tags, matching files are automatically selected per group.")
    parser.add_argument("-dl", "--default_label",       default="default", help="Label for the default YODA file in output and plots")
    parser.add_argument("-dc", "--default_color",       default="black", help="Color used for the default dataset in plots (name or hex). Default is black")
    parser.add_argument("-e",  "--envelope",            nargs=2, metavar=("up.yoda", "dn.yoda"), help="Check if reference values are within envelope (up.yoda dn.yoda)")
    parser.add_argument("-t",  "--tags",                nargs="+", default=None, help="If set, only use YODA files with any of these tags from the given directory")
    parser.add_argument(       "--subdir",              action="store_true", default=False, help="If set, also process all subdirectories in the given directory")
    parser.add_argument("-o",  "--outdir",              default="chi2_plots", help="Output directory for chi2 plots")
    parser.add_argument(       "--group_by_analyses",   action="store_true", default=False, help="Group files by analyses instead of by filename suffix (after last '-')")
    parser.add_argument("-v",  "--debug",               action="store_true", default=False, help="Enable debug output")
    args = parser.parse_args()

    print("Starting chi2 analysis...")

    try:
        loader = yodaLoader()
        if args.weights is not None:
            weights = read_weights(args.weights)
            if args.debug:
                print(f"Loaded weights from {args.weights}")
        else:
            weights = None
            if args.debug:
                print("No weights file provided, using default weight of 1.0 for all observables/bins")
            if args.weighted:
                print("Warning: Weighted chi2 calculation is enabled, but no weights file is provided")
                args.weighted = False
                
        if args.analyses is not None:
            analyses = read_analyses(args.analyses)
            if args.debug:
                print(f"Loaded analyses from {args.analyses}: {sorted(analyses)}")
        else:
            analyses = None
            if args.debug:
                print("No analyses file provided, using all available analyses")
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return 1
    except Exception as e:
        print(f"Error reading files: {e}")
        return 1

    if args.default:
        default_path = Path(args.default)
        if ('/' in args.default or '\\' in args.default) and not default_path.exists():
            print(f"Warning: Default path {args.default} does not exist")
            args.default = None

    if args.tags and args.labels:
        if len(args.tags) != len(args.labels):
            print(f"Warning: Number of tags ({len(args.tags)}) does not match number of labels ({len(args.labels)}), falling back to filename-based labeling...")
        elif args.debug:
            print(f"Using tag-label mapping: {dict(zip(args.tags, args.labels))}")

    if args.tags and args.colors:
        if len(args.tags) != len(args.colors):
            print(f"Warning: Number of tags ({len(args.tags)}) does not match number of colors ({len(args.colors)}), falling back to default plasma colormap...")
            args.colors = None
        elif args.debug:
            print(f"Using tag-color mapping: {dict(zip(args.tags, args.colors))}")

    if args.colors and not args.tags and len(args.yoda_files) > 1:
        if len(args.colors) != len(args.yoda_files):
            print(f"Warning: Number of colors ({len(args.colors)}) does not match number of YODA files ({len(args.yoda_files)}), falling back to default plasma colormap...")
            args.colors = None

    valid_bins = None
    if args.envelope:
        try:
            valid_bins = process_envelope(args, loader, weights)
            if valid_bins is None:
                return 1
        except Exception as e:
            print(f"Error processing envelope: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            return 1

    try:
        if len(args.yoda_files) == 1 and Path(args.yoda_files[0]).is_dir():
            return process_directory(args, loader, weights, analyses, valid_bins, group_by_name=not args.group_by_analyses, debug=args.debug)
        else:
            return process_files(args, loader, weights, analyses, valid_bins, debug=args.debug)
    except Exception as e:
        print(f"Error processing files: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
    

def process_envelope(args, loader, weights):
    """Check if reference values are within envelope bounds."""
    up_file, dn_file = args.envelope
    
    if not Path(up_file).exists():
        print(f"Error: Envelope up file '{up_file}' does not exist")
        return None
    if not Path(dn_file).exists():
        print(f"Error: Envelope down file '{dn_file}' does not exist")
        return None
    
    try:
        valid_bins = loader.get_valid_bins(up_file, dn_file, weights, debug=args.debug)
        return valid_bins
        
    except Exception as e:
        print(f"Error processing envelope: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return None
    

def process_default_files(default_arg, loader, weights, analyses, valid_bins=None, debug=False):
    """Process default files from a path, directory, or tag."""
    if debug: print(f"Processing default files from: {default_arg}")
    
    default_path = Path(default_arg)
    default_files = []
    
    if default_path.is_file():
        default_files = [default_path]
        if debug: print(f"  Using single default file: {default_path}")
    
    elif default_path.is_dir():
        default_files = list(default_path.glob("*.yoda")) + list(default_path.glob("*.yoda.gz"))
        if debug: print(f"  Found {len(default_files)} default files in directory")
    
    else:
        current_dir = Path(".")
        all_yoda_files = list(current_dir.glob("*.yoda")) + list(current_dir.glob("*.yoda.gz"))
        default_files = [f for f in all_yoda_files if default_arg in f.name]
        if debug: print(f"  Found {len(default_files)} default files matching tag '{default_arg}'")
    
    default_file_analyses = {}
    default_file_results = {}
    
    for dfile in default_files:
        try:
            differences, squared_errors, bin_weights, bin_names, analyses_found = loader.get_bin_differences(
                str(dfile), weights, analyses, include_ref_error=True, debug=debug)
            chi2, ndf = global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=False, valid_bins=valid_bins, debug=debug)
            obs_chi2s = obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=False, valid_bins=valid_bins, debug=debug)
            valid_chi2s = [v for v in obs_chi2s.values() if v != -1]
            
            default_file_analyses[dfile] = analyses_found
            default_file_results[dfile] = {
                'chi2': chi2,
                'ndf': ndf,
                'obs_chi2s': obs_chi2s,
                'valid_chi2s': valid_chi2s,
                'differences': differences,
                'squared_errors': squared_errors,
                'bin_weights': bin_weights,
                'bin_names': bin_names
            }
            
            if debug: print(f"  Processed default file: {dfile.name} ({len(analyses)} analyses)")
                
        except Exception as e:
            if debug: print(f"  Failed to process default file {dfile}: {e}")
            continue
    
    return default_file_analyses, default_file_results


def process_directory(args, loader, weights, analyses, valid_bins=None, group_by_name=True, debug=False):
    """Process a directory containing YODA files."""
    if debug: print(f"Processing directory: {args.yoda_files[0]}")

    yoda_path = Path(args.yoda_files[0])
    reduced_chi2_dict = {}
    avg_chi2_dict = {}
    ndf_dict = {}

    yoda_files = list(yoda_path.glob("*.yoda")) + list(yoda_path.glob("*.yoda.gz"))
    if args.tags:
        yoda_files = [f for f in yoda_files if any(tag in f.name for tag in args.tags)]
    
    file_analyses = {}
    file_results = {}
    
    for yfile in yoda_files:
        differences, squared_errors, bin_weights, bin_names, analyses_found = loader.get_bin_differences(
            str(yfile), weights, analyses, include_ref_error=True, debug=debug)
        file_analyses[yfile] = analyses_found
        chi2, ndf = global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=debug)
        obs_chi2s = obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=debug)
        valid_chi2s = [v for v in obs_chi2s.values() if v != -1]

        file_results[yfile] = {
            'chi2': chi2,
            'ndf': ndf,
            'obs_chi2s': obs_chi2s,
            'valid_chi2s': valid_chi2s
        }

        key = yfile.stem
        reduced_chi2_dict[key] = chi2 / ndf if ndf > 0 else float('nan')
        avg_chi2_dict[key] = np.mean(valid_chi2s) if valid_chi2s else float('nan')
        ndf_dict[key] = ndf

    if args.subdir:
        for subfolder in yoda_path.iterdir():
            if subfolder.is_dir():
                yoda_files_sub = list(subfolder.glob("*.yoda")) + list(subfolder.glob("*.yoda.gz"))
                if args.tags:
                    yoda_files_sub = [f for f in yoda_files_sub if any(tag in f.name for tag in args.tags)]
                if not yoda_files_sub:
                    continue
                for yfile in yoda_files_sub:
                    differences, squared_errors, bin_weights, bin_names, analyses_found = loader.get_bin_differences(
                        str(yfile), weights, analyses, include_ref_error=True, debug=debug)
                    file_analyses[yfile] = analyses_found
                    chi2, ndf = global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=debug)
                    obs_chi2s = obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=debug)
                    valid_chi2s = [v for v in obs_chi2s.values() if v != -1]

                    file_results[yfile] = {
                        'chi2': chi2,
                        'ndf': ndf,
                        'obs_chi2s': obs_chi2s,
                        'valid_chi2s': valid_chi2s
                    }

                    key = subfolder.name
                    reduced_chi2_dict[key] = chi2 / ndf if ndf > 0 else float('nan')
                    avg_chi2_dict[key] = np.mean(valid_chi2s) if valid_chi2s else float('nan')
                    ndf_dict[key] = ndf

    reduced_chi2_dict = {key: reduced_chi2_dict[key] for key in sorted(reduced_chi2_dict.keys())}

    print("\nInput directory: " + str(yoda_path.resolve()))
    print(("Weighted" if args.weighted else "Unweighted") + " chi2 summary:\n")

    table = []
    for key in reduced_chi2_dict:
        table.append([
            key,
            f"{reduced_chi2_dict[key]:.2f}",
            f"{avg_chi2_dict[key]:.2f}",
            f"{ndf_dict[key]:.0f}"
        ])
    print(tabulate(table, headers=["Label", "Reduced chi2", "Averaged chi2", "ndf"], tablefmt="github"))

    if reduced_chi2_dict:
        min_reduced_key = min(reduced_chi2_dict, key=lambda k: reduced_chi2_dict[k])
        min_avg_key = min(avg_chi2_dict, key=lambda k: avg_chi2_dict[k])
        print()
        print(f"Min. reduced chi2: {reduced_chi2_dict[min_reduced_key]:.2f} (key: {min_reduced_key})")
        print(f"Min. averaged chi2: {avg_chi2_dict[min_avg_key]:.2f} (key: {min_avg_key})")

    default_file_analyses = {}
    default_file_results = {}
    
    if args.default:
        try:
            default_file_analyses, default_file_results = process_default_files(
                args.default, loader, weights, analyses, valid_bins, debug=debug)
            
            if default_file_analyses:
                print(f"\nProcessed {len(default_file_analyses)} default files for potential group matching.")
            else:
                print(f"\nWarning: No valid default files found from: {args.default}")
                args.default = None
        except Exception as e:
            print(f"Error processing default files: {e}")
            args.default = None

    if args.plots:
        apply_rivet_style(use_tex_preference=args.use_tex)
        print()
        if not file_analyses:
            print("No YODA files found for plotting.")
            return 0
        
        if group_by_name:
            groups = group_yoda_files_by_name(list(file_analyses.keys()), debug=debug)
        else:
            groups = group_yoda_files_by_analyses(file_analyses, debug=debug)
        
        if not groups:
            print("No valid YODA files found for grouping and plotting.")
            return 0
        
        if group_by_name:
            sorted_groups = sorted(groups.items(), key=lambda x: x[0])
        else:
            sorted_groups = groups.items()
        
        for group_idx, (signature, group_files) in enumerate(sorted_groups):
            if group_by_name:
                group_name = signature
            else:
                endings = set()
                for yfile in group_files:
                    name = yfile.stem
                    if name.endswith('.yoda'):
                        name = name[:-5]
                    if name.endswith('.yoda.gz'):
                        name = name[:-8]
                    if '-' in name:
                        ending = name.split('-')[-1]
                        endings.add(ending)
                if len(endings) > 1:
                    group_name = f"group_{group_idx+1}"
                else:
                    group_name = list(endings)[0] if endings else f"group_{group_idx+1}"
            
            if args.tags:
                group_files = sort_files_by_tag_order(group_files, args.tags)
            
            group_labels = []
            group_colors = []
            for yfile in group_files:
                if args.tags and args.labels and len(args.tags) == len(args.labels):
                    tag_label = get_label_for_file(yfile, args.tags, args.labels)
                    if tag_label:
                        group_labels.append(tag_label)
                    else:
                        group_labels.append(yfile.stem)
                else:
                    name = yfile.stem
                    if name.endswith('.yoda'):
                        name = name[:-5]
                    if name.endswith('.yoda.gz'):
                        name = name[:-8]
                    
                    if group_by_name:
                        parts = name.split('-')
                        if len(parts) >= 2:
                            label = parts[-2]
                        else:
                            label = name
                    else:
                        parts = name.split('-')
                        if len(parts) >= 2:
                            endings = set()
                            for f in group_files:
                                fname = f.stem
                                if fname.endswith('.yoda'):
                                    fname = fname[:-5]
                                if fname.endswith('.yoda.gz'):
                                    fname = fname[:-8]
                                if '-' in fname:
                                    endings.add(fname.split('-')[-1])
                            
                            if len(endings) > 1:
                                label = parts[-2] + "-" + parts[-1]
                            else:
                                label = parts[-2]
                        else:
                            label = name
                    group_labels.append(label)
                
                if args.tags and args.colors and len(args.tags) == len(args.colors):
                    tag_color = get_color_for_file(yfile, args.tags, args.colors)
                    if tag_color:
                        group_colors.append(tag_color)
                    else:
                        group_colors.append(None)
                else:
                    group_colors.append(None)
            
            group_chi2_plots = []
            group_valid_chi2s = []
            group_summaries = []
            
            for yfile, label in zip(group_files, group_labels):
                if yfile in file_results:
                    results = file_results[yfile]
                    chi2 = results['chi2']
                    ndf = results['ndf']
                    obs_chi2s = results['obs_chi2s']
                    valid_chi2s = results['valid_chi2s']
                    
                    group_valid_chi2s.append(valid_chi2s)
                    group_summaries.append({
                        "label": label,
                        "Global chi2": f"{chi2:.2f}",
                        "Degrees of freedom (ndf)": f"{int(ndf)}",
                        "Reduced chi2": f"{chi2 / ndf:.2f}" if ndf > 0 else "undefined (ndf = 0)",
                        "Averaged chi2s over all obs.": f"{np.mean(valid_chi2s):.2f}" if valid_chi2s else "undefined (no valid observables)"
                    })
                    
                    chi2_plot = create_chi2_plot_from_obs(obs_chi2s)
                    group_chi2_plots.append(chi2_plot)
                else:
                    print(f"Warning: No precomputed results found for {yfile}, skipping...")
            
            has_valid_data = any(group_chi2_plots) and any(any(valid_chi2s) for valid_chi2s in group_valid_chi2s)
            if group_chi2_plots and has_valid_data:
                group_outdir = os.path.join(args.outdir, group_name)
                print(f"Creating plots for {group_name} in {group_outdir}...")
                
                group_chi2_plot_def = None
                group_valid_chi2s_def = None
                default_label = args.default_label
                
                if default_file_analyses:
                    matching_default_file = find_matching_default_file(
                        signature, group_name, default_file_analyses, 
                        group_by_name=group_by_name, debug=debug)
                    
                    if matching_default_file and matching_default_file in default_file_results:
                        default_results = default_file_results[matching_default_file]
                        group_valid_chi2s_def = default_results['valid_chi2s']
                        
                        group_chi2_plot_def = create_chi2_plot_from_obs(default_results['obs_chi2s'])
                        
                        if debug:
                            print(f"  Using default file: {matching_default_file.name}")
                    else:
                        if debug:
                            print(f"  No matching default file found for group {group_name}")
                
                filtered_colors = [c for c in group_colors if c is not None]
                if len(filtered_colors) == len(group_labels):
                    final_colors = group_colors
                elif len(filtered_colors) > 0 and len(filtered_colors) != len(group_labels):
                    print(f"Warning: Only {len(filtered_colors)} colors specified for {len(group_labels)} files in group {group_name}, using default plasma colormap")
                    final_colors = None
                else:
                    final_colors = None
                
                plot_chi2_per_analysis(group_chi2_plots, group_labels, final_colors, chi2_plot_def=group_chi2_plot_def, default_label=default_label, default_color=args.default_color, outdir=group_outdir, debug=debug)
                plot_chi2_distribution(group_valid_chi2s, group_labels, final_colors, valid_chi2s_def=group_valid_chi2s_def, default_label=default_label, default_color=args.default_color, outdir=group_outdir, debug=debug)
                create_index_html(group_outdir, group_summaries, group_chi2_plots)
            else:
                print(f"Skipping {group_name}: no valid data to plot")
            print()
        create_master_index_html(args.outdir, groups, args.default_label if args.default_label else None, group_by_name)

    return 0


def process_files(args, loader, weights, analyses, valid_bins=None, debug=False):
    """Process individual YODA files."""
    if debug: print(f"Processing files: {args.yoda_files}")

    if args.labels and len(args.labels) == len(args.yoda_files):
        labels = args.labels
    else:
        if args.labels:
            print(f"Warning: Number of labels ({len(args.labels)}) does not match number of YODA files ({len(args.yoda_files)}), using file stems as labels")
        labels = [Path(yoda_file).stem for yoda_file in args.yoda_files]

    if args.colors and len(args.colors) == len(args.yoda_files):
        colors = args.colors
    else:
        if args.colors:
            print(f"Warning: Number of colors ({len(args.colors)}) does not match number of YODA files ({len(args.yoda_files)}), using default plasma colormap")
        colors = None

    all_valid_chi2s = []
    all_chi2_plots = []
    summaries = []

    for idx, yoda_file in enumerate(args.yoda_files):
        yoda_path = Path(yoda_file)
        if yoda_path.is_dir():
            print(f"Warning: {yoda_file} is a directory, please provide a specific YODA file, skipping...")
            continue
        if not yoda_path.exists():
            print(f"Warning: YODA file {yoda_file} does not exist, skipping...")
            continue
        
        label = labels[idx]

        differences, squared_errors, bin_weights, bin_names, _ = loader.get_bin_differences(
            str(yoda_file), weights, analyses=analyses, include_ref_error=True, debug=debug)
        chi2, ndf = global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=debug)
        obs_chi2s = obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=debug)
        valid_chi2s = [v for v in obs_chi2s.values() if v != -1]

        all_valid_chi2s.append(valid_chi2s)
        summaries.append({
            "label": label,
            "Global chi2": f"{chi2:.2f}",
            "Degrees of freedom (ndf)": f"{int(ndf)}",
            "Reduced chi2": f"{chi2 / ndf:.2f}" if ndf > 0 else "undefined (ndf = 0)",
            "Averaged chi2s over all obs.": f"{np.mean(valid_chi2s):.2f}" if valid_chi2s else "undefined (no valid observables)"
        })

        print(f"\nResults for {label}:")
        for k, v in summaries[-1].items():
            if k != "label":
                print(f"  {k}: {v}")

        if args.plots:
            chi2_plot = create_chi2_plot_from_obs(obs_chi2s)
            all_chi2_plots.append(chi2_plot)

    chi2_plot_def = None
    valid_chi2s_def = None
    if args.default:
        try:
            _, default_file_results = process_default_files(
                args.default, loader, weights, analyses, valid_bins, debug=debug)
            
            if default_file_results:
                default_file = list(default_file_results.keys())[0]
                default_results = default_file_results[default_file]
                
                chi2_def = default_results['chi2']
                ndf_def = default_results['ndf']
                obs_chi2s_def = default_results['obs_chi2s']
                valid_chi2s_def = default_results['valid_chi2s']
                
                summaries.append({
                    "label": args.default_label,
                    "Global chi2": f"{chi2_def:.2f}",
                    "Degrees of freedom (ndf)": f"{int(ndf_def)}",
                    "Reduced chi2": f"{chi2_def / ndf_def:.2f}" if ndf_def > 0 else "undefined (ndf = 0)",
                    "Averaged chi2s over all obs.": f"{np.mean(valid_chi2s_def):.2f}" if valid_chi2s_def else "undefined (no valid observables)"
                })

                print(f"\nResults for {args.default_label}:")
                for k, v in summaries[-1].items():
                    if k != "label":
                        print(f"  {k}: {v}")

                if args.plots:
                    chi2_plot_def = create_chi2_plot_from_obs(obs_chi2s_def)
            else:
                print(f"Warning: No valid default files found from: {args.default}")
                
        except Exception as e:
            print(f"Error processing default files: {e}")
            if debug:
                import traceback
                traceback.print_exc()

    if args.plots:
        apply_rivet_style(use_tex_preference=args.use_tex)
        if not all_chi2_plots:
            print("No valid YODA files found. Skipping plot and HTML creation")
        else:
            print("\nCreating plots...")
            plot_chi2_per_analysis(all_chi2_plots, labels, colors, chi2_plot_def=chi2_plot_def, default_label=args.default_label, default_color=args.default_color, outdir=args.outdir, debug=debug)
            plot_chi2_distribution(all_valid_chi2s, labels, colors, valid_chi2s_def=valid_chi2s_def, default_label=args.default_label, default_color=args.default_color, outdir=args.outdir, debug=debug)

            print("\nCreating index.html...")
            create_index_html(args.outdir, summaries, all_chi2_plots)
    
    return 0


if __name__ == "__main__":
    main()