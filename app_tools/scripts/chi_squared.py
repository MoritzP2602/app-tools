
import os
import yoda
import rivet
import argparse
import datetime
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from pathlib import Path
from tabulate import tabulate

if tuple(map(int, yoda.__version__.split("."))) < (2, 1, 0):
    print("Warning: This script is optimized for YODA 2.1.0. You are using YODA", yoda.__version__, "... please double-check your results!")


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
                raise FileNotFoundError(f"Neither {gz_path} nor {yoda_path} found.")
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

    def get_bin_differences(self, _yd, weights=None, include_ref_error=True, debug=False):
        """
        Compute differences between data and reference YODA files.

        :param _yd: Path to the YODA file containing data.
        :param weights: Optional weights for the observables.
        :param include_ref_error: Whether to include reference errors in the computation.
        :param debug: Whether to print debug information.
        :return: Tuple of differences, squared errors, bin weights, and bin names.
        """
        if debug:
            print(f"Executing get_bin_differences for {_yd}")

        yd = yoda.readYODA(_yd)
        differences = []
        squared_errors = []
        bin_names = []
        bin_weights = []
        yoda_version = tuple(map(int, yoda.__version__.split(".")))

        for a in yd:
            if weights is not None and (a not in weights and not any(k.startswith(a + "#") for k in weights)):
                if debug:
                    print(f"  Skipping observable {a} (not in weights file)")
                continue
            try:
                analysis_name = a.split("/")[1].split(":")[0]
            except (IndexError, AttributeError):
                if debug:
                    print(f"  Failed to parse analysis name from {a}")
                continue
            
            obs_weight = 1.0
            if weights and a in weights:
                obs_weight = weights[a]
            
            try:
                ref_yoda = self.load(analysis_name)
                a_ref = f"/REF/{analysis_name}/{a.split('/')[-1]}"
                obj = yd[a]
                ref_obj = ref_yoda.get(a_ref)
                
                if ref_obj is None:
                    if debug:
                        print(f"  Reference object not found for {a_ref}")
                    continue
            except (FileNotFoundError, KeyError) as e:
                if debug:
                    print(f"  Error loading reference data for {analysis_name}: {e}")
                continue

            if debug:
                print(f"  Processing: {obj}, /REF: {ref_obj}")

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
                if debug:
                    print(f"  Data and reference bins do not match in length: {len(data_bins)} vs {len(ref_bins)} for observable {a}")
                continue
            
            n_bins = min(len(data_bins), len(ref_bins))
            for i in range(n_bins):
                bin_name = data_bins[i]
                if data_errs[i] == 0 or np.isnan(data_errs[i]):
                    if debug:
                        print(f"  Skipping bin {bin_name} due to data error=0 or nan (data_err={data_errs[i]})")
                    continue
                
                diff = data_vals[i] - ref_vals[i]
                if include_ref_error:
                    err = np.sqrt(data_errs[i] ** 2 + ref_errs[i] ** 2)
                else:
                    err = data_errs[i]
                
                bin_weight = obs_weight
                if weights and bin_name in weights:
                    if debug:
                        print(f"  Using weight for bin {bin_name}: {weights[bin_name]}")
                    bin_weight = weights[bin_name]
                
                differences.append(diff)
                squared_errors.append(err ** 2)
                bin_names.append(bin_name)
                bin_weights.append(bin_weight)
        
        return np.array(differences), np.array(squared_errors), np.array(bin_weights), bin_names
    
    def get_valid_bins(self, up_file, dn_file, weights=None, debug=False):
        """
        Check if reference values are within envelope bounds.
        
        :param up_file: Path to YODA file with maximum values
        :param dn_file: Path to YODA file with minimum values  
        :param weights: Dictionary of weights (optional)
        :param debug: If True, print debug info
        :return: List of valid bin names
        """
        if debug:
            print(f"Executing get_valid_bins for {up_file}, {dn_file}")

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
                    if debug:
                        print(f"  Up object not found for {obs_name}")
                    continue
                if dn_obj is None:
                    if debug:
                        print(f"  Down object not found for {obs_name}")
                    continue
                if ref_obj is None:
                    if debug:
                        print(f"  Reference object not found for {ref_path}")
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
                    if debug:
                        print(f"  Bin count mismatch for {obs_name}: up={len(up_bins)}, dn={len(dn_bins)}, ref={len(ref_bins)}")
                    continue
                
                for i in range(len(up_bins)):
                    bin_name = up_bins[i]
                    up_val = up_vals[i]
                    dn_val = dn_vals[i]
                    ref_val = ref_vals[i]
                    
                    if np.isnan(up_val) or np.isnan(dn_val) or np.isnan(ref_val):
                        if debug:
                            print(f"  NaN value in bin {bin_name}: up={up_val}, dn={dn_val}, ref={ref_val}")
                        continue
                    
                    min_val = min(up_val, dn_val)
                    max_val = max(up_val, dn_val)
                    
                    if min_val <= ref_val <= max_val:
                        valid_bins.append(bin_name)
                    else:
                        invalid_bins.append((bin_name, ref_val, min_val, max_val))
                        
            except (FileNotFoundError, KeyError) as e:
                if debug:
                    print(f"  Error processing {obs_name}: {e}")
                continue
            except Exception as e:
                if debug:
                    print(f"  Unexpected error processing {obs_name}: {e}")
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
    if debug:
        print("Executing obs_chi2")

    obs_bins = {}
    chi2s = {}
    
    for idx, name in enumerate(bin_names):
        obs = name.split("#")[0]
        if valid_bins is not None and name not in valid_bins:
            if debug:
                print(f"  Skipping bin {name} (not in valid envelope bins)")
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
                raise ValueError("  Error: Bin weights > 1 encountered in weighted mode.")
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
        
        if debug:
            print(f"  Observable: {obs}, chi2: {chi2s[obs]}, ndf: {ndf}")
    
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
    if debug:
        print("Executing global_chi2")

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
            raise ValueError("  Error: Bin weights > 1 encountered in weighted mode.")
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


def plot_chi2_per_analysis(all_chi2_plots, labels, chi2_plot_def=None, default_label="default", outdir="chi2_plots"):
    """
    Plot chi2 per analysis for multiple tunes, optionally comparing to default.
    """
    os.makedirs(outdir, exist_ok=True)
    reds = plt.get_cmap('Reds')
    colors = [reds(0.4 + 0.5 * i / max(1, len(labels)-1)) for i in range(len(labels))]

    all_analysis = set()
    for chi2_plot in all_chi2_plots:
        all_analysis.update(chi2_plot.keys())
    if chi2_plot_def:
        all_analysis.update(chi2_plot_def.keys())

    for analysis_name in sorted(all_analysis):
        tune_bin_ids = []
        tune_chi2s = []
        for chi2_plot in all_chi2_plots:
            if analysis_name in chi2_plot:
                bin_ids, chi2_values = zip(*chi2_plot[analysis_name])
                tune_bin_ids.append(bin_ids)
                tune_chi2s.append(chi2_values)
            else:
                tune_bin_ids.append(())
                tune_chi2s.append(())

        ref_idx = next((i for i, b in enumerate(tune_bin_ids) if b), 0)
        bin_ids = tune_bin_ids[ref_idx] if tune_bin_ids[ref_idx] else []
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
                2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, figsize=(8, 6))

            for i, (chi2s, label) in enumerate(zip(tune_chi2s, labels)):
                if not chi2s:
                    continue
                chunk_chi2s = chi2s[start:end]
                ax_top.plot(chunk_bin_ids, chunk_chi2s, marker='o', markersize=4, linestyle='', label=label, color=colors[i])
                ax_top.plot(chunk_bin_ids, chunk_chi2s, linestyle='-', alpha=0.5, color=colors[i])
            if chi2_values_def is not None:
                chunk_chi2_values_def = chi2_values_def[start:end]
                ax_top.plot(chunk_bin_ids, chunk_chi2_values_def, marker='o', markersize=4, color='cornflowerblue', linestyle='', label=default_label)
                ax_top.plot(chunk_bin_ids, chunk_chi2_values_def, linestyle='-', color='cornflowerblue', alpha=0.5)
            ax_top.set_ylabel(r'$\chi^2 / \mathrm{ndf}$')
            if n_chunks == 1:
                ax_top.set_title(f'{analysis_name}')
            else:
                ax_top.set_title(f'{analysis_name} (part {chunk_idx + 1}/{n_chunks})')
            ax_top.legend()
            ax_top.grid(True, axis='y')
            ax_top.tick_params(axis='x', which='both', bottom=True, top=False, length=3.5, direction='in', labelbottom=False)

            if chi2_values_def is not None:
                for i, (chi2s, label) in enumerate(zip(tune_chi2s, labels)):
                    if not chi2s:
                        continue
                    chunk_chi2s = np.array(chi2s[start:end])
                    chunk_chi2_values_def = np.array(chi2_values_def[start:end])
                    ratio = chunk_chi2s / chunk_chi2_values_def
                    ax_bottom.plot(chunk_bin_ids, ratio, marker='o', markersize=4, linestyle='', label=f"{label}/{default_label}", color=colors[i])
                    ax_bottom.plot(chunk_bin_ids, ratio, linestyle='-', alpha=0.5, color=colors[i])
                ax_bottom.axhline(1, color='black', linestyle='--', alpha=0.2)
                ax_bottom.set_ylabel(f'tune/{default_label}')
                ax_bottom.set_xlabel('observable')
                ax_bottom.set_ylim(0.0, 2.0)
                ax_bottom.grid(True, axis='y')
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


def plot_chi2_distribution(all_valid_chi2s, labels, valid_chi2s_def=None, default_label="default", outdir="chi2_plots"):
    """
    Plot the distribution of chi2 values (log10) for multiple YODA files.
    """
    os.makedirs(outdir, exist_ok=True)
    all_logs = []
    for chi2s in all_valid_chi2s:
        all_logs.extend(np.log10(chi2s))
    if valid_chi2s_def is not None:
        all_logs.extend(np.log10(valid_chi2s_def))
    bins = np.linspace(np.min(all_logs), np.max(all_logs), 51)

    all_hists = []
    for chi2s in all_valid_chi2s:
        hist, _ = np.histogram(np.log10(chi2s), bins=bins, density=True)
        all_hists.append(hist)
    if valid_chi2s_def is not None:
        hist_def, _ = np.histogram(np.log10(valid_chi2s_def), bins=bins, density=True)
        all_hists.append(hist_def)
    global_ylim = (0, max([np.max(h) for h in all_hists]) * 1.08)

    n_plots = len(all_valid_chi2s) + (1 if valid_chi2s_def is not None else 0)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, max(5, 2 * n_plots)), sharex=True, gridspec_kw={'hspace': 0})

    if n_plots == 1:
        axes = [axes]

    reds = plt.get_cmap('Reds')
    colors = [reds(0.4 + 0.5 * i / max(1, len(all_valid_chi2s)-1)) for i in range(len(all_valid_chi2s))]

    plot_idx = 0
    for chi2s, label, color in zip(all_valid_chi2s, labels, colors):
        hist, bin_edges = np.histogram(np.log10(chi2s), bins=bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax = axes[plot_idx]
        ax.bar(bin_centers, hist, width=(bin_edges[1]-bin_edges[0]), alpha=0.2, color=color, label=None)
        ax.step(bin_edges, np.append(hist, hist[-1]), where='post', color=color, linewidth=1, alpha=0.7, label=label)
        ax.set_ylabel('Density')
        ax.set_ylim(global_ylim)
        ax.legend(loc='upper right')
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))
        plot_idx += 1

    if valid_chi2s_def is not None:
        hist_def, bin_edges_def = np.histogram(np.log10(valid_chi2s_def), bins=bins, density=True)
        bin_centers_def = 0.5 * (bin_edges_def[:-1] + bin_edges_def[1:])
        ax = axes[plot_idx]
        ax.bar(bin_centers_def, hist_def, width=(bin_edges_def[1]-bin_edges_def[0]), alpha=0.2, color='cornflowerblue', label=None)
        ax.step(bin_edges_def, np.append(hist_def, hist_def[-1]), where='post', color='cornflowerblue', linewidth=1, alpha=0.7, label=default_label)
        ax.set_ylabel('Density')
        ax.set_ylim(global_ylim)
        ax.legend(loc='upper right')
        ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, prune='both'))

    axes[-1].set_xlabel(r'$\log_{10}(\chi^2 / \mathrm{ndf})$')
    fname = f"{outdir}/chi2_distribution.pdf"
    plt.savefig(fname, dpi=300)
    if fname.endswith('.pdf'):
        plt.savefig(fname.replace('.pdf', '.png'), dpi=150)
    plt.close(fig)
    print(f"Plot of Chi2 distribution saved to {outdir}/chi2_distribution.pdf/png")


def create_index_html(outdir="chi2_plots", summaries=None, all_chi2_plots=None):
    """
    Create an index.html file in the output directory to access all plot files.
    """
    plot_files = []
    for fname in sorted(os.listdir(outdir)):
        if fname.endswith((".pdf", ".png", ".jpg", ".jpeg", ".svg")):
            plot_files.append(fname)

    from collections import defaultdict
    exp_analysis_groups = defaultdict(list)
    for fname in plot_files:
        parts = fname.split('_')
        if parts[0] == "chi2":
            continue
        if len(parts) >= 3:
            group = "_".join(parts[:3])
        elif len(parts) >= 2:
            group = "_".join(parts[:2])
        else:
            group = parts[0]
        exp_analysis_groups[group].append(fname)

    now = datetime.datetime.now().strftime("%A, %d. %B %Y %H:%M")
    html = f"""<html>
    <head>
        <title>Plots from chi_squared.py</title>
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
        <h2>Plots from chi_squared.py</h2>
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
        <p>Generated at {now}</p>
        <p>Created with command: <pre>{" ".join(os.sys.argv)}</pre></p>
        </footer>
    </body>
    </html>
    """

    with open(os.path.join(outdir, "index.html"), "w") as f:
        f.write(html)
    print(f"Created {os.path.join(outdir, 'index.html')}")


def main():
    parser = argparse.ArgumentParser(description="Compute global chi2 from YODA file and weights.")
    parser.add_argument("weights_file", help="Path to the weights file")
    parser.add_argument("yoda_files", nargs="+", help="Path(s) to one or more YODA files or a directory")
    parser.add_argument("--labels", nargs="+", help="Labels for each YODA file (same order)")
    parser.add_argument("--weighted", action="store_true", default=False, help="Enable weighted chi2 calculation")
    parser.add_argument("--plots", action="store_true", default=False, help="Enable plotting of chi2 per analysis")
    parser.add_argument("--default", default=None, help="Path to a YODA file for comparison of chi2 (default: None)")
    parser.add_argument("--default-label", default="default", help="Label for the default YODA file in output and plots")
    parser.add_argument("--envelope", nargs=2, metavar=("up.yoda", "dn.yoda"), help="Check if reference values are within envelope (up.yoda dn.yoda)")
    parser.add_argument("-o", "--outdir", default=None, help="Output directory for chi2 plots")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug output")
    args = parser.parse_args()

    print("Starting chi2 analysis...")

    try:
        loader = yodaLoader()
        weights = read_weights(args.weights_file)
    except FileNotFoundError as e:
        print(f"Error: Weights file not found: {e}")
        return 1
    except Exception as e:
        print(f"Error reading weights file: {e}")
        return 1

    if args.default:
        if not Path(args.default).exists():
            print(f"\nWarning: YODA file (default) {args.default} does not exist.")
            args.default = None
    
    if args.outdir is None:
        parent_folder = Path.cwd().name
        args.outdir = f"{parent_folder}.chi2"

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
            return process_directory(args, loader, weights, valid_bins)
        else:
            return process_files(args, loader, weights, valid_bins)
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


def process_directory(args, loader, weights, valid_bins=None):
    """Process a directory containing YODA files."""
    yoda_path = Path(args.yoda_files[0])
    reduced_chi2_dict = {}
    avg_chi2_dict = {}
    ndf_dict = {}

    yoda_files = list(yoda_path.glob("*.yoda"))
    for yfile in yoda_files:
        differences, squared_errors, bin_weights, bin_names = loader.get_bin_differences(
            str(yfile), weights, include_ref_error=True, debug=args.debug)
        chi2, ndf = global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
        obs_chi2s = obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
        valid_chi2s = [v for v in obs_chi2s.values() if v != -1]

        key = yfile.stem
        reduced_chi2_dict[key] = chi2 / ndf if ndf > 0 else float('nan')
        avg_chi2_dict[key] = np.mean(valid_chi2s) if valid_chi2s else float('nan')
        ndf_dict[key] = ndf

    for subfolder in yoda_path.iterdir():
        if subfolder.is_dir():
            yoda_files = list(subfolder.glob("*.yoda"))
            if not yoda_files:
                continue
            for yfile in yoda_files:
                differences, squared_errors, bin_weights, bin_names = loader.get_bin_differences(
                    str(yfile), weights, include_ref_error=True, debug=args.debug)
                chi2, ndf = global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
                obs_chi2s = obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
                valid_chi2s = [v for v in obs_chi2s.values() if v != -1]

                key = subfolder.name
                reduced_chi2_dict[key] = chi2 / ndf if ndf > 0 else float('nan')
                avg_chi2_dict[key] = np.mean(valid_chi2s) if valid_chi2s else float('nan')
                ndf_dict[key] = ndf

    reduced_chi2_dict = {key: reduced_chi2_dict[key] for key in sorted(reduced_chi2_dict.keys())}

    print("\n" + "Input directory: " + str(yoda_path.resolve()))
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

    if args.default:
        differences_def, squared_errors_def, bin_weights_def, bin_names_def = loader.get_bin_differences(
            args.default, weights, include_ref_error=True, debug=args.debug)
        chi2_def, ndf_def = global_chi2(differences_def, squared_errors_def, bin_weights_def, bin_names_def, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
        obs_chi2s_def = obs_chi2(differences_def, squared_errors_def, bin_weights_def, bin_names_def, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
        valid_chi2s_def = [v for v in obs_chi2s_def.values() if v != -1]
        print()
        print(f"Label: {args.default_label}")
        print(f"  Degrees of freedom (ndf): {ndf_def:.0f}")
        print(f"  Reduced chi2: {chi2_def / ndf_def if ndf_def > 0 else 'undefined (ndf = 0)'}")
        print(f"  Averaged chi2s over all obs.: {np.mean(valid_chi2s_def) if valid_chi2s_def else 'undefined (no valid observables)'}")
    
    return 0

def process_files(args, loader, weights, valid_bins=None):
    """Process individual YODA files."""
    if args.labels and len(args.labels) == len(args.yoda_files):
        labels = args.labels
    else:
        labels = [Path(yoda_file).stem for yoda_file in args.yoda_files]

    all_valid_chi2s = []
    all_chi2_plots = []
    summaries = []

    for idx, yoda_file in enumerate(args.yoda_files):
        yoda_path = Path(yoda_file)
        if yoda_path.is_dir():
            print(f"\nWarning: {yoda_file} is a directory, please provide a specific YODA file. Skipping.")
            continue
        if not yoda_path.exists():
            print(f"\nWarning: YODA file {yoda_file} does not exist. Skipping.")
            continue
        
        label = labels[idx]

        differences, squared_errors, bin_weights, bin_names = loader.get_bin_differences(
            str(yoda_file), weights, include_ref_error=True, debug=args.debug)
        chi2, ndf = global_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
        obs_chi2s = obs_chi2(differences, squared_errors, bin_weights, bin_names, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
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
            chi2_plot = {}
            for obs, chi2_value in obs_chi2s.items():
                if chi2_value == -1:
                    continue
                analysis_name = obs.split("/")[1]
                bin_id = obs.split("/")[2].split("-")[0]
                if analysis_name not in chi2_plot:
                    chi2_plot[analysis_name] = []
                existing_bin_ids = [b[0] for b in chi2_plot[analysis_name]]
                new_bin_id = bin_id
                suffix = ord('b')
                while new_bin_id in existing_bin_ids:
                    new_bin_id = f"{bin_id}-{chr(suffix)}"
                    suffix += 1
                chi2_plot[analysis_name].append((new_bin_id, chi2_value))
            all_chi2_plots.append(chi2_plot)

    chi2_plot_def = None
    valid_chi2s_def = None
    
    if args.default:
        differences_def, squared_errors_def, bin_weights_def, bin_names_def = loader.get_bin_differences(
            args.default, weights, include_ref_error=True, debug=args.debug)
        chi2_def, ndf_def = global_chi2(differences_def, squared_errors_def, bin_weights_def, bin_names_def, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
        obs_chi2s_def = obs_chi2(differences_def, squared_errors_def, bin_weights_def, bin_names_def, weighted=args.weighted, valid_bins=valid_bins, debug=args.debug)
        valid_chi2s_def = [v for v in obs_chi2s_def.values() if v != -1]
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
            chi2_plot_def = {}
            for obs, chi2_value in obs_chi2s_def.items():
                if chi2_value == -1:
                    continue
                analysis_name = obs.split("/")[1]
                bin_id = obs.split("/")[2].split("-")[0]
                if analysis_name not in chi2_plot_def:
                    chi2_plot_def[analysis_name] = []
                existing_bin_ids = [b[0] for b in chi2_plot_def[analysis_name]]
                new_bin_id = bin_id
                suffix = ord('b')
                while new_bin_id in existing_bin_ids:
                    new_bin_id = f"{bin_id}-{chr(suffix)}"
                    suffix += 1
                chi2_plot_def[analysis_name].append((new_bin_id, chi2_value))

    if args.plots:
        if not all_chi2_plots:
            print("\nNo valid YODA files found. Skipping plot and HTML creation.")
        else:
            print("\nCreating plots...")
            plot_chi2_per_analysis(all_chi2_plots, labels, chi2_plot_def=chi2_plot_def, default_label=args.default_label, outdir=args.outdir)
            plot_chi2_distribution(all_valid_chi2s, labels, valid_chi2s_def=valid_chi2s_def, default_label=args.default_label, outdir=args.outdir)

            print("\nCreating index.html...")
            create_index_html(args.outdir, summaries, all_chi2_plots)
    
    return 0


if __name__ == "__main__":
    main()