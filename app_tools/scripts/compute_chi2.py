import numpy as np
import argparse
import json
import os
import re
import sys
import rivet
import yoda
from pathlib import Path
from tabulate import tabulate


MIN_YODA = (2, 1, 0)
def check_yoda_version():
	if tuple(int(x) for x in yoda.__version__.split(".")[:3]) < MIN_YODA:
		raise RuntimeError(f"YODA >= {'.'.join(map(str, MIN_YODA))} is required, found {yoda.__version__}")


def read_weights(wfile):
	"""Read observable/bin weights from text file."""

	weights = {}
	with open(wfile, "r") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			parts = line.split()
			if len(parts) < 2:
				continue
			weights[parts[0]] = float(parts[1])
	return weights


def read_analyses(afile):
	"""Read analyses allow-list from text file."""

	analyses = set()
	with open(afile, "r") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			analyses.add(line)
	return analyses

def is_estimate(obj):
	return "Estimate" in type(obj).__name__

def is_histo(obj):
	return "Histo" in type(obj).__name__

def is_profile(obj):
	return "Profile" in type(obj).__name__

def is_scatter(obj):
	return "Scatter" in type(obj).__name__

def accepted_type(obj):
	"""Check if object is a supported YODA type."""

	if not hasattr(obj, "type") or not callable(obj.type):
		return False
	if is_estimate(obj):
		return True
	if is_histo(obj):
		return True
	if is_profile(obj):
		return True
	if is_scatter(obj):
		return True
	return False


class YodaLoader:
	"""YODA helper for loading refs and extracting bin/value/error arrays."""

	def __init__(self):
		self.paths = rivet.getAnalysisRefPaths()
		self.yodas = {}
		self.error_counts = {}

	def load(self, analysis_name):
		if analysis_name in self.yodas:
			return self.yodas[analysis_name]

		ref_yoda = None
		for base_path in self.paths:
			gz_path = f"{base_path}/{analysis_name}.yoda.gz"
			yoda_path = f"{base_path}/{analysis_name}.yoda"
			if os.path.exists(gz_path):
				ref_yoda = yoda.readYODA(gz_path)
				break
			if os.path.exists(yoda_path):
				ref_yoda = yoda.readYODA(yoda_path)
				break

		if ref_yoda is None:
			raise FileNotFoundError(
				f"Reference YODA for {analysis_name} not found")

		self.yodas[analysis_name] = ref_yoda
		return ref_yoda
	
	def extract_bins_profile(self, obj):
		if hasattr(obj, "path") and callable(obj.path):
			path = obj.path()
		else:
			raise TypeError(f"Unsupported YODA object {type(obj).__name__}: missing callable path() method.")
		
		# --- Note ------------------------------------------------------------------------------- #
		# For Profile objects, mkScatter() and point.errs() returns errors equal to zero.          #
	    # This means, we have to use Profile specific methods in this case (tested for Profile1D). #
		# ---------------------------------------------------------------------------------------- #

		dim = obj.dim()
		if dim != 2:
			print(f"Warning: Profile object with dim != 2 encountered. "
		 		  f"Please double check the results for this observable, obj: {obj}.")
		bin_dict = {}
		for i, bin in enumerate(obj.bins()):
			bin_name = f"{path}#{i}"
			if hasattr(bin, "mean") and callable(bin.mean):
				val = bin.mean(dim)
			else: val = np.nan
			if hasattr(bin, "stdErr") and callable(bin.stdErr):
				err = bin.stdErr(dim)
			else: err = np.nan
			bin_dict[bin_name] = (val, err)
		return bin_dict
	
	def extract_bins_scatter(self, obj):
		if hasattr(obj, "path") and callable(obj.path):
			path = obj.path()
		else:
			raise TypeError(f"Unsupported YODA object {type(obj).__name__}: missing callable path() method.")
		if not hasattr(obj, "mkScatter") or not callable(obj.mkScatter):
			raise TypeError(f"Unsupported YODA object {type(obj).__name__}: missing callable mkScatter() method.")
		scatter = obj.mkScatter()
		
		dim = scatter.dim()
		bin_dict = {}
		for i, p in enumerate(scatter.points()):
			bin_name = f"{path}#{i}"
			if hasattr(p, "val") and callable(p.val):
				val = p.val(dim - 1)
			else: val = np.nan
			if hasattr(p, "errs") and callable(p.errs):
				err = sum(p.errs(dim - 1)) / 2
			else: err = np.nan
			bin_dict[bin_name] = (val, err)
		return bin_dict

	def extract_bins(self, obj):	
		if is_estimate(obj):
			return self.extract_bins_scatter(obj)
		elif is_scatter(obj):
			return self.extract_bins_scatter(obj)
		elif is_histo(obj):
			return self.extract_bins_scatter(obj)
		elif is_profile(obj):
			return self.extract_bins_profile(obj)
		else:
			raise TypeError(f"Unsupported YODA object type: {type(obj).__name__}")

	def get_bin_differences(self, yoda_file, weights_dict=None, analyses_set=None, include_ref_error=True, debug=False):
		if debug: print(f"Executing get_bin_differences for {yoda_file}")

		self.error_counts[yoda_file] = {}

		if analyses_set:
			ana_union = "|".join(re.escape(a) for a in sorted(analyses_set))
			yd = yoda.readYODA(yoda_file, patterns=rf"^/({ana_union})(:.*)?/.*$")
		else:
			yd = yoda.readYODA(yoda_file)

		obs_weights_set = set()
		if weights_dict is not None:
			for key in weights_dict.keys():
				obs_weights_set.add(key.split("#")[0])

		obs_dict = {}
		for obs_name in yd.keys():
			if weights_dict is not None and (obs_name not in obs_weights_set):
				if debug: print(f"  Skipping observable {obs_name}. Not in weights file.")
				self.error_counts[yoda_file]["Observable not in weights file"] = self.error_counts[yoda_file].get("Observable not in weights file", 0) + 1
				continue

			try:
				analysis_name = obs_name.split("/")[1].split(":")[0]
			except (IndexError, AttributeError):
				if debug: print(f"  Skipping observable {obs_name}. Failed to parse analysis name.")
				self.error_counts[yoda_file]["Failed to parse analysis name"] = self.error_counts[yoda_file].get("Failed to parse analysis name", 0) + 1
				continue
			if analyses_set is not None and analysis_name not in analyses_set:
				if debug: print(f"  Skipping observable {obs_name}. Analysis {analysis_name} not in analyses list.")
				self.error_counts[yoda_file]["Analysis name not in list"] = self.error_counts[yoda_file].get("Analysis name not in list", 0) + 1
				continue

			try:
				yd_ref  = self.load(analysis_name)
				obs_ref = f"/REF/{analysis_name}/{obs_name.split('/')[-1]}"
				histogram     = yd[obs_name]
				histogram_ref = yd_ref.get(obs_ref)
				if histogram_ref is None:
					if debug: print(f"  Skipping observable {obs_name}. Reference data not found in /REF/{analysis_name}.")
					self.error_counts[yoda_file]["Reference data not found"] = self.error_counts[yoda_file].get("Reference data not found", 0) + 1
					continue
			except (FileNotFoundError, KeyError) as e:
				if debug: print(f"  Skipping observable {obs_name}. Error loading reference data for {analysis_name}: {e}.")
				self.error_counts[yoda_file]["Error loading reference data"] = self.error_counts[yoda_file].get("Error loading reference data", 0) + 1
				continue

			if debug: print(f"  Processing observable {obs_name}, type {histogram.type()}, "
							f"with reference {obs_ref}, type {histogram_ref.type()}.")
			if not (accepted_type(histogram) and accepted_type(histogram_ref)):
				if debug: print(f"  Skipping observable {obs_name}. Unsupported object type in one of the datasets.")
				self.error_counts[yoda_file]["Unsupported object type"] = self.error_counts[yoda_file].get("Unsupported object type", 0) + 1
				continue

			bin_dict = self.extract_bins(histogram)
			bin_dict_ref = self.extract_bins(histogram_ref)

			if len(bin_dict) != len(bin_dict_ref):
				if debug: print(f"  Skipping observable {obs_name}. Bin count mismatch between data and reference: "
						  		f"data = {len(bin_dict)}, ref = {len(bin_dict_ref)}.")
					
				self.error_counts[yoda_file]["Bin count mismatch"] = self.error_counts[yoda_file].get("Bin count mismatch", 0) + 1
				continue

			obs_dict[obs_name] = {}
			for bin_name, (data_val, data_err) in bin_dict.items():
				bin_name_ref = f"/REF" + bin_name if not bin_name.startswith("/REF") else bin_name
				if bin_name_ref not in bin_dict_ref:
					if debug: print(f"    Skipping bin {bin_name}. Not found in reference for {obs_name}.")
					self.error_counts[yoda_file]["Bin not found in reference data"] = self.error_counts[yoda_file].get("Bin not found in reference data", 0) + 1
					continue

				ref_val, ref_err = bin_dict_ref[bin_name_ref]
				diff = data_val - ref_val
				err = np.sqrt(data_err ** 2 + ref_err ** 2) if include_ref_error else data_err
				if err == 0 or np.isnan(err):
					if debug: print(f"    Skipping bin {bin_name}. Invalid combined error {err}.")
					self.error_counts[yoda_file]["Invalid combined error"] = self.error_counts[yoda_file].get("Invalid combined error", 0) + 1
					continue

				if not weights_dict:
					bin_weight = 1.0
				else:
					bin_weight = weights_dict.get(bin_name, weights_dict.get(obs_name, 0.0))
				obs_dict[obs_name][bin_name] = (diff, err, bin_weight)
		return obs_dict

	def get_valid_bins(self, up_file, dn_file, weights_dict=None, analyses_set=None, debug=False):
		if debug: print(f"Executing get_valid_bins for {up_file}, {dn_file}")

		obs_weights_set = set()
		if weights_dict is not None:
			for key in weights_dict.keys():
				obs_weights_set.add(key.split("#")[0])

		if obs_weights_set:
			patterns = [rf"^{re.escape(o)}$" for o in obs_weights_set]
			yd_up = yoda.readYODA(up_file, patterns=patterns)
			yd_dn = yoda.readYODA(dn_file, patterns=patterns)
		else:
			yd_up = yoda.readYODA(up_file)
			yd_dn = yoda.readYODA(dn_file)

		valid_bins_list, invalid_bins_list = [], []
		obs_weights_set = obs_weights_set if obs_weights_set else set(yd_up.keys())

		for obs_name in obs_weights_set:
			try:
				analysis_name = obs_name.split("/")[1].split(":")[0]
			except (IndexError, AttributeError):
				if debug: print(f"  Skipping observable {obs_name}. Failed to parse analysis name.")
				self.error_counts["Envelope"]["Failed to parse analysis name"] = self.error_counts["Envelope"].get("Failed to parse analysis name", 0) + 1
				continue
			if analyses_set is not None and analysis_name not in analyses_set:
				if debug: print(f"  Skipping observable {obs_name}. Analysis {analysis_name} not in analyses list.")
				self.error_counts["Envelope"]["Analysis name not in list"] = self.error_counts["Envelope"].get("Analysis name not in list", 0) + 1
				continue

			try:
				analysis_name = obs_name.split("/")[1].split(":")[0]
				ref_yoda = self.load(analysis_name)
				ref_path = f"/REF/{analysis_name}/{obs_name.split('/')[-1]}"

				obj_up  = yd_up.get(obs_name)
				obj_dn  = yd_dn.get(obs_name)
				obj_ref = ref_yoda.get(ref_path)

				if obj_up is None or obj_dn is None or obj_ref is None:
					if debug: print(f"  Skipping observable {obs_name}. Missing object found.")
					self.error_counts["Envelope"]["Observable data not found"] = self.error_counts["Envelope"].get("Observable data not found", 0) + 1
					continue

				if debug: print(f"  Processing observable {obs_name} for envelope validation, type up: {obj_up.type()}, "
								f"dn: {obj_dn.type()}, reference type {obj_ref.type()}.")
				if not (accepted_type(obj_up) and accepted_type(obj_dn) and accepted_type(obj_ref)):
					if debug: print(f"  Skipping observable {obs_name}. Unsupported object type in one of the datasets.")
					self.error_counts["Envelope"]["Unsupported object type"] = self.error_counts["Envelope"].get("Unsupported object type", 0) + 1
					continue

				bin_dict_up  = self.extract_bins(obj_up)
				bin_dict_dn  = self.extract_bins(obj_dn)
				bin_dict_ref = self.extract_bins(obj_ref)

				if len(bin_dict_up) != len(bin_dict_dn) or len(bin_dict_up) != len(bin_dict_ref):
					if debug: print(f"  Skipping observable {obs_name}. Bin count mismatch: "
							  		f"up = {len(bin_dict_up)}, dn = {len(bin_dict_dn)}, ref = {len(bin_dict_ref)}.")
					self.error_counts["Envelope"]["Bin count mismatch"] = self.error_counts["Envelope"].get("Bin count mismatch", 0) + 1
					continue

				for bin_name in bin_dict_up.keys():
					bin_name_ref = f"/REF" + bin_name if not bin_name.startswith("/REF") else bin_name
					if bin_name not in bin_dict_dn or bin_name_ref not in bin_dict_ref:
						if debug: print(f"    Skipping bin {bin_name}. Not found in all datasets for {obs_name}.")
						self.error_counts["Envelope"]["Bin not found in all datasets"] = self.error_counts["Envelope"].get("Bin not found in all datasets", 0) + 1
						continue
					up_val, dn_val, ref_val = bin_dict_up[bin_name][0], bin_dict_dn[bin_name][0], bin_dict_ref[bin_name_ref][0]
					if np.isnan(up_val) or np.isnan(dn_val) or np.isnan(ref_val):
						if debug: print(f"    Skipping bin {bin_name}. NaN value encountered in one of the datasets for {obs_name}.")
						self.error_counts["Envelope"]["NaN value in datasets"] = self.error_counts["Envelope"].get("NaN value in datasets", 0) + 1
						continue

					min_val, max_val = min(up_val, dn_val), max(up_val, dn_val)
					if min_val <= ref_val <= max_val:
						valid_bins_list.append(bin_name)
					else:
						invalid_bins_list.append(bin_name)

			except Exception as e:
				if debug: print(f"  Skipping observable {obs_name}. Error: {e}.")
				self.error_counts["Envelope"]["Error processing observable"] = self.error_counts["Envelope"].get("Error processing observable", 0) + 1
				continue

		if debug: print(f"  Envelope validation: {len(valid_bins_list)} valid bins, "
				  		f"{len(invalid_bins_list)} invalid bins (total: {len(valid_bins_list) + len(invalid_bins_list)})")
		return valid_bins_list


def global_chi2(obs_dict, weighted=False, valid_bins=None, debug=False):
	"""Compute global chi2 and ndf over all bins."""

	if debug: print("Executing global_chi2")

	chi2_sum = 0.0
	ndf_sum  = 0.0

	if debug: print(f"  Processing all observables with total bins: "
				 	+ str(sum(len(bins) for bins in obs_dict.values())))
	for _, bins in obs_dict.items():
		for bin_name, (diff, err, weight) in bins.items():
			if valid_bins is not None and bin_name not in valid_bins:
				if debug: print(f"    Skipping bin {bin_name}. Not in valid (enveloped) bins list.")
				continue
			if err == 0 or np.isnan(err):
				if debug: print(f"    Skipping bin {bin_name}. Invalid error {err}.")
				continue
			if weight <= 0 or np.isnan(weight):
				if debug: print(f"    Skipping bin {bin_name}. NaN or zero weight {weight}.")
				continue

			den = err ** 2
			term = (diff ** 2) / den
			if not np.isfinite(term):
				if debug: print(f"    Skipping bin {bin_name}. Non-finite term encountered.")
				continue
			if weighted:
				chi2_sum += (weight ** 2) * term
				ndf_sum  += weight ** 2
			else:
				chi2_sum += term
				ndf_sum  += 1.0
	if debug: print(f"  Global chi2: {chi2_sum}, ndf: {ndf_sum}.")
	return chi2_sum, ndf_sum


def obs_chi2(obs_dict, weighted=False, valid_bins=None, debug=False):
	"""Compute reduced chi2 per observable path."""

	if debug: print("Executing obs_chi2")

	chi2_dict = {}
	for obs, bins in obs_dict.items():
		chi2_sum = 0.0
		ndf_sum  = 0.0

		if debug: print(f"  Processing observable {obs} with {len(bins)} bins.")
		for bin_name, (diff, err, weight) in bins.items():
			if valid_bins is not None and bin_name not in valid_bins:
				if debug: print(f"    Skipping bin {bin_name}. Not in valid (enveloped) bins list.")
				continue
			if err == 0 or np.isnan(err):
				if debug: print(f"    Skipping bin {bin_name}. Invalid error {err}.")
				continue
			if weight <= 0 or np.isnan(weight):
				if debug: print(f"    Skipping bin {bin_name}. NaN or zero weight {weight}.")
				continue

			term = (diff ** 2) / err ** 2
			if not np.isfinite(term):
				if debug: print(f"    Skipping bin {bin_name}. Non-finite term encountered.")
				continue
			if weighted:
				chi2_sum += (weight ** 2) * term
				ndf_sum  += weight ** 2
			else:
				chi2_sum += term
				ndf_sum  += 1.0

		chi2_dict[obs] = (chi2_sum, ndf_sum)
		if debug: print(f"  Results for observable {obs}: chi2: {chi2_dict[obs][0]}, ndf: {chi2_dict[obs][1]}.")
	return chi2_dict


def analyses_chi2(obs_chi2s, debug=False):
	"""Breakdown obs_chi2s by analyses."""

	if debug: print("Executing analyses_chi2")

	chi2_dict = {}
	for obs_name, (chi2, ndf) in obs_chi2s.items():
		try:
			analysis = obs_name.split("/")[1].split(":")[0]
		except (IndexError, AttributeError):
			analysis = "unknown"
		if analysis not in chi2_dict:
			chi2_dict[analysis] = (0.0, 0.0)
		previous_chi2, previous_ndf = chi2_dict[analysis]
		chi2_dict[analysis] = (previous_chi2 + chi2, previous_ndf + ndf)
	if debug: print(f"  Results for analyses breakdown: {chi2_dict}.")
	return chi2_dict


def process_envelope(args, loader, weights_dict, analyses_set=None):
	"""Process envelope files and return valid bins."""

	up_file, dn_file = args.envelope
	loader.error_counts["Envelope"] = {}

	if not Path(up_file).exists():
		print(f"Error: Envelope up file '{up_file}' does not exist!")
		return None
	if not Path(dn_file).exists():
		print(f"Error: Envelope down file '{dn_file}' does not exist!")
		return None
	try:
		return loader.get_valid_bins(up_file, dn_file, weights_dict, analyses_set=analyses_set, debug=args.debug)
	except Exception as e:
		print(f"Error processing envelope: {e}")
		return None


def process_single_file(args, loader, yoda_file, 
						label, weights_dict=None, weighted=False, valid_bins=None, analyses_set=None):
	"""Process a single YODA file and compute chi2 summary and per-observable chi2s."""

	obs_dict = loader.get_bin_differences(str(yoda_file),
		weights_dict=weights_dict, analyses_set=analyses_set, include_ref_error=True, debug=args.debug)

	chi2, ndf = global_chi2(obs_dict,
		weighted=weighted, valid_bins=valid_bins, debug=args.debug)
	reduced_chi2 = (chi2 / ndf) if ndf > 0 else np.nan

	obs_chi2s = obs_chi2( obs_dict, 
		weighted=weighted, valid_bins=valid_bins, debug=args.debug)
	valid_obs = [chi2/ndf for chi2, ndf in obs_chi2s.values() if ndf > 0 and np.isfinite(chi2)]
	avg_obs_chi2 = np.mean(valid_obs) if valid_obs else np.nan

	analyses_chi2s = {}
	if "analyses" in set(args.cli_output): 
		analyses_chi2s = analyses_chi2(obs_chi2s, debug=args.debug)

	summary = {
		"source"       : str(yoda_file),
		"label"        : label,
		"global_chi2"  : float(chi2),
		"ndf"          : float(ndf),
		"reduced_chi2" : float(reduced_chi2),
		"avg_obs_chi2" : float(avg_obs_chi2),
		"analysis_chi2": analyses_chi2s
	}

	obs_rows = []
	for obs_name, (chi2, ndf) in sorted(obs_chi2s.items()):
		if ndf > 0 and np.isfinite(chi2):
			obs_rows.append((str(yoda_file), label, obs_name, chi2, ndf))
	return summary, obs_rows


def collect_yoda_files(yoda_inputs, tags=None, labels=None, include_subdirs=False):
	"""Collect YODA files and assign labels during collection."""

	def build_label_from_path(path):
		name = Path(path).name
		if name.endswith(".yoda.gz"):
			return name[:-len(".yoda.gz")]
		if name.endswith(".yoda"):
			return name[:-len(".yoda")]
		return Path(path).stem
	
	def assign_label(file_path, input_index, matched_tags, label_mode):
		if label_mode == "tag":
			if len(matched_tags) > 1:
				chosen = matched_tags[0]
				print(f"Warning: file '{file_path}' matched multiple tags {matched_tags}, using first match '{chosen}'.")
				return tag_to_label[chosen]
			return tag_to_label[matched_tags[0]]
		if label_mode == "input":
			return labels[input_index]
		return build_label_from_path(file_path)

	label_mode = "stem"
	tag_to_label = {}
	if tags:
		if labels is not None and len(labels) == len(tags):
			label_mode = "tag"
			tag_to_label = dict(zip(tags, labels))
		elif labels is not None:
			print(f"Warning: tags count ({len(tags)}) does not match labels count ({len(labels)}), using file stems.")
	else:
		if labels is not None and len(labels) == len(yoda_inputs):
			label_mode = "input"
		elif labels is not None:
			print(f"Warning: labels count ({len(labels)}) does not match input count ({len(yoda_inputs)}), using file stems.")

	files = []
	assigned_labels = []
	for input_index, raw_input in enumerate(yoda_inputs):
		path = Path(raw_input)
		if path.is_dir():
			pattern = "**/*" if include_subdirs else "*"
			candidates = sorted([p for p in path.glob(f"{pattern}.yoda") if p.is_file()])
			candidates.extend(sorted([p for p in path.glob(f"{pattern}.yoda.gz") if p.is_file()]))
		else:
			candidates = [path]

		for file_path in candidates:
			if not file_path.exists():
				print(f"Warning: file not found, skipping: {file_path}.")
				continue
			if not file_path.is_file():
				print(f"Warning: input is not a file, skipping: {file_path}.")
				continue

			matched_tags = [t for t in tags if t in file_path.name] if tags else []
			if tags and not matched_tags:
				continue
			if file_path in files:
				print(f"Warning: the file '{file_path}' is included in multiple inputs, skipping duplicate occurrence.")
				continue
			files.append(file_path)

			assigned_labels.append(assign_label(file_path, input_index, matched_tags, label_mode))
	return files, assigned_labels


def print_table(summaries, show_analyses=False, show_sources=False):
	"""Print chi2 summary table to console."""

	table = []
	chi2_dict = {"red": {}, "avg": {}}

	for s in summaries:
		table.append(
			[
				f"{s['label']}" if not show_sources else f"{s['source']}",
				f"{s['global_chi2']:.3f}",
				f"{s['ndf']:.0f}",
				f"{s['reduced_chi2']:.3f}" if np.isfinite(s["reduced_chi2"]) else "nan",
				f"{s['avg_obs_chi2']:.3f}" if np.isfinite(s["avg_obs_chi2"]) else "nan",
			]
		)
		chi2_dict["red"][s["label"]] = s["reduced_chi2"] if np.isfinite(s["reduced_chi2"]) else np.inf
		chi2_dict["avg"][s["label"]] = s["avg_obs_chi2"] if np.isfinite(s["avg_obs_chi2"]) else np.inf

		if show_analyses:
			for a in s["analysis_chi2"].keys():
				table.append(
					[
						f"- {a}",
						f"{s['analysis_chi2'][a][0]:.3f}",
						f"{s['analysis_chi2'][a][1]:.0f}",
						f"{(s['analysis_chi2'][a][0] / s['analysis_chi2'][a][1]):.3f}" if s['analysis_chi2'][a][1] > 0 else "nan",
						""
					]
				)
	
	print(tabulate(table, 
				headers=["Label" if not show_sources else "Source", "Global chi2", "ndf", "Reduced chi2", "Average chi2"], 
				tablefmt="simple_outline"))
	print()
	min_red_key = min(chi2_dict["red"], key=chi2_dict["red"].get)
	if np.isfinite(chi2_dict["red"][min_red_key]):
		print(f"Min. reduced chi2: {chi2_dict['red'][min_red_key]:.2f} "
				f"(label: {min_red_key}, source: {s['source']})")
	min_avg_key = min(chi2_dict["avg"], key=chi2_dict["avg"].get)
	if np.isfinite(chi2_dict["avg"][min_avg_key]):
		print(f"Min. average chi2: {chi2_dict['avg'][min_avg_key]:.2f} "
				f"(label: {min_avg_key}, source: {s['source']})")
	if np.isfinite(chi2_dict["red"][min_red_key]) or np.isfinite(chi2_dict["avg"][min_avg_key]):
		print()
	return


def print_error_summary(loader):
	"""Print summary of skipped observables by reason."""

	print("YODA loader skips summary:")
	for label, counts in loader.error_counts.items():
		if len(counts) == 0:
			continue
		print(f"File: {label}")
		for reason, count in sorted(counts.items()):
			print(f"  {reason}: {count}")
	print()
	return


def write_chi2_json(output_file, summaries, obs_stats, command):
	"""Write chi2 results to JSON file."""

	outpath = Path(output_file)
	outpath.parent.mkdir(parents=True, exist_ok=True)

	def safe_float(x):
		if np.isfinite(x):
			return float(x)
		return None

	payload = {
		"format": "CHI2JSON",
		"command": command,
		"summaries": [
			{
				"source"      : str(s["source"]),
				"label"       : str(s["label"]),
				"global_chi2" : safe_float(s["global_chi2"]),
				"ndf"         : int(s["ndf"]),
				"reduced_chi2": safe_float(s["reduced_chi2"]),
				"avg_obs_chi2": safe_float(s["avg_obs_chi2"]),
			}
			for s in summaries
		],
		"observables": [
			{
				"source"      : str(source),
				"label"       : str(label),
				"observable"  : str(obs),
				"chi2"        : safe_float(chi2),
				"ndf"         : int(ndf),
				"reduced_chi2": safe_float(chi2 / ndf),
			}
			for source, label, obs, chi2, ndf in obs_stats
		],
	}

	with outpath.open("w") as f:
		json.dump(payload, f, indent=2, allow_nan=False)
		f.write("\n")

	print(f"Wrote chi2 JSON to {outpath}.")
	return


def main():
	print("Starting chi2 computation...\n")
	try:
		check_yoda_version()
	except Exception as e:
		print(f"Error: {e}.")
		return 1

	parser = argparse.ArgumentParser(description="Compute chi2 from YODA files and write results to a chi2.json file.")
	parser.add_argument("yoda_files", nargs="+", help="YODA files or directories containing YODA files")
	parser.add_argument("-w" , "--weights", default=None, help="Path to weights file")
	parser.add_argument("-a" , "--analyses", default=None, help="Path to analyses filter file")
	parser.add_argument("-e" , "--envelope", nargs=2, metavar=("up.yoda", "dn.yoda"), help="Envelope files for valid-bin filtering")
	parser.add_argument("-t" , "--tags", nargs="+", default=None, help="Only keep files with matching tags (directory mode)")
	parser.add_argument("-l" , "--labels", nargs="+", default=None, help="Labels for inputs. With --tags: #labels must equal #tags and labels are assigned by tag match. Without --tags: #labels must equal #yoda_files arguments (directory labels are applied to all files from that input).")
	parser.add_argument("-o" , "--output", default="chi2.json", help="Output chi2 JSON file")
	parser.add_argument("-s" , "--subdir", action="store_true", default=False, help="Include subdirectories when input is a directory")
	parser.add_argument("-wd", "--weighted", action="store_true", default=False, help="Use weighted chi2 computation")
	parser.add_argument("-co", "--CLI-output", dest="cli_output", nargs="+", choices=["analyses", "sources", "errors"], default=[], help="Additional CLI output")
	parser.add_argument("-v" , "--debug", action="store_true", default=False, help="Enable debug output")
	args = parser.parse_args()
	command = " ".join(sys.argv)

	"""Initialize YODA loader and read weights/analyses files if provided"""
	try:
		loader = YodaLoader()
		weights  = read_weights(args.weights)   if args.weights  else None
		analyses = read_analyses(args.analyses) if args.analyses else None
		if args.debug:
			print("Rivet reference search paths:")
			for i, path in enumerate(loader.paths, start=1):
				print(f"  [{i}] {path}")
	except Exception as e:
		print(f"Error reading weights/analyses files: {e}.")
		return 1

	if args.weighted and weights is None:
		print("Warning: --weighted requested without weights file. Falling back to unweighted mode.")
		args.weighted = False

	"""If envelope option is used, determine valid bins before processing files"""
	valid_bins = None
	if args.envelope:
		valid_bins = process_envelope(args, loader, weights)
		if valid_bins is None:
			print("Error processing envelope files/no valid bins found.")
			return 1

	"""Collect YODA files to process, applying tags and assigning labels"""
	yoda_files, labels = collect_yoda_files(args.yoda_files, 
										 tags=args.tags, 
										 labels=args.labels,
										 include_subdirs=args.subdir)
	if not yoda_files:
		print("No YODA files found.")
		return 1

	"""Process each YODA file and collect summaries and observable-level stats for JSON output"""
	summaries = []
	obs_stats = []

	for yoda_file, label in zip(yoda_files, labels):
		summary, obs_rows = process_single_file(args, loader, yoda_file, 
										  label, weights_dict=weights, weighted=args.weighted, valid_bins=valid_bins, analyses_set=analyses)
		summaries.append(summary)
		obs_stats.extend(obs_rows)

	"""Print results table and write JSON output"""
	print_table(summaries, 
			 show_analyses=("analyses" in set(args.cli_output)), 
			 show_sources=("sources" in set(args.cli_output)))
	if "errors" in set(args.cli_output): print_error_summary(loader)
	write_chi2_json(args.output, summaries, obs_stats, command)
	return 0


if __name__ == "__main__":
    main()
