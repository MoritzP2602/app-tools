import numpy as np
import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from tabulate import tabulate

try:
	import rivet
	import yoda
except ImportError as e:
	raise ImportError(f"Required module not found: {e.name}. Please install RIVET and YODA >= 2.1.0.") from e


MIN_YODA = (2, 1, 0)
def check_yoda_version():
	if tuple(int(x) for x in yoda.__version__.split(".")[:3]) < MIN_YODA:
		raise RuntimeError(f"YODA >= {'.'.join(map(str, MIN_YODA))} is required, found {yoda.__version__}")


def read_text_lines(path):
	"""Yield non-empty, non-comment lines from a text file (stripped)."""

	with open(path, "r") as f:
		for line in f:
			line = line.strip()
			if not line or line.startswith("#"):
				continue
			yield line


def read_weights(wfile):
	"""Read observable/bin weights from text file."""

	weights = {}
	for line in read_text_lines(wfile):
		parts = line.split()
		if len(parts) < 2:
			continue
		weights[parts[0]] = float(parts[1])
	return weights


def read_analyses(afile):
	"""Read analyses allow-list from text file."""

	return set(read_text_lines(afile))


def read_output_command(output_file):
	"""Read previous creation command from an existing CHI2JSON output file."""

	outpath = Path(output_file)
	if not outpath.exists() or not outpath.is_file():
		return None
	try:
		with outpath.open("r", encoding="utf-8") as f:
			data = json.load(f)
	except Exception:
		return None

	if data.get("format") != "CHI2JSON":
		return None
	command = data.get("command")
	if not command:
		return None
	return str(command)


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
	return is_estimate(obj) or is_histo(obj) or is_profile(obj) or is_scatter(obj)


def split_variation_tag(path):
	"""Split '/A/B[...]' into ('/A/B', '...')."""
	if not isinstance(path, str):
		return path, None
	m = re.match(r"^(.*)\[([^\[\]]+)\]$", path)
	if not m:
		return path, None
	return m.group(1), m.group(2)

def strip_variation_tag(path):
	"""Remove final '[...]' before optional bin suffix '#N'."""
	if not isinstance(path, str):
		return path
	return re.sub(r"\[[^\[\]]+\](#\d+)?$", r"\1", path)

def tag_sort_key(tag):
	m = re.match(r"^(.*?)(\d+)$", tag)
	if m:
		return (m.group(1), int(m.group(2)))
	return (tag, -1)


class YodaLoader:
	"""YODA helper for loading refs and extracting bin/value/error arrays."""

	def __init__(self):
		self.paths = rivet.getAnalysisRefPaths()
		self.yodas = {}
		self.error_counts = defaultdict(lambda: defaultdict(int))

	def load(self, analysis_name):
		try:
			base_name = str(analysis_name).split(":")[0]
		except Exception:
			base_name = analysis_name

		if base_name in self.yodas:
			return self.yodas[base_name]

		ref_yoda = None
		for base_path in self.paths:
			gz_path = f"{base_path}/{base_name}.yoda.gz"
			yoda_path = f"{base_path}/{base_name}.yoda"
			if os.path.exists(gz_path):
				ref_yoda = yoda.readYODA(gz_path)
				break
			if os.path.exists(yoda_path):
				ref_yoda = yoda.readYODA(yoda_path)
				break

		if ref_yoda is None:
			raise FileNotFoundError(
				f"Reference YODA for {base_name} not found")

		self.yodas[base_name] = ref_yoda
		return ref_yoda

	def get_variation_tags(self, yoda_file, pattern, analyses_set=None):
		tags = set()
		yd = yoda.readYODA(yoda_file)
		for obs_name in yd.keys():
			bare_obs_name, tag = split_variation_tag(obs_name)
			if tag is None or pattern not in tag:
				continue
			try:
				analysis_name = bare_obs_name.split("/")[1].split(":")[0]
			except (IndexError, AttributeError):
				continue
			if analyses_set is not None and analysis_name not in analyses_set:
				continue
			tags.add(tag)
		return sorted(tags, key=tag_sort_key)

	def extract_bins_profile(self, obj):
		if hasattr(obj, "path") and callable(obj.path):
			path = obj.path()
		else:
			raise TypeError(f"Unsupported YODA object {type(obj).__name__}: missing callable path() method.")

		# --- Note ----------------------------------------------------------------------------- #
		# For Profile objects, mkScatter() and point.errs() returns errors equal to zero.        #
		# This means we have to use Profile specific methods in this case (tested for Profile1D). #
		# --------------------------------------------------------------------------------------- #

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
		if is_profile(obj):
			return self.extract_bins_profile(obj)
		if is_estimate(obj) or is_scatter(obj) or is_histo(obj):
			return self.extract_bins_scatter(obj)
		raise TypeError(f"Unsupported YODA object type: {type(obj).__name__}")

	def get_bin_differences(self, yoda_file, weights_dict=None, analyses_set=None, pattern=None, debug=False):
		if debug: print(f"Executing get_bin_differences for {yoda_file}")

		skips = self.error_counts[yoda_file]

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
			bare_obs_name, tag = split_variation_tag(obs_name)
			obs_name_ref = bare_obs_name if pattern is not None else obs_name
			if pattern is not None:
				if tag is None or pattern not in tag:
					if debug: print(f"  Skipping observable {obs_name}. Does not match pattern '{pattern}'.")
					skips["Observable [tag] does not match pattern"] += 1
					continue

			if weights_dict is not None and (bare_obs_name not in obs_weights_set):
				if debug: print(f"  Skipping observable {obs_name}. Not in weights file.")
				skips["Observable not in weights file"] += 1
				continue

			try:
				analysis_name = obs_name_ref.split("/")[1].split(":")[0]
			except (IndexError, AttributeError):
				if debug: print(f"  Skipping observable {obs_name}. Failed to parse analysis name.")
				skips["Failed to parse analysis name"] += 1
				continue
			if analyses_set is not None and analysis_name not in analyses_set:
				if debug: print(f"  Skipping observable {obs_name}. Analysis {analysis_name} not in analyses list.")
				skips["Analysis name not in list"] += 1
				continue

			try:
				yd_ref  = self.load(analysis_name)
				obs_ref = f"/REF/{analysis_name}/{obs_name_ref.split('/')[-1]}"
				histogram     = yd[obs_name]
				histogram_ref = yd_ref.get(obs_ref)
				if histogram_ref is None:
					if debug: print(f"  Skipping observable {obs_name}. Reference data not found in /REF/{analysis_name}.")
					skips["Reference data not found"] += 1
					continue
			except (FileNotFoundError, KeyError) as e:
				if debug: print(f"  Skipping observable {obs_name}. Error loading reference data for {analysis_name}: {e}.")
				skips["Error loading reference data"] += 1
				continue

			if debug: print(f"  Processing observable {obs_name}, type {histogram.type()}, "
							f"with reference {obs_ref}, type {histogram_ref.type()}.")
			if not (accepted_type(histogram) and accepted_type(histogram_ref)):
				if debug: print(f"  Skipping observable {obs_name}. Unsupported object type in one of the datasets.")
				skips["Unsupported object type"] += 1
				continue

			bin_dict = self.extract_bins(histogram)
			bin_dict_ref = self.extract_bins(histogram_ref)

			if len(bin_dict) != len(bin_dict_ref):
				if debug: print(f"  Skipping observable {obs_name}. Bin count mismatch between data and reference: "
								f"data = {len(bin_dict)}, ref = {len(bin_dict_ref)}.")
				skips["Bin count mismatch"] += 1
				continue

			for bin_name, (data_val, data_err) in bin_dict.items():
				bare_bin_name = strip_variation_tag(bin_name)
				try:
					obs_suffix = bare_bin_name.split("/")[-1]
				except Exception:
					obs_suffix = bare_bin_name
				bin_name_ref = f"/REF/{analysis_name}/{obs_suffix}"
				if bin_name_ref not in bin_dict_ref:
					if debug: print(f"    Skipping bin {bin_name}. Not found in reference for {obs_name}.")
					skips["Bin not found in reference data"] += 1
					continue

				ref_val, ref_err = bin_dict_ref[bin_name_ref]
				diff = data_val - ref_val
				err = np.sqrt(data_err ** 2 + ref_err ** 2)
				if err == 0 or np.isnan(err):
					if debug: print(f"    Skipping bin {bin_name}. Invalid combined error {err}.")
					skips["Invalid combined error"] += 1
					continue

				if not weights_dict:
					bin_weight = 1.0
				else:
					bin_weight = weights_dict.get(bare_bin_name, weights_dict.get(bare_obs_name, 0.0))
				key = bare_bin_name if pattern is not None else bin_name
				obs_dict.setdefault(obs_name, {})[key] = (diff, err, bin_weight)
		return obs_dict

	def get_valid_bins(self, up_file, dn_file, weights_dict=None, analyses_set=None, debug=False):
		if debug: print(f"Executing get_valid_bins for {up_file}, {dn_file}")

		skips = self.error_counts["Envelope"]

		obs_weights_set = set()
		if weights_dict is not None:
			for key in weights_dict.keys():
				obs_weights_set.add(key.split("#")[0])

		yd_up = yoda.readYODA(up_file)
		yd_dn = yoda.readYODA(dn_file)

		valid_bins_list, invalid_bins_list = [], []
		candidate_obs = set(yd_up.keys())

		for obs_name in candidate_obs:
			if weights_dict is not None and (obs_name not in obs_weights_set):
				skips["Observable not in weights file"] += 1
				continue
			try:
				analysis_name = obs_name.split("/")[1].split(":")[0]
			except (IndexError, AttributeError):
				if debug: print(f"  Skipping observable {obs_name}. Failed to parse analysis name.")
				skips["Failed to parse analysis name"] += 1
				continue
			if analyses_set is not None and analysis_name not in analyses_set:
				if debug: print(f"  Skipping observable {obs_name}. Analysis {analysis_name} not in analyses list.")
				skips["Analysis name not in list"] += 1
				continue

			try:
				ref_yoda = self.load(analysis_name)
				ref_path = f"/REF/{analysis_name}/{obs_name.split('/')[-1]}"

				obj_up  = yd_up.get(obs_name)
				obj_dn  = yd_dn.get(obs_name)
				obj_ref = ref_yoda.get(ref_path)

				if obj_up is None or obj_dn is None or obj_ref is None:
					if debug: print(f"  Skipping observable {obs_name}. Missing object found.")
					skips["Observable data not found"] += 1
					continue

				if debug: print(f"  Processing observable {obs_name} for envelope validation, type up: {obj_up.type()}, "
								f"dn: {obj_dn.type()}, reference type {obj_ref.type()}.")
				if not (accepted_type(obj_up) and accepted_type(obj_dn) and accepted_type(obj_ref)):
					if debug: print(f"  Skipping observable {obs_name}. Unsupported object type in one of the datasets.")
					skips["Unsupported object type"] += 1
					continue

				bin_dict_up  = self.extract_bins(obj_up)
				bin_dict_dn  = self.extract_bins(obj_dn)
				bin_dict_ref = self.extract_bins(obj_ref)

				if len(bin_dict_up) != len(bin_dict_dn) or len(bin_dict_up) != len(bin_dict_ref):
					if debug: print(f"  Skipping observable {obs_name}. Bin count mismatch: "
									f"up = {len(bin_dict_up)}, dn = {len(bin_dict_dn)}, ref = {len(bin_dict_ref)}.")
					skips["Bin count mismatch"] += 1
					continue

				for bin_name in bin_dict_up.keys():
					try:
						obs_suffix = bin_name.split("/")[-1]
					except Exception:
						obs_suffix = bin_name
					bin_name_ref = f"/REF/{analysis_name}/{obs_suffix}"
					if bin_name not in bin_dict_dn or bin_name_ref not in bin_dict_ref:
						if debug: print(f"    Skipping bin {bin_name}. Not found in all datasets for {obs_name}.")
						skips["Bin not found in all datasets"] += 1
						continue
					up_val, dn_val, ref_val = bin_dict_up[bin_name][0], bin_dict_dn[bin_name][0], bin_dict_ref[bin_name_ref][0]
					if np.isnan(up_val) or np.isnan(dn_val) or np.isnan(ref_val):
						if debug: print(f"    Skipping bin {bin_name}. NaN value encountered in one of the datasets for {obs_name}.")
						skips["NaN value in datasets"] += 1
						continue

					min_val, max_val = min(up_val, dn_val), max(up_val, dn_val)
					if min_val <= ref_val <= max_val:
						valid_bins_list.append(bin_name)
					else:
						invalid_bins_list.append(bin_name)

			except Exception as e:
				if debug: print(f"  Skipping observable {obs_name}. Error: {e}.")
				skips["Error processing observable"] += 1
				continue

		if debug: print(f"  Envelope validation: {len(valid_bins_list)} valid bins, "
						f"{len(invalid_bins_list)} invalid bins (total: {len(valid_bins_list) + len(invalid_bins_list)})")
		return valid_bins_list


def chi2_terms(bins, weighted=False, valid_bins=None, debug=False):
	"""Sum chi2 contributions over a flat dict of bins. Returns (chi2_sum, ndf_sum)."""

	chi2_sum = 0.0
	ndf_sum  = 0.0
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
		chi2_sum += (weight ** 2) * term if weighted else term
		ndf_sum  += 1.0
	return chi2_sum, ndf_sum


def global_chi2(obs_dict, weighted=False, valid_bins=None, debug=False):
	"""Compute global chi2 and ndf over all bins."""

	if debug: print("Executing global_chi2")
	if debug: print(f"  Processing all observables with total bins: "
					+ str(sum(len(bins) for bins in obs_dict.values())))

	chi2_sum = 0.0
	ndf_sum  = 0.0
	for _, bins in obs_dict.items():
		chi2, ndf = chi2_terms(bins, weighted=weighted, valid_bins=valid_bins, debug=debug)
		chi2_sum += chi2
		ndf_sum  += ndf
	if debug: print(f"  Global chi2: {chi2_sum}, ndf: {ndf_sum}.")
	return chi2_sum, ndf_sum


def obs_chi2(obs_dict, weighted=False, valid_bins=None, debug=False):
	"""Compute reduced chi2 per observable path."""

	if debug: print("Executing obs_chi2")

	chi2_dict = {}
	for obs, bins in obs_dict.items():
		if debug: print(f"  Processing observable {obs} with {len(bins)} bins.")
		chi2_sum, ndf_sum = chi2_terms(bins, weighted=weighted, valid_bins=valid_bins, debug=debug)
		chi2_dict[obs] = (chi2_sum, ndf_sum)
		if debug: print(f"  Results for observable {obs}: chi2: {chi2_sum}, ndf: {ndf_sum}.")
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
			chi2_dict[analysis] = (0.0, 0.0, 0.0, 0.0)
		previous_chi2, previous_ndf, previous_avg, previous_count = chi2_dict[analysis]
		reduced_chi2 = (chi2 / ndf) if ndf > 0 and np.isfinite(chi2) else np.nan
		if np.isfinite(reduced_chi2):
			previous_avg += reduced_chi2
			previous_count += 1.0
		chi2_dict[analysis] = (previous_chi2 + chi2, previous_ndf + ndf, previous_avg, previous_count)
	if debug: print(f"  Results for analyses breakdown: {chi2_dict}.")
	return chi2_dict


def process_envelope(args, loader, weights_dict, analyses_set=None):
	"""Process envelope files and return valid bins."""

	up_file, dn_file = args.envelope
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


def build_summary(source, label, obs_dict, weighted, valid_bins, show_analyses, debug):
	"""Compute (summary_dict, obs_data_list) for a single (source, label, obs_dict) group."""

	chi2, ndf = global_chi2(obs_dict, weighted=weighted, valid_bins=valid_bins, debug=debug)
	reduced_chi2 = (chi2 / ndf) if ndf > 0 else np.nan

	obs_chi2s = obs_chi2(obs_dict, weighted=weighted, valid_bins=valid_bins, debug=debug)
	valid_obs = [c / n for c, n in obs_chi2s.values() if n > 0 and np.isfinite(c)]
	average_chi2 = np.mean(valid_obs) if valid_obs else np.nan

	analyses_chi2s = analyses_chi2(obs_chi2s, debug=debug) if show_analyses else {}

	summary = {
		"source"       : str(source),
		"label"        : str(label),
		"global_chi2"  : float(chi2),
		"ndf"          : float(ndf),
		"reduced_chi2" : float(reduced_chi2),
		"average_chi2" : float(average_chi2),
		"analysis_chi2": analyses_chi2s,
	}
	obs_data = []
	for obs_name, (c, n) in sorted(obs_chi2s.items()):
		if n > 0 and np.isfinite(c):
			obs_data.append((str(source), str(label), str(obs_name), c, n))
	return summary, obs_data


def process_single_file(args, loader, yoda_file, label,
						weights_dict=None, weighted=False, valid_bins=None,
						analyses_set=None, pattern=None, show_analyses=False):
	"""Process a single YODA file. Returns (list_of_summaries, list_of_obs_data).

	Without --pattern: produces one summary. With --pattern: produces one summary per matched tag.
	"""

	obs_dict_all = loader.get_bin_differences(str(yoda_file),
		weights_dict=weights_dict, analyses_set=analyses_set, pattern=pattern, debug=args.debug)

	if pattern is None:
		summary, obs_data = build_summary(yoda_file, label, obs_dict_all,
			weighted=weighted, valid_bins=valid_bins, show_analyses=show_analyses, debug=args.debug)
		return [summary], obs_data

	grouped_by_tag = {}
	for obs_name, bins in obs_dict_all.items():
		_, tag = split_variation_tag(obs_name)
		if tag is None:
			continue
		grouped_by_tag.setdefault(tag, {})[obs_name] = bins

	if not grouped_by_tag:
		print(f"\nWarning: No observable tags matching --pattern '{pattern}' found in {yoda_file}, skipping.")
		return [], []

	summaries = []
	obs_data_all = []
	for tag in sorted(grouped_by_tag.keys(), key=tag_sort_key):
		summary, obs_data = build_summary(f"{yoda_file}[{tag}]", f"{label}[{tag}]", grouped_by_tag[tag],
			weighted=weighted, valid_bins=valid_bins, show_analyses=show_analyses, debug=args.debug)
		summaries.append(summary)
		obs_data_all.extend(obs_data)
	return summaries, obs_data_all


def collect_yoda_files(yoda_inputs, tags=None, labels=None, depth=0):
	"""Collect YODA files and assign labels during collection."""

	if depth < 0:
		print("Warning: --depth must be >= 0, using 0.")
		depth = 0

	def collect_from_dir(path_obj, max_depth):
		found = []
		for root, dirs, files_in_root in os.walk(path_obj):
			root_path = Path(root)
			rel_parts = root_path.relative_to(path_obj).parts
			cur_depth = len(rel_parts)
			if cur_depth > max_depth:
				dirs[:] = []
				continue
			for name in files_in_root:
				if name.endswith(".yoda") or name.endswith(".yoda.gz"):
					found.append(root_path / name)
			if cur_depth == max_depth:
				dirs[:] = []
		return sorted(found)

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
				print(f"Warning: File '{file_path}' matched multiple tags {matched_tags}, using first match '{chosen}'.")
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
			print(f"Warning: Tags count ({len(tags)}) does not match labels count ({len(labels)}), using file stems.")
	else:
		if labels is not None and len(labels) == len(yoda_inputs):
			label_mode = "input"
		elif labels is not None:
			print(f"Warning: Labels count ({len(labels)}) does not match input count ({len(yoda_inputs)}), using file stems.")

	files = []
	assigned_labels = []
	for input_index, raw_input in enumerate(yoda_inputs):
		path = Path(raw_input)
		if path.is_dir():
			candidates = collect_from_dir(path, depth)
		else:
			candidates = [path]

		for file_path in candidates:
			if not file_path.exists():
				print(f"Warning: File not found, skipping: {file_path}.")
				continue
			if not file_path.is_file():
				print(f"Warning: Input is not a file, skipping: {file_path}.")
				continue

			matched_tags = [t for t in tags if t in file_path.name] if tags else []
			if tags and not matched_tags:
				continue
			if file_path in files:
				print(f"Warning: The file '{file_path}' is included in multiple inputs, skipping duplicate occurrence.")
				continue
			files.append(file_path)

			assigned_labels.append(assign_label(file_path, input_index, matched_tags, label_mode))
	return files, assigned_labels


def print_table(summaries, show_analyses=False, show_sources=False):
	"""Print chi2 summary table to console."""

	table = []

	for s in summaries:
		table.append(
			[
				f"{s['label']}" if not show_sources else f"{s['source']}",
				f"{s['global_chi2']}" if np.isfinite(s["global_chi2"]) else "nan",
				f"{s['ndf']:.0f}",
				f"{s['reduced_chi2']}" if np.isfinite(s["reduced_chi2"]) else "nan",
				f"{s['average_chi2']}" if np.isfinite(s["average_chi2"]) else "nan",
			]
		)
		if show_analyses:
			for a in s["analysis_chi2"].keys():
				analysis_chi2, analysis_ndf, analysis_avg_sum, analysis_count = s["analysis_chi2"][a]
				analysis_avg = (analysis_avg_sum / analysis_count) if analysis_count > 0 else np.nan
				table.append(
					[
						f"- {a}",
						f"{analysis_chi2}",
						f"{analysis_ndf:.0f}",
						f"{(analysis_chi2 / analysis_ndf)}" \
							if analysis_ndf > 0 else "nan",
						f"{analysis_avg}" if np.isfinite(analysis_avg) else "nan"
					]
				)

	print(tabulate(table,
				headers=["Label" if not show_sources else "Source", "Global chi2", "ndf", "Reduced chi2", "Average chi2"],
				tablefmt="simple_outline", floatfmt=".3f", numalign="decimal"))
	print()
	finite_red = [s for s in summaries if np.isfinite(s["reduced_chi2"])]
	finite_avg = [s for s in summaries if np.isfinite(s["average_chi2"])]
	if finite_red:
		best_red = min(finite_red, key=lambda x: x["reduced_chi2"])
		print(f"Min. reduced chi2: {best_red['reduced_chi2']:.2f} "
				f"(label: {best_red['label']}, source: {best_red['source']})")
	if finite_avg:
		best_avg = min(finite_avg, key=lambda x: x["average_chi2"])
		print(f"Min. average chi2: {best_avg['average_chi2']:.2f} "
				f"(label: {best_avg['label']}, source: {best_avg['source']})")
	if finite_red or finite_avg:
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
				"source"      : s["source"],
				"label"       : s["label"],
				"global_chi2" : safe_float(s["global_chi2"]),
				"ndf"         : safe_float(s["ndf"]),
				"reduced_chi2": safe_float(s["reduced_chi2"]),
				"average_chi2": safe_float(s["average_chi2"]),
			}
			for s in summaries
		],
		"observables": [
			{
				"source"      : source,
				"label"       : label,
				"observable"  : obs_name,
				"chi2"        : safe_float(chi2),
				"ndf"         : safe_float(ndf),
				"reduced_chi2": safe_float(chi2 / ndf),
			}
			for source, label, obs_name, chi2, ndf in obs_stats
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

	parser = argparse.ArgumentParser(
		description="Compute chi2 from YODA files and write results to a JSON file.",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  compute_chi2.py "file1.yoda" --CLI-output analyses
  compute_chi2.py results/ --depth 1
  compute_chi2.py results/ --tags tag1 tag2 --labels label1 label2
  compute_chi2.py results/ --envelope up.yoda dn.yoda --weights weights.txt --weighted

Input handling:
  - yoda_files accepts files and/or directories
  - If a directory is provided, all .yoda/.yoda.gz files are collected recursively up to --depth
  - Weights (-w) and analyses (-a) files can be used to filter analyses/observables and provide obs/bin weights
    - Weighted mode (--weighted) applies weights in chi2 computation (bin weight squared)
  - Envelope option (-e) allows specifying up/down YODA files to determine valid bins
    - Only bins where the reference value is between the up/down values are considered in the chi2 computation

Labels and tags:
  - With --tags, only files with matching tags in their filename are collected from the provided inputs
	- Labels are assigned based on tag match (len(labels) must match len(tags))
  - Without --tags, labels map by input argument (len(labels) must match len(yoda_files args))
  - If counts do not match, file stems are used as labels

Output:
  - Results are printed in a table format to the console
  - Chi2 details are written to a JSON file (default: chi2.json) with summaries and per-observable stats
  - Additional output options are available using --table-output (analyses, sources) and --error-summary
		"""
	)
	parser.add_argument("yoda_files", nargs="+", help="YODA files or directories containing YODA files")
	parser.add_argument("-w" , "--weights", default=None, help="Path to weights file")
	parser.add_argument("-a" , "--analyses", default=None, help="Path to analyses filter file")
	parser.add_argument("-e" , "--envelope", nargs=2, metavar=("up.yoda", "dn.yoda"), help="Envelope files for valid-bin filtering")
	parser.add_argument("-t" , "--tags", nargs="+", default=None, help="Only keep files with matching tags (directory mode)")
	parser.add_argument("-l" , "--labels", nargs="+", default=None, help="Labels for inputs. With --tags: #labels must equal #tags and labels are assigned by tag match. Without --tags: #labels must equal #yoda_files arguments.")
	parser.add_argument("-o" , "--output", default="chi2.json", help="Output chi2 JSON file")
	parser.add_argument("--weighted", action="store_true", default=False, help="Use weights in chi2 computation")
	parser.add_argument("--pattern", default=None, help="Only use observables with [TAG] containing this string; process each matching TAG separately")
	parser.add_argument("--depth", type=int, default=0, help="Directory recursion depth for collecting YODA files (0 = only given directory)")
	parser.add_argument("--table-output", dest="table_output", nargs="+", choices=["analyses", "sources"], default=[], help="Configure table output")
	parser.add_argument("--error-summary", action="store_true", default=False, help="Display error summary for each file")
	parser.add_argument("-v" , "--debug", action="store_true", default=False, help="Enable debug output")
	args = parser.parse_args()
	command = " ".join(os.sys.argv)

	table_output_set = set(args.table_output)
	show_analyses = "analyses" in table_output_set
	show_sources  = "sources"  in table_output_set

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


	"""Collect YODA files to process, applying tags and assigning labels"""
	yoda_files, labels = collect_yoda_files(args.yoda_files,
										 tags=args.tags, labels=args.labels, depth=args.depth)
	if not yoda_files:
		print("No YODA files found.")
		return 1

	"""If envelope option is used, determine valid bins once before processing files"""
	valid_bins = None
	if args.envelope:
		valid_bins = process_envelope(args, loader, weights, analyses_set=analyses)
		if valid_bins is None:
			print("Error processing envelope files/no valid bins found.")
			return 1

	"""Process each YODA file and collect summaries and observable-level stats for JSON output"""
	summaries = []
	obs_stats = []
	total_files = len(yoda_files)
	for idx, (yoda_file, label) in enumerate(zip(yoda_files, labels), start=1):
		print(f"\rProcessing {idx}/{total_files} files...", end='', flush=True)
		file_summaries, file_obs_data = process_single_file(args, loader, yoda_file, label,
			weights_dict=weights, weighted=args.weighted, valid_bins=valid_bins,
			analyses_set=analyses, pattern=args.pattern, show_analyses=show_analyses)
		summaries.extend(file_summaries)
		obs_stats.extend(file_obs_data)
		print('\r' + ' ' * 50 + '\r', end='', flush=True)
	if not summaries:
		print("No matching observables/files found to compute chi2.")
		return 1

	"""Print results table and write JSON output"""
	print_table(summaries, show_analyses=show_analyses, show_sources=show_sources)
	if args.error_summary: print_error_summary(loader)
	outpath = Path(args.output)
	if outpath.exists():
		if outpath.is_file():
			print(f"Output file already exists.")
			previous_cmd = read_output_command(outpath)
			if previous_cmd:
				print(f"  Previous output file command:")
				print(f"    {previous_cmd}")
			outpath.unlink()
			print(f"  Removed existing output file: {outpath}.\n")
		else:
			print(f"Error: Output path exists and is not a file: {outpath}. "
				  f"Please remove or specify a different output path.")
			return 1
	write_chi2_json(args.output, summaries, obs_stats, command)
	return 0


if __name__ == "__main__":
	main()
