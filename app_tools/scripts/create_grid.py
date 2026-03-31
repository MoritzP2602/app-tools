
import numpy as np
import os
import sys
import glob
import json
import re
import argparse
import itertools
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict, deque


def fail(msg):
    print(f"Error: {msg}.")
    sys.exit(1)

def is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)

def is_param_ref(x):
    return isinstance(x, str) and x.strip() != ""


@dataclass
class Sector:
    sector_id: int
    bounds: OrderedDict = field(default_factory=OrderedDict)


@dataclass
class GridPoint:
    values: OrderedDict
    sector_id: int = 0
    sector_bounds: OrderedDict = field(default_factory=OrderedDict)
    index: str = ""


class ParameterModel:
    def __init__(self, ordered_specs):
        self.specs = ordered_specs
        self.order = list(ordered_specs.keys())
        self.validate_refs_and_cycles()

    @classmethod
    def from_file(cls, path):
        if not os.path.exists(path):
            fail(f"Parameter file '{path}' not found")

        if path.endswith(".json"):
            with open(path) as f:
                raw = json.load(f, object_pairs_hook=OrderedDict)
            return cls(cls.parse_json_specs(raw, path))

        specs = OrderedDict()
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                name = parts[0]
                lo, hi = float(parts[1]), float(parts[2])
                if lo > hi:
                    fail(f"Invalid bounds for '{name}' in '{path}' ({lo} > {hi})")
                specs[name] = {"kind": "range", "bounds": [lo, hi]}

        if not specs:
            fail(f"No valid parameters found in '{path}'")
        return cls(specs)

    @staticmethod
    def parse_json_specs(raw, path):
        if not isinstance(raw, dict) or not raw:
            fail(f"No valid parameters found in '{path}'")

        specs = OrderedDict()
        for name, val in raw.items():
            if not isinstance(val, list) or len(val) < 2:
                fail(f"Parameter '{name}' must be a list with at least 2 entries in '{path}'")

            if len(val) == 2:
                a, b = val
                if is_number(a) and is_number(b):
                    lo, hi = float(a), float(b)
                    if lo > hi:
                        fail(f"Invalid bounds for '{name}' ({lo} > {hi}) in '{path}'")
                    specs[name] = {"kind": "range", "bounds": [lo, hi]}
                    continue

                if (is_number(a) or is_param_ref(a)) and (is_number(b) or is_param_ref(b)):
                    specs[name] = {"kind": "dynamic", "bounds": [a, b]}
                    continue

                fail(f"Invalid bounds definition for '{name}' in '{path}'")

            if all(is_number(x) for x in val):
                points = [float(x) for x in val]
                if any(points[i] >= points[i + 1] for i in range(len(points) - 1)):
                    fail(f"Breakpoints for sectorized parameter '{name}' must be strictly increasing in '{path}'")
                specs[name] = {"kind": "sector", "breakpoints": points}
                continue

            fail(f"Unsupported definition for parameter '{name}' in '{path}'")
        return specs

    def validate_refs_and_cycles(self):
        names = set(self.specs.keys())
        edges = defaultdict(set)
        indeg = {n: 0 for n in names}

        for name, spec in self.specs.items():
            if spec["kind"] != "dynamic":
                continue
            lo, hi = spec["bounds"]
            refs = []
            if is_param_ref(lo):
                refs.append(lo)
            if is_param_ref(hi):
                refs.append(hi)

            for ref in refs:
                if ref not in names:
                    fail(f"Parameter '{name}' references unknown parameter '{ref}'")
                if ref == name:
                    fail(f"Parameter '{name}' cannot reference itself")
                if name not in edges[ref]:
                    edges[ref].add(name)
                    indeg[name] += 1

        q = deque([n for n in self.order if indeg[n] == 0])
        topo = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in edges[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        if len(topo) != len(self.order):
            fail("Cyclic dynamic parameter dependencies detected")

        self.sampling_order = topo
        return

    def has_dynamic(self):
        return any(spec["kind"] == "dynamic" for spec in self.specs.values())

    def build_sectors(self):
        sector_params = [name for name, spec in self.specs.items() if spec["kind"] == "sector"]
        if not sector_params:
            return [Sector(sector_id=0, bounds=OrderedDict())]

        interval_options = []
        for name in sector_params:
            b = self.specs[name]["breakpoints"]
            interval_options.append([(b[i], b[i + 1]) for i in range(len(b) - 1)])

        sectors = []
        for sid, combo in enumerate(itertools.product(*interval_options)):
            bounds = OrderedDict()
            for pname, (lo, hi) in zip(sector_params, combo):
                bounds[pname] = [float(lo), float(hi)]
            sectors.append(Sector(sector_id=sid, bounds=bounds))
        return sectors

    def resolve_bound_expr(self, expr, sampled_values):
        if is_number(expr):
            return float(expr)
        return float(sampled_values[expr])

    def resolve_bounds_for_param(self, param_name, sampled_values, sector_bounds):
        spec = self.specs[param_name]
        if spec["kind"] == "sector":
            lo, hi = sector_bounds[param_name]
            return float(lo), float(hi)
        if spec["kind"] == "range":
            lo, hi = spec["bounds"]
            return float(lo), float(hi)

        lo_expr, hi_expr = spec["bounds"]
        lo = self.resolve_bound_expr(lo_expr, sampled_values)
        hi = self.resolve_bound_expr(hi_expr, sampled_values)
        return float(lo), float(hi)


class ParameterGrid:
    def __init__(self, points=None, parameter_order=None):
        self.points = points or []
        self.parameter_order = parameter_order or self._infer_parameter_order()

    def _infer_parameter_order(self):
        if not self.points:
            return []
        return list(self.points[0].values.keys())

    def parameter_names(self):
        if self.parameter_order:
            return list(self.parameter_order)
        return self._infer_parameter_order()

    @staticmethod
    def _allocate_points_per_sector(total, nsectors):
        base = total // nsectors
        rem = total % nsectors
        return [base + (1 if i < rem else 0) for i in range(nsectors)]

    @classmethod
    def from_sampling(cls, model, num_points, sampling_mode="random", seed=None):
        if num_points <= 0:
            fail("--num-points must be a positive integer")

        if seed is not None and sampling_mode == "random":
            np.random.seed(seed)

        sectors = model.build_sectors()
        allocation = cls._allocate_points_per_sector(num_points, len(sectors))
        points = []

        for sector, n_in_sector in zip(sectors, allocation):
            if n_in_sector <= 0:
                continue
            if sampling_mode == "uniform":
                points.extend(cls._sample_uniform_for_sector(model, n_in_sector, sector))
            else:
                points.extend(cls._sample_random_for_sector(model, n_in_sector, sector))

        return cls(points=points, parameter_order=model.order)

    @classmethod
    def _sample_random_for_sector(cls, model, npoints, sector):
        points = []
        for _ in range(npoints):
            sampled = OrderedDict()
            for name in model.sampling_order:
                lo, hi = model.resolve_bounds_for_param(name, sampled, sector.bounds)
                if lo > hi:
                    fail(f"Invalid dynamic bounds for parameter '{name}' in sector {sector.sector_id} ({lo} > {hi})")
                sampled[name] = float(np.random.uniform(lo, hi))
            points.append(GridPoint(values=sampled, sector_id=sector.sector_id, sector_bounds=sector.bounds))
        return points

    @classmethod
    def _sample_uniform_for_sector(cls, model, npoints, sector):
        if model.has_dynamic():
            fail("uniform mode is incompatible with dynamic bounds")

        names = model.order
        ndim = len(names)
        n_per_dim = int(np.ceil(npoints ** (1 / ndim)))

        while True:
            grids = []
            for name in names:
                lo, hi = model.resolve_bounds_for_param(name, {}, sector.bounds)
                if n_per_dim == 1:
                    grids.append([lo])
                else:
                    step = (hi - lo) / (n_per_dim - 1)
                    grids.append([lo + i * step for i in range(n_per_dim)])

            all_points = list(itertools.product(*grids))
            if len(all_points) >= npoints:
                out = []
                for tup in all_points[:npoints]:
                    out.append(GridPoint(values=OrderedDict(zip(names, tup)),
                            sector_id=sector.sector_id,
                            sector_bounds=sector.bounds))
                return out
            n_per_dim += 1

    @classmethod
    def from_directory(cls, directory_path, params_filename="params.dat", limit=None):
        if not os.path.isdir(directory_path):
            fail(f"Directory '{directory_path}' not found")

        points = []
        subfolders = sorted([d for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))])
        if limit is not None:
            subfolders = subfolders[:limit]

        for sub in subfolders:
            param_file = os.path.join(directory_path, sub, params_filename)
            if not os.path.exists(param_file):
                continue
            values = OrderedDict()
            try:
                with open(param_file, "r") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            values[parts[0]] = float(parts[1])
            except (ValueError, IOError) as exc:
                print(f"Warning: failed reading {param_file}: {exc}.")
                continue

            if values:
                points.append(GridPoint(values=values, sector_id=0, sector_bounds=OrderedDict(), index=sub))

        if not points:
            fail(f"No valid parameter files found in '{directory_path}'")
        return cls(points=points)

    @classmethod
    def from_table(cls, table_path):
        if not os.path.exists(table_path):
            fail(f"Table file '{table_path}' not found")

        points = []
        header = None
        current_sector = 0
        current_sector_bounds = OrderedDict()
        interval_bounds_by_param_and_idx = {}

        try:
            with open(table_path, "r") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue

                    if line.startswith("#"):
                        lower = line.lower()
                        maybe_header = line[1:].strip().split()
                        if header is None and maybe_header and maybe_header[0].lower() == "index":
                            header = maybe_header
                            continue

                        if lower.startswith("# sectorized parameter"):
                            m = re.match(r"^#\s*sectorized parameter\s+(\S+)\s*:\s*(.*)$", line, re.IGNORECASE)
                            if m:
                                pname = m.group(1)
                                tail = m.group(2)
                                idx_map = {}
                                for idx_str, lo_str, hi_str in re.findall(
                                    r"(\d+)\s*=\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]",
                                    tail,
                                ):
                                    idx_map[int(idx_str)] = (float(lo_str), float(hi_str))
                                if idx_map:
                                    interval_bounds_by_param_and_idx[pname] = idx_map
                            continue

                        if lower.startswith("# sector"):
                            parts = lower.replace(":", " ").split()
                            try:
                                current_sector = int(parts[2])
                            except (ValueError, IndexError):
                                pass

                            current_sector_bounds = OrderedDict()
                            rhs = line.split(":", 1)
                            if len(rhs) == 2:
                                for pname, idx_str in re.findall(r"([A-Za-z0-9_]+)\s*\(\s*(\d+)\s*\)", rhs[1]):
                                    idx = int(idx_str)
                                    if pname in interval_bounds_by_param_and_idx and idx in interval_bounds_by_param_and_idx[pname]:
                                        lo, hi = interval_bounds_by_param_and_idx[pname][idx]
                                        current_sector_bounds[pname] = [float(lo), float(hi)]
                            continue

                        continue

                    cols = line.split()
                    if header is None:
                        header = cols
                        continue

                    if len(cols) != len(header):
                        print(f"Warning: Skipping malformed line in '{table_path}': {raw.rstrip()}")
                        continue

                    row = dict(zip(header, cols))
                    values = OrderedDict()
                    idx = ""
                    for k, v in row.items():
                        lk = k.lower()
                        if lk == "index":
                            idx = v
                            continue
                        if lk == "sector":
                            try:
                                current_sector = int(float(v))
                            except ValueError:
                                pass
                            continue
                        values[k] = float(v)

                    points.append(GridPoint(
                            values=values,
                            sector_id=int(current_sector),
                            sector_bounds=OrderedDict((k, [b[0], b[1]]) for k, b in current_sector_bounds.items()),
                            index=idx))

        except (IOError, ValueError) as exc:
            fail(f"Cannot parse table file '{table_path}': {exc}")

        if not points:
            fail(f"No valid parameter rows found in '{table_path}'")
        return cls(points=points)

    @classmethod
    def from_minmax(cls, model, defaults):
        for name, spec in model.specs.items():
            if spec["kind"] != "range":
                fail(f"minmax mode only supports fixed [min,max] ranges. '{name}' is '{spec['kind']}'")

        param_order = sorted(model.specs.keys())
        points = []
        info_lines = []
        info_pad_width = 1 + int(np.ceil(np.log10(2 * len(param_order))))

        for idx, param_name in enumerate(param_order):
            lo, hi = model.specs[param_name]["bounds"]
            for bound_idx, bound_name in enumerate(["min", "max"]):
                values = OrderedDict(defaults.copy())
                values[param_name] = lo if bound_idx == 0 else hi
                points.append(GridPoint(values=values, sector_id=0, sector_bounds=OrderedDict()))
                info_idx = f"{2 * idx + bound_idx}".zfill(info_pad_width)
                info_lines.append(f"{info_idx}\t{param_name}: {bound_name}")
        return cls(points=points), info_lines

    def assign_sequential_indices(self):
        if not self.points:
            return
        pad = 1 + int(np.ceil(np.log10(len(self.points))))
        for i, point in enumerate(self.points):
            point.index = str(i).zfill(pad)
        return

    def write_scan(self, outdir, template_name, template_content, params_filename="params.dat"):
        if not self.points:
            fail("No points to write")

        os.makedirs(outdir, exist_ok=True)
        self.assign_sequential_indices()

        for point in self.points:
            point_dir = os.path.join(outdir, point.index)
            os.makedirs(point_dir, exist_ok=True)

            with open(os.path.join(point_dir, params_filename), "w") as pf:
                for key, value in sorted(point.values.items(), key=lambda kv: kv[0]):
                    pf.write(f"{key} {float(value):e}\n")

            try:
                rendered = template_content.format(**point.values)
            except KeyError as exc:
                fail(f"Template parameter missing in point {point.index}: {exc}")

            with open(os.path.join(point_dir, template_name), "w") as tf:
                tf.write(rendered)
        return

    def write_lookup_table(self, outdir):
        if not self.points:
            return

        self.assign_sequential_indices()
        outdir_clean = outdir[:-1] if outdir.endswith('/') else outdir
        table_file = os.path.join(outdir_clean, f"{outdir_clean.split('/')[-1]}.dat")
        param_names = sorted(self.parameter_names())

        grouped = OrderedDict()
        for point in self.points:
            grouped.setdefault(int(point.sector_id), []).append(point)

        sectorized_intervals = OrderedDict()
        for points in grouped.values():
            if not points:
                continue
            bounds = points[0].sector_bounds
            for pname, (lo, hi) in bounds.items():
                sectorized_intervals.setdefault(pname, set()).add((float(lo), float(hi)))

        interval_index_by_param = OrderedDict()
        for pname, ivals in sectorized_intervals.items():
            ordered = sorted(ivals)
            interval_index_by_param[pname] = {ival: idx for idx, ival in enumerate(ordered)}

        with open(table_file, "w") as f:
            f.write("# INDEX\tSECTOR\t" + "  ".join(param_names) + "\n\n")

            for pname, idx_map in interval_index_by_param.items():
                inv = sorted([(idx, ival) for ival, idx in idx_map.items()], key=lambda x: x[0])
                tail = ", ".join([f"{idx} = [{lo:.6e},{hi:.6e}]" for idx, (lo, hi) in inv])
                f.write(f"# sectorized parameter {pname}: {tail}\n")
            if interval_index_by_param:
                f.write("\n")

            for sid, points in grouped.items():
                sector_bounds = points[0].sector_bounds
                if sector_bounds and interval_index_by_param:
                    tokens = []
                    for pname, (lo, hi) in sector_bounds.items():
                        idx = interval_index_by_param[pname][(float(lo), float(hi))]
                        tokens.append(f"{pname} ({idx})")
                    f.write(f"# sector {sid}: " + " ".join(tokens) + "\n")
                else:
                    f.write(f"# sector {sid}\n")

                for point in points:
                    values = [f"{float(point.values[name]):.6e}" for name in param_names]
                    f.write(f"{point.index}\t{sid}\t" + "\t".join(values) + "\n")
                f.write("\n")
        return

    @staticmethod
    def _load_nominal_values(nominal_path):
        try:
            if nominal_path.endswith(".json"):
                with open(nominal_path) as f:
                    nominal_values = json.load(f)
            else:
                nominal_values = {}
                with open(nominal_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) >= 2:
                            nominal_values[parts[0]] = float(parts[1])
        except (json.JSONDecodeError, ValueError, IOError) as exc:
            fail(f"Invalid nominal file format in '{nominal_path}': {exc}")

        if not nominal_values:
            fail(f"No valid parameters found in nominal file '{nominal_path}'")
        return nominal_values

    def _sector_bounds_by_id(self):
        by_id = OrderedDict()
        for point in self.points:
            if point.sector_id not in by_id:
                by_id[point.sector_id] = point.sector_bounds
        return by_id

    def _prepare_nominal_values_by_sector(self, nominal_values):
        if not self.points:
            return {}

        sector_bounds_by_id = self._sector_bounds_by_id()
        sector_ids = list(sector_bounds_by_id.keys())
        param_names = self.parameter_names()

        sectorized_params = set()
        for bounds in sector_bounds_by_id.values():
            for name in bounds.keys():
                sectorized_params.add(name)

        intervals_by_param = {}
        interval_index_by_param = {}
        for pname in sectorized_params:
            intervals = set()
            for sid in sector_ids:
                bounds = sector_bounds_by_id[sid]
                if pname not in bounds:
                    fail(f"Sector information for parameter '{pname}' is incomplete")
                lo, hi = bounds[pname]
                intervals.add((float(lo), float(hi)))
            intervals_sorted = sorted(intervals)
            intervals_by_param[pname] = intervals_sorted
            interval_index_by_param[pname] = {interval: idx for idx, interval in enumerate(intervals_sorted)}

        by_sector = OrderedDict((sid, {}) for sid in sector_ids)

        for pname in param_names:
            if pname not in nominal_values:
                fail(f"Nominal value for parameter '{pname}' is missing")

            entry = nominal_values[pname]
            is_sectorized = pname in sectorized_params

            if isinstance(entry, list):
                if len(entry) == 0 or not all(is_number(x) for x in entry):
                    fail(f"Nominal entry for '{pname}' must be numeric or list of numeric values")
                values = [float(x) for x in entry]

                if not is_sectorized:
                    if len(values) != 1:
                        fail(f"Parameter '{pname}' is not sectorized, but nominal list has {len(values)} values")
                    for sid in sector_ids:
                        by_sector[sid][pname] = values[0]
                    continue

                n_intervals = len(intervals_by_param[pname])
                if len(values) == 1:
                    for sid in sector_ids:
                        by_sector[sid][pname] = values[0]
                elif len(values) == n_intervals:
                    idx_map = interval_index_by_param[pname]
                    for sid in sector_ids:
                        lo, hi = sector_bounds_by_id[sid][pname]
                        idx = idx_map[(float(lo), float(hi))]
                        by_sector[sid][pname] = values[idx]
                else:
                    fail(
                        f"Nominal list for sectorized parameter '{pname}' has {len(values)} values, expected 1 or {n_intervals}"
                    )
                continue

            if not is_number(entry):
                fail(f"Nominal entry for '{pname}' must be numeric or list of numeric values")

            for sid in sector_ids:
                by_sector[sid][pname] = float(entry)
        return by_sector

    def write_reweighting(self, nominal_path, template_name, template_content, outdir):
        os.makedirs(outdir, exist_ok=True)
        nominal_values = self._load_nominal_values(nominal_path)
        nominal_by_sector = self._prepare_nominal_values_by_sector(nominal_values)

        grouped = OrderedDict()
        for point in self.points:
            grouped.setdefault(int(point.sector_id), []).append(point)

        combined = OrderedDict()
        pad_width = 1 + int(np.ceil(np.log10(max(1, len(self.points)))))
        for sid, points in grouped.items():
            sector_dir = os.path.join(outdir, str(sid).zfill(pad_width))
            os.makedirs(sector_dir, exist_ok=True)

            param_dict = OrderedDict()
            for name in self.parameter_names():
                param_dict[name] = [float(nominal_by_sector[sid][name])]
            for point in points:
                for name in param_dict.keys():
                    param_dict[name].append(float(point.values[name]))

            rendered = template_content.format(**param_dict)
            with open(os.path.join(sector_dir, template_name), "w") as tf:
                tf.write(rendered)

            if not combined:
                combined = OrderedDict((k, []) for k in param_dict.keys())
            for k, vals in param_dict.items():
                combined[k].extend(vals[1:])

        variations_file = os.path.join(outdir, "variations.dat")
        with open(variations_file, "w") as f:
            for pname, values in combined.items():
                values_str = ", ".join([f"{float(v):e}" for v in values])
                f.write(f"{pname} [{values_str}]\n")
        return


def load_template(path):
    if not os.path.exists(path):
        fail(f"Template file '{path}' not found")
    try:
        with open(path, "r") as f:
            return os.path.basename(path), f.read()
    except IOError as exc:
        fail(f"Cannot read template file '{path}': {exc}")
    return


def extract_params_from_minimum_file(path):
    if not os.path.exists(path):
        return {}
    params = {}
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        params[parts[0]] = float(parts[1])
                    except ValueError:
                        continue
                else:
                    break
    except IOError:
        return {}
    return params


def run_tune_mode(scan_dir, template_name, template_content, tune_prefix, defaults_path, outdir, precision):
    if not os.path.isdir(scan_dir):
        fail(f"Scan directory '{scan_dir}' not found")

    os.makedirs(outdir, exist_ok=True)

    if defaults_path is not None:
        if not os.path.exists(defaults_path):
            fail(f"Defaults file '{defaults_path}' not found")
        with open(defaults_path, "r") as f:
            defaults = json.load(f)
        default_dir = os.path.join(outdir, "default")
        os.makedirs(default_dir, exist_ok=True)
        with open(os.path.join(default_dir, template_name), "w") as f:
            f.write(template_content.format(**defaults))

    tune_subdirs = sorted(
        [d for d in os.listdir(scan_dir) if d.startswith(tune_prefix) and os.path.isdir(os.path.join(scan_dir, d))])
    if not tune_subdirs:
        fail(f"No '{tune_prefix}*' subdirectories found in '{scan_dir}'")

    for subdir in tune_subdirs:
        full_subdir = os.path.join(scan_dir, subdir)
        min_files = glob.glob(os.path.join(full_subdir, "minimum_*.txt"))
        tune_dat = os.path.join(full_subdir, "tune.dat")

        if min_files:
            source_file = min_files[0]
        elif os.path.exists(tune_dat):
            source_file = tune_dat
        else:
            continue

        params = extract_params_from_minimum_file(source_file)
        if not params:
            print(f"Warning: no parameters in {source_file}, skipping")
            continue

        if precision is not None:
            params = {k: round(v, precision) for k, v in params.items()}

        target = os.path.join(outdir, subdir)
        if os.path.exists(target):
            print(f"Skipping {target} (already exists)")
            continue
        os.makedirs(target, exist_ok=True)

        try:
            rendered = template_content.format(**params)
        except KeyError as exc:
            print(f"Warning: missing parameter {exc} for {source_file}, skipping")
            continue

        with open(os.path.join(target, template_name), "w") as f:
            f.write(rendered)
    return


def confirm_overwrite(path):
    if os.path.exists(path):
        response = input(f"Warning: Output directory '{path}' already exists. Continue? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(1)
    return


def write_minmax_info(outdir, info_lines, info_filename="params.dat"):
    os.makedirs(outdir, exist_ok=True)
    info_path = os.path.join(outdir, info_filename)
    with open(info_path, "w") as f:
        f.write("# INDEX\tPARAMETER: min/max\n")
        for line in info_lines:
            f.write(line + "\n")
    return


def build_parser():
    epilog = """
SUBCOMMANDS:
  sample    Sample points from parameter bounds (random or uniform)
  import    Load points from table or existing directory
  tune      Extract parameter sets from tune scan directory
  minmax    Generate min/max points for each parameter

Examples:
  Sample 100 random points from parameter ranges:
    create_grid.py sample parameters.json TEMPLATE.yaml -n 100
  Sample points with seed and write lookup table:
    create_grid.py sample parameters.json TEMPLATE.yaml -n 100 --seed 42 --table
  Generate reweighting runcards using nominal values:
    create_grid.py sample parameters.json TEMPLATE.yaml -n 100 --nominal nominal.json
  Uniform grid sampling:
    create_grid.py sample parameters.json TEMPLATE.yaml -n 100 --sampling uniform
  Import from existing table:
    create_grid.py import newscan.dat TEMPLATE.yaml --nominal nominal.json -o output
  Import from existing scan directory:
    create_grid.py import newscan/ TEMPLATE.yaml
  Generate runcards from tune results:
    create_grid.py tune tunes/ TEMPLATE.yaml --tune-prefix tune_
  Generate runcards with min/max extremes:
    create_grid.py minmax parameters.json TEMPLATE.yaml --defaults defaults.json

Advanced Features:
  Dynamic parameter bounds (sample/minmax):
    - JSON bounds can reference other parameters.
    - Example: "paramA": ["paramB", 2.0] constrains paramA based on paramB's sampled value.
  Sectorized parameters (sample only):
    - Breakpoints define sampling sectors for specific parameters.
    - Example: "paramC": [0.0, 1.0, 2.0] creates sectors [0.0,1.0] and [1.0,2.0],
      distributing points uniformly across sectors.
  Reweighting output:
    - Use --nominal to generate reweighting runcards and variations.dat file.
    - Supports per-sector nominal values via lists in the nominal parameter file.
  Lookup table:
    - Use --table to create a summary table mapping points to parameters and sectors.
    """
    
    parser = argparse.ArgumentParser(description="Create and export parameter grids.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=epilog)
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample", help="Sample points from parameter model")
    p_sample.add_argument("parameters", help="Parameter definition file (.json/.txt)")
    p_sample.add_argument("template", help="Template file")
    p_sample.add_argument("-n", "--num-points", required=True, type=int, help="Total number of points")
    p_sample.add_argument("--sampling", choices=["random", "uniform"], default="random")
    p_sample.add_argument("--seed", type=int, help="Random seed (random sampling only)")
    p_sample.add_argument("-o", "--outdir", default="newscan")
    p_sample.add_argument("-t", "--table", action="store_true", help="Write lookup table")
    p_sample.add_argument("-r", "--nominal", help="Nominal parameter file for reweighting output")

    p_import = sub.add_parser("import", help="Import points from table or existing directory")
    p_import.add_argument("input", help="Input table (.dat) or directory")
    p_import.add_argument("template", help="Template file")
    p_import.add_argument("-n", "--num-points", type=int, help="Optional point limit when input is directory")
    p_import.add_argument("-o", "--outdir", default="newscan")
    p_import.add_argument("-t", "--table", action="store_true", help="Write lookup table")
    p_import.add_argument("-r", "--nominal", help="Nominal parameter file for reweighting output")

    p_tune = sub.add_parser("tune", help="Build templates from tune scan directory")
    p_tune.add_argument("scan_dir", help="Directory containing tune results")
    p_tune.add_argument("template", help="Template file")
    p_tune.add_argument("-o", "--outdir", default="newscan")
    p_tune.add_argument("-d", "--defaults", help="Optional defaults JSON")
    p_tune.add_argument("--tune-prefix", default="tune_", help="Required prefix for tune subdirectories")
    p_tune.add_argument("--precision", type=int, default=3, help="Decimal rounding for imported tune params")

    p_minmax = sub.add_parser("minmax", help="Generate min/max points per parameter")
    p_minmax.add_argument("parameters", help="Parameter definition file (.json/.txt)")
    p_minmax.add_argument("template", help="Template file")
    p_minmax.add_argument("-d", "--defaults", required=True, help="Defaults JSON used as baseline")
    p_minmax.add_argument("-o", "--outdir", default="newscan")

    return parser


def print_sampling_summary(args, source_kind, grid, model=None):
    npoints = len(grid.points)
    sector_ids = sorted({int(p.sector_id) for p in grid.points})
    nsectors = len(sector_ids)

    if source_kind == "parameter-file":
        source_msg = f"parameter-file: {args.parameters}"
    elif source_kind == "table-file":
        source_msg = f"table-file: {args.input}"
    else:
        source_msg = f"existing directory: {args.input}"

    dynamic_params = []
    sectorized_params = []
    if model is not None:
        dynamic_params = [name for name, spec in model.specs.items() if spec["kind"] == "dynamic"]
        sectorized_params = [name for name, spec in model.specs.items() if spec["kind"] == "sector"]
    else:
        sec_names = set()
        for point in grid.points:
            for name in point.sector_bounds.keys():
                sec_names.add(name)
        sectorized_params = sorted(sec_names)

    print("Run summary:")
    if args.command == "sample":
        print(f"  Mode: {args.sampling}")
    else:
        print("  Mode: import")
    print(f"  Parameter source: {source_msg}")
    if getattr(args, "nominal", None):
        print(f"  Nominal source: {args.nominal}")
    if args.command == "sample" and args.sampling == "random" and args.seed is not None:
        print(f"  Random seed: {args.seed}")
    print(f"  Number of points: {npoints}")
    if nsectors > 1:
        print(f"  Number of sectors: {nsectors}")
    if model is None and source_kind in ["table-file", "directory"]:
        print("  Dynamic parameters: unknown (input is NOT generated from parameter file)")
    elif dynamic_params:
        print("  Dynamic parameters: " + ", ".join(dynamic_params))
    if sectorized_params:
        print("  Sectorized parameters: " + ", ".join(sectorized_params))
    print()
    return


def print_non_sampling_summary(command, source_msg, defaults=None, tune_prefix=None):
    print("Run summary:")
    print(f"  Mode: {command}")
    print(f"  Parameter source: {source_msg}")
    if defaults:
        print(f"  Default source: {defaults}")
    if tune_prefix:
        print(f"  Tune prefix: {tune_prefix}")
    print()
    return


def print_outputs(outdir, wrote_table=False, wrote_reweighting=False):
    outdir_clean = outdir[:-1] if outdir.endswith('/') else outdir
    outputs = [outdir_clean]
    if wrote_table:
        outputs.append(os.path.join(outdir_clean, f"{os.path.basename(outdir_clean)}.dat"))
    if wrote_reweighting:
        rw_dir = outdir_clean + ".rew"
        outputs.append(rw_dir)
        outputs.append(os.path.join(rw_dir, "variations.dat"))

    print("Created outputs:")
    for p in outputs:
        print(f"  - {p}")
    return


def main():
    print("Starting grid generation...\n")

    parser = build_parser()
    args   = parser.parse_args()

    if args.command == "sample":
        confirm_overwrite(args.outdir)
        template_name, template_content = load_template(args.template)
        model = ParameterModel.from_file(args.parameters)
        grid = ParameterGrid.from_sampling(model, args.num_points, args.sampling, args.seed)
        grid.write_scan(args.outdir, template_name, template_content)

        wrote_table       = False
        wrote_reweighting = False
        if args.table:
            grid.write_lookup_table(args.outdir)
            wrote_table = True
        if args.nominal:
            grid.write_reweighting(args.nominal, template_name, template_content, args.outdir.rstrip("/") + ".rew")
            wrote_reweighting = True
        print_sampling_summary(args, source_kind="parameter-file", grid=grid, model=model)
        print_outputs(args.outdir, wrote_table, wrote_reweighting)
        return

    if args.command == "import":
        confirm_overwrite(args.outdir)
        template_name, template_content = load_template(args.template)

        if os.path.isdir(args.input):
            grid = ParameterGrid.from_directory(args.input, limit=args.num_points)
            source_kind = "directory"
        elif os.path.isfile(args.input) and args.input.endswith(".dat"):
            grid = ParameterGrid.from_table(args.input)
            source_kind = "table-file"
        else:
            fail("Import input must be a directory or a .dat table file")
        grid.write_scan(args.outdir, template_name, template_content)

        wrote_table       = False
        wrote_reweighting = False
        if args.table:
            grid.write_lookup_table(args.outdir)
            wrote_table = True
        if args.nominal:
            grid.write_reweighting(args.nominal, template_name, template_content, args.outdir.rstrip("/") + ".rew")
            wrote_reweighting = True
        print_sampling_summary(args, source_kind=source_kind, grid=grid, model=None)
        print_outputs(args.outdir, wrote_table, wrote_reweighting)
        return

    if args.command == "tune":
        template_name, template_content = load_template(args.template)
        run_tune_mode(scan_dir=args.scan_dir,
            template_name=template_name,
            template_content=template_content,
            tune_prefix=args.tune_prefix,
            defaults_path=args.defaults,
            outdir=args.outdir,
            precision=args.precision)
        print_non_sampling_summary(command="tune",
            source_msg=f"existing directory: {args.scan_dir}",
            defaults=args.defaults,
            tune_prefix=args.tune_prefix)
        print_outputs(args.outdir, wrote_table=False, wrote_reweighting=False)
        return

    if args.command == "minmax":
        confirm_overwrite(args.outdir)
        template_name, template_content = load_template(args.template)
        with open(args.defaults, "r") as f:
            defaults = json.load(f)
        model = ParameterModel.from_file(args.parameters)
        grid, info_lines = ParameterGrid.from_minmax(model, defaults)
        grid.write_scan(args.outdir, template_name, template_content)
        write_minmax_info(args.outdir, info_lines, info_filename="params.dat")
        print_non_sampling_summary(command="minmax",
            source_msg=f"parameter-file: {args.parameters}",
            defaults=args.defaults,)
        print_outputs(args.outdir, wrote_table=False, wrote_reweighting=False)
        return


if __name__ == "__main__":
    main()
