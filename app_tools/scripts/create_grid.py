
import os
import sys
import glob
import json
import re
import numpy as np
import argparse
import itertools
from collections import OrderedDict, defaultdict, deque


def write_params(param_list, templates, outdir, fname="params.dat"):
    if not param_list:
        print("Error: No parameters to write!")
        sys.exit(1)
        
    for num, params in enumerate(param_list):
        npad = f"{num}".zfill(1 + int(np.ceil(np.log10(len(param_list)))))
        outd = os.path.join(outdir, npad)
        outf = os.path.join(outd, fname)

        if not os.path.exists(outd):
            os.makedirs(outd)

        with open(outf, "w") as pf:
            for key, value in params.items():
                if key.startswith("_") or key == "N":
                    continue
                pf.write(f"{key} {float(value):e}\n")

        params["N"] = npad
        template_params = {k: v for k, v in params.items() if not k.startswith("_")}
        for template_name, template_content in templates.items():
            txt = template_content.format(**template_params)
            tname = os.path.join(outd, template_name)
            with open(tname, "w") as tf:
                tf.write(txt)


def load_params_from_folders(outdir="newscan", fname="params.dat", npoints=None):
    if not os.path.exists(outdir):
        print(f"Error: Directory '{outdir}' not found!")
        sys.exit(1)
        
    param_dicts = []
    subfolders = sorted([d for d in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, d))])
    
    if npoints is not None:
        if len(subfolders) < npoints:
            print(f"Error: Only {len(subfolders)} subfolders found, but {npoints} requested")
            sys.exit(1)
        subfolders = subfolders[:npoints]
    
    for sub in subfolders:
        param_file = os.path.join(outdir, sub, fname)
        if not os.path.exists(param_file):
            continue
        params = {}
        try:
            with open(param_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        key, val = parts[0], parts[1]
                        params[key] = float(val)
        except (ValueError, IOError) as e:
            print(f"Warning: Error reading {param_file}: {e}.")
            continue
        if params:
            params["_sector_id"] = 0
            params["_sector_bounds"] = OrderedDict()
            param_dicts.append(params)
    
    if not param_dicts:
        print(f"Error: No valid parameter files found in '{outdir}'!")
        sys.exit(1)
        
    return param_dicts


def extract_params_from_minimum_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found!")
        return {}
        
    params = {}
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error: Cannot read file '{filepath}': {e}!")
        return {}
        
    for line in lines:
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                val = float(parts[1])
                params[key] = val
            except ValueError:
                continue
        else:
            break
    return params


def tune_mode(scan_dir, template_path, tune_tag, defaults_path=None, outdir="newscan", precision=3):
    if not os.path.exists(scan_dir):
        print(f"Error: Scan directory '{scan_dir}' not found!")
        sys.exit(1)
        
    defaults = None
    if defaults_path is not None:
        if not os.path.exists(defaults_path):
            print(f"Error: Defaults file '{defaults_path}' not found!")
            sys.exit(1)
        try:
            with open(defaults_path, "r") as f:
                defaults = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: Cannot read defaults file '{defaults_path}': {e}!")
            sys.exit(1)

    try:
        with open(template_path, "r") as f:
            template_content = f.read()
    except IOError as e:
        print(f"Error: Cannot read template file '{template_path}': {e}!")
        sys.exit(1)
        
    template_name = os.path.basename(template_path)
    out_base = os.path.join(outdir)
    os.makedirs(out_base, exist_ok=True)

    if defaults is not None:
        out_dir = os.path.join(out_base, "default")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, template_name)
        try:
            filled = template_content.format(**defaults)
        except KeyError as e:
            print(f"Error: Missing parameter {e} in defaults file!")
            sys.exit(1)
        with open(out_file, "w") as f:
            f.write(filled)
        print(f"Wrote default values to {out_file}.")

    tune_subdirs = sorted([d for d in os.listdir(scan_dir) 
                          if tune_tag in d and os.path.isdir(os.path.join(scan_dir, d))])

    if not tune_subdirs:
        print(f"Error: No {tune_tag}* subdirectories found in '{scan_dir}'!")
        sys.exit(1)

    for subdir in tune_subdirs:
        full_subdir = os.path.join(scan_dir, subdir)
        min_files = glob.glob(os.path.join(full_subdir, "minimum_*.txt"))
        tune_dat = os.path.join(full_subdir, "tune.dat")
        
        if min_files:
            min_file = min_files[0]
        elif os.path.exists(tune_dat):
            min_file = tune_dat
        else:
            continue
            
        params = extract_params_from_minimum_file(min_file)
        if not params:
            print(f"Warning: No parameters found in {min_file}, skipping...")
            continue
        
        if precision is not None:
            params = {k: round(v, precision) for k, v in params.items()}

        out_dir = os.path.join(out_base, subdir)
        if os.path.exists(out_dir):
            print(f"Skipping {out_dir} (already exists)...")
            continue

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, template_name)

        try:
            filled = template_content.format(**params)
        except KeyError as e:
            print(f"Error: Missing parameter {e} for {min_file}!")
            continue

        with open(out_file, "w") as f:
            f.write(filled)
        print(f"Wrote {out_file}")


def minmax_mode(boxdef, defaults_path, template_path, outdir="minmaxscan", infofile_path=None):
    model = ParameterModel.from_file(boxdef)

    for name, spec in model.specs.items():
        if spec["kind"] != "range":
            print(f"Error: minmax mode only supports fixed [min,max] ranges. Parameter '{name}' is '{spec['kind']}'.")
            sys.exit(1)

    if not os.path.exists(defaults_path):
        print(f"Error: Defaults file '{defaults_path}' not found!")
        sys.exit(1)
        
    try:
        with open(defaults_path, "r") as f:
            defaults = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error: Cannot read defaults file '{defaults_path}': {e}!")
        sys.exit(1)

    param_order = sorted(model.specs.keys())

    param_sets = []
    info_lines = []
    for idx, param_name in enumerate(param_order):
        lo, hi = model.specs[param_name]["bounds"]
        for bound_idx, bound_name in enumerate(["min", "max"]):
            params = defaults.copy()
            params[param_name] = lo if bound_idx == 0 else hi
            param_sets.append(params)
            info_lines.append(f"{2*idx + bound_idx}\t{param_name}: {bound_name}")

    if infofile_path is not None:
        info_dir = os.path.dirname(infofile_path)
        if info_dir:
            os.makedirs(info_dir, exist_ok=True)
        try:
            with open(infofile_path, "w") as f:
                f.write("index\tparameter: bound\n")
                for line in info_lines:
                    f.write(line + "\n")
        except IOError as e:
            print(f"Warning: Cannot write info file '{infofile_path}': {e}.")

    try:
        with open(template_path, "r") as f:
            template_content = f.read()
    except IOError as e:
        print(f"Error: Cannot read template file '{template_path}': {e}!")
        sys.exit(1)
        
    template_name = os.path.basename(template_path)
    templates = {template_name: template_content}
    write_params(param_sets, templates, outdir)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _is_param_ref(x):
    return isinstance(x, str) and x.strip() != ""


def _physical_param_items(params):
    for k, v in params.items():
        if k.startswith("_") or k == "N":
            continue
        yield k, v


def _physical_param_names(params):
    return [k for k, _ in _physical_param_items(params)]


def _zfill_for_count(n):
    if n <= 1:
        return 1
    return 1 + int(np.ceil(np.log10(n)))


class ParameterModel:
    def __init__(self, ordered_specs):
        self.specs = ordered_specs
        self.order = list(ordered_specs.keys())
        self._validate_refs_and_cycles()

    @classmethod
    def from_file(cls, boxdef):
        if boxdef.endswith(".json"):
            with open(boxdef) as f:
                raw = json.load(f, object_pairs_hook=OrderedDict)
            return cls(cls._parse_json_specs(raw, boxdef))

        specs = OrderedDict()
        with open(boxdef) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                name = parts[0]
                lo = float(parts[1])
                hi = float(parts[2])
                if lo > hi:
                    print(f"Error: Invalid bounds for '{name}' in '{boxdef}' ({lo} > {hi})!")
                    sys.exit(1)
                specs[name] = {"kind": "range", "bounds": [float(lo), float(hi)]}

        if not specs:
            print(f"Error: No valid parameters found in '{boxdef}'!")
            sys.exit(1)
        return cls(specs)

    @staticmethod
    def _parse_json_specs(raw, boxdef):
        if not isinstance(raw, dict) or len(raw) == 0:
            print(f"Error: No valid parameters found in '{boxdef}'!")
            sys.exit(1)

        specs = OrderedDict()
        for name, val in raw.items():
            if not isinstance(val, list) or len(val) < 2:
                print(f"Error: Parameter '{name}' must be a list with at least 2 entries in '{boxdef}'!")
                sys.exit(1)

            if len(val) == 2:
                a, b = val
                if _is_number(a) and _is_number(b):
                    lo, hi = float(a), float(b)
                    if lo > hi:
                        print(f"Error: Invalid bounds for '{name}' ({lo} > {hi}) in '{boxdef}'!")
                        sys.exit(1)
                    specs[name] = {"kind": "range", "bounds": [lo, hi]}
                    continue

                if ((_is_number(a) or _is_param_ref(a)) and (_is_number(b) or _is_param_ref(b))):
                    specs[name] = {"kind": "dynamic", "bounds": [a, b]}
                    continue

                print(f"Error: Invalid bounds definition for '{name}' in '{boxdef}'!")
                sys.exit(1)

            if all(_is_number(x) for x in val):
                points = [float(x) for x in val]
                if any(points[i] >= points[i + 1] for i in range(len(points) - 1)):
                    print(f"Error: Breakpoints for sectorized parameter '{name}' must be strictly increasing in '{boxdef}'!")
                    sys.exit(1)
                specs[name] = {"kind": "sector", "breakpoints": points}
                continue

            print(f"Error: Unsupported definition for parameter '{name}' in '{boxdef}'!")
            sys.exit(1)

        return specs

    def _validate_refs_and_cycles(self):
        names = set(self.specs.keys())
        edges = defaultdict(set)
        indeg = {n: 0 for n in names}

        for name, spec in self.specs.items():
            if spec["kind"] != "dynamic":
                continue
            lo, hi = spec["bounds"]
            refs = []
            if _is_param_ref(lo):
                refs.append(lo)
            if _is_param_ref(hi):
                refs.append(hi)

            for ref in refs:
                if ref not in names:
                    print(f"Error: Parameter '{name}' references unknown parameter '{ref}' in dynamic bounds!")
                    sys.exit(1)
                if ref == name:
                    print(f"Error: Parameter '{name}' cannot reference itself in dynamic bounds!")
                    sys.exit(1)
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
            print("Error: Cyclic dynamic parameter dependencies detected!")
            sys.exit(1)

        self.sampling_order = topo

    def has_dynamic(self):
        return any(spec["kind"] == "dynamic" for spec in self.specs.values())

    def build_sector_list(self):
        sector_params = [name for name, spec in self.specs.items() if spec["kind"] == "sector"]
        if not sector_params:
            return [{"id": 0, "bounds": OrderedDict()}]

        interval_options = []
        for name in sector_params:
            b = self.specs[name]["breakpoints"]
            intervals = [(b[i], b[i + 1]) for i in range(len(b) - 1)]
            interval_options.append(intervals)

        sectors = []
        sid = 0
        for combo in itertools.product(*interval_options):
            bounds = OrderedDict()
            for pname, (lo, hi) in zip(sector_params, combo):
                bounds[pname] = [float(lo), float(hi)]
            sectors.append({"id": sid, "bounds": bounds})
            sid += 1
        return sectors

    def _resolve_bound_expr(self, expr, sampled_values):
        if _is_number(expr):
            return float(expr)
        return float(sampled_values[expr])

    def resolve_bounds_for_param(self, param_name, sampled_values, sector_bounds):
        spec = self.specs[param_name]
        kind = spec["kind"]

        if kind == "sector":
            lo, hi = sector_bounds[param_name]
            return float(lo), float(hi)

        if kind == "range":
            lo, hi = spec["bounds"]
            return float(lo), float(hi)

        lo_expr, hi_expr = spec["bounds"]
        lo = self._resolve_bound_expr(lo_expr, sampled_values)
        hi = self._resolve_bound_expr(hi_expr, sampled_values)
        return float(lo), float(hi)


def _allocate_points_per_sector(total, nsectors):
    base = total // nsectors
    rem = total % nsectors
    return [base + (1 if i < rem else 0) for i in range(nsectors)]


def _sample_random_for_sector(model, npoints, sector):
    if npoints <= 0:
        return []

    points = []
    for _ in range(npoints):
        sampled = OrderedDict()
        for name in model.sampling_order:
            lo, hi = model.resolve_bounds_for_param(name, sampled, sector["bounds"])
            if lo > hi:
                print(f"Error: Invalid dynamic bounds for parameter '{name}' in sector {sector['id']} ({lo} > {hi}).")
                sys.exit(1)
            sampled[name] = float(np.random.uniform(lo, hi))

        sampled["_sector_id"] = int(sector["id"])
        sampled["_sector_bounds"] = sector["bounds"]
        points.append(sampled)

    return points


def _sample_uniform_for_sector(model, npoints, sector):
    if npoints <= 0:
        return []
    if model.has_dynamic():
        print("Error: uniform mode is currently incompatible with dynamic parameter bounds.")
        sys.exit(1)

    names = model.order
    ndim = len(names)
    n_per_dim = int(np.ceil(npoints ** (1 / ndim)))

    while True:
        grids = []
        for name in names:
            lo, hi = model.resolve_bounds_for_param(name, {}, sector["bounds"])
            if n_per_dim == 1:
                grids.append([lo])
            else:
                step = (hi - lo) / (n_per_dim - 1)
                grids.append([lo + i * step for i in range(n_per_dim)])

        all_points = list(itertools.product(*grids))
        if len(all_points) >= npoints:
            out = []
            for tup in all_points[:npoints]:
                d = OrderedDict(zip(names, tup))
                d["_sector_id"] = int(sector["id"])
                d["_sector_bounds"] = sector["bounds"]
                out.append(d)
            return out
        n_per_dim += 1


def sample_from_model(model, npoints, mode):
    sectors = model.build_sector_list()
    allocation = _allocate_points_per_sector(npoints, len(sectors))

    points = []
    for sector, nsec in zip(sectors, allocation):
        if mode == "uniform":
            points.extend(_sample_uniform_for_sector(model, nsec, sector))
        else:
            points.extend(_sample_random_for_sector(model, nsec, sector))
    return points


def load_params_from_table(table_path):
    params = []
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
                            for idx_str, lo_str, hi_str in re.findall(r"(\d+)\s*=\s*\[\s*([^,\]]+)\s*,\s*([^\]]+)\s*\]", tail):
                                try:
                                    idx_map[int(idx_str)] = (float(lo_str), float(hi_str))
                                except ValueError:
                                    pass
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

                cols = line.split()
                if header is None:
                    header = cols
                    continue

                if len(cols) != len(header):
                    print(f"Warning: Skipping malformed line in '{table_path}': {raw.rstrip()}")
                    continue

                row = dict(zip(header, cols))
                p = OrderedDict()
                for k, v in row.items():
                    lk = k.lower()
                    if lk == "index":
                        p["N"] = v
                        continue
                    if lk == "sector":
                        try:
                            current_sector = int(float(v))
                        except ValueError:
                            pass
                        continue
                    p[k] = float(v)

                p["_sector_id"] = int(current_sector)
                p["_sector_bounds"] = OrderedDict((k, [v[0], v[1]]) for k, v in current_sector_bounds.items())
                params.append(p)

    except (IOError, ValueError) as e:
        print(f"Error: Cannot parse table file '{table_path}': {e}!")
        sys.exit(1)

    if not params:
        print(f"Error: No valid parameter rows found in '{table_path}'!")
        sys.exit(1)
    return params


def write_lookup_table(param_list, outdir):
    if not param_list:
        return

    outdir_clean = outdir[:-1] if outdir.endswith('/') else outdir
    table_file = os.path.join(outdir_clean, f"{outdir_clean.split('/')[-1]}.dat")
    param_names = sorted(_physical_param_names(param_list[0]))

    grouped = OrderedDict()
    for p in param_list:
        sid = int(p.get("_sector_id", 0))
        grouped.setdefault(sid, []).append(p)

    sectorized_intervals = OrderedDict()
    for points in grouped.values():
        if not points:
            continue
        bounds = points[0].get("_sector_bounds", OrderedDict())
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
            sector_bounds = points[0].get("_sector_bounds", OrderedDict())
            if sector_bounds and interval_index_by_param:
                tokens = []
                for pname, (lo, hi) in sector_bounds.items():
                    idx = interval_index_by_param[pname][(float(lo), float(hi))]
                    tokens.append(f"{pname} ({idx})")
                f.write(f"# sector {sid}: " + " ".join(tokens) + "\n")
            else:
                f.write(f"# sector {sid}\n")
            for p in points:
                idx = p.get("N", "")
                values = [f"{float(p[name]):.6e}" for name in param_names]
                f.write(f"{idx}\t{sid}\t" + "\t".join(values) + "\n")
            f.write("\n")


def load_nominal_parameter_values(nominal_path):
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
    except (json.JSONDecodeError, ValueError, IOError) as e:
        print(f"Error: Invalid parameter file format in '{nominal_path}': {e}!")
        sys.exit(1)

    if not nominal_values:
        print(f"Error: No valid parameters found in '{nominal_path}'!")
        sys.exit(1)
    return nominal_values


def _get_sector_bounds_by_id(param_list):
    by_id = OrderedDict()
    for p in param_list:
        sid = int(p.get("_sector_id", 0))
        if sid not in by_id:
            by_id[sid] = p.get("_sector_bounds", OrderedDict())
    return by_id


def _prepare_nominal_values_by_sector(param_list, nominal_values):
    if not param_list:
        return {}

    sector_bounds_by_id = _get_sector_bounds_by_id(param_list)
    sector_ids = list(sector_bounds_by_id.keys())
    param_names = _physical_param_names(param_list[0])

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
                print(f"Error: Sector information for parameter '{pname}' is incomplete.")
                sys.exit(1)
            lo, hi = bounds[pname]
            intervals.add((float(lo), float(hi)))
        intervals_sorted = sorted(intervals)
        intervals_by_param[pname] = intervals_sorted
        interval_index_by_param[pname] = {
            interval: idx for idx, interval in enumerate(intervals_sorted)
        }

    by_sector = OrderedDict((sid, {}) for sid in sector_ids)

    for pname in param_names:
        if pname not in nominal_values:
            print(f"Error: Nominal value for parameter '{pname}' is missing in reweighting input!")
            sys.exit(1)

        entry = nominal_values[pname]
        is_sectorized = pname in sectorized_params

        if isinstance(entry, list):
            if len(entry) == 0 or not all(_is_number(x) for x in entry):
                print(f"Error: Nominal entry for '{pname}' must be numeric or list of numeric values!")
                sys.exit(1)

            values = [float(x) for x in entry]

            if not is_sectorized:
                if len(values) == 1:
                    for sid in sector_ids:
                        by_sector[sid][pname] = values[0]
                else:
                    print(
                        f"Error: Parameter '{pname}' is not sectorized, but nominal list has {len(values)} values."
                    )
                    sys.exit(1)
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
                print(
                    f"Error: Nominal list for sectorized parameter '{pname}' has {len(values)} values, "
                    f"but expected 1 or {n_intervals}."
                )
                sys.exit(1)
            continue

        if not _is_number(entry):
            print(f"Error: Nominal entry for '{pname}' must be numeric or list of numeric values!")
            sys.exit(1)

        value = float(entry)
        for sid in sector_ids:
            by_sector[sid][pname] = value

    return by_sector


def build_reweighting_dict(points, sector_nominal_values):
    d = OrderedDict()
    for name in _physical_param_names(points[0]):
        d[name] = [float(sector_nominal_values[name])]
    for p in points:
        for name in d.keys():
            d[name].append(float(p[name]))
    return d


def write_reweighting_runcards_by_sector(param_list, templates, nominal_values, outdir="newscan.rew"):
    os.makedirs(outdir, exist_ok=True)
    grouped = OrderedDict()
    for p in param_list:
        sid = int(p.get("_sector_id", 0))
        grouped.setdefault(sid, []).append(p)

    nominal_by_sector = _prepare_nominal_values_by_sector(param_list, nominal_values)
    combined = OrderedDict()
    pad_width = _zfill_for_count(len(param_list))
    for sid, points in grouped.items():
        sector_dir = os.path.join(outdir, str(sid).zfill(pad_width))
        os.makedirs(sector_dir, exist_ok=True)
        param_dict = build_reweighting_dict(points, nominal_by_sector[sid])
        for template_name, template_content in templates.items():
            txt = template_content.format(**param_dict)
            with open(os.path.join(sector_dir, template_name), "w") as tf:
                tf.write(txt)

        if not combined:
            combined = OrderedDict((k, []) for k in param_dict.keys())
        for k, vals in param_dict.items():
            combined[k].extend(vals[1:])
    return combined


def write_variations_file(param_dict, outdir="newscan.rew"):
    variations_file = os.path.join(outdir, "variations.dat")
    with open(variations_file, "w") as f:
        for param_name, values in param_dict.items():
            values_str = ", ".join([f"{float(v):e}" for v in values])
            f.write(f"{param_name} [{values_str}]\n")


def _validate_mode_compatibility(args, is_json, is_dat, is_directory):
    if args.mode == "tune":
        if not is_directory:
            print("Error: tune mode requires a scan directory with tune_* subdirectories!")
            sys.exit(1)
        if args.npoints is not None:
            print("Warning: npoints argument ignored in tune mode.")
        if args.seed is not None:
            print("Warning: --seed argument ignored in tune mode.")
        if args.table:
            print("Warning: --table argument ignored in tune mode.")
        return

    if args.mode == "minmax":
        if not is_json:
            print("Error: minmax mode requires a parameter file (json/txt)!")
            sys.exit(1)
        if not args.default:
            print("Error: minmax mode requires --default defaults.json file!")
            sys.exit(1)
        return

    if args.default is not None:
        print("Warning: --default argument ignored in random/uniform mode.")
    if args.tune_tag is not None:
        print("Warning: --tune_tag argument ignored in random/uniform mode.")

    if is_directory or is_dat:
        if args.npoints is not None:
            print("Warning: npoints argument ignored for directory/.dat input.")
        if args.seed is not None:
            print("Warning: --seed argument ignored for directory/.dat input.")
        return

    if is_json:
        if args.npoints is None or args.npoints <= 0:
            print("Error: npoints must be a positive integer for random/uniform mode with parameter file!")
            sys.exit(1)
        if args.mode == "uniform" and args.seed is not None:
            print("Warning: --seed argument ignored in uniform mode.")
        return

    print("Error: requires either a parameter file (json/txt), a table (.dat), or a newscan directory!")
    sys.exit(1)


def _summarize_input_and_plan(args, source_kind, model, param_list):
    npoints = len(param_list)
    sector_ids = sorted({int(p.get("_sector_id", 0)) for p in param_list})
    nsectors = len(sector_ids)

    if source_kind == "parameter-file":
        source_msg = f"parameter-file: {args.parameters}"
    elif source_kind == "table-file":
        source_msg = f"table-file: {args.parameters}"
    else:
        source_msg = f"existing directory: {args.parameters}"

    dynamic_params = []
    sectorized_params = []

    if model is not None:
        dynamic_params = [name for name, spec in model.specs.items() if spec["kind"] == "dynamic"]
        sectorized_params = [name for name, spec in model.specs.items() if spec["kind"] == "sector"]
    else:
        sec_names = set()
        for p in param_list:
            for k in p.get("_sector_bounds", {}).keys():
                sec_names.add(k)
        sectorized_params = sorted(sec_names)

    print("\nRun summary:")
    print(f"  Mode: {args.mode}")
    print(f"  Parameter source: {source_msg}")
    if args.reweighting:
        print(f"  Nominal source: {args.reweighting}")
    if args.mode == "random" and source_kind == "parameter-file" and args.seed is not None:
        print(f"  Random seed: {args.seed}")
    print(f"  Number of points: {npoints}")
    
    if nsectors > 1:
        print(f"  Number of sectors: {nsectors}")

    if model is None and source_kind in ["table-file", "directory"]:
        print("  Dynamic parameters: unknown (input is pre-generated, not boundary model)")
    elif dynamic_params:
        print("  Dynamic parameters: " + ", ".join(dynamic_params))

    if sectorized_params:
        print("  Sectorized parameters: " + ", ".join(sectorized_params))


def _print_created_outputs(args, wrote_table, wrote_reweighting):
    outdir_clean = args.outdir[:-1] if args.outdir.endswith('/') else args.outdir
    outputs = [outdir_clean]

    if wrote_table:
        table_file = os.path.join(outdir_clean, f"{outdir_clean.split('/')[-1]}.dat")
        outputs.append(table_file)

    if wrote_reweighting:
        rw_dir = outdir_clean + ".rew"
        outputs.append(rw_dir)
        outputs.append(os.path.join(rw_dir, "variations.dat"))

    print("\nCreated outputs:")
    for path in outputs:
        print(f"  - {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Sample and create templates for parameter grid generation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  create_grid.py parameters.json TEMPLATE.yaml 20
  create_grid.py parameters.json TEMPLATE.yaml 20 --seed 42 --table
  create_grid.py parameters.json TEMPLATE.yaml 20 --table --reweighting nominal.json -o scan
  create_grid.py newscan TEMPLATE.yaml --mode tune --default defaults.json
  create_grid.py newscan.dat TEMPLATE.yaml --reweighting nominal.json -o output

Modes:
  random      Sample points randomly from parameter ranges (default)
  uniform     Sample points on uniform grid
  tune        Process existing scan directory with tune_* subdirectories
  minmax      Generate min/max points for each parameter

Use --table to create a lookup table for all generated points.
Use --reweighting to generate (sector-wise) reweighting runcards with per-sector nominal values.
        """
    )
    parser.add_argument("parameters", help="Parameter file (.json/.txt), table (.dat), newscan_dir, or scan_dir (tune mode)")
    parser.add_argument("template", help="Template file")
    parser.add_argument("npoints", nargs="?", type=int, help="Total points for sampling (json/txt + random/uniform)")
    parser.add_argument("--mode", choices=["random", "uniform", "tune", "minmax"], default="random", help="Sampling mode (default: random)")
    parser.add_argument("-s", "--seed", type=int, help="Random seed (random mode)")
    parser.add_argument("-d", "--default", help="Defaults.json file (tune/minmax mode)")
    parser.add_argument("-o", "--outdir", default="newscan", help="Output directory name (default: newscan)")
    parser.add_argument("--tune-tag", help="Prefix for tune directories (default: tune_)")
    parser.add_argument("--precision", type=int, default=3, help="Number of decimal places for parameters in tune mode")
    parser.add_argument("-t", "--table", action="store_true", help="Create a lookup table for all generated points")
    parser.add_argument("-r", "--reweighting", help="Nominal parameter set for (sector-wise) reweighting runcards")
    args = parser.parse_args()

    if not os.path.exists(args.parameters):
        print(f"Error: Parameters input '{args.parameters}' not found!")
        sys.exit(1)
    if not os.path.exists(args.template):
        print(f"Error: Template file '{args.template}' not found!")
        sys.exit(1)
    if args.mode in ["random", "uniform", "minmax"] and os.path.exists(args.outdir):
        response = input(f"Warning: Output directory '{args.outdir}' already exists. Continue? [y/N] ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(1)

    is_directory = os.path.isdir(args.parameters)
    is_dat = os.path.isfile(args.parameters) and args.parameters.endswith(".dat")
    is_json_txt = os.path.isfile(args.parameters) and args.parameters.endswith((".json", ".txt"))
    _validate_mode_compatibility(args, is_json_txt, is_dat, is_directory)

    if args.mode == "tune":
        if args.tune_tag is None:
            args.tune_tag = "tune_"
        tune_mode(args.parameters, args.template, args.tune_tag, args.default, args.outdir, args.precision)
        return

    if args.mode == "minmax":
        infofile_path = os.path.join(args.outdir, "params.dat")
        minmax_mode(args.parameters, args.default, args.template, args.outdir, infofile_path=infofile_path)
        return

    model = None
    source_kind = None
    if is_directory:
        print(f"Loading parameters from: {args.parameters}...")
        param_list = load_params_from_folders(args.parameters, npoints=args.npoints)
        source_kind = "directory"
    elif is_dat:
        print(f"Loading parameters from table: {args.parameters}...")
        param_list = load_params_from_table(args.parameters)
        source_kind = "table-file"
    else:
        print("Sampling new parameters...")
        if args.seed is not None and args.mode == "random":
            np.random.seed(args.seed)
            print(f"Using random seed: {args.seed}")
        model = ParameterModel.from_file(args.parameters)
        param_list = sample_from_model(model, args.npoints, args.mode)
        source_kind = "parameter-file"

    _summarize_input_and_plan(args, source_kind, model, param_list)

    try:
        with open(args.template, "r") as f:
            template_content = f.read()
    except IOError as e:
        print(f"Error: Cannot read template file '{args.template}': {e}!")
        sys.exit(1)

    template_name = os.path.basename(args.template)
    templates = {template_name: template_content}
    write_params(param_list, templates, args.outdir)

    wrote_reweighting = False
    if args.reweighting:
        nominal_values = load_nominal_parameter_values(args.reweighting)
        reweight_outdir = args.outdir.rstrip("/") + ".rew"
        combined = write_reweighting_runcards_by_sector(param_list, templates, nominal_values, outdir=reweight_outdir)
        write_variations_file(combined, outdir=reweight_outdir)
        wrote_reweighting = True

    wrote_table = False
    if args.table:
        write_lookup_table(param_list, args.outdir)
        wrote_table = True

    _print_created_outputs(args, wrote_table, wrote_reweighting)


if __name__ == "__main__":
    main()
