
import os
import sys
import glob
import json
import numpy as np
import argparse
import itertools


def write_params(param_list, templates, outdir, fname="params.dat"):
    if not param_list:
        print("Error: No parameters to write")
        sys.exit(1)
        
    for num, params in enumerate(param_list):
        npad = f"{num}".zfill(1 + int(np.ceil(np.log10(len(param_list)))))
        outd = os.path.join(outdir, npad)
        outf = os.path.join(outd, fname)

        if not os.path.exists(outd):
            os.makedirs(outd)

        with open(outf, "w") as pf:
            for key, value in params.items():
                pf.write(f"{key} {value:e}\n")

        params["N"] = npad
        for template_name, template_content in templates.items():
            txt = template_content.format(**params)
            tname = os.path.join(outd, template_name)
            with open(tname, "w") as tf:
                tf.write(txt)


def write_lookup_table(param_list, outdir):
    """Write a lookup table with folder indices and parameter values."""
    if not param_list:
        return
        
    table_file = os.path.join(outdir, "params.dat")
    
    all_param_names = param_list[0].keys()
    param_names = sorted([name for name in all_param_names 
                         if isinstance(param_list[0][name], (int, float))])
    
    try:
        with open(table_file, "w") as f:
            f.write("index\t" + "\t".join(param_names) + "\n")
            
            for num, params in enumerate(param_list):
                npad = f"{num}".zfill(1 + int(np.ceil(np.log10(len(param_list)))))
                values = []
                for name in param_names:
                    value = params[name]
                    if isinstance(value, (int, float)):
                        values.append(f"{value:.6e}")
                    else:
                        values.append(str(value))
                f.write(f"{npad}\t" + "\t".join(values) + "\n")

    except IOError as e:
        print(f"Warning: Cannot write lookup table '{table_file}': {e}")


def sample_random(boxdef, npoints):
    boundaries = load_parameter_boundaries(boxdef)
    param_order = sorted(boundaries.keys())
    xmin = [boundaries[x][0] for x in param_order]
    xmax = [boundaries[x][1] for x in param_order]

    random_points = np.random.uniform(low=xmin, high=xmax, size=(npoints, len(xmin)))
    return [dict(zip(param_order, x)) for x in random_points]


def sample_uniform(boxdef, npoints):
    boundaries = load_parameter_boundaries(boxdef)

    param_order = sorted(boundaries.keys())
    xmin = [boundaries[x][0] for x in param_order]
    xmax = [boundaries[x][1] for x in param_order]

    ndim = len(param_order)
    n_per_dim = int(np.ceil(npoints ** (1/ndim)))

    grids = []
    for lo, hi in zip(xmin, xmax):
        if n_per_dim == 1:
            grids.append([lo])
        else:
            step = (hi - lo) / (n_per_dim - 1)
            grids.append([lo + i * step for i in range(n_per_dim)])

    all_points = list(itertools.product(*grids))
    all_points = all_points[:npoints]
    return [dict(zip(param_order, x)) for x in all_points]


def load_params_from_folders(outdir="newscan", fname="params.dat", npoints=None):
    if not os.path.exists(outdir):
        print(f"Error: Directory '{outdir}' not found")
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
            print(f"Warning: Error reading {param_file}: {e}")
            continue
        if params:
            param_dicts.append(params)
    
    if not param_dicts:
        print(f"Error: No valid parameter files found in '{outdir}'")
        sys.exit(1)
        
    return param_dicts


def extract_params_from_minimum_file(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found")
        return {}
        
    params = {}
    try:
        with open(filepath) as f:
            lines = f.readlines()
    except IOError as e:
        print(f"Error: Cannot read file '{filepath}': {e}")
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


def load_parameter_boundaries(boxdef):
    try:
        with open(boxdef) as f:
            c = f.read(1)
            is_json = c == "{"
    except FileNotFoundError:
        print(f"Error: Parameter file '{boxdef}' not found")
        sys.exit(1)

    try:
        if is_json:
            with open(boxdef) as f:
                boundaries = json.load(f)
        else:
            with open(boxdef) as f:
                lines = [line.strip().split() for line in f if line.strip() and not line.startswith("#")]
                boundaries = {x[0]: [float(x[1]), float(x[2])] for x in lines if len(x) >= 3}
    except (json.JSONDecodeError, ValueError, IndexError) as e:
        print(f"Error: Invalid parameter file format in '{boxdef}': {e}")
        sys.exit(1)
    
    if not boundaries:
        print(f"Error: No valid parameters found in '{boxdef}'")
        sys.exit(1)
    
    return boundaries


def tune_mode(scan_dir, template_path, tune_tag, defaults_path=None, outdir="newscan"):
    if not os.path.exists(scan_dir):
        print(f"Error: Scan directory '{scan_dir}' not found")
        sys.exit(1)
        
    defaults = None
    if defaults_path is not None:
        if not os.path.exists(defaults_path):
            print(f"Error: Defaults file '{defaults_path}' not found")
            sys.exit(1)
        try:
            with open(defaults_path, "r") as f:
                defaults = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error: Cannot read defaults file '{defaults_path}': {e}")
            sys.exit(1)

    try:
        with open(template_path, "r") as f:
            template_content = f.read()
    except IOError as e:
        print(f"Error: Cannot read template file '{template_path}': {e}")
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
            print(f"Error: Missing parameter {e} in defaults file")
            sys.exit(1)
        with open(out_file, "w") as f:
            f.write(filled)
        print(f"Wrote default values to {out_file}")

    tune_subdirs = sorted([d for d in os.listdir(scan_dir) 
                          if tune_tag in d and os.path.isdir(os.path.join(scan_dir, d))])

    if not tune_subdirs:
        print(f"Error: No {tune_tag}* subdirectories found in '{scan_dir}'")
        sys.exit(1)

    for subdir in tune_subdirs:
        full_subdir = os.path.join(scan_dir, subdir)
        min_files = glob.glob(os.path.join(full_subdir, "minimum_*.txt"))
        if not min_files:
            continue
        min_file = min_files[0]
        params = extract_params_from_minimum_file(min_file)
        if not params:
            print(f"Warning: No parameters found in {min_file}, skipping")
            continue

        out_dir = os.path.join(out_base, subdir)
        if os.path.exists(out_dir):
            print(f"Skipping {out_dir} (already exists)")
            continue

        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, template_name)

        try:
            filled = template_content.format(**params)
        except KeyError as e:
            print(f"Error: Missing parameter {e} for {min_file}")
            continue

        with open(out_file, "w") as f:
            f.write(filled)
        print(f"Wrote {out_file}")


def minmax_mode(boxdef, defaults_path, template_path, outdir="minmaxscan", infofile_path=None):
    boundaries = load_parameter_boundaries(boxdef)

    if not os.path.exists(defaults_path):
        print(f"Error: Defaults file '{defaults_path}' not found")
        sys.exit(1)
        
    try:
        with open(defaults_path, "r") as f:
            defaults = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error: Cannot read defaults file '{defaults_path}': {e}")
        sys.exit(1)

    param_order = sorted(boundaries.keys())

    param_sets = []
    info_lines = []
    for idx, param_name in enumerate(param_order):
        for bound_idx, bound_name in enumerate(["min", "max"]):
            params = defaults.copy()
            params[param_name] = boundaries[param_name][bound_idx]
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
            print(f"Warning: Cannot write info file '{infofile_path}': {e}")

    try:
        with open(template_path, "r") as f:
            template_content = f.read()
    except IOError as e:
        print(f"Error: Cannot read template file '{template_path}': {e}")
        sys.exit(1)
        
    template_name = os.path.basename(template_path)
    templates = {template_name: template_content}
    write_params(param_sets, templates, outdir)


def main():
    parser = argparse.ArgumentParser(description="Sample and instantiate templates for parameter grid generation.")
    parser.add_argument("parameters", help="Parameter box (json/txt), newscan_dir (existing params), or scan_dir (for tune mode)")
    parser.add_argument("template", help="Template file")
    parser.add_argument("npoints", nargs="?", type=int, help="Number of points to sample (only for random/uniform mode with json/txt parameters)")
    parser.add_argument("--mode", choices=["random", "uniform", "tune", "minmax"], default="random", help="Sampling mode (default: random)")
    parser.add_argument("-s", "--seed", type=int, help="Random seed (for random mode)")
    parser.add_argument("-d", "--default", help="Defaults.json file (for tune mode)")
    parser.add_argument("-o", "--outdir", default="newscan", help="Output directory name (default: newscan)")
    parser.add_argument("--tune_tag", help="Prefix for tune directories (default: tune_)")
    parser.add_argument("--table", action="store_true", help="Create a lookup table (params.dat) with folder indices and parameter values (only for random/uniform modes)")
    args = parser.parse_args()

    if not os.path.exists(args.parameters):
        print(f"Error: Parameters input '{args.parameters}' not found")
        sys.exit(1)

    if not os.path.exists(args.template):
        print(f"Error: Template file '{args.template}' not found")
        sys.exit(1)

    if args.mode in ["random", "uniform", "minmax"]:
        if os.path.exists(args.outdir):
            print(f"Error: Output directory '{args.outdir}' already exists. Please remove it or choose a different name.")
            sys.exit(1)

    print(f"Running in {args.mode} mode")

    is_json_txt = args.parameters.endswith(('.json', '.txt'))
    is_directory = os.path.isdir(args.parameters)
    
    if args.mode == "tune":
        is_scan_dir = is_directory
        is_newscan_dir = False
    else:
        is_scan_dir = False
        is_newscan_dir = is_directory

    if args.mode == "tune":
        if not is_scan_dir:
            print("Error: tune mode requires a scan directory with tune_* subdirectories")
            sys.exit(1)
        if args.npoints is not None:
            print("Warning: npoints argument ignored in tune mode (will process all tune directories)")
        if args.seed is not None:
            print("Warning: --seed argument ignored in tune mode")
        if args.table:
            print("Warning: --table argument ignored in tune mode")

    elif args.mode == "minmax":
        if not is_json_txt:
            print("Error: minmax mode requires a parameter file (json/txt)")
            sys.exit(1)
        if not args.default:
            print("Error: minmax mode requires --default defaults.json file")
            sys.exit(1)
        if args.npoints is not None:
            print("Warning: npoints argument ignored in minmax mode (automatically set to 2 * number of parameters)")
        if args.seed is not None:
            print("Warning: --seed argument ignored in minmax mode")
        if args.tune_tag is not None:
            print("Warning: --tune_tag argument ignored in minmax mode")
        if args.table:
            print("Warning: --table argument ignored in minmax mode (automatically creates lookup table)")
    
    elif args.mode in ["random", "uniform"]:
        if is_newscan_dir:
            if args.npoints is not None:
                print("Warning: npoints argument ignored when using newscan directory (will use all available parameters)")
            if args.seed is not None:
                print("Warning: --seed argument ignored when using newscan directory")
        elif is_json_txt:
            if args.npoints is None:
                print("Error: npoints required for random/uniform mode with parameter file")
                sys.exit(1)
            if args.npoints <= 0:
                print("Error: Number of points must be positive")
                sys.exit(1)
            if args.mode == "uniform" and args.seed is not None:
                print("Warning: --seed argument ignored in uniform mode with parameter file")
        else:
            print("Error: random/uniform mode requires either a parameter file (json/txt) or newscan directory")
            sys.exit(1)
        if args.default is not None:
            print("Warning: --default argument ignored in random/uniform mode")
        if args.tune_tag is not None:
            print("Warning: --tune_tag argument ignored in random/uniform mode")

    if args.mode == "tune":
        print(f"Loading parameters from: {args.parameters}...")
        if args.tune_tag is None:
            args.tune_tag = "tune_"
        tune_mode(args.parameters, args.template, args.tune_tag, args.default, args.outdir)
        return

    if args.mode == "minmax":
        infofile_path = os.path.join(args.outdir, "params.dat")
        print("Sampling new parameters...")
        minmax_mode(args.parameters, args.default, args.template, args.outdir, infofile_path=infofile_path)
        return

    if is_newscan_dir:
        print(f"Loading parameters from: {args.parameters}...")
        param_list = load_params_from_folders(args.parameters, npoints=args.npoints)
    else:
        print("Sampling new parameters...")
        if args.seed is not None and args.mode == "random":
            np.random.seed(args.seed)
            print(f"Using random seed: {args.seed}")
        
        if args.mode == "uniform":
            param_list = sample_uniform(args.parameters, args.npoints)
        else:
            param_list = sample_random(args.parameters, args.npoints)

    try:
        with open(args.template, "r") as f:
            template_content = f.read()
    except IOError as e:
        print(f"Error: Cannot read template file '{args.template}': {e}")
        sys.exit(1)

    template_name = os.path.basename(args.template)
    templates = {template_name: template_content}
    write_params(param_list, templates, args.outdir)
    
    if args.table:
        write_lookup_table(param_list, args.outdir)


if __name__ == "__main__":
    main()