import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import json
import os
import re
import shutil
import datetime
import itertools
from pathlib import Path

mpl.use('Agg')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['TeX Gyre Pagella']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'TeX Gyre Pagella'
mpl.rcParams['mathtext.bf'] = 'TeX Gyre Pagella:bold'
mpl.rcParams['mathtext.it'] = 'TeX Gyre Pagella:italic'
mpl.rcParams['mathtext.default'] = 'it'


def to_float_or_nan(value):
    """Convert value to float, returning NaN for unsupported/missing values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return np.nan
    return out if np.isfinite(out) else np.nan


def read_output_command(outdir):
    """Read previous creation command from outdir/index.html if available."""
    index_path = Path(outdir) / "index.html"
    if not index_path.exists() or not index_path.is_file():
        return None
    try:
        text = index_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    match = re.search(r"Created with command:\s*<pre>(.*?)</pre>", text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group(1).strip() or None


def read_chi2_json(path):
    """Read chi2 data from JSON file and directly build plotting structure."""

    with open(path, "r") as f:
        data = json.load(f)
    
    if data.get("format") != "CHI2JSON":
        raise ValueError(f"Invalid JSON format. Expected 'CHI2JSON', got '{data.get('format', 'missing')}'.")
    
    source_command = data.get("command")
    summaries  = list(data.get("summaries", []))
    labels     = []
    series_ids = []
    data_dict  = {}

    for obs in data.get("observables", []):
        source    = obs.get("source", "unknown")
        label     = obs.get("label", "unknown")
        series_id = (label, source)
        obs_name  = obs.get("observable", "unknown")
        reduced_chi2 = obs.get("reduced_chi2")
        if reduced_chi2 is None:
            chi2 = to_float_or_nan(obs.get("chi2", np.nan))
            ndf  = to_float_or_nan(obs.get("ndf", np.nan))
            reduced_chi2 = chi2 / ndf if ndf > 0 and np.isfinite(chi2) else np.nan
        else:
            reduced_chi2 = to_float_or_nan(reduced_chi2)

        parts = obs_name.split("/")
        if len(parts) < 3:
            continue
        analysis_name = parts[1]
        obs_id        = parts[2]

        if label not in labels:
            labels.append(label)
        if series_id not in series_ids:
            series_ids.append(series_id)
        data_dict.setdefault(analysis_name, {}).setdefault(series_id, []).append((obs_id, reduced_chi2))

    for analysis_name, series_map in data_dict.items():
        for series_id, obs_tuples in series_map.items():
            obs_ids = [o[0] for o in obs_tuples]
            short_ids = []
            for obs_id in obs_ids:
                if '-' in obs_id:
                    short_id = obs_id.split('-')[0]
                else:
                    short_id = obs_id
                short_ids.append(short_id)
            if len(set(short_ids)) == len(obs_ids):
                series_map[series_id] = [(short_id, chi2_value) for (_, chi2_value), short_id in zip(obs_tuples, short_ids)]
    return summaries, data_dict, labels, series_ids, source_command


def merge_chi2_json_files(json_paths):
    """Read and merge multiple CHI2JSON files."""

    merged_data_dict  = {}
    source_commands   = []
    merged_summaries  = []
    merged_labels     = []
    merged_series_ids = []

    for path in json_paths:
        summaries, data_dict, labels, series_ids, source_command = read_chi2_json(path)
        merged_summaries.extend(summaries)
        merged_labels.extend([label for label in labels if label not in merged_labels])
        merged_series_ids.extend([sid for sid in series_ids if sid not in merged_series_ids])

        if source_command:
            source_commands.append(f"{path}: {source_command}")
        else:
            source_commands.append(f"{path}: (no command in JSON)")

        for analysis_name, series_map in data_dict.items():
            tgt_series_map = merged_data_dict.setdefault(analysis_name, {})
            for series_id, obs_tuples in series_map.items():
                tgt_series_map.setdefault(series_id, []).extend(obs_tuples)
    return merged_summaries, merged_data_dict, merged_labels, merged_series_ids, source_commands


def build_series_labels(series_ids):
    """Build display labels and source notes for duplicate label names."""

    grouped = {}
    for sid in series_ids:
        label, _ = sid
        grouped.setdefault(label, []).append(sid)

    series_labels = {}
    duplicate_series_notes = []
    for label, group in grouped.items():
        if len(group) == 1:
            series_labels[group[0]] = label
            continue
        for idx, sid in enumerate(group, start=1):
            label_display = f"{label} ({idx})"
            series_labels[sid] = label_display
            duplicate_series_notes.append((label_display, sid[1]))
    return series_labels, duplicate_series_notes


def filter_labels(labels, series_ids, requested_labels=None):
    """Filter labels select matching internal series IDs."""

    if requested_labels:
        missing = [x for x in requested_labels if x not in labels]
        if missing:
            print(f"Warning: requested labels not found in chi2.json and ignored: {missing}.")
        labels = [x for x in requested_labels if x in labels]
    selected_series_ids = [sid for sid in series_ids if sid[0] in labels]
    return labels, selected_series_ids


def extract_numeric_sort_key(obs_id):
    """Extract numeric values from observable ID for sorting."""

    nums = re.findall(r'\d+', obs_id)
    if nums:
        return tuple(int(n) for n in nums)
    return (float('inf'),)


def plot_chi2_per_analysis(data_dict, series_ids, series_labels, default_label=None,
                           log_scale=False, output_dir='chi2-plots'):
    """Create one plot per analysis showing chi2/ndf for each observable."""

    os.makedirs(output_dir)
    chunk_size = 100
    
    for analysis_name in sorted(data_dict.keys()):
        analysis_data = data_dict[analysis_name]
        if not analysis_data:
            continue

        plot_series = [sid for sid in series_ids if sid in analysis_data]
        if not plot_series:
            continue

        series_to_bin_data = {}
        all_bin_ids = set()
        for sid in plot_series:
            obs_list = analysis_data[sid]
            obs_list_sorted = sorted(obs_list, key=lambda x: extract_numeric_sort_key(x[0]))
            bin_data = dict(obs_list_sorted)
            series_to_bin_data[sid] = bin_data
            all_bin_ids.update(bin_data.keys())
        default_series_id = None
        default_bin_data  = None
        if default_label:
            default_candidates = [sid for sid in plot_series if sid[0] == default_label]
            if len(default_candidates) == 1:
                default_series_id = default_candidates[0]
                default_bin_data  = series_to_bin_data.get(default_series_id)
            elif len(default_candidates) > 1:
                print(f"Warning: analysis '{analysis_name}' has multiple series with default label '{default_label}'. "
                      f"Skipping ratio subplot for this analysis.")

        bin_ids = sorted(all_bin_ids, key=extract_numeric_sort_key)
        n_bins  = len(bin_ids)
        if n_bins == 0:
            continue

        n_chunks = (n_bins + chunk_size - 1) // chunk_size

        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        colors = ['#3366FF', '#EE3311', '#33BB33', '#FF9933', 
                  '#9933FF', '#FF33CC', '#00CCCC', '#FFCC00']
        default_color = '#898C89'

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end   = min((chunk_idx + 1) * chunk_size, n_bins)
            chunk_bin_ids  = bin_ids[start:end]
            bin_ids_length = max((len(str(bid)) for bid in chunk_bin_ids), default=1)

            has_ratio_plot = default_bin_data is not None and len(plot_series) > 1
            n_legend_items = len(plot_series)
            ncols_legend   = min(3, n_legend_items)
            nrows_legend   = (n_legend_items + ncols_legend - 1) // ncols_legend
            
            title_height   = 0.6
            legend_height  = 0.4 * nrows_legend
            bottom_height  = max(0.8, 0.15 * bin_ids_length)
            axes_height    = 4.0 + (1.2 if has_ratio_plot else 0.0)
            total_height   = (title_height + legend_height) + axes_height + bottom_height

            top_frac       = (title_height + legend_height) / total_height + 0.02
            bottom_frac    = bottom_height / total_height

            if has_ratio_plot:
                fig, (ax, ax_ratio) = plt.subplots(2, 1, sharex=True, figsize=(16, total_height),
                    gridspec_kw={'height_ratios': [4.0, 1.2], 'hspace': 0.0})
                plt.subplots_adjust(left=0.07, right=0.98, bottom=bottom_frac, top=1-top_frac, hspace=0.0)
            else:
                fig, ax = plt.subplots(figsize=(16, total_height))
                plt.subplots_adjust(left=0.07, right=0.98, bottom=bottom_frac, top=1-top_frac)

            title_pos = (total_height - title_height / 2.0 + 0.1) / total_height
            title = f'{analysis_name}' if n_chunks == 1 else f'{analysis_name} ({chunk_idx + 1}/{n_chunks})'
            fig.suptitle(title, fontsize=24, y=title_pos, fontweight='bold')
            ax.set_ylabel(r'$\chi^2 / \mathrm{ndf}$', fontsize=20)
            ax.set_yscale('log' if log_scale else 'linear')

            plot_index = 0
            for sid in plot_series:
                bin_data = series_to_bin_data[sid]
                chi2_values = [to_float_or_nan(bin_data.get(bid, np.nan)) for bid in chunk_bin_ids]

                valid_indices = [j for j, val in enumerate(chi2_values) if np.isfinite(val)]
                if not valid_indices:
                    continue
                valid_x    = np.array(valid_indices)
                valid_chi2 = [chi2_values[j] for j in valid_indices]

                is_default = bool(default_series_id and sid == default_series_id)
                label      = series_labels.get(sid, sid[0])
                if is_default:
                    color      = default_color
                    marker     = 'h'
                    markersize = 5
                    linewidth  = 1.5
                    linealpha  = 0.4
                    if 'default' not in label.lower():
                        label  = f'{label} (default)'
                else:
                    color      = colors[plot_index % len(colors)]
                    marker     = markers[plot_index % len(markers)]
                    markersize = 6
                    linewidth  = 2.0
                    linealpha  = 0.4
                    plot_index += 1
                ax.plot(valid_x, valid_chi2, marker=marker, linewidth=0.0,
                        label=label, color=color, markersize=markersize, alpha=1.0)
                ax.plot(valid_x, valid_chi2, linestyle='-', marker=None,
                        color=color, linewidth=linewidth, alpha=linealpha)

                if has_ratio_plot and not is_default:
                    default_values = [to_float_or_nan(default_bin_data.get(bid, np.nan)) for bid in chunk_bin_ids]
                    ratio_values = []
                    for num, den in zip(chi2_values, default_values):
                        if np.isfinite(num) and np.isfinite(den) and den > 0:
                            ratio_values.append(num / den)
                        else:
                            ratio_values.append(np.nan)

                    valid_ratio_indices = [j for j, val in enumerate(ratio_values) if np.isfinite(val)]
                    if valid_ratio_indices:
                        ratio_x = np.array(valid_ratio_indices)
                        ratio_y = [ratio_values[j] for j in valid_ratio_indices]
                        ax_ratio.plot(ratio_x, ratio_y, marker=marker, linewidth=0.0,
                                      color=color, markersize=5, alpha=1.0)
                        ax_ratio.plot(ratio_x, ratio_y, linestyle='-', marker=None,
                                      color=color, linewidth=1.5, alpha=0.5)

            ax.minorticks_on()
            x_positions = np.arange(len(chunk_bin_ids))
            x_axis = ax_ratio if has_ratio_plot else ax
            if len(chunk_bin_ids) <= 40:
                x_axis.set_xticks(x_positions)
                x_axis.set_xticklabels(chunk_bin_ids, rotation=90, fontsize=20)
                x_axis.xaxis.set_minor_locator(plt.NullLocator())
            else:
                step = max(1, len(chunk_bin_ids) // 20)
                xticks = x_positions[::step]
                xticklabels = [chunk_bin_ids[i] for i in range(0, len(chunk_bin_ids), step)]
                x_axis.set_xticks(xticks)
                x_axis.set_xticklabels(xticklabels, rotation=90, fontsize=20)
                x_axis.set_xticks(x_positions, minor=True)

            if log_scale:
                ax.set_ylim(min(ax.get_ylim()[0], 1.0), max(ax.get_ylim()[1], 1.0))
                ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
                ax.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10.0))
                ax.yaxis.set_minor_formatter(mticker.NullFormatter())

            if has_ratio_plot:
                ax.tick_params(axis='x', labelbottom=False)
                ax_ratio.axhline(1.0, color=default_color, linestyle='--', alpha=0.4)
                ax_ratio.set_ylabel(f'data/default', fontsize=20)
                ax_ratio.set_ylim(0.0, 2.0)
                ax_ratio.set_yticks([0.0, 1.0])
                ax_ratio.tick_params(axis='both', which='major', labelsize=20, direction='in',
                                     length=5, top=True, right=True)
                ax_ratio.tick_params(axis='both', which='minor', direction='in',
                                     length=3, top=True, right=True)
                fig.align_ylabels([ax, ax_ratio])

            ax.legend(loc='lower left', fontsize=20, frameon=False, ncols=ncols_legend,
                      bbox_to_anchor=(0, 1.01), borderaxespad=0)
            ax.tick_params(axis='both', which='major', labelsize=20, direction='in',
                           length=5, top=True, right=True)
            ax.tick_params(axis='both', which='minor', direction='in',
                           length=3, top=True, right=True)
            ax.tick_params(axis='y', which='minor', direction='in',
                           length=3, top=True, right=True)

            safe_name = analysis_name.replace('/', '_').replace(' ', '_')
            if n_chunks == 1:
                output_file_pdf = os.path.join(output_dir, f'{safe_name}_chi2_plot.pdf')
                output_file_png = os.path.join(output_dir, f'{safe_name}_chi2_plot.png')
            else:
                output_file_pdf = os.path.join(output_dir, f'{safe_name}_chi2_plot_{chunk_idx + 1}.pdf')
                output_file_png = os.path.join(output_dir, f'{safe_name}_chi2_plot_{chunk_idx + 1}.png')

            plt.savefig(output_file_pdf, format='pdf', dpi=600)
            plt.savefig(output_file_png, format='png', dpi=600)

            if n_chunks == 1: 
                print(f"Created plot for {analysis_name}: {output_file_pdf}/png")
            else: 
                print(f"Created plot for {analysis_name} ({chunk_idx + 1}/{n_chunks}): {output_file_pdf}/png")
            plt.close(fig)
    print()
    return


def create_index_html(command, summaries=None, data_dict=None, series_ids=None, series_labels=None,
                      duplicate_series_notes=None, default_label=None,
                      input_file=None, source_command=None, output_dir="chi2-plots"):
    """Create an index.html file in the output directory to access all plot files."""

    def output_value(value):
        if value is None:
            return "nan"
        try:
            f = float(value)
        except (TypeError, ValueError):
            return str(value)
        return f"{f:.2f}" if np.isfinite(f) else "nan"

    plot_files = []
    for file_name in sorted(os.listdir(output_dir)):
        if file_name.endswith((".pdf", ".png")):
            plot_files.append(file_name)

    exp_analysis_groups = {}
    for file_name in plot_files:
        if file_name.startswith("chi2_"):
            continue
        if "_chi2_plot" in file_name:
            group = file_name.split("_chi2_plot")[0]
        else:
            parts = file_name.split('_')
            if len(parts) >= 3:
                group = "_".join(parts[:3])
            elif len(parts) >= 2:
                group = "_".join(parts[:2])
            else:
                group = parts[0]
        if group not in exp_analysis_groups:
            exp_analysis_groups[group] = []
        exp_analysis_groups[group].append(file_name)

    now = datetime.datetime.now().strftime("%A, %d. %B %Y %H:%M")
    html = f"""<html>
    <head>
        <title>{output_dir}</title>
        <style>
        html {{ font-family: sans-serif; }}
        img {{ border: 0; }}
        a {{ text-decoration: none; font-weight: bold; }}
        table {{ border-collapse: collapse; margin-bottom: 2em; margin-left: 1.5em; }}
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
        <h2>{output_dir}</h2>"""

    if input_file:
        html += f"""
        <p>Input files: {input_file}"""
    if source_command:
        html += f"""
        <pre>{source_command}</pre></p>"""

    if summaries:
        html += f"""
        <h3>Summary</h3>
        <table>
            <tr><th>Source</th><th>Label</th><th>Global $\chi^2$</th><th>ndf</th><th>Reduced $\chi^2$</th><th>Averaged $\chi^2$ over all obs.</th></tr>"""
        for summary in summaries:
            html += f"""
            <tr>
                <td>{summary.get('source','')}</td>
                <td>{summary.get('label','')}</td>
                <td>{output_value(summary.get('global_chi2'))}</td>
                <td>{output_value(summary.get('ndf'))}</td>
                <td>{output_value(summary.get('reduced_chi2'))}</td>
                <td>{output_value(summary.get('average_chi2'))}</td>
            </tr>"""
        html += f"""
        </table>"""

    series_ids             = series_ids             or []
    series_labels          = series_labels          or {}
    duplicate_series_notes = duplicate_series_notes or []
    data_dict              = data_dict              or {}
    for group, files in sorted(exp_analysis_groups.items()):
        html += f"""
        <h3>{group}</h3>"""
        pngs = [f for f in files if f.endswith('.png')]
        avg_entries = []

        for sid in series_ids:
            bin_tuples = data_dict.get(group, {}).get(sid, [])
            chi2_vals = [v for _, v in bin_tuples if np.isfinite(v) and v != -1]
            avg = (sum(chi2_vals) / len(chi2_vals)) if chi2_vals else None
            avg_entries.append((sid, avg))

        if any(avg is not None for _, avg in avg_entries):
            html += f"""
        <div style='font-size:1.1em; margin-bottom:0.3em;'><b>Average chi2 per YODA:</b><ul style='margin:0.2em 0 0.5em 1em;'>"""
            for sid, avg in avg_entries:
                if avg is not None:
                    label = series_labels.get(sid, sid[0])
                    if default_label and sid[0] == default_label and 'default' not in label.lower():
                        label = f'{label} (default)'
                    html += f'<li>{label}: {avg:.2f}</li>'
            html += '</ul></div>'
        for png in sorted(pngs):
            pdf = png.replace('.png', '.pdf')
            html += f"""
        <div style='margin-bottom:1.5em;'><img src='{png}' alt='{png}' style='max-width:1000px;'><br>"""
            if pdf in files:
                html += f'<a href="{pdf}">Download PDF</a>'
            html += '</div>'

    if duplicate_series_notes:
        html += """
        <h3>Label clarification</h3>
        <ul>"""
        for label, source in duplicate_series_notes:
            html += f"<li>{label}: {source}</li>"
        html += '</ul>'

    html += f"""    
        <footer style="clear:both; margin-top:3em; padding-top:3em">
        <p>Generated at {now}</p>
        <p>Created with command: <pre>{command}</pre></p>
        </footer>
    </body>"""
    html += "\n</html>"

    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    print(f"Created {os.path.join(output_dir, 'index.html')}")
    return


def grid(args):
    """Plot chi2 values on a grid."""

    def read_grid_table(path):
        points = []
        header = None
        with open(path, "r") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                if line.startswith("#"):
                    maybe_header = line[1:].strip().split()
                    if header is None and maybe_header and maybe_header[0].lower() == "index":
                        header = maybe_header
                    continue
                cols = line.split()
                if header is None:
                    header = cols
                    continue
                if len(cols) != len(header):
                    raise ValueError(f"Malformed grid row in '{path}': {raw.rstrip()}")
                row = dict(zip(header, cols))
                point_index = ""
                values = {}
                for k, v in row.items():
                    lk = k.lower()
                    if lk == "index":
                        point_index = v
                        continue
                    if lk == "sector":
                        continue
                    values[k] = float(v)
                points.append({"index": point_index, "values": values})
        if not points:
            raise ValueError(f"No valid parameter rows found in '{path}'.")
        parameter_names = list(points[0]["values"].keys())
        if len(parameter_names) < 2:
            raise ValueError("Grid plot requires at least 2 parameters in the grid file.")
        return points, parameter_names
    
    def chi2_from_summaries(summaries):
            if not summaries:
                raise ValueError("No summaries found in chi2.json")
            chi2_by_label = {}
            for summary in summaries:
                label = str(summary.get("label", "")).strip()
                if not label:
                    raise ValueError("Found summary entry without label in chi2.json")
                if label in chi2_by_label:
                    raise ValueError(f"Duplicate summary label '{label}' found in chi2.json")
                reduced = summary.get("reduced_chi2", None)
                if reduced is None:
                    chi2 = to_float_or_nan(summary.get("global_chi2"))
                    ndf = to_float_or_nan(summary.get("ndf"))
                    reduced = chi2 / ndf if np.isfinite(chi2) and np.isfinite(ndf) and ndf > 0 else np.nan
                chi2_by_label[label] = to_float_or_nan(reduced)
            return chi2_by_label

    def plot_chi2_grid(grid_path, chi2_json_path, fmt="pdf", dpi=150):
        points, parameter_names = read_grid_table(grid_path)
        summaries, _, _, _, _   = read_chi2_json(chi2_json_path)
        chi2_by_label = chi2_from_summaries(summaries)

        grid_indices = [str(p["index"]) for p in points]
        grid_index_set = set(grid_indices)
        chi2_label_set = set(chi2_by_label.keys())

        if len(grid_indices) != len(chi2_by_label):
            raise ValueError(f"Grid/chi2 size mismatch: grid has {len(grid_indices)} points, "
                            f"chi2.json has {len(chi2_by_label)} summaries")

        missing_in_chi2 = sorted(grid_index_set - chi2_label_set)
        extra_in_chi2   = sorted(chi2_label_set - grid_index_set)
        if missing_in_chi2 or extra_in_chi2:
            msg = []
            if missing_in_chi2:
                msg.append(f"missing in chi2.json: {missing_in_chi2}")
            if extra_in_chi2:
                msg.append(f"missing in grid file: {extra_in_chi2}")
            raise ValueError("Label mismatch between grid file and chi2.json (" + "; ".join(msg) + ").")

        x_lookup = {}
        y_lookup = {}
        for point in points:
            for name, value in point["values"].items():
                x_lookup.setdefault(name, set()).add(float(value))
                y_lookup.setdefault(name, set()).add(float(value))

        grid_name = Path(grid_path).name
        if grid_name.endswith(".grid.dat"):
            stem = grid_name[:-len(".grid.dat")]
        elif grid_name.endswith(".dat"):
            stem = grid_name[:-len(".dat")]
        else:
            stem = Path(grid_path).stem
        outdir  = str(Path(grid_path).with_name(f"{stem}.chi2.plots"))
        outpath = Path(outdir)
        if outpath.exists():
            raise ValueError(f"Output path already exists: {outpath}.")
        os.makedirs(outdir)

        pairs = list(itertools.combinations(parameter_names, 2))
        for xname, yname in pairs:
            xv = [float(p["values"][xname]) for p in points]
            yv = [float(p["values"][yname]) for p in points]
            cv = [to_float_or_nan(chi2_by_label[str(p["index"])]) for p in points]
            finite_mask = np.isfinite(cv)
            xv = [x for x, keep in zip(xv, finite_mask) if keep]
            yv = [y for y, keep in zip(yv, finite_mask) if keep]
            cv = [c for c, keep in zip(cv, finite_mask) if keep]
            if not cv:
                raise ValueError(f"Grid for pair ({xname}, {yname}) contains no finite chi2/ndf values.")
            if np.any(np.asarray(cv, dtype=float) <= 0):
                raise ValueError(f"Grid for pair ({xname}, {yname}) contains non-positive chi2/ndf values. "
                                 f"Log10 color scale requires strictly positive values.")

            z_pos = np.asarray([val for val in cv if np.isfinite(val) and val > 0], dtype=float)
            vmin  = float(np.min(z_pos))
            vmax  = float(np.max(z_pos))
            if np.isclose(vmin, vmax):
                vmin = max(vmin * 0.9, 1e-12)
                vmax = vmax * 1.1
            color_norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

            try:
                import matplotlib.tri as mtri
            except ImportError as e:
                raise ValueError(f"matplotlib.tri is required for grid interpolation: {e}.")

            fig, ax = plt.subplots(figsize=(6.0, 6.0))
            triangulation = mtri.Triangulation(xv, yv)
            mesh = ax.tripcolor(triangulation, cv, cmap="viridis_r", norm=color_norm, shading="gouraud")
            ax.scatter(xv, yv, s=16, c="white", edgecolors="black", linewidths=0.4, zorder=3)

            ax.set_box_aspect(1)
            ax.set_xlabel(xname, fontsize=14)
            ax.set_ylabel(yname, fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12, length=5)
            cbar = fig.colorbar(mesh, ax=ax)
            cbar.set_label(r"$\chi^2 / \mathrm{ndf}$", fontsize=14)
            cbar.formatter = mticker.FuncFormatter(lambda val, _: f"{val:g}")
            cbar.update_ticks()
            x_values = sorted(x_lookup.get(xname, []))
            y_values = sorted(y_lookup.get(yname, []))
            if x_values:
                x_margin = 0.05 * (max(x_values) - min(x_values) if len(x_values) > 1 else 1.0)
                ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
            if y_values:
                y_margin = 0.05 * (max(y_values) - min(y_values) if len(y_values) > 1 else 1.0)
                ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
            fig.tight_layout()

            fname = f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(xname))}__{re.sub(r'[^A-Za-z0-9_.-]+', '_', str(yname))}.{fmt}"
            fpath = os.path.join(outdir, fname)
            fig.savefig(fpath, dpi=dpi)
            plt.close(fig)
            print(f"Created plot: {fpath}")
        print()
        return outdir

    forbidden_options = []
    for option in os.sys.argv[1:]:
        if option == "--grid" or option.startswith("--grid="):
            continue
        if option.startswith("-"):
            forbidden_options.append(option)
    if forbidden_options:
        print("Error: --grid mode does not allow any additional options.")
        print(f"       Forbidden option(s): {' '.join(forbidden_options)}")
        return 1
    if len(args.chi2_json) != 1:
        print("Error: --grid mode requires exactly one chi2.json file.")
        return 1
    grid_path = Path(args.grid)
    json_path = Path(args.chi2_json[0])
    if not grid_path.exists():
        print(f"Error: grid file does not exist: {grid_path}.")
        return 1
    if not json_path.exists():
        print(f"Error: input file does not exist: {json_path}.")
        return 1
    try:
        plot_chi2_grid(str(grid_path), str(json_path), fmt="pdf", dpi=150)
    except ValueError as e:
        print(f"Error: {e}.")
        return 1
    return 0

def main():
    print("Starting chi2 plotting...\n")

    parser = argparse.ArgumentParser(
        description="Plot chi2 data from a JSON file and generate an HTML report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  plot_chi2.py chi2.json --outdir output_directory
  plot_chi2.py chi2.json --default-label default_label --log
  plot_chi2.py chi2_file1.json chi2_file2.json --labels label1 label2

Input handling:
  - Reads results from compute_chi2.py in JSON format (CHI2JSON)
  - Labels (-l) can be used to filter and reorder the data to plot
  - Specifying a default label (-d) adds ratio subplots comparing other labels to the default

Output:
  - Creates one plot per analysis (PDF + PNG) and an index.html summary
  - Output directory is overwritten if it already exists, can be specified with -o/--outdir
        """
    )
    parser.add_argument("chi2_json", nargs="+", help="Input chi2.json file(s) (from compute_chi2.py)")
    parser.add_argument("-o", "--outdir", default="chi2-plots", help="Output directory for plots")
    parser.add_argument("-l", "--labels", nargs="+", default=None, help="Subset/order of labels to plot")
    parser.add_argument("-d", "--default-label", default=None, help="Label to use as default/reference in ratio plots")
    parser.add_argument("--log", action="store_true", help="Use logarithmic scale for y-axis")
    parser.add_argument("--grid", default=None, help="Grid table (.dat, e.g. newscan.grid.dat). Plot chi2 grid.")
    # parser.add_argument("-v", "--debug", action="store_true", default=False, help="Enable debug output")
    args = parser.parse_args()
    command = " ".join(os.sys.argv)

    if args.grid:
        return grid(args)

    outpath = Path(args.outdir)
    if outpath.exists():
        if outpath.is_dir():
            print(f"Output directory already exists.")
            previous_cmd = read_output_command(outpath)
            if previous_cmd:
                print(f"  Previous output directory command (from index.html):")
                print(f"    {previous_cmd}")
            shutil.rmtree(outpath)
            print(f"  Removed existing output directory: {outpath}.\n")
        else:
            print(f"Output path exists and is not a directory: {outpath}. "
                  f"Please remove or choose a different output directory.")
            return 1

    json_paths = [Path(p) for p in args.chi2_json]
    missing = [str(p) for p in json_paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"Error: input file does not exist: {p}.")
        return 1

    summaries, data_dict, available_labels, available_series_ids, source_commands = merge_chi2_json_files([str(p) for p in json_paths])
    if not data_dict:
        print("Error: no observable chi2 data found in input file(s).")
        return 1

    labels, series_ids = filter_labels(available_labels, available_series_ids, args.labels)
    if not labels:
        print("Error: no labels selected for plotting.")
        return 1
    if not series_ids:
        print("Error: selected labels do not map to any plot series.")
        return 1
    series_labels, duplicate_series_notes = build_series_labels(series_ids)

    default_label = None
    if args.default_label:
        if args.default_label not in available_labels:
            print(f"Warning: default label '{args.default_label}' not found in JSON, available labels: {available_labels}. "
                  f"Continuing without default reference.")
        else:
            default_label = args.default_label
            if default_label not in labels:
                labels.insert(0, default_label)

    plot_chi2_per_analysis(data_dict, series_ids, series_labels,
                           default_label=default_label,
                           log_scale=args.log,
                           output_dir=args.outdir)
    create_index_html(command, 
                      summaries=summaries, 
                      data_dict=data_dict, 
                      series_ids=series_ids,
                      series_labels=series_labels,
                      duplicate_series_notes=duplicate_series_notes,
                      default_label=default_label,
                      input_file=", ".join(str(p) for p in json_paths),
                      source_command="\n".join(source_commands),
                      output_dir=args.outdir)
    return 0


if __name__ == "__main__":
    main()
