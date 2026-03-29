import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import json
import shutil
from pathlib import Path
import numpy as np
import argparse
import datetime

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


def read_chi2_json(path):
    """Read chi2 data from JSON file and directly build plotting structure."""

    with open(path, "r") as f:
        data = json.load(f)
    
    if data.get("format") != "CHI2JSON":
        raise ValueError(f"Invalid JSON format. Expected 'CHI2JSON', got '{data.get('format', 'missing')}'.")
    
    source_command = data.get("command")
    summaries = list(data.get("summaries", []))
    labels    = set()
    data_dict = {}
    for obs in data.get("observables", []):
        label        = obs.get("label", "unknown")
        obs_name     = obs.get("observable", "unknown")
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
        labels.add(label)
        data_dict.setdefault(analysis_name, {}).setdefault(label, []).append((obs_id, reduced_chi2))

    for analysis_name, label_map in data_dict.items():
        for label, obs_tuples in label_map.items():
            counts = {}
            for obs_id, _ in obs_tuples:
                counts[obs_id] = counts.get(obs_id, 0) + 1
            duplicate_obs = sorted([o for o, c in counts.items() if c > 1])
            if duplicate_obs:
                print(f"Warning: found duplicate observable IDs for analysis '{analysis_name}', label '{label}'. "
                      f"Later entries overwrite earlier ones in plotting. This can happen if the same label "
                      f"is used for multiple YODA files with overlapping analyses and observables.")
    for analysis_name, label_map in data_dict.items():
        for label, obs_tuples in label_map.items():
            obs_ids = [o[0] for o in obs_tuples]

            short_ids = []
            for obs_id in obs_ids:
                if '-' in obs_id:
                    short_id = obs_id.split('-')[0]
                else:
                    short_id = obs_id
                short_ids.append(short_id)

            if len(set(short_ids)) == len(obs_ids):
                label_map[label] = [(short_id, chi2_value) for (_, chi2_value), short_id in zip(obs_tuples, short_ids)]
    return summaries, data_dict, sorted(labels), source_command


def filter_labels(labels, requested_labels=None):
    """Filter available labels from JSON file based on user requested labels."""

    if requested_labels:
        missing = [x for x in requested_labels if x not in labels]
        if missing:
            print(f"Warning: requested labels not found in chi2.json and ignored: {missing}.")
        labels = [x for x in requested_labels if x in labels]
    return labels


def extract_numeric_sort_key(obs_id):
    """Extract numeric values from observable ID for sorting."""

    import re
    nums = re.findall(r'\d+', obs_id)
    if nums:
        return tuple(int(n) for n in nums)
    return (float('inf'),)


def plot_chi2_per_analysis(data_dict, labels, default_label=None, log_scale=False, output_dir='chi2-plots'):
    """Create one plot per analysis showing chi2/ndf for each observable."""

    os.makedirs(output_dir, exist_ok=True)
    chunk_size = 100
    
    for analysis_name in sorted(data_dict.keys()):
        analysis_data = data_dict[analysis_name]
        if not analysis_data:
            continue

        plot_labels = [l for l in labels if l in analysis_data]
        if not plot_labels:
            continue

        label_to_bin_data = {}
        all_bin_ids = set()
        for label in plot_labels:
            obs_list = analysis_data[label]
            obs_list_sorted = sorted(obs_list, key=lambda x: extract_numeric_sort_key(x[0]))
            bin_data = dict(obs_list_sorted)
            label_to_bin_data[label] = bin_data
            all_bin_ids.update(bin_data.keys())
        default_bin_data = label_to_bin_data.get(default_label) if default_label else None

        bin_ids = sorted(all_bin_ids, key=extract_numeric_sort_key)
        n_bins  = len(bin_ids)
        if n_bins == 0:
            continue

        n_chunks = (n_bins + chunk_size - 1) // chunk_size

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p']
        colors = ['#3366FF', '#EE3311', '#33BB33', '#FF9933', 
                  '#9933FF', '#FF33CC', '#00CCCC', '#FFCC00']
        default_color = '#898C89'

        for chunk_idx in range(n_chunks):
            start = chunk_idx * chunk_size
            end   = min((chunk_idx + 1) * chunk_size, n_bins)
            chunk_bin_ids  = bin_ids[start:end]
            bin_ids_length = max((len(str(bid)) for bid in chunk_bin_ids), default=1)

            has_ratio_plot = default_bin_data is not None and len(plot_labels) > 1
            n_legend_items = len(plot_labels)
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
            for label in plot_labels:
                bin_data = label_to_bin_data[label]
                chi2_values = [to_float_or_nan(bin_data.get(bid, np.nan)) for bid in chunk_bin_ids]

                valid_indices = [j for j, val in enumerate(chi2_values) if np.isfinite(val)]
                if not valid_indices:
                    continue
                valid_x    = np.array(valid_indices)
                valid_chi2 = [chi2_values[j] for j in valid_indices]

                is_default = bool(default_label and label == default_label)
                if is_default:
                    color = default_color
                    marker = 'h'
                    markersize = 5
                    linewidth = 1.5
                    line_alpha = 0.4
                    label_display = label if 'default' in label.lower() else f'{label} (default)'
                else:
                    color = colors[plot_index % len(colors)]
                    marker = markers[plot_index % len(markers)]
                    markersize = 6
                    linewidth = 2.0
                    line_alpha = 0.4
                    label_display = label
                    plot_index += 1
                ax.plot(valid_x, valid_chi2, marker=marker, linewidth=0.0,
                        label=label_display, color=color, markersize=markersize, alpha=1.0)
                ax.plot(valid_x, valid_chi2, linestyle='-', marker=None,
                        color=color, linewidth=linewidth, alpha=line_alpha)

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
                print(f"Created plot for {analysis_name}: {output_file_pdf}")
            else: 
                print(f"Created plot for {analysis_name} ({chunk_idx + 1}/{n_chunks}): {output_file_pdf}")
            plt.close(fig)
    return


def create_index_html(command, summaries=None, data_dict=None, labels=None, default_label=None,
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

    labels    = labels or []
    data_dict = data_dict or {}
    for group, files in sorted(exp_analysis_groups.items()):
        html += f"""
        <h3>{group}</h3>"""
        pngs = [f for f in files if f.endswith('.png')]
        avg_entries = []

        labels_for_avg = []
        default_in_group = default_label and (default_label in data_dict.get(group, {}))
        # if default_in_group:
        #     labels_for_avg.append(default_label)
        # labels_for_avg.extend([l for l in labels if l != default_label])
        labels_for_avg.extend([l for l in labels])

        for label in labels_for_avg:
            bin_tuples = data_dict.get(group, {}).get(label, [])
            chi2_vals = [v for _, v in bin_tuples if np.isfinite(v) and v != -1]
            avg = (sum(chi2_vals) / len(chi2_vals)) if chi2_vals else None
            avg_entries.append((label, avg))

        if any(avg is not None for _, avg in avg_entries):
            html += f"""
        <div style='font-size:1.1em; margin-bottom:0.3em;'><b>Average chi2 per YODA:</b><ul style='margin:0.2em 0 0.5em 1em;'>"""
            for label_name, avg in avg_entries:
                if avg is not None:
                    if default_label and label_name == default_label:
                        label_display = label_name if 'default' in label_name.lower() else f'{label_name} (default)'
                    else:
                        label_display = label_name
                    html += f'<li>{label_display}: {avg:.2f}</li>'
            html += '</ul></div>'
        for png in sorted(pngs):
            pdf = png.replace('.png', '.pdf')
            html += f"""
        <div style='margin-bottom:1.5em;'><img src='{png}' alt='{png}' style='max-width:1000px;'><br>"""
            if pdf in files:
                html += f'<a href="{pdf}">Download PDF</a>'
            html += '</div>'

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


def main():
    parser = argparse.ArgumentParser(
        description="Plot chi2 data from a JSON file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  plot_chi2.py chi2.json ---outdir output_directory
  plot_chi2.py chi2.json --labels label1 label2
  plot_chi2.py chi2.json --default-label default_label --log

Input handling:
  - Reads results from compute_chi2.py in JSON format (CHI2JSON)
  - Labels (-l) can be used to filter and reorder the data to plot
  - Specifying a default label (-d) adds ratio subplots comparing other labels to the default

Output:
  - Creates one plot per analysis (PDF + PNG) and an index.html summary
  - Output directory is overwritten if it already exists, can be specified with -o/--outdir
        """
    )
    parser.add_argument("chi2_json", help="Input chi2.json file (from compute_chi2.py)")
    parser.add_argument("-o", "--outdir", default="chi2-plots", help="Output directory for plots")
    parser.add_argument("-l", "--labels", nargs="+", default=None, help="Subset/order of labels to plot")
    parser.add_argument("-d", "--default-label", default=None, help="Label to use as default/reference in ratio plots")
    parser.add_argument("--log", action="store_true", help="Use logarithmic scale for y-axis")
    # parser.add_argument("-v", "--debug", action="store_true", default=False, help="Enable debug output")
    args = parser.parse_args()
    command = " ".join(os.sys.argv)

    outpath = Path(args.outdir)
    if outpath.exists():
        if outpath.is_dir():
            shutil.rmtree(outpath)
            print(f"Removed existing output directory: {outpath}")
        else:
            outpath.unlink()
            print(f"Removed existing output path: {outpath}")

    json_path = Path(args.chi2_json)
    if not json_path.exists():
        print(f"Error: input file does not exist: {json_path}.")
        return 1

    summaries, data_dict, available_labels, source_command = read_chi2_json(json_path)
    if not data_dict:
        print("Error: no observable chi2 data found in input file.")
        return 1

    labels = filter_labels(available_labels, args.labels)
    if not labels:
        print("Error: no labels selected for plotting.")
        return 1

    default_label = None
    if args.default_label:
        if args.default_label not in available_labels:
            print(f"Warning: default label '{args.default_label}' not found in JSON, available labels: {available_labels}. "
                  f"Continuing without default reference.")
        else:
            default_label = args.default_label
            if default_label not in labels:
                labels.insert(0, default_label)

    plot_chi2_per_analysis(data_dict, 
                           labels,
                           default_label=default_label,
                           log_scale=args.log,
                           output_dir=args.outdir)
    create_index_html(command, 
                      summaries=summaries, 
                      data_dict=data_dict, 
                      labels=labels, 
                      default_label=default_label,
                      input_file=str(json_path),
                      source_command=source_command,
                      output_dir=args.outdir)
    print(f"Plots written to {args.outdir}.")
    return 0


if __name__ == "__main__":
    main()
