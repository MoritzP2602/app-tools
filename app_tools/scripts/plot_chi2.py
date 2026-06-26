import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import argparse
import json
import os
import re
import sys
import shutil
import datetime
from pathlib import Path
from matplotlib import font_manager as fm

mpl.use('Agg')


MARKERS = ['o', 's', '^', 'D', 'v', '<', '>']
COLORS = ['#EE3311', '#3366FF', '#33BB33', '#FF9933',
          '#9933FF', '#FF33CC', '#00CCCC', '#FFCC00']
DEFAULT_COLOR = '#898C89'
CHUNK_SIZE = 100


def configure_fonts():
    """Apply TeX Gyre Pagella with a DejaVu Serif fallback."""
    mpl.rcParams['font.family'] = 'serif'
    try:
        fm.findfont('TeX Gyre Pagella', fallback_to_default=False)
        mpl.rcParams['font.serif'] = ['TeX Gyre Pagella']
        mpl.rcParams['mathtext.fontset'] = 'custom'
        mpl.rcParams['mathtext.rm'] = 'TeX Gyre Pagella'
        mpl.rcParams['mathtext.bf'] = 'TeX Gyre Pagella:bold'
        mpl.rcParams['mathtext.it'] = 'TeX Gyre Pagella:italic'
        mpl.rcParams['mathtext.default'] = 'it'
    except Exception:
        mpl.rcParams['font.serif'] = ['DejaVu Serif']
        mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
        mpl.rcParams['mathtext.default'] = 'it'

configure_fonts()


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
        chi2 = to_float_or_nan(obs.get("chi2", np.nan))
        ndf  = to_float_or_nan(obs.get("ndf", np.nan))
        if reduced_chi2 is None:
            reduced_chi2 = chi2 / ndf if np.isfinite(chi2) and np.isfinite(ndf) and ndf > 0 else np.nan
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
        data_dict.setdefault(analysis_name, {}).setdefault(series_id, []).append((obs_id, chi2, ndf, reduced_chi2))

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
                series_map[series_id] = [(short_id, chi2, ndf, reduced_chi2) for (_, chi2, ndf, reduced_chi2), short_id in zip(obs_tuples, short_ids)]
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


def assign_superscripts_per_analysis(data_dict, series_ids):
    """Assign superscripts per analysis, not globally.
    
    For each analysis, if the same label appears more than once within that
    analysis, assign superscripts. Superscripts are independent per analysis.
    
    Returns: {analysis_name: {series_id: display_label_html}}
    """
    per_analysis_labels_html = {}
    per_analysis_labels_mpl = {}
    
    for analysis_name, series_map in data_dict.items():
        analysis_series_ids = [sid for sid in series_ids if sid in series_map]
        
        grouped = {}
        for sid in analysis_series_ids:
            label = sid[0]
            grouped.setdefault(label, []).append(sid)
        
        analysis_labels = {}
        analysis_labels_mpl = {}
        for label, group in grouped.items():
            if len(group) == 1:
                analysis_labels[group[0]] = label
                analysis_labels_mpl[group[0]] = label
            else:
                for idx, sid in enumerate(group, start=1):
                    analysis_labels[sid] = f"{label}<sup>{idx}</sup>"
                    analysis_labels_mpl[sid] = f"{label}$^{{{idx}}}$"
        
        per_analysis_labels_html[analysis_name] = analysis_labels
        per_analysis_labels_mpl[analysis_name] = analysis_labels_mpl
    
    return per_analysis_labels_html, per_analysis_labels_mpl


def build_per_analysis_chi2_table(analysis_name, series_ids, per_analysis_labels, data_dict):
    """Build an HTML table showing per-analysis chi2 statistics."""
    analysis_data = data_dict.get(analysis_name, {})
    if not analysis_data:
        return ""

    analysis_series_ids = [sid for sid in series_ids if sid in analysis_data]
    if not analysis_series_ids:
        return ""

    analysis_label_map = per_analysis_labels.get(analysis_name, {})

    table_html = r"""        <table>
            <tr><th>Source</th><th>Label</th><th>Global $\chi^2$</th><th>ndf</th><th>Reduced $\chi^2$</th><th>Averaged $\chi^2$ over all obs.</th></tr>"""

    for sid in analysis_series_ids:
        label_text = analysis_label_map.get(sid, sid[0])
        source = sid[1]
        bin_tuples = analysis_data[sid]

        if not bin_tuples:
            continue

        chi2_list = []
        ndf_list = []
        reduced_chi2_list = []
        for _, chi2, ndf, reduced_chi2 in bin_tuples:
            if np.isfinite(reduced_chi2) and reduced_chi2 != -1:
                chi2_list.append(chi2)
                ndf_list.append(ndf)
            reduced_chi2_list.append(reduced_chi2)

        if not chi2_list:
            continue

        chi2_sum_obs = sum(chi2_list)
        total_ndf_obs = sum(ndf_list)
        reduced_chi2_obs = chi2_sum_obs / total_ndf_obs if total_ndf_obs > 0 else np.nan

        avg_chi2 = sum(reduced_chi2_list) / len(reduced_chi2_list)

        table_html += f"""
            <tr>
                <td>{source}</td>
                <td>{label_text}</td>
                <td>{format_value(chi2_sum_obs)}</td>
                <td>{format_value(total_ndf_obs)}</td>
                <td>{format_value(reduced_chi2_obs)}</td>
                <td>{format_value(avg_chi2)}</td>
            </tr>"""

    table_html += """
        </table>"""

    return table_html


def filter_labels(labels, series_ids, requested_labels):
    """Filter labels select matching internal series IDs."""

    if requested_labels:
        missing = [x for x in requested_labels if x not in labels]
        if missing:
            print(f"Warning: Requested labels not found in chi2.json and ignored: {missing}.")
        labels = [x for x in requested_labels if x in labels]
    selected_series_ids = [sid for sid in series_ids if sid[0] in labels]
    return labels, selected_series_ids


def extract_numeric_sort_key(obs_id):
    """Extract numeric values from observable ID for sorting."""

    nums = re.findall(r'\d+', obs_id)
    if nums:
        return tuple(int(n) for n in nums)
    return (float('inf'),)


def series_style(is_default, plot_index):
    """Return marker/color/size/linewidth for a series."""
    if is_default:
        return {
            'color': DEFAULT_COLOR,
            'marker': 'h',
            'markersize': 5,
            'linewidth': 1.5,
            'linealpha': 0.4,
        }
    return {
        'color': COLORS[plot_index % len(COLORS)],
        'marker': MARKERS[plot_index % len(MARKERS)],
        'markersize': 6,
        'linewidth': 2.0,
        'linealpha': 0.4,
    }


def prepare_series_data(analysis_data, series_ids, default_label, analysis_name):
    """Build per-series bin maps and resolve the default series for ratio plots."""
    plot_series = [sid for sid in series_ids if sid in analysis_data]
    if not plot_series:
        return None

    series_to_bin_data = {}
    all_bin_ids = set()
    for sid in plot_series:
        obs_list_sorted = sorted(analysis_data[sid], key=lambda x: extract_numeric_sort_key(x[0]))
        bin_data = {obs_id: reduced_chi2 for obs_id, chi2, ndf, reduced_chi2 in obs_list_sorted}
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
            print(f"Warning: Analysis '{analysis_name}' has multiple series with default label '{default_label}'. "
                  f"Skipping ratio subplot for this analysis.")

    bin_ids = sorted(all_bin_ids, key=extract_numeric_sort_key)
    return {
        'plot_series'      : plot_series,
        'series_to_bin_data': series_to_bin_data,
        'bin_ids'          : bin_ids,
        'default_series_id': default_series_id,
        'default_bin_data' : default_bin_data,
    }


def compute_figure_layout(plot_series, has_ratio_plot, bin_ids_length):
    """Return figure-size and adjustment fractions plus legend column count."""
    n_legend_items = len(plot_series)
    max_label_len  = max((len(sid[0]) for sid in plot_series), default=0)
    if max_label_len < 20:
        ncols_legend = min(3, n_legend_items)
    elif max_label_len < 40:
        ncols_legend = min(2, n_legend_items)
    else:
        ncols_legend = 1
    nrows_legend = (n_legend_items + ncols_legend - 1) // ncols_legend

    title_height  = 0.6
    legend_height = 0.4 * nrows_legend
    bottom_height = max(0.8, 0.15 * bin_ids_length)
    axes_height   = 4.0 + (1.2 if has_ratio_plot else 0.0)
    total_height  = (title_height + legend_height) + axes_height + bottom_height

    return {
        'ncols_legend' : ncols_legend,
        'total_height' : total_height,
        'top_frac'     : (title_height + legend_height) / total_height + 0.02,
        'bottom_frac'  : bottom_height / total_height,
        'title_pos'    : (total_height - title_height / 2.0 + 0.1) / total_height,
    }


def plot_series_on_axes(ax, ax_ratio, plot_series, series_to_bin_data, chunk_bin_ids,
                         default_series_id, default_bin_data, analysis_label_map):
    """Draw all series on the main axis (and ratio axis if ax_ratio is given)."""
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
        label      = analysis_label_map.get(sid, sid[0])
        if is_default and 'default' not in label.lower():
            label = f'{label} (default)'
        style = series_style(is_default, plot_index)
        if not is_default:
            plot_index += 1

        ax.plot(valid_x, valid_chi2, marker=style['marker'], linewidth=0.0,
                label=label, color=style['color'], markersize=style['markersize'], alpha=1.0)
        ax.plot(valid_x, valid_chi2, linestyle='-', marker=None,
                color=style['color'], linewidth=style['linewidth'], alpha=style['linealpha'])

        if ax_ratio is not None and not is_default:
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
                ax_ratio.plot(ratio_x, ratio_y, marker=style['marker'], linewidth=0.0,
                              color=style['color'], markersize=5, alpha=1.0)
                ax_ratio.plot(ratio_x, ratio_y, linestyle='-', marker=None,
                              color=style['color'], linewidth=1.5, alpha=0.5)


def configure_axes(ax, ax_ratio, chunk_bin_ids, log_scale, ncols_legend):
    """Apply tick formatting, legend, and ratio-axis styling."""
    has_ratio_plot = ax_ratio is not None
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
        ax_ratio.axhline(1.0, color=DEFAULT_COLOR, linestyle='--', alpha=0.4)
        ax_ratio.set_ylabel(f'data/default', fontsize=20)
        ax_ratio.set_ylim(0.0, 2.0)
        ax_ratio.set_yticks([0.0, 1.0])
        ax_ratio.tick_params(axis='both', which='major', labelsize=20, direction='in',
                             length=5, top=True, right=True)
        ax_ratio.tick_params(axis='both', which='minor', direction='in',
                             length=3, top=True, right=True)

    ax.legend(loc='lower left', fontsize=20, frameon=False, ncols=ncols_legend,
              bbox_to_anchor=(0, 1.01), borderaxespad=0)
    ax.tick_params(axis='both', which='major', labelsize=20, direction='in',
                   length=5, top=True, right=True)
    ax.tick_params(axis='both', which='minor', direction='in',
                   length=3, top=True, right=True)
    ax.tick_params(axis='y', which='minor', direction='in',
                   length=3, top=True, right=True)


def plot_chi2_per_analysis(data_dict, series_ids, per_analysis_labels, default_label,
                           log_scale, output_dir):
    """Create one plot per analysis showing chi2/ndf for each observable."""
    os.makedirs(output_dir)

    for analysis_name in sorted(data_dict.keys()):
        analysis_data = data_dict[analysis_name]
        if not analysis_data:
            continue

        prepared = prepare_series_data(analysis_data, series_ids, default_label, analysis_name)
        if prepared is None:
            continue
        plot_series        = prepared['plot_series']
        series_to_bin_data = prepared['series_to_bin_data']
        bin_ids            = prepared['bin_ids']
        default_series_id  = prepared['default_series_id']
        default_bin_data   = prepared['default_bin_data']

        n_bins = len(bin_ids)
        if n_bins == 0:
            continue

        n_chunks = (n_bins + CHUNK_SIZE - 1) // CHUNK_SIZE
        for chunk_idx in range(n_chunks):
            start = chunk_idx * CHUNK_SIZE
            end   = min((chunk_idx + 1) * CHUNK_SIZE, n_bins)
            chunk_bin_ids  = bin_ids[start:end]
            bin_ids_length = max((len(str(bid)) for bid in chunk_bin_ids), default=1)

            has_ratio_plot = default_bin_data is not None and len(plot_series) > 1
            layout = compute_figure_layout(plot_series, has_ratio_plot, bin_ids_length)

            if has_ratio_plot:
                fig, (ax, ax_ratio) = plt.subplots(2, 1, sharex=True, figsize=(16, layout['total_height']),
                    gridspec_kw={'height_ratios': [4.0, 1.2], 'hspace': 0.0})
                plt.subplots_adjust(left=0.07, right=0.98, bottom=layout['bottom_frac'],
                                    top=1 - layout['top_frac'], hspace=0.0)
            else:
                fig, ax = plt.subplots(figsize=(16, layout['total_height']))
                ax_ratio = None
                plt.subplots_adjust(left=0.07, right=0.98, bottom=layout['bottom_frac'],
                                    top=1 - layout['top_frac'])

            title = f'{analysis_name}' if n_chunks == 1 else f'{analysis_name} ({chunk_idx + 1}/{n_chunks})'
            fig.suptitle(title, fontsize=24, y=layout['title_pos'], fontweight='bold')
            ax.set_ylabel(r'$\chi^2 / \mathrm{ndf}$', fontsize=20)
            ax.set_yscale('log' if log_scale else 'linear')

            plot_series_on_axes(ax, ax_ratio, plot_series, series_to_bin_data, chunk_bin_ids,
                                 default_series_id, default_bin_data,
                                 per_analysis_labels.get(analysis_name, {}))
            configure_axes(ax, ax_ratio, chunk_bin_ids, log_scale, layout['ncols_legend'])

            if has_ratio_plot:
                fig.align_ylabels([ax, ax_ratio])

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
                print(f"Created plot for {analysis_name} [{chunk_idx + 1}/{n_chunks}]: {output_file_pdf}/png")
            plt.close(fig)
    print()
    return


def format_value(value):
    """Format a numeric value for HTML display, returning 'nan' for missing/non-finite."""
    if value is None:
        return "nan"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{f:.2f}" if np.isfinite(f) else "nan"


def group_plot_files(plot_files):
    """Group plot file names by analysis (chunk siblings stay together)."""
    groups = {}
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
        groups.setdefault(group, []).append(file_name)
    return groups


def create_index_html(command, summaries, data_dict, series_ids, per_analysis_labels,
                      input_file, source_command, output_dir):
    """Create an index.html file in the output directory to access all plot files."""

    plot_files = sorted(f for f in os.listdir(output_dir) if f.endswith((".pdf", ".png")))
    exp_analysis_groups = group_plot_files(plot_files)

    now = datetime.datetime.now().strftime("%A, %d. %B %Y %H:%M")

    parts = []
    parts.append(f"""<html>
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
        <h2>{output_dir}</h2>""")

    if input_file:
        parts.append(f"""
        <p>Input files: {input_file}""")
    if source_command:
        parts.append(f"""
        <pre>{source_command}</pre></p>""")

    if summaries:
        parts.append(r"""
        <h3>Summary</h3>
        <table>
            <tr><th>Source</th><th>Label</th><th>Global $\chi^2$</th><th>ndf</th><th>Reduced $\chi^2$</th><th>Averaged $\chi^2$ over all obs.</th></tr>""")
        for summary in summaries:
            parts.append(f"""
            <tr>
                <td>{summary.get('source','')}</td>
                <td>{summary.get('label','')}</td>
                <td>{format_value(summary.get('global_chi2'))}</td>
                <td>{format_value(summary.get('ndf'))}</td>
                <td>{format_value(summary.get('reduced_chi2'))}</td>
                <td>{format_value(summary.get('average_chi2'))}</td>
            </tr>""")
        parts.append("""
        </table>""")
        parts.append(r"""
        <p style="margin-left: 1.5em; max-width: 60em; font-size: 0.95em;">
            <strong>Reduced $\chi^2$:</strong> total $\chi^2$ over all bins divided by the total number of bins,
            $\chi^2_{\mathrm{red}} = \dfrac{\sum_{o\in\mathrm{obs}} \chi^2_o}{\sum_{o\in\mathrm{obs}}\mathrm{ndf}_o}$.<br>
            <strong>Averaged $\chi^2$:</strong> mean of the per-observable reduced $\chi^2$ values,
            $\chi^2_{\mathrm{avg}} = \dfrac{1}{N_{\mathrm{obs}}}\sum_{o\in\mathrm{obs}} \dfrac{\chi^2_o}{\mathrm{ndf}_o}$.
        </p>""")

    for group, files in sorted(exp_analysis_groups.items()):
        parts.append(f"""
        <h3>{group}</h3>""")

        table_html = build_per_analysis_chi2_table(group, series_ids, per_analysis_labels, data_dict)
        if table_html:
            parts.append("\n" + table_html)
        
        pngs = [f for f in files if f.endswith('.png')]
        for png in sorted(pngs):
            pdf = png.replace('.png', '.pdf')
            parts.append(f"""
        <div style='margin-bottom:1.5em;'><img src='{png}' alt='{png}' style='max-width:1000px;'><br>""")
            if pdf in files:
                parts.append(f'<a href="{pdf}">Download PDF</a>')
            parts.append('</div>')

    parts.append(f"""    
        <footer style="clear:both; margin-top:3em; padding-top:3em">
        <p>Generated at {now}</p>
        <p>Created with command: <pre>{command}</pre></p>
        </footer>
    </body>
</html>""")

    html = "".join(parts)
    with open(os.path.join(output_dir, "index.html"), "w") as f:
        f.write(html)
    print(f"Created {os.path.join(output_dir, 'index.html')}")
    return


# --- Interactive html creation --------------------------------------------------------- #

def build_interactive_payload(data_dict, series_ids, per_analysis_labels, default_label, log_scale):
    """Build the JSON-serializable data payload embedded into interactive.html."""

    PLOTLY_MARKER_SYMBOLS = {'o': 'circle', 's': 'square', '^': 'triangle-up', 'D': 'diamond',
                         'v': 'triangle-down', '<': 'triangle-left', '>': 'triangle-right',
                         'h': 'hexagon'}

    def finite_or_none(value):
        """Convert value to a finite float or None, so it can be embedded as JSON."""
        out = to_float_or_nan(value)
        return float(out) if np.isfinite(out) else None
    
    analyses = {}
    for analysis_name in sorted(data_dict.keys()):
        analysis_data = data_dict[analysis_name]
        if not analysis_data:
            continue
        prepared = prepare_series_data(analysis_data, series_ids, None, analysis_name)
        if prepared is None:
            continue
        plot_series        = prepared['plot_series']
        bin_ids            = prepared['bin_ids']
        series_to_bin_data = prepared['series_to_bin_data']

        default_candidates = [sid for sid in plot_series if default_label and sid[0] == default_label]
        default_series_id  = default_candidates[0] if len(default_candidates) == 1 else None

        analysis_label_map = per_analysis_labels.get(analysis_name, {})
        series_entries = []
        plot_index = 0
        for sid in plot_series:
            bin_data = series_to_bin_data[sid]
            chi2_map = {obs_id: chi2 for obs_id, chi2, ndf, _ in analysis_data[sid]}
            ndf_map  = {obs_id: ndf for obs_id, chi2, ndf, _ in analysis_data[sid]}

            is_default = bool(default_series_id and sid == default_series_id)
            label      = analysis_label_map.get(sid, sid[0])
            if is_default and 'default' not in label.lower():
                label = f'{label} (default)'
            style = series_style(is_default, plot_index)
            if not is_default:
                plot_index += 1
            series_entries.append({'label'     : label,
                                   'source'    : sid[1],
                                   'values'    : [finite_or_none(bin_data.get(bid)) for bid in bin_ids],
                                   'chi2'      : [finite_or_none(chi2_map.get(bid)) for bid in bin_ids],
                                   'ndf'       : [finite_or_none(ndf_map.get(bid)) for bid in bin_ids],
                                   'color'     : style['color'],
                                   'symbol'    : PLOTLY_MARKER_SYMBOLS.get(style['marker'], 'circle'),
                                   'is_default': is_default})
        analyses[analysis_name] = {'bin_ids': [str(bid) for bid in bin_ids], 'series': series_entries}
    return {'default_label': default_label, 'log_scale': bool(log_scale), 'analyses': analyses}

INTERACTIVE_TEMPLATE = r"""<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>__TITLE__ - interactive [EXPERIMENTAL]</title>
        <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
        <style>
        html { font-family: sans-serif; }
        a { text-decoration: none; font-weight: bold; }
        .controls { display: flex; flex-wrap: wrap; gap: 1.2em; align-items: center; margin: 0.8em 0; }
        .controls > div { display: flex; align-items: center; gap: 0.4em; }
        .controls select { max-width: 28em; }
        button { padding: 2px 12px; cursor: pointer; }
        button.active { background: #3366FF; color: #fff; border: 1px solid #3366FF; }
        #series-panel { display: flex; flex-wrap: wrap; gap: 0.4em 1.6em; margin: 0.6em 0 1em 0; }
        .series-row { display: flex; align-items: center; gap: 0.4em; }
        .series-row input[type=color] { width: 2.2em; height: 1.7em; padding: 0; border: 1px solid #ccc; cursor: pointer; }
        .series-row small { color: #666; }
        footer { clear: both; margin-top: 3em; padding-top: 1em; }
        </style>
    </head>
    <body>
        <h2>__TITLE__ &mdash; interactive</h2>
        <p><a href="index.html">Back to static report</a></p>
        <div class="controls">
            <div><label for="analysis-select">Analysis:</label><select id="analysis-select"></select></div>
            <div><button id="log-toggle" title="Toggle logarithmic y-axis">log</button></div>
            <div><button id="ratio-toggle" title="Toggle ratio subplot">ratio</button></div>
            <div><label for="ref-select">Reference:</label><select id="ref-select"></select></div>
            <div><label for="sort-select">Bin order:</label>
                <select id="sort-select" title="Sort bins by decreasing χ²/ndf of the reference series">
                    <option value="natural">natural</option>
                    <option value="chi2">by reference &chi;&sup2;/ndf</option>
                </select>
            </div>
            <div><label><input type="checkbox" id="unity-line"> &chi;&sup2;/ndf = 1 line</label></div>
            <div><button id="reset-button" title="Reset all settings for the current analysis">Reset</button></div>
        </div>
        <div id="series-panel"></div>
        <div id="plot" style="width:100%;height:750px;"></div>
        <p style="color:#666; font-size:0.9em;">Tip: click legend entries to hide/show a series, use the checkboxes
        and colour pickers above, drag to zoom.</p>
        <footer>
        <p>Generated at __NOW__</p>
        <p>Created with command: <pre>__COMMAND__</pre></p>
        </footer>
        <script>
        const DATA = __PAYLOAD__;

        const plotDiv        = document.getElementById('plot');
        const analysisSelect = document.getElementById('analysis-select');
        const refSelect      = document.getElementById('ref-select');
        const sortSelect     = document.getElementById('sort-select');
        const logButton      = document.getElementById('log-toggle');
        const ratioButton    = document.getElementById('ratio-toggle');
        const unityCheckbox  = document.getElementById('unity-line');
        const resetButton    = document.getElementById('reset-button');
        const seriesPanel    = document.getElementById('series-panel');

        const analysisNames = Object.keys(DATA.analyses);
        const state = {
            colors       : [],
            visible      : [],
            ref          : -1,
            sort         : 'natural',
            logScale     : DATA.log_scale,
            showUnityLine: false,
            showRatioPlot: true,
        };
        let currentAnalysis = analysisNames[0];
        let traceSeriesIndex = [];

        function plainLabel(series) {
            return series.label.replace(/<sup>(.*?)<\/sup>/g, ' ($1)').replace(/<[^>]*>/g, '');
        }

        function hexToRgba(hex, alpha) {
            const v = hex.replace('#', '');
            const r = parseInt(v.substring(0, 2), 16);
            const g = parseInt(v.substring(2, 4), 16);
            const b = parseInt(v.substring(4, 6), 16);
            return 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')';
        }

        function getState(name) {
            const analysis = DATA.analyses[name];
            while (state.colors.length < analysis.series.length) {
                const i = state.colors.length;
                state.colors.push(analysis.series[i].color);
                state.visible.push(true);
            }
            return state;
        }

        function binPermutation(name) {
            const analysis = DATA.analyses[name];
            const st = getState(name);
            const indices = analysis.bin_ids.map((bid, j) => j);
            if (st.sort !== 'chi2' || st.ref < 0) {
                return indices;
            }
            const ref = st.ref >= 0 ? analysis.series[st.ref] : null;
            const score = analysis.bin_ids.map((bid, j) => {
                if (ref !== null) {
                    return ref.values[j] !== null ? ref.values[j] : -1;
                }
                let best = -1;
                analysis.series.forEach((s, i) => {
                    if (st.visible[i] && s.values[j] !== null && s.values[j] > best) best = s.values[j];
                });
                return best;
            });
            indices.sort((a, b) => score[b] - score[a]);
            return indices;
        }

        function buildTraces(name) {
            const analysis = DATA.analyses[name];
            const st = getState(name);
            const hasRatio = st.ref >= 0 && analysis.series.length > 1;
            const showRatioPlot = hasRatio && st.showRatioPlot;
            const perm = binPermutation(name);
            const orderedBins = perm.map(j => analysis.bin_ids[j]);
            const traces = [];
            traceSeriesIndex = [];

            analysis.series.forEach((s, i) => {
                traces.push({
                    x: orderedBins,
                    y: perm.map(j => s.values[j]),
                    customdata: perm.map(j => [s.chi2[j], s.ndf[j]]),
                    name: s.label,
                    mode: 'lines+markers',
                    marker: {color: st.colors[i], symbol: s.symbol, size: 9},
                    line: {color: hexToRgba(st.colors[i], 0.4), width: 2},
                    visible: st.visible[i] ? true : 'legendonly',
                    legendgroup: 'series' + i,
                    yaxis: 'y',
                    hovertemplate: '%{x}<br>χ²/ndf = %{y:.3f}<br>χ² = %{customdata[0]:.2f}, ndf = %{customdata[1]}' +
                                   '<extra>' + plainLabel(s) + '</extra>',
                });
                traceSeriesIndex.push(i);
            });

            if (showRatioPlot) {
                const ref = analysis.series[st.ref];
                analysis.series.forEach((s, i) => {
                    if (i === st.ref) return;
                    const ratio = perm.map(j => {
                        const v = s.values[j];
                        const d = ref.values[j];
                        return (v !== null && d !== null && d > 0) ? v / d : null;
                    });
                    traces.push({
                        x: orderedBins,
                        y: ratio,
                        name: s.label,
                        mode: 'lines+markers',
                        marker: {color: st.colors[i], symbol: s.symbol, size: 7},
                        line: {color: hexToRgba(st.colors[i], 0.5), width: 1.5},
                        visible: st.visible[i] ? true : 'legendonly',
                        legendgroup: 'series' + i,
                        showlegend: false,
                        yaxis: 'y2',
                        hovertemplate: '%{x}<br>ratio = %{y:.3f}<extra>' +
                                       plainLabel(s) + ' / ' + plainLabel(ref) + '</extra>',
                    });
                    traceSeriesIndex.push(i);
                });
            }
            return {traces: traces, hasRatio: hasRatio, showRatioPlot: showRatioPlot, orderedBins: orderedBins};
        }

        function buildLayout(name, hasRatio, showRatioPlot, orderedBins) {
            const st = getState(name);
            const shapes = [];
            if (st.showUnityLine) {
                shapes.push({type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0: 1, y1: 1,
                             line: {color: '#898C89', width: 1, dash: 'dash'}});
            }
            if (showRatioPlot) {
                shapes.push({type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: 1, y1: 1,
                             line: {color: '#898C89', width: 1, dash: 'dash'}});
            }
            const layout = {
                title: {text: name, font: {size: 22}},
                hovermode: 'closest',
                legend: {orientation: 'h', x: 0, y: 1.02, xanchor: 'left', yanchor: 'bottom'},
                margin: {l: 80, r: 30, t: 120, b: 100},
                uirevision: name,
                xaxis: {
                    type: 'category',
                    categoryorder: 'array',
                    categoryarray: orderedBins,
                    tickangle: -90,
                    automargin: true,
                    anchor: showRatioPlot ? 'y2' : 'y',
                },
                yaxis: {
                    title: {text: 'χ² / ndf'},
                    type: st.logScale ? 'log' : 'linear',
                    domain: showRatioPlot ? [0.30, 1.0] : [0.0, 1.0],
                    automargin: true,
                },
                shapes: shapes,
            };
            if (showRatioPlot) {
                layout.yaxis2 = {
                    title: {text: 'data / reference'},
                    domain: [0.0, 0.24],
                    range: [0.0, 2.0],
                    automargin: true,
                };
            }
            return layout;
        }

        function render() {
            const st = getState(currentAnalysis);
            const built = buildTraces(currentAnalysis);
            const layout = buildLayout(currentAnalysis, built.hasRatio, built.showRatioPlot, built.orderedBins);
            const config = {
                responsive: true,
                displaylogo: false,
                scrollZoom: true,
                toImageButtonOptions: {format: 'png', scale: 2,
                                       filename: currentAnalysis.replace(/[\/ ]/g, '_') + '_chi2_interactive'},
            };
            Plotly.react(plotDiv, built.traces, layout, config);
            logButton.classList.toggle('active', st.logScale);
            ratioButton.classList.toggle('active', built.showRatioPlot);
            ratioButton.disabled = !built.hasRatio;
            ratioButton.title = built.hasRatio
                ? (built.showRatioPlot ? 'Hide ratio subplot' : 'Show ratio subplot')
                : 'Select a reference to enable the ratio subplot';
            unityCheckbox.checked = st.showUnityLine;
        }

        function rebuildControls() {
            const analysis = DATA.analyses[currentAnalysis];
            const st = getState(currentAnalysis);

            refSelect.innerHTML = '';
            const noneOption = document.createElement('option');
            noneOption.value = '-1';
            noneOption.textContent = '(none)';
            refSelect.appendChild(noneOption);
            analysis.series.forEach((s, i) => {
                const option = document.createElement('option');
                option.value = String(i);
                option.textContent = plainLabel(s) + '  [' + s.source + ']';
                refSelect.appendChild(option);
            });
            refSelect.value = String(st.ref);
            if (st.ref < 0) {
                st.sort = 'natural';
            }
            sortSelect.value = st.sort;
            sortSelect.disabled = st.ref < 0;
            logButton.classList.toggle('active', st.logScale);
            ratioButton.classList.toggle('active', st.showRatioPlot);
            unityCheckbox.checked = st.showUnityLine;

            seriesPanel.innerHTML = '';
            analysis.series.forEach((s, i) => {
                const row = document.createElement('div');
                row.className = 'series-row';

                const checkbox = document.createElement('input');
                checkbox.type = 'checkbox';
                checkbox.checked = st.visible[i];
                checkbox.addEventListener('change', () => {
                    st.visible[i] = checkbox.checked;
                    render();
                });

                const colorInput = document.createElement('input');
                colorInput.type = 'color';
                colorInput.value = st.colors[i];
                colorInput.addEventListener('input', () => {
                    st.colors[i] = colorInput.value;
                    render();
                });

                const text = document.createElement('span');
                text.innerHTML = s.label + ' <small>[' + s.source + ']</small>';

                row.appendChild(checkbox);
                row.appendChild(colorInput);
                row.appendChild(text);
                seriesPanel.appendChild(row);
            });
        }

        function syncCheckboxes() {
            const st = getState(currentAnalysis);
            seriesPanel.querySelectorAll('input[type=checkbox]').forEach((checkbox, i) => {
                checkbox.checked = st.visible[i];
            });
        }

        if (analysisNames.length === 0) {
            plotDiv.textContent = 'No chi2 data available.';
        } else {
            analysisNames.forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = name;
                analysisSelect.appendChild(option);
            });
            analysisSelect.value = currentAnalysis;

            analysisSelect.addEventListener('change', () => {
                currentAnalysis = analysisSelect.value;
                rebuildControls();
                render();
            });
            refSelect.addEventListener('change', () => {
                getState(currentAnalysis).ref = parseInt(refSelect.value, 10);
                rebuildControls();
                render();
            });
            sortSelect.addEventListener('change', () => {
                getState(currentAnalysis).sort = sortSelect.value;
                render();
            });
            ratioButton.addEventListener('click', () => {
                const st = getState(currentAnalysis);
                st.showRatioPlot = !st.showRatioPlot;
                rebuildControls();
                render();
            });
            logButton.addEventListener('click', () => {
                getState(currentAnalysis).logScale = !getState(currentAnalysis).logScale;
                render();
            });
            unityCheckbox.addEventListener('change', () => {
                getState(currentAnalysis).showUnityLine = unityCheckbox.checked;
                render();
            });
            resetButton.addEventListener('click', () => {
                const analysis = DATA.analyses[currentAnalysis];
                state.colors = analysis.series.map(s => s.color);
                state.visible = analysis.series.map(() => true);
                state.ref = analysis.series.findIndex(s => s.is_default);
                state.sort = 'natural';
                state.logScale = DATA.log_scale;
                state.showUnityLine = false;
                state.showRatioPlot = true;
                rebuildControls();
                render();
            });

            rebuildControls();
            render();
            plotDiv.on('plotly_restyle', (event) => {
                const update = event[0];
                const indices = event[1];
                if (!update || update.visible === undefined || !indices) return;
                const values = Array.isArray(update.visible) ? update.visible : [update.visible];
                const st = getState(currentAnalysis);
                indices.forEach((traceIdx, k) => {
                    const seriesIdx = traceSeriesIndex[traceIdx];
                    if (seriesIdx === undefined) return;
                    st.visible[seriesIdx] = values[k % values.length] === true;
                });
                syncCheckboxes();
            });
        }
        </script>
    </body>
</html>"""

def create_interactive_html(command, payload, output_dir):
    """Create interactive.html, a Plotly-based interactive view of the chi2 data."""
    payload_json = json.dumps(payload).replace("</", "<\\/")
    now = datetime.datetime.now().strftime("%A, %d. %B %Y %H:%M")
    html = (INTERACTIVE_TEMPLATE
            .replace("__TITLE__", str(output_dir))
            .replace("__NOW__", now)
            .replace("__COMMAND__", command)
            .replace("__PAYLOAD__", payload_json))
    out_path = os.path.join(output_dir, "interactive.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Created {out_path} [EXPERIMENTAL]")
    add_interactive_link_to_index(output_dir)
    return

def add_interactive_link_to_index(output_dir):
    """Add a link to interactive.html at the bottom of an existing index.html."""
    index_path = Path(output_dir) / "index.html"
    if not index_path.exists():
        return
    text = index_path.read_text(encoding="utf-8")
    link_html = ('<br><p><a href="interactive.html">Open interactive version of these plots</a></p></body>')
    if "interactive.html" not in text and "    </body>" in text:
        text = text.replace("    </body>", link_html, 1)
        index_path.write_text(text, encoding="utf-8")
    return

# --------------------------------------------------------------------------------------- #



def build_parser():
    """Build the argument parser for per-analysis chi2 plots."""
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
    parser.add_argument("-i", "--interactive", action="store_true", help="Additionally create an interactive plot page [EXPERIMENTAL]")
    return parser


def main():
    print("Starting chi2 plotting...\n")

    parser = build_parser()
    args = parser.parse_args()
    command = " ".join(sys.argv)

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
    default_label = None
    if args.default_label:
        if args.default_label not in available_labels:
            print(f"Warning: Default label '{args.default_label}' not found in JSON, available labels: {available_labels}. "
                  f"Continuing without default reference.")
        else:
            default_label = args.default_label
            if default_label not in labels:
                labels.insert(0, default_label)
                series_ids = [sid for sid in available_series_ids if sid[0] in labels]
    per_analysis_labels_html, per_analysis_labels_mpl = assign_superscripts_per_analysis(data_dict, series_ids)

    plot_chi2_per_analysis(data_dict, series_ids, per_analysis_labels_mpl,
                           default_label=default_label,
                           log_scale=args.log,
                           output_dir=args.outdir)
    create_index_html(command,
                      summaries=summaries,
                      data_dict=data_dict,
                      series_ids=series_ids,
                      per_analysis_labels=per_analysis_labels_html,
                      input_file=", ".join(str(p) for p in json_paths),
                      source_command="\n".join(source_commands),
                      output_dir=args.outdir)
    if args.interactive:
        payload = build_interactive_payload(data_dict, series_ids, per_analysis_labels_html,
                                            default_label=default_label, log_scale=args.log)
        create_interactive_html(command, payload, args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
