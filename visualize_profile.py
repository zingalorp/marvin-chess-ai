import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

PROFILE_FILE = "data_profile.json"
RETENTION_FILE = "retention_rates.json"
DEFAULT_OUTPUT = "profile_summary.txt"
DEFAULT_METADATA = {
    "elo_min": 600,
    "elo_max": 2900,
    "elo_step": 100,
    "ply_bands": [20, 40, 60, 80],
    "clock_bands": [15, 60, 300, 600],
}


def parse_key(key):
    return tuple(map(int, key.split('_')))


def load_profile(path):
    with open(path, 'r') as f:
        data = json.load(f)

    if "bucket_counts" in data:
        return {k: int(v) for k, v in data["bucket_counts"].items()}, data.get("metadata", {})
    return {k: int(v) for k, v in data.items()}, {}


def load_retention(path):
    if not Path(path).exists():
        return {}
    with open(path, 'r') as f:
        data = json.load(f)
    if "retention_rates" in data:
        return {k: float(v) for k, v in data["retention_rates"].items()}, data.get("metadata", {})
    return {k: float(v) for k, v in data.items()}, {}


def compute_scale(metadata):
    total_files = metadata.get("total_files")
    profiled_files = metadata.get("profiled_files")
    if total_files and profiled_files:
        return total_files / profiled_files
    return None


def summarize_dimensions(bucket_counts):
    elo_counts = defaultdict(int)
    ply_counts = defaultdict(int)
    clock_counts = defaultdict(int)
    inc_counts = defaultdict(int)

    for key, count in bucket_counts.items():
        b_elo, b_ply, b_clk, b_inc = parse_key(key)
        elo_counts[b_elo] += count
        ply_counts[b_ply] += count
        clock_counts[b_clk] += count
        inc_counts[b_inc] += count

    return elo_counts, ply_counts, clock_counts, inc_counts


def format_number(value):
    if value is None:
        return "-"
    if value >= 1000:
        return f"{value:,.0f}"
    return f"{value:.2f}" if isinstance(value, float) and value != int(value) else str(int(value))


def bucket_to_ranges(key, metadata):
    b_elo, b_ply, b_clk, b_inc = parse_key(key)

    elo_min = metadata.get("elo_min", DEFAULT_METADATA["elo_min"])
    elo_max = metadata.get("elo_max", DEFAULT_METADATA["elo_max"])
    elo_step = metadata.get("elo_step", DEFAULT_METADATA["elo_step"])
    ply_bands = metadata.get("ply_bands", DEFAULT_METADATA["ply_bands"])
    clock_bands = metadata.get("clock_bands", DEFAULT_METADATA["clock_bands"])

    max_elo_bucket = (elo_max - elo_min) // elo_step + 1
    if b_elo == 0:
        elo_label = f"< {elo_min}"
    elif b_elo >= max_elo_bucket:
        elo_label = f">= {elo_max}"
    else:
        low = elo_min + (b_elo - 1) * elo_step
        high = low + elo_step
        elo_label = f"{low}-{high - 1}"

    def band_label(bands, idx, suffix=""):
        if idx < len(bands):
            upper = bands[idx]
            lower = bands[idx - 1] if idx > 0 else 0
            return f"{lower}-{upper}{suffix}"
        lower = bands[-1] if bands else 0
        return f">= {lower}{suffix}"

    ply_label = band_label(ply_bands, b_ply, " ply")
    clock_suffix = "s"
    clock_label = band_label(clock_bands, b_clk, clock_suffix)
    inc_label = "with inc" if b_inc else "no inc"

    return f"ELO {elo_label} | Ply {ply_label} | Clock {clock_label} | {inc_label}"


def render_bucket_table(bucket_counts, retention_rates, scale, metadata, log):
    header = f"{'Bucket Description':<70}{'Count':>12}{'Projected':>14}{'KeepProb':>10}"
    log()
    log("Bucket Breakdown")
    log(header)
    log("-" * len(header))

    items = sorted(bucket_counts.items(), key=lambda kv: kv[1], reverse=True)
    for key, count in items:
        count = bucket_counts[key]
        projected = count * scale if scale else None
        keep = retention_rates.get(key)
        proj_text = format_number(projected) if projected is not None else '-'
        keep_text = f"{keep:.3f}" if keep is not None else '-'
        description = bucket_to_ranges(key, metadata)
        log(f"{description:<70}{format_number(count):>12}{proj_text:>14}{keep_text:>10}")


def maybe_plot(args, elo_counts, ply_counts, clock_counts, log):
    if not args.save_plot:
        return

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    elo_keys = sorted(elo_counts.keys())
    elo_vals = [elo_counts[k] for k in elo_keys]
    elo_labels = [f"{600 + k * 100}" for k in elo_keys]
    axs[0].bar(elo_keys, elo_vals, color='skyblue')
    axs[0].set_title('ELO Distribution')
    axs[0].set_xticks(elo_keys)
    axs[0].set_xticklabels(elo_labels, rotation=45)

    ply_labels = ["<20", "20-40", "40-60", "60-80", "80+"]
    ply_keys = sorted(ply_counts.keys())
    ply_vals = [ply_counts[k] for k in ply_keys]
    axs[1].bar(ply_keys, ply_vals, color='salmon')
    axs[1].set_title('Ply Distribution')
    axs[1].set_xticks(ply_keys)
    axs[1].set_xticklabels([ply_labels[k] if k < len(ply_labels) else f"{k}" for k in ply_keys])

    clk_labels = ["<15s", "<60s", "<5m", "<10m", ">10m"]
    clk_keys = sorted(clock_counts.keys())
    clk_vals = [clock_counts[k] for k in clk_keys]
    axs[2].bar(clk_keys, clk_vals, color='lightgreen')
    axs[2].set_title('Clock Distribution')
    axs[2].set_xticks(clk_keys)
    axs[2].set_xticklabels([clk_labels[k] if k < len(clk_labels) else f"{k}" for k in clk_keys])

    plt.tight_layout()
    plt.savefig(args.save_plot)
    log(f"Saved plot to {args.save_plot}")


def main():
    parser = argparse.ArgumentParser(description="Summarize profiler output and retention rates.")
    parser.add_argument("--profile", default=PROFILE_FILE, help="Path to data_profile.json")
    parser.add_argument("--retention", default=RETENTION_FILE, help="Path to retention_rates.json")
    parser.add_argument("--save-plot", help="Optional path to save distribution plot")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Path to write full text report")
    args = parser.parse_args()

    report_lines = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    bucket_counts, profile_meta = load_profile(args.profile)
    retention_rates, retention_meta = load_retention(args.retention)

    log(f"Loaded {len(bucket_counts)} buckets from {args.profile}.")
    total_moves = sum(bucket_counts.values())
    log(f"Total Moves Profiled: {total_moves:,}")

    if profile_meta:
        log()
        log("Profiler Metadata:")
        for key, value in profile_meta.items():
            log(f"  {key}: {value}")

    if retention_rates:
        log()
        log(f"Loaded {len(retention_rates)} retention entries from {args.retention}.")
        meta = retention_meta.get("global_target_rows")
        if meta:
            log(f"  Global Target Rows: {meta:,}")

    scale = compute_scale(profile_meta)
    if scale:
        log()
        log(f"Projected scale factor (total/profiled files): {scale:.2f}x")

    render_bucket_table(bucket_counts, retention_rates, scale, profile_meta, log)

    sorted_buckets = sorted(bucket_counts.items(), key=lambda kv: kv[1], reverse=True)
    log()
    log("Top 10 Buckets:")
    for key, count in sorted_buckets[:10]:
        prob = retention_rates.get(key)
        prob_text = f" (keep {prob:.2%})" if prob is not None else ""
        desc = bucket_to_ranges(key, profile_meta)
        log(f"  {desc}: {count:,}{prob_text}")

    elo_counts, ply_counts, clock_counts, inc_counts = summarize_dimensions(bucket_counts)
    log()
    log("Increment Distribution:")
    for inc_bucket, count in sorted(inc_counts.items()):
        label = "No Increment" if inc_bucket == 0 else ">0 Increment"
        log(f"  {label}: {count:,}")

    maybe_plot(args, elo_counts, ply_counts, clock_counts, log)

    with open(args.output, 'w') as f:
        f.write("\n".join(report_lines))
    log()
    log(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()