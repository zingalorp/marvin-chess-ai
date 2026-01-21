import json
import math
from pathlib import Path

# --- Configuration ---
DATA_PROFILE_FILE = Path("data_profile.json")
OUTPUT_FILE = Path("retention_rates.json")
GLOBAL_TARGET_ROWS = 2_000_000_000  # set to None to disable per-bucket targets
MIN_RETENTION_FLOOR = 0.03     # never go below this probability (3%)


def load_profile(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")

    with path.open() as f:
        data = json.load(f)

    if "bucket_counts" in data:
        counts = data["bucket_counts"]
        metadata = data.get("metadata", {})
    else:
        counts = data
        metadata = {}

    # ensure ints
    counts = {k: int(v) for k, v in counts.items()}
    return counts, metadata


def compute_scale(metadata):
    total_files = metadata.get("total_files")
    profiled_files = metadata.get("profiled_files")
    if total_files and profiled_files and profiled_files > 0:
        return total_files / profiled_files
    return 1.0


def compute_target_per_bucket(num_buckets):
    if not num_buckets:
        return None
    if GLOBAL_TARGET_ROWS is None:
        return None
    return GLOBAL_TARGET_ROWS / num_buckets


def main():
    bucket_counts, metadata = load_profile(DATA_PROFILE_FILE)
    if not bucket_counts:
        raise RuntimeError("Profile has no bucket counts. Run profile_data.py first.")

    scale_factor = compute_scale(metadata)
    projected_counts = {
        key: value * scale_factor
        for key, value in bucket_counts.items()
    }

    num_buckets = len(bucket_counts)
    target_per_bucket = compute_target_per_bucket(num_buckets)

    retention_rates = {}
    total_projected = 0.0
    for key, projected in projected_counts.items():
        total_projected += projected
        if projected <= 0:
            retention_rates[key] = 0.0
            continue
        if target_per_bucket is None:
            raw_rate = 1.0
        else:
            raw_rate = target_per_bucket / projected
        retention_rates[key] = float(
            min(1.0, max(MIN_RETENTION_FLOOR, raw_rate))
        )

    output_payload = {
        "metadata": {
            "global_target_rows": GLOBAL_TARGET_ROWS,
            "target_rows_per_bucket": target_per_bucket,
            "min_retention_floor": MIN_RETENTION_FLOOR,
            "scale_factor": scale_factor,
            "profile_metadata": metadata,
            "profile_total_rows": sum(bucket_counts.values()),
            "projected_total_rows": total_projected,
            "bucket_count": num_buckets,
        },
        "retention_rates": retention_rates,
    }

    with OUTPUT_FILE.open('w') as f:
        json.dump(output_payload, f, indent=2)

    print(f"Wrote retention rates for {num_buckets} buckets to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
