import chess
import chess.pgn
import json
import os
import glob
import random
import math
from collections import defaultdict
from tqdm import tqdm
import multiprocessing

# --- Configuration ---
INPUT_DIR = "data/raw"
OUTPUT_FILE = "data_profile.json"
SAMPLE_RATE = 0.05  # Process 5% of files for profiling
MAX_SAMPLE_FILES = 24  # Upper bound so profiling finishes quickly
SINGLE_FILE_MODE = False  # Enable holistic profiling by default

# --- Constants ---
# ELO Bands: 1200 to 2900 in 100-point increments (games below 1200 are skipped)
ELO_MIN = 1200
ELO_MAX = 2900
ELO_STEP = 100

# Ply Bands: 0-20, 20-40, 40-60, 60-80, 80+
PLY_BANDS = [20, 40, 60, 80]

# Clock Bands: <15s, <60s, <300s, <600s, >600s
CLOCK_BANDS = [15, 60, 300, 600]

# --- Helper Functions ---

def parse_time_control(tc_string):
    if not tc_string or tc_string == "-": return None
    try:
        if "+" in tc_string:
            base, inc = tc_string.split("+")
            return int(base), int(inc)
        else:
            return int(tc_string), 0
    except:
        return None

def get_bucket_key(elo, ply, clock, inc):
    # 1. ELO Bucket
    if elo < ELO_MIN: b_elo = 0
    elif elo >= ELO_MAX: b_elo = (ELO_MAX - ELO_MIN) // ELO_STEP + 1
    else: b_elo = (elo - ELO_MIN) // ELO_STEP + 1
    
    # 2. Ply Bucket
    b_ply = len(PLY_BANDS)
    for i, limit in enumerate(PLY_BANDS):
        if ply < limit:
            b_ply = i
            break
            
    # 3. Clock Bucket
    b_clk = len(CLOCK_BANDS)
    for i, limit in enumerate(CLOCK_BANDS):
        if clock < limit:
            b_clk = i
            break
            
    # 4. Increment Bucket (Binary)
    b_inc = 1 if inc > 0 else 0
    
    # Return tuple key (JSON compatible when converted to string)
    return f"{b_elo}_{b_ply}_{b_clk}_{b_inc}"

def process_file(file_path, position=None):
    local_counts = defaultdict(int)
    
    try:
        file_size = os.path.getsize(file_path)
        pbar = tqdm(
            total=file_size,
            unit='B',
            unit_scale=True,
            desc=os.path.basename(file_path),
            position=position,
            leave=False
        )

        with open(file_path) as pgn_file:
            game_count = 0
            while True:
                if pbar and game_count % 1000 == 0: # Update less frequently for speed
                    pbar.n = pgn_file.tell()
                    pbar.refresh()

                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception:
                    continue
                
                if game is None: break
                game_count += 1
                
                headers = game.headers
                
                # Metadata Filters
                tc_str = headers.get("TimeControl", "")
                tc_params = parse_time_control(tc_str)
                if not tc_params: continue
                base, inc = tc_params
                
                if (base + 40 * inc) < 180: continue # Bullet Filter
                
                try:
                    white_elo = int(headers.get("WhiteElo", 0))
                    black_elo = int(headers.get("BlackElo", 0))
                except:
                    continue

                if white_elo < 1200 or black_elo < 1200:
                    continue
                
                # Simulation
                node = game
                white_clock = int(base)
                black_clock = int(base)
                ply = 0
                
                while True:
                    next_node = node.next()
                    if next_node is None: break
                    
                    # OPTIMIZATION: Determine turn via Ply count
                    is_white = (ply % 2 == 0)
                    
                    # Determine Active Player Stats
                    if is_white:
                        active_elo = white_elo
                        active_clock = white_clock
                    else:
                        active_elo = black_elo
                        active_clock = black_clock
                    
                    # Bucket Key
                    key = get_bucket_key(active_elo, ply, active_clock, inc)
                    local_counts[key] += 1
                    
                    # Update State
                    clk_comment = next_node.clock()
                    if clk_comment is not None:
                        clk_after = int(float(clk_comment))
                        if is_white:
                            white_clock = clk_after
                        else:
                            black_clock = clk_after
                            
                    ply += 1
                    node = next_node
        
        pbar.close()
                    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        
    return local_counts


def process_file_worker(args):
    return process_file(*args)

def main():
    # 1. Select Files
    all_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pgn")))
    total_files = len(all_files)
    if total_files == 0:
        print("No PGN files found!")
        return

    if SINGLE_FILE_MODE:
        print("Running in SINGLE FILE MODE (Fast Profiling)...")
        selected_files = [all_files[0]]
    else:
        sample_size = max(1, int(total_files * SAMPLE_RATE))
        sample_size = min(sample_size, MAX_SAMPLE_FILES)
        random.seed(42)
        selected_files = random.sample(all_files, sample_size)

    print(f"Profiling {len(selected_files)} files (out of {total_files})...")

    # 2. Run Profiling
    global_counts = defaultdict(int)

    if SINGLE_FILE_MODE or len(selected_files) == 1:
        results = [process_file(selected_files[0], position=0)]
    else:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        tasks = [(file_path, idx) for idx, file_path in enumerate(selected_files)]

        with multiprocessing.Pool(processes=num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_file_worker, tasks),
                    total=len(selected_files)
                )
            )

    # 3. Aggregate Results
    print("Aggregating results...")
    for res in results:
        for k, v in res.items():
            global_counts[k] += v

    total_moves = sum(global_counts.values())

    metadata = {
        "total_files": total_files,
        "profiled_files": len(selected_files),
        "sample_rate": SAMPLE_RATE,
        "max_sample_files": MAX_SAMPLE_FILES,
        "single_file_mode": SINGLE_FILE_MODE,
        "total_moves_profiled": total_moves,
        "unique_buckets": len(global_counts),
        "elo_min": ELO_MIN,
        "elo_max": ELO_MAX,
        "elo_step": ELO_STEP,
        "ply_bands": PLY_BANDS,
        "clock_bands": CLOCK_BANDS,
    }

    bucket_counts = {k: int(v) for k, v in global_counts.items()}

    payload = {
        "metadata": metadata,
        "bucket_counts": bucket_counts,
    }

    print(f"Saving profile to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(payload, f, indent=2)

    print("Done.")
    print(f"Total Moves Profiled: {total_moves:,}")
    print(f"Unique Buckets Found: {len(global_counts)}")

if __name__ == "__main__":
    main()
