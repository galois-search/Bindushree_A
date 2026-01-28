import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager, Lock
import galois

# ✅ Define parameters
m = 12  # Field order
num_pieces = 5  # Number of pieces to divide the range into
start_index = 1  # Start index of combinations to process (1-based)
end_index = 25  # End index of combinations to process (1-based)
num_cores = cpu_count()  # Number of CPU cores to use

# ✅ Create a folder named with the value of m
folder_name = str(m)
os.makedirs(folder_name, exist_ok=True)

# ✅ Get MATLAB-compatible primitive polynomial
primitive_poly = galois.matlab_primitive_poly(2, m)
GF = galois.GF(2 ** m, irreducible_poly=primitive_poly, repr="power")

if 25 <= m <= 63:
    gf_compiled = GF.compile("jit-calculate")
elif m < 25:
    gf_compiled = GF.compile("jit-lookup")
    gf_table = {int(alpha): i for i, alpha in enumerate(GF.elements)}
else:
    gf_compiled = GF.compile("python-calculate")

# ✅ Function to compute autocorrelation using sequence differences
def compute_difference(sequence):
    N = len(sequence)
    results = []
    for k in range(N):
        shifted_sequence = np.roll(sequence, k)
        agreements = np.sum(sequence == shifted_sequence)
        disagreements = N - agreements
        result = agreements - disagreements
        results.append(result)
    return results

# ✅ Compute balance of a sequence
def calculate_balance(sequence):
    d = np.sum(sequence)
    return len(sequence) - 2 * d

# ✅ Function to get power representation safely
def get_power_representation(value):
    if value == 0:
        return None
    field_element = GF(value)
    try:
        return int(field_element.log())
    except AttributeError:
        return None

# ✅ Function to create pieces based on number of pieces
def create_pieces(num_pieces):
    total_elements = 2 ** m - 1
    elements_per_piece = total_elements // num_pieces
    pieces = []
    for i in range(num_pieces):
        start = i * elements_per_piece + 1
        end = (i + 1) * elements_per_piece if i != num_pieces - 1 else total_elements
        pieces.append((start, end))
    with open(f"{folder_name}/pieces.txt", "w") as f:
        for i, (start, end) in enumerate(pieces, 1):
            f.write(f"Piece {i}: From {start} to {end}\n")
    return pieces

# ✅ Function to generate all combinations and save to file
def generate_all_combinations(num_pieces):
    combinations = []
    with open("all_piece_combinations.txt", "w") as f:
        for x0_piece in range(1, num_pieces + 1):
            for mu_piece in range(1, num_pieces + 1):
                combinations.append((x0_piece, mu_piece))
                f.write(f"x0_piece={x0_piece}, mu_piece={mu_piece}\n")
    return combinations

# ✅ Function to load selected combinations from range
def load_selected_combinations_from_range(start_idx, end_idx, file_path="all_piece_combinations.txt"):
    combos = []
    with open(file_path, "r") as f:
        all_lines = f.readlines()
        selected_lines = all_lines[start_idx - 1:end_idx]
        for line in selected_lines:
            if "x0_piece=" in line and "mu_piece=" in line:
                parts = line.strip().split(",")
                x0 = int(parts[0].split("=")[1])
                mu = int(parts[1].split("=")[1])
                combos.append((x0, mu))
    print(f"🔢 Loaded combos #{start_idx} to #{end_idx} ({len(combos)} total)")
    return combos

# ✅ Function to process a single combination
def process_combination(params):
    power_x0, power_mu, power_x0_range, power_mu_range = params
    lock = Lock()
    start_time = time.perf_counter()

    alpha = GF.primitive_element
    Xn = alpha ** power_x0
    R = alpha ** power_mu
    one_ele = alpha ** (2 ** m - 1)

    sequence = []
    unique_powers = set()

    while True:
        int_value = int(Xn)
        if m < 25:
            power_repr = gf_table.get(int_value, -1)
        else:
            power_repr = get_power_representation(int_value) or -1

        if power_repr in unique_powers:
            break

        sequence.append(int_value)
        unique_powers.add(power_repr)
        Xn = Xn * R * (Xn + one_ele)

    binary_sequence = [int(x) % 2 for x in sequence]
    balance = calculate_balance(binary_sequence)
    unique_vals = compute_difference(binary_sequence)
    unique_vals = [abs(val) for val in unique_vals]
    acr_max = max(set(unique_vals[1:])) if len(unique_vals) > 1 else 0
    execution_time = time.perf_counter() - start_time

    result = {
        "power_x0": int(power_x0),
        "power_mu": int(power_mu),
        "period": int(len(binary_sequence)),
        "acr_max": int(acr_max),
        "balance": int(balance)
    }

    file_name = f"{folder_name}/seq_out(m_{m})_{power_x0_range[0]}_to_{power_x0_range[-1]}_to_{power_mu_range[0]}_to_{power_mu_range[-1]}.txt"
    with lock:
        with open(file_name, "a") as file:
            file.write(json.dumps(result) + "\n")

    return result

# ✅ Main execution
if __name__ == "__main__":
    start_timestamp = datetime.now()
    print(f"⏳ Code execution started at {start_timestamp.strftime('%I:%M %p on %d/%m/%Y')}")
    total_start = time.perf_counter()

    pieces = create_pieces(num_pieces)
    all_combinations = generate_all_combinations(num_pieces)
    selected_combos = load_selected_combinations_from_range(start_index, end_index)

    manager = Manager()
    queue = manager.Queue()
    data_list = []

    for selected_piece_x0, selected_piece_mu in selected_combos:
        piece_start_x0, piece_end_x0 = pieces[selected_piece_x0 - 1]
        piece_start_mu, piece_end_mu = pieces[selected_piece_mu - 1]

        power_x0_range = range(piece_start_x0, piece_end_x0 + 1)
        power_mu_range = range(piece_start_mu, piece_end_mu + 1)

        file_name_base = f"{folder_name}/seq_out(m_{m})_{piece_start_x0}_to_{piece_end_x0}_to_{piece_start_mu}_to_{piece_end_mu}"
        if not os.path.exists(file_name_base + ".txt"):
            with open(file_name_base + ".txt", "w") as file:
                file.write(f"Field order m: {m}\n")
                file.write(f"power_x0 range: {piece_start_x0} to {piece_end_x0}\n")
                file.write(f"power_mu range: {piece_start_mu} to {piece_end_mu}\n")
                print("txt file initialized")

        param_list = [(x0, mu, power_x0_range, power_mu_range) for x0 in power_x0_range for mu in power_mu_range]

        with Pool(processes=num_cores) as pool:
            results = pool.map(process_combination, param_list)

        disp_timestamp = datetime.now()
        print(f"⏳ DataFrame time starts now {disp_timestamp.strftime('%I:%M %p on %d/%m/%Y')}")

        df = pd.DataFrame((results), columns=["power_x0", "power_mu", "period", "acr_max", "balance"])
        #df.to_parquet(file_name_base + ".parquet", engine="pyarrow", compression="snappy")
        #df.to_excel(file_name_base + ".xlsx", index=False)
        #print(f"✅ Data saved: {file_name_base}.parquet / .xlsx")

        #max_rows_per_file = 1_000_000
        #for i, start in enumerate(range(0, len(df), max_rows_per_file)):
        #    chunk = df.iloc[start:start + max_rows_per_file]
        #   chunk_filename = f"{file_name_base}_part{i + 1}.xlsx"
        #    chunk.to_excel(chunk_filename, index=False)
        #    print(f"✅ Saved {chunk_filename} ({len(chunk)} rows)")

        #df_filtered = df.loc[df.groupby("period")["acr_max"].idxmin()]
        #df_filtered.to_parquet(file_name_base.replace("seq_out", "filtered_seq_out") + ".parquet", engine="pyarrow", compression="snappy")
        #df_filtered.to_excel(file_name_base.replace("seq_out", "filtered_seq_out") + ".xlsx", index=False)
        #print("✅ Filtered Parquet successfully converted to Excel!")
        disp_timestamp = datetime.now()
        print(f"⏳ Completed one run of for loop now {disp_timestamp.strftime('%I:%M %p on %d/%m/%Y')}")

    total_time = time.perf_counter() - total_start
    print(f"\n✅ All computations completed in {total_time:.4f} seconds.")
    dispall_timestamp = datetime.now()
    print(f"⏳ Completed all runs now {dispall_timestamp.strftime('%I:%M %p on %d/%m/%Y')}")