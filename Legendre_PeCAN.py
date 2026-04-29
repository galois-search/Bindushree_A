import numpy as np
import pandas as pd
import os

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"
REPORT_FILE = "LPeCAN_LNS_ALL_Results.txt"
CSV_SUMMARY = "LPeCAN_LNS_Results_Summary.csv"


def get_unified_seed(n):
    p = n
    while not all(p % i != 0 for i in range(2, int(p ** 0.5) + 1)): p += 1
    s = np.ones(p)
    for i in range(1, p):
        if pow(i, (p - 1) // 2, p) != 1: s[i] = -1
    seed = np.roll(s[:n], n // 4)
    return seed.astype(float), p


def get_metrics(s):
    n = len(s)
    S = np.fft.fft(s)
    acf = np.real(np.fft.ifft(S * np.conj(S)))
    rounded_acf = np.round(acf).astype(int)
    if n <= 2:
        return 0, rounded_acf
    psl = np.max(np.abs(rounded_acf[1:n // 2 + 1]))
    return int(psl), rounded_acf


def write_wrapped_bits(f, seq, label):
    f.write(f"\n{label}\n")
    bin_str = "".join(['1' if x == 1 else '0' for x in seq])
    for i in range(0, len(bin_str), 100):
        f.write(bin_str[i:i + 100] + "\n")


def load_lengths_smart():
    df = pd.read_csv(CSV_FILE)
    col_name = df.columns[0]
    all_lengths = pd.to_numeric(df[col_name], errors='coerce').dropna().astype(int).tolist()
    unique_lengths = sorted(list(set(all_lengths)))
    print(f"PeCAN+LNS: {len(unique_lengths)} unique lengths: {unique_lengths}")
    return unique_lengths


def run_pecan_lns_batch():
    lengths = load_lengths_smart()
    results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("PeCAN + LNS COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input CSV: {CSV_FILE}\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lengths: {lengths}\n\n")

        for n in lengths:
            print(f"\nProcessing N={n}")
            initial_seq, p = get_unified_seed(n)
            init_psl, _ = get_metrics(initial_seq)

            current_s = initial_seq.copy()
            best_s = initial_seq.copy()
            best_psl = init_psl
            stagnation = 0

            max_iter = min(1000, n * 50)
            for iteration in range(1, max_iter + 1):
                S = np.fft.fft(current_s)
                noise = (stagnation / max_iter) * 0.1
                target_mag = np.sqrt(n) + np.random.normal(0, noise, n)
                S_ideal = target_mag * np.exp(1j * np.angle(S))

                s_cont = np.real(np.fft.ifft(S_ideal))
                candidate_s = np.where(s_cont >= 0, 1.0, -1.0)

                curr_psl, _ = get_metrics(candidate_s)

                if curr_psl < best_psl:
                    best_psl = curr_psl
                    best_s = candidate_s.copy()
                    stagnation = 0
                else:
                    stagnation += 1

                if stagnation > 200 and stagnation % 50 == 0 and n > 10:
                    uncertainty = np.abs(s_cont)
                    threshold = np.percentile(uncertainty, 1)
                    candidate_s[uncertainty < threshold] *= -1

                current_s = candidate_s

            f.write(f"\n{'#' * 80}\n")
            f.write(f"N = {n} | Base Prime: {p}\n")
            f.write(f"Initial PSL: {init_psl} --> Final PSL: {best_psl}\n")
            f.write(f"{'#' * 80}\n")
            write_wrapped_bits(f, initial_seq, "--- INITIAL SEQUENCE ---")
            write_wrapped_bits(f, best_s, "--- OPTIMIZED SEQUENCE ---")

            results.append({
                'Length': n, 'Base_Prime': p, 'Initial_PSL': init_psl,
                'Optimized_PSL': best_psl, 'Improvement': init_psl - best_psl
            })
            print(f"N={n}: {init_psl} --> {best_psl}")

        f.write(f"\nSUMMARY:\n")
        for r in results:
            f.write(f"N={r['Length']:2}: {r['Initial_PSL']:2} --> {r['Optimized_PSL']:2}\n")

    print(f"\n📄 TEXT: {REPORT_FILE}")

    # CSV SUMMARY
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(CSV_SUMMARY, index=False)
    print(f"📊 CSV SUMMARY: {CSV_SUMMARY}")
    print(summary_df)


if __name__ == "__main__":
    run_pecan_lns_batch()