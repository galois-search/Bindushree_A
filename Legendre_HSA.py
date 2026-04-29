import numpy as np
import pandas as pd
import math
import os

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"
REPORT_FILE = "LHybridSA_ALL_Results.txt"
CSV_SUMMARY = "LHybridSA_Results_Summary.csv"


def get_unified_seed(n):
    p = n
    while not all(p % i != 0 for i in range(2, int(p ** 0.5) + 1)): p += 1
    s = np.ones(p)
    for i in range(1, p):
        if pow(i, (p - 1) // 2, p) != 1: s[i] = -1
    return np.roll(s[:n], n // 4).astype(float), p


def safe_psl(x):
    n = len(x)
    if n <= 2:
        return 0
    acf = np.round(np.real(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x)))))
    return int(np.max(np.abs(acf[1:n // 2 + 1])))


def write_wrapped_bits(f, seq, label):
    f.write(f"\n{label}\n")
    bin_str = "".join(['1' if x > 0 else '0' for x in seq])
    for i in range(0, len(bin_str), 100):
        f.write(bin_str[i:i + 100] + "\n")


def load_lengths_smart():
    df = pd.read_csv(CSV_FILE)
    col_name = df.columns[0]
    all_lengths = pd.to_numeric(df[col_name], errors='coerce').dropna().astype(int).tolist()
    unique_lengths = sorted(list(set(all_lengths)))
    print(f"Hybrid SA: {len(unique_lengths)} unique lengths")
    return unique_lengths


def run_hybrid_sa_batch():
    lengths = load_lengths_smart()
    results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("HYBRID SIMULATED ANNEALING REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input CSV: {CSV_FILE}\n")

        for n in lengths:
            print(f"\nProcessing N={n}")
            initial_seq, p = get_unified_seed(n)
            init_psl = safe_psl(initial_seq)

            current = initial_seq.copy()
            cur_psl = init_psl
            best_s = initial_seq.copy()
            best_psl = init_psl
            temp = 45.0

            max_iter = min(15000, n * 20)
            for i in range(max_iter):
                new = current.copy()
                idx = np.random.randint(0, n, size=min(4, n))
                new[idx] *= -1
                new_psl = safe_psl(new)

                if new_psl < cur_psl or np.random.rand() < math.exp((cur_psl - new_psl) / temp):
                    current, cur_psl = new, new_psl
                    if cur_psl < best_psl:
                        best_psl, best_s = cur_psl, current.copy()
                temp *= 0.9998

            f.write(f"\n{'#' * 80}\n")
            f.write(f"N = {n} | Prime: {p}\n")
            f.write(f"Initial PSL: {init_psl} --> Final: {best_psl}\n")
            f.write(f"{'#' * 80}\n")
            write_wrapped_bits(f, initial_seq, "--- INITIAL ---")
            write_wrapped_bits(f, best_s, "--- OPTIMIZED ---")

            results.append({
                'Length': n, 'Base_Prime': p, 'Initial_PSL': init_psl,
                'Optimized_PSL': best_psl, 'Improvement': init_psl - best_psl
            })
            print(f"N={n}: {init_psl} --> {best_psl}")

    print(f"\n📄 TEXT: {REPORT_FILE}")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(CSV_SUMMARY, index=False)
    print(f"📊 CSV: {CSV_SUMMARY}")
    print(summary_df)


if __name__ == "__main__":
    run_hybrid_sa_batch()