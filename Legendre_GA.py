import numpy as np
import pandas as pd
import os

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"
REPORT_FILE = "LGA_ALL_Results.txt"
CSV_SUMMARY = "LGA_Results_Summary.csv"


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
    print(f"GA: {len(unique_lengths)} unique lengths")
    return unique_lengths


def run_ga_batch():
    lengths = load_lengths_smart()
    results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("GENETIC ALGORITHM REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input CSV: {CSV_FILE}\n")

        for n in lengths:
            print(f"\nProcessing N={n}")
            initial_seed, p = get_unified_seed(n)
            init_psl = safe_psl(initial_seed)

            pop_size = min(10, n)
            population = [np.roll(initial_seed, i * (n // pop_size)) for i in range(pop_size)]

            generations = min(60, n * 2)
            for gen in range(generations):
                population.sort(key=safe_psl)
                next_gen = population[:2]
                while len(next_gen) < pop_size:
                    p1, p2 = population[0], population[np.random.randint(1, min(4, len(population)))]
                    cp = np.random.randint(n)
                    child = np.concatenate([p1[:cp], p2[cp:]])
                    if np.random.rand() < 0.15:
                        child[np.random.randint(n)] *= -1
                    next_gen.append(child)
                population = next_gen

            final_seq = population[0]
            final_psl = safe_psl(final_seq)

            f.write(f"\n{'#' * 80}\n")
            f.write(f"N = {n} | Prime: {p}\n")
            f.write(f"Initial PSL: {init_psl} --> Final: {final_psl}\n")
            f.write(f"{'#' * 80}\n")
            write_wrapped_bits(f, initial_seed, "--- INITIAL SEED ---")
            write_wrapped_bits(f, final_seq, "--- OPTIMIZED ---")

            results.append({
                'Length': n, 'Base_Prime': p, 'Initial_PSL': init_psl,
                'Optimized_PSL': final_psl, 'Improvement': init_psl - final_psl
            })
            print(f"N={n}: {init_psl} --> {final_psl}")

    print(f"\n📄 TEXT: {REPORT_FILE}")

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(CSV_SUMMARY, index=False)
    print(f"📊 CSV: {CSV_SUMMARY}")
    print(summary_df)


if __name__ == "__main__":
    run_ga_batch()