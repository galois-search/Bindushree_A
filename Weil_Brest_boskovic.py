import numpy as np
import pandas as pd
import time
import os

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"  # Updated to .csv
REPORT_FILE = "BrestBoskovic_Weil_ALL_Results.txt"
CSV_SUMMARY = "BrestBoskovic_Weil_Results_Summary.csv"


def get_best_weil_seed(n):
    def is_prime(num):
        if num < 2: return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0: return False
        return True

    def get_primitive_roots(p):
        roots = [];
        phi = p - 1;
        temp = phi;
        factors = [];
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0: temp //= d
            d += 1
        if temp > 1: factors.append(temp)
        for res in range(2, p):
            ok = True
            for f in factors:
                if pow(res, phi // f, p) == 1:
                    ok = False;
                    break
            if ok: roots.append(res)
        return roots

    p = n
    while not is_prime(p): p += 1
    roots = get_primitive_roots(p)
    best_psl = float('inf');
    best_seed = None;
    best_g = None
    for g in roots:
        s = np.ones(p)
        for i in range(p):
            val = (pow(i, g, p) + 1) % p
            s[i] = 1 if (val == 0 or pow(val, (p - 1) // 2, p) == 1) else -1
        cand = np.roll(s[:n], n // 4)
        curr_psl = int(
            np.max(np.abs(np.round(np.real(np.fft.ifft(np.fft.fft(cand) * np.conj(np.fft.fft(cand)))))[1:n // 2 + 1])))
        if curr_psl < best_psl:
            best_psl, best_seed, best_g = curr_psl, cand.astype(float), g
    return best_seed, p, best_g, n // 4


def safe_psl(acf, n):
    if n <= 2: return 0
    sidelobes = slice(1, n // 2 + 1)
    return int(np.max(np.abs(acf[sidelobes])))


def write_wrapped_bits(f, seq, label):
    f.write(f"\n{label}\n")
    bin_str = "".join(['1' if x > 0 else '0' for x in seq])
    for i in range(0, len(bin_str), 100):
        f.write(bin_str[i:i + 100] + "\n")


def load_lengths_smart():
    print(f"🔍 Loading from: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)  # CSV reader
        print(f"📊 CSV loaded successfully!")
    except:
        df = pd.read_excel(CSV_FILE)  # Fallback to Excel
        print(f"📊 Excel loaded successfully!")

    print(f"📊 Columns: {list(df.columns)}")
    col_name = df.columns[0]
    all_lengths = pd.to_numeric(df[col_name], errors='coerce').dropna().astype(int).tolist()
    unique_lengths = sorted(list(set(all_lengths)))

    print(f"📈 All lengths ({len(all_lengths)}): {all_lengths}")
    print(f"🎯 Unique lengths ({len(unique_lengths)}): {unique_lengths}")
    return unique_lengths


def run_brest_boskovic_batch():
    lengths = load_lengths_smart()
    excel_results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("BREST-BOSKOVIC | WEIL SEED COMPREHENSIVE REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input: {CSV_FILE}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Unique Lengths: {lengths}\n\n")
        f.write("=" * 80 + "\n\n")

        for n in lengths:
            print(f"\n{'=' * 60}")
            print(f"Processing N={n}")

            try:
                initial_seq, p, g, shift = get_best_weil_seed(n)
                f_init = np.fft.fft(initial_seq)
                acf = np.round(np.real(np.fft.ifft(f_init * np.conj(f_init)))).astype(float)
                init_psl = safe_psl(acf, n)

                seq = initial_seq.copy()
                best_psl = init_psl
                best_seq = seq.copy()
                gamma = 4.0

                print(f"Prime: {p} | Root: {g} | Shift: {shift} | Init PSL: {init_psl}")

                max_steps = min(300001, n * 10000)
                for step in range(1, max_steps):
                    idx = np.random.randint(0, n)
                    diff = -2 * seq[idx]
                    indices_minus = (idx - np.arange(n)) % n
                    indices_plus = (idx + np.arange(n)) % n
                    delta = diff * (seq[indices_minus] + seq[indices_plus])

                    sidelobes = slice(1, n // 2 + 1)
                    if n > 2 and np.sum(np.abs(acf[sidelobes] + delta[sidelobes]) ** gamma) < np.sum(
                            np.abs(acf[sidelobes]) ** gamma):
                        seq[idx] *= -1
                        acf += delta
                        curr_psl = safe_psl(acf, n)
                        if curr_psl < best_psl:
                            best_psl = curr_psl
                            best_seq = seq.copy()
                            gamma = min(18, gamma + 0.1)

                f.write(f"\n{'#' * 80}\n")
                f.write(f"N = {n}\n")
                f.write(f"Base Prime: {p} | Primitive Root: {g} | Shift: {shift}\n")
                f.write(f"Initial PSL: {init_psl} --> Final PSL: {best_psl}\n")
                f.write(f"Status: {'Trivial' if n <= 3 else 'Optimized'}\n")
                f.write(f"{'#' * 80}\n")

                write_wrapped_bits(f, initial_seq, "--- INITIAL WEIL SEED ---")
                write_wrapped_bits(f, best_seq, "--- OPTIMIZED SEQUENCE ---")

                excel_results.append({
                    'Length': n,
                    'Base_Prime': p,
                    'Primitive_Root': g,
                    'Shift': shift,
                    'Initial_PSL': init_psl,
                    'Optimized_PSL': best_psl,
                    'Improvement': init_psl - best_psl
                })

                print(f"N={n}: {init_psl} --> {best_psl}")

            except Exception as e:
                print(f"Error N={n}: {e}")
                excel_results.append({
                    'Length': n, 'Base_Prime': 'ERROR', 'Primitive_Root': 'ERROR',
                    'Shift': 'ERROR', 'Initial_PSL': 'ERROR', 'Optimized_PSL': 'ERROR',
                    'Improvement': 0
                })

        f.write(f"\n{'=' * 100}\nSUMMARY TABLE\n{'=' * 100}\n")
        f.write(f"N      | Prime   | Root | Shift | Init PSL | Opt PSL | Improve\n")
        f.write("-" * 100 + "\n")
        for row in excel_results:
            n, p, g, shift, init, opt, imp = row.values()
            f.write(f"{n:6} | {p:8} | {g:4} | {shift:5} | {init:7} | {opt:7} | {imp:7}\n")

    summary_df = pd.DataFrame(excel_results)
    summary_df.to_csv(CSV_SUMMARY, index=False)

    print(f"\n✅ TEXT REPORT: {REPORT_FILE}")
    print(f"✅ CSV SUMMARY: {CSV_SUMMARY}")
    print("\n📋 CSV Preview:")
    print(summary_df.head())


if __name__ == "__main__":
    print("🚀 Brest-Boskovic (WEIL SEED) - Text + CSV Summary")
    start_time = time.time()
    run_brest_boskovic_batch()
    print(f"\n⏱️ Total Time: {time.time() - start_time:.1f}s")
    print("✅ COMPLETE!")