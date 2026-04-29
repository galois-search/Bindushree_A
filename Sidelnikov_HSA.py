import numpy as np
import pandas as pd
import time
import os
import math

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"
REPORT_FILE = "Sidelnikov_Hybrid_SA_ALL_Results.txt"
CSV_SUMMARY = "Sidelnikov_Hybrid_SA_Summary.csv"


def get_best_sidelnikov_seed(n):
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
    all_roots = get_primitive_roots(p)
    roots_to_scan = all_roots[:25]

    best_psl = float('inf');
    best_seed = None;
    best_alpha = None

    for alpha in roots_to_scan:
        s = np.ones(p)
        log_table = {pow(alpha, i, p): i for i in range(p - 1)}
        for i in range(p - 1):
            val = (pow(alpha, i, p) + 1) % p
            if val == 0:
                s[i] = -1
            else:
                exponent = log_table[val]
                s[i] = 1 if (exponent % 2 == 0) else -1

        cand = np.roll(s[:n], n // 4)
        f = np.fft.fft(cand)
        acf = np.round(np.real(np.fft.ifft(f * np.conj(f)))).astype(int)
        curr_psl = int(np.max(np.abs(acf[1:n // 2 + 1])))
        if curr_psl < best_psl:
            best_psl, best_seed, best_alpha = curr_psl, cand.astype(float), alpha

    return best_seed, p, best_alpha, n // 4


def safe_psl(x, n):
    f = np.fft.fft(x)
    acf = np.round(np.real(np.fft.ifft(f * np.conj(f)))).astype(int)
    return int(np.max(np.abs(acf[1:n // 2 + 1])))


def write_wrapped_bits(f, seq, label):
    f.write(f"\n{label}\n")
    bin_str = "".join(['1' if x > 0 else '0' for x in seq])
    for i in range(0, len(bin_str), 100):
        f.write(bin_str[i:i + 100] + "\n")


def load_lengths_smart():
    print(f"🔍 Loading from: {CSV_FILE}")
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"📊 CSV loaded!")
    except:
        df = pd.read_excel(CSV_FILE, engine='openpyxl')
        print(f"📊 Excel loaded!")

    col_name = df.columns[0]
    all_lengths = pd.to_numeric(df[col_name], errors='coerce').dropna().astype(int).tolist()
    unique_lengths = sorted(list(set([n for n in all_lengths if n > 2])))

    print(f"📈 All: {len(all_lengths)} | Valid N>2: {len(unique_lengths)}: {unique_lengths}")
    return unique_lengths


def run_hybrid_sa_batch():
    lengths = load_lengths_smart()
    excel_results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("HYBRID SIMULATED ANNEALING | SIDELNIKOV SEED BATCH REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Input: {CSV_FILE}\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Lengths: {lengths}\n\n")

        for n in lengths:
            print(f"\n{'=' * 60}\nProcessing N={n}")
            try:
                seed, p, alpha, shift = get_best_sidelnikov_seed(n)
                init_psl = safe_psl(seed, n)

                # EXACT SAME OPTIMIZATION LOGIC
                curr = seed.copy();
                cur_psl = init_psl
                best_s = seed.copy();
                best_psl = init_psl
                temp = 35.0

                print(f"Alpha: {alpha} | Prime: {p} | Init PSL: {init_psl}")

                for i in range(20000):
                    new = curr.copy()
                    new[np.random.randint(n)] *= -1

                    n_psl = safe_psl(new, n)
                    delta = n_psl - cur_psl

                    if delta < 0 or np.random.rand() < math.exp(-delta / temp):
                        curr, cur_psl = new, n_psl
                        if cur_psl < best_psl:
                            best_psl, best_s = cur_psl, curr.copy()

                temp *= 0.9997

                f.write(f"\n{'#' * 80}\n")
                f.write(f"N={n} | Prime={p} | Alpha={alpha} | Shift={shift}\n")
                f.write(f"Initial PSL: {init_psl} → Final: {best_psl}\n")
                write_wrapped_bits(f, seed, "--- INITIAL SIDELNIKOV (Alpha {}) ---".format(alpha))
                write_wrapped_bits(f, best_s, "--- OPTIMIZED SEQUENCE ---")

                excel_results.append({
                    'Length': n, 'Base_Prime': p, 'Alpha': alpha, 'Shift': shift,
                    'Initial_PSL': init_psl, 'Optimized_PSL': best_psl, 'Improvement': init_psl - best_psl
                })
                print(f"N={n}: {init_psl} → {best_psl}")

            except Exception as e:
                print(f"Error N={n}: {e}")
                excel_results.append({'Length': n, 'Base_Prime': 'ERROR', 'Alpha': 'ERROR', 'Shift': 'ERROR',
                                      'Initial_PSL': 0, 'Optimized_PSL': 0, 'Improvement': 0})

        # SUMMARY
        f.write(f"\n{'=' * 100}\nSUMMARY\n{'=' * 100}\n")
        f.write("N | Prime | Alpha | Shift | Init | Opt | Improve\n")
        for row in excel_results:
            vals = list(row.values())
            f.write(
                f"{vals[0]:3} | {vals[1]:6} | {vals[2]:5} | {vals[3]:5} | {vals[4]:4} | {vals[5]:4} | {vals[6]:7}\n")

    pd.DataFrame(excel_results).to_csv(CSV_SUMMARY, index=False)
    print(f"\n✅ REPORT: {REPORT_FILE}\n✅ SUMMARY: {CSV_SUMMARY}")


if __name__ == "__main__":
    print("🚀 Hybrid Simulated Annealing (SIDELNIKOV SEED)")
    start_time = time.time()
    run_hybrid_sa_batch()
    print(f"⏱️ {time.time() - start_time:.1f}s")