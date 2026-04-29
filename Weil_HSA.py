import numpy as np
import pandas as pd
import time
import math
import os

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"
REPORT_FILE = "Hybrid_SA_Weil_ALL_Results.txt"
CSV_SUMMARY = "Hybrid_SA_Weil_Results_Summary.csv"


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
    if n <= 2:
        return 0
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
        df = pd.read_csv(CSV_FILE)
        print(f"📊 CSV loaded successfully!")
    except Exception as e:
        print(f"CSV failed: {e}. Trying Excel...")
        try:
            df = pd.read_excel(CSV_FILE, engine='openpyxl')
            print(f"📊 Excel loaded successfully!")
        except:
            raise ValueError(f"Cannot load {CSV_FILE}")

    print(f"📊 Columns: {list(df.columns)}")
    col_name = df.columns[0]
    all_lengths = pd.to_numeric(df[col_name], errors='coerce').dropna().astype(int).tolist()
    unique_lengths = sorted(list(set(all_lengths)))

    print(f"📈 All lengths ({len(all_lengths)}): {all_lengths}")
    print(f"🎯 Unique lengths ({len(unique_lengths)}): {unique_lengths}")
    return unique_lengths


def run_sa_batch():
    lengths = load_lengths_smart()
    excel_results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("HYBRID SIMULATED ANNEALING | WEIL SEED REPORT\n")
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
                acf_init = np.round(np.real(np.fft.ifft(f_init * np.conj(f_init)))).astype(float)
                init_psl = safe_psl(acf_init, n)

                # EXACT SAME HYBRID SA LOGIC FROM YOUR ORIGINAL
                curr = initial_seq.copy()
                cur_psl = init_psl
                best_s = curr.copy()
                best_psl = init_psl
                temp = 40.0

                print(f"Prime: {p} | Root: {g} | Shift: {shift} | Init PSL: {init_psl}")

                improvements = 0
                for i in range(20000):
                    new = curr.copy()
                    if np.random.rand() < 0.8:  # Single flip (80%)
                        new[np.random.randint(n)] *= -1
                    else:  # Block flip (20%)
                        sz = np.random.randint(2, min(8, n))
                        st = np.random.randint(0, n - sz)
                        new[st:st + sz] *= -1

                    f_new = np.fft.fft(new)
                    acf_new = np.round(np.real(np.fft.ifft(f_new * np.conj(f_new)))).astype(float)
                    n_psl = safe_psl(acf_new, n)

                    if n_psl < cur_psl or np.random.rand() < math.exp((cur_psl - n_psl) / temp):
                        curr = new.copy()
                        cur_psl = n_psl
                        if cur_psl < best_psl:
                            best_psl = cur_psl
                            best_s = curr.copy()
                            improvements += 1
                            if improvements % 100 == 0:
                                print(f"  Iter {i}: Best PSL = {best_psl}")
                    temp *= 0.9998

                f.write(f"\n{'#' * 80}\n")
                f.write(f"N = {n}\n")
                f.write(f"Base Prime: {p} | Root: {g} | Shift: {shift}\n")
                f.write(f"Initial PSL: {init_psl} --> Final PSL: {best_psl}\n")
                f.write(f"Iterations: 20000 | Initial Temp: 40.0 | Cooling: 0.9998\n")
                f.write(f"{'#' * 80}\n")

                write_wrapped_bits(f, initial_seq, "--- INITIAL WEIL SEED ---")
                write_wrapped_bits(f, best_s, "--- OPTIMIZED SEQUENCE ---")

                excel_results.append({
                    'Length': n,
                    'Base_Prime': p,
                    'Primitive_Root': g,
                    'Shift': shift,
                    'Initial_PSL': init_psl,
                    'Optimized_PSL': best_psl,
                    'Improvement': init_psl - best_psl
                })

                print(f"N={n}: {init_psl} → {best_psl} (Improve: {init_psl - best_psl})")

            except Exception as e:
                print(f"❌ Error N={n}: {e}")
                excel_results.append({
                    'Length': n, 'Base_Prime': 'ERROR', 'Primitive_Root': 'ERROR',
                    'Shift': 'ERROR', 'Initial_PSL': 'ERROR', 'Optimized_PSL': 'ERROR',
                    'Improvement': 0
                })

        # SUMMARY TABLE
        f.write(f"\n{'=' * 120}\nSUMMARY TABLE\n{'=' * 120}\n")
        f.write(f"N\t|\tPrime\t|\tRoot\t|\tShift\t|\tInit PSL\t|\tOpt PSL\t|\tImprove\n")
        f.write("-" * 120 + "\n")
        for row in excel_results:
            vals = [str(row[k]) for k in
                    ['Length', 'Base_Prime', 'Primitive_Root', 'Shift', 'Initial_PSL', 'Optimized_PSL', 'Improvement']]
            f.write("\t|\t".join(vals) + "\n")

    summary_df = pd.DataFrame(excel_results)
    summary_df.to_csv(CSV_SUMMARY, index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ TEXT REPORT: {REPORT_FILE}")
    print(f"✅ CSV SUMMARY: {CSV_SUMMARY}")
    print(f"📊 Processed {len(lengths)} lengths")
    print("\n📋 Preview:")
    print(summary_df.head().to_string(index=False))


if __name__ == "__main__":
    print("🚀 Hybrid SA (WEIL SEED) - Batch Processing")
    start_time = time.time()
    run_sa_batch()
    print(f"\n⏱️ TOTAL TIME: {time.time() - start_time:.1f}s")
    print("🎉 COMPLETE!")