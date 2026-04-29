import numpy as np
import pandas as pd
import time
import os

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"  # FIXED: Your CSV path
REPORT_FILE = "PeCAN_LNS_Weil_ALL_Results.txt"
CSV_SUMMARY = "PeCAN_LNS_Weil_Results_Summary.csv"


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
        df = pd.read_csv(CSV_FILE)  # TRY CSV FIRST ✅
        print(f"📊 CSV loaded successfully!")
    except Exception as e:
        print(f"CSV failed: {e}. Trying Excel...")
        try:
            df = pd.read_excel(CSV_FILE, engine='openpyxl')  # FALLBACK
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


def run_pecan_lns_batch():
    lengths = load_lengths_smart()
    excel_results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("PECAN + LNS | WEIL SEED COMPREHENSIVE REPORT\n")
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

                # EXACT SAME PECAN+LNS LOGIC FROM YOUR ORIGINAL
                curr = initial_seq.copy()
                best_s = curr.copy()
                best_psl = init_psl
                stagnation = 0

                print(f"Prime: {p} | Root: {g} | Shift: {shift} | Init PSL: {init_psl}")

                for i in range(1, 1001):
                    S = np.fft.fft(curr)
                    noise = (stagnation / 1000) * 0.1
                    S_ideal = (np.sqrt(n) + np.random.normal(0, noise, n)) * np.exp(1j * np.angle(S))
                    s_cont = np.real(np.fft.ifft(S_ideal))
                    cand = np.where(s_cont >= 0, 1.0, -1.0)

                    f_cand = np.fft.fft(cand)
                    acf_cand = np.round(np.real(np.fft.ifft(f_cand * np.conj(f_cand)))).astype(float)
                    c_psl = safe_psl(acf_cand, n)

                    if c_psl < best_psl:
                        best_psl = c_psl
                        best_s = cand.copy()
                        stagnation = 0
                        print(f"  Iter {i}: PSL improved to {best_psl}")
                    else:
                        stagnation += 1

                    if stagnation > 100 and stagnation % 50 == 0:
                        low_percentile = np.percentile(np.abs(s_cont), 1)
                        cand[np.abs(s_cont) < low_percentile] *= -1
                    curr = cand

                # WRITE RESULTS
                f.write(f"\n{'#' * 80}\n")
                f.write(f"N = {n}\n")
                f.write(f"Base Prime: {p} | Primitive Root: {g} | Shift: {shift}\n")
                f.write(f"Initial PSL: {init_psl} --> Final PSL: {best_psl}\n")
                f.write(f"Iterations: 1000 | Stagnation threshold: 100\n")
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
        f.write(f"\n{'=' * 120}\n")
        f.write("SUMMARY TABLE\n")
        f.write(f"{'=' * 120}\n")
        f.write(f"N\t|\tPrime\t|\tRoot\t|\tShift\t|\tInit PSL\t|\tOpt PSL\t|\tImprove\n")
        f.write("-" * 120 + "\n")
        for row in excel_results:
            n, p, g, sh, init, opt, imp = [str(row[k]) for k in
                                           ['Length', 'Base_Prime', 'Primitive_Root', 'Shift', 'Initial_PSL',
                                            'Optimized_PSL', 'Improvement']]
            f.write(f"{n}\t|\t{p}\t|\t{g}\t|\t{sh}\t|\t{init}\t|\t{opt}\t|\t{imp}\n")

    # SAVE CSV
    summary_df = pd.DataFrame(excel_results)
    summary_df.to_csv(CSV_SUMMARY, index=False)

    print(f"\n{'=' * 60}")
    print(f"✅ TEXT REPORT SAVED: {REPORT_FILE}")
    print(f"✅ SUMMARY CSV SAVED: {CSV_SUMMARY}")
    print(f"📊 Processed {len(lengths)} unique lengths")
    print("\n📋 CSV Preview:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    print("🚀 PeCAN + LNS (WEIL SEED) - Batch Processing")
    start_time = time.time()
    run_pecan_lns_batch()
    print(f"\n⏱️  TOTAL TIME: {time.time() - start_time:.1f} seconds")
    print("🎉 ALL COMPLETE!")