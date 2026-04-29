import numpy as np
import pandas as pd
import time
import os

CSV_FILE = r"C:\Users\Dileep D\Downloads\length.csv"
REPORT_FILE = "GA_Weil_ALL_Results.txt"
CSV_SUMMARY = "GA_Weil_Results_Summary.csv"


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


def run_ga_batch():
    lengths = load_lengths_smart()
    excel_results = []

    with open(REPORT_FILE, "w", encoding='utf-8') as f:
        f.write("GENETIC ALGORITHM | WEIL SEED COMPREHENSIVE REPORT\n")
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

                # EXACT SAME GA LOGIC FROM YOUR ORIGINAL
                pop = [np.roll(initial_seq, i * 5) for i in range(12)]

                print(f"Prime: {p} | Root: {g} | Shift: {shift} | Init PSL: {init_psl}")

                for gen in range(60):
                    # Fitness sorting
                    pop.sort(key=lambda x: safe_psl(
                        np.round(np.real(np.fft.ifft(np.fft.fft(x) * np.conj(np.fft.fft(x))))).astype(float), n))

                    if gen % 10 == 0:
                        current_best = safe_psl(
                            np.round(np.real(np.fft.ifft(np.fft.fft(pop[0]) * np.conj(np.fft.fft(pop[0]))))).astype(
                                float), n)
                        print(f"  Gen {gen}: Best PSL = {current_best}")

                    next_gen = pop[:2]  # Elitism - keep top 2

                    while len(next_gen) < 12:
                        p1, p2 = pop[0], pop[np.random.randint(1, 4)]
                        cp = np.random.randint(n)
                        child = np.concatenate([p1[:cp], p2[cp:]])
                        if np.random.rand() < 0.2:  # 20% mutation
                            child[np.random.randint(n)] *= -1
                        next_gen.append(child)

                    pop = next_gen

                # Final best
                final_seq = pop[0]
                f_final = np.fft.fft(final_seq)
                acf_final = np.round(np.real(np.fft.ifft(f_final * np.conj(f_final)))).astype(float)
                final_psl = safe_psl(acf_final, n)

                f.write(f"\n{'#' * 80}\n")
                f.write(f"N = {n}\n")
                f.write(f"Base Prime: {p} | Root: {g} | Shift: {shift}\n")
                f.write(f"Initial PSL: {init_psl} --> Final PSL: {final_psl}\n")
                f.write(f"Generations: 60 | Population: 12 | Elitism: Top 2 | Mutation: 20%\n")
                f.write(f"{'#' * 80}\n")

                write_wrapped_bits(f, initial_seq, "--- INITIAL WEIL SEED ---")
                write_wrapped_bits(f, final_seq, "--- OPTIMIZED SEQUENCE ---")

                excel_results.append({
                    'Length': n,
                    'Base_Prime': p,
                    'Primitive_Root': g,
                    'Shift': shift,
                    'Initial_PSL': init_psl,
                    'Optimized_PSL': final_psl,
                    'Improvement': init_psl - final_psl
                })

                print(f"N={n}: {init_psl} → {final_psl} (Improve: {init_psl - final_psl})")

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
    print("🚀 Genetic Algorithm (WEIL SEED) - Batch Processing")
    start_time = time.time()
    run_ga_batch()
    print(f"\n⏱️ TOTAL TIME: {time.time() - start_time:.1f}s")
    print("🎉 COMPLETE!")