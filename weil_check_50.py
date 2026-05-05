import numpy as np
import os


def get_first_50_weils(n):
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

    print(f"p={p}, Total Weils={len(roots)}")
    first_50_seqs = []
    first_50_g = []

    # Take FIRST 50 Weils (with their n//4 shift as original did)
    for g_idx in range(min(50, len(roots))):
        g = roots[g_idx]
        s = np.ones(p)
        for i in range(p):
            val = (pow(i, g, p) + 1) % p
            s[i] = 1 if (val == 0 or pow(val, (p - 1) // 2, p) == 1) else -1
        seq = np.roll(s[:n], n // 4).astype(float)  # Same shift as original
        first_50_seqs.append(seq)
        first_50_g.append(g)
        print(f"Weil {g_idx + 1}/50 (g={g}) generated")

    return first_50_seqs, p, first_50_g


def compute_psl(seq, n):
    return int(np.max(np.abs(np.round(np.real(np.fft.ifft(np.fft.fft(seq) * np.conj(np.fft.fft(seq)))))[1:n // 2 + 1])))


def compute_cc_matrix(sequences, n):
    """Compute cross-correlation matrix"""
    num_seqs = len(sequences)
    cc_matrix = np.zeros((num_seqs, num_seqs), dtype=int)
    for i in range(num_seqs):
        for j in range(num_seqs):
            if i == j:
                cc_matrix[i, j] = 0  # Diagonal = 0 (auto-correlation)
            else:
                diff_seq = sequences[i] - sequences[j]
                cc_matrix[i, j] = compute_psl(diff_seq, n)
    return cc_matrix


def run_brest_boskovic():
    n = int(input("Enter N: "))

    # Get FIRST 50 Weils out of 5121
    sequences, p, gs = get_first_50_weils(n)
    num_seqs = len(sequences)

    initial_psls = [compute_psl(seq, n) for seq in sequences]

    print(f"\nFirst {num_seqs} Weils from p={p}:")
    for i, (psl, g) in enumerate(zip(initial_psls, gs)):
        print(f"  Weil {i + 1} (g={g}): PSL={psl}")

    # FILE 1: INITIAL 50 WEILS
    with open(f"Weil_First50_N{n}_p{p}.txt", "w") as f:
        f.write(f"FIRST 50 WEIL SEQUENCES | N={n} | p={p}\n")
        f.write(f"Out of {len(gs)} total Weils (first 50 primitive roots)\n\n")
        for i, (seq, psl, g) in enumerate(zip(sequences, initial_psls, gs)):
            f.write(f"WEIL {i + 1} (g={g}) | PSL={psl} | shifted n//4\n")
            s_str = "".join(['1' if x > 0 else '0' for x in seq])
            for j in range(0, len(s_str), 100):
                f.write(s_str[j:j + 100] + "\n")
            f.write("\n")

    # OPTIMIZE ALL 50 (YOUR EXACT LOGIC)
    optimized_sequences = []
    optimized_psls = []

    print("\n=== OPTIMIZING FIRST 50 WEILS ===")
    for seq_idx in range(num_seqs):
        print(f"\nWeil {seq_idx + 1} (g={gs[seq_idx]}, Init PSL={initial_psls[seq_idx]})")

        # YOUR 100% UNCHANGED OPTIMIZATION
        seq = sequences[seq_idx].copy()
        init_psl = initial_psls[seq_idx]
        best_seq = seq.copy()
        best_psl = init_psl
        gamma = 4.0
        f_vec = np.fft.fft(seq)
        acf = np.round(np.real(np.fft.ifft(f_vec * np.conj(f_vec)))).astype(float)

        for step in range(1, 100001):
            idx = np.random.randint(0, n)
            diff = -2 * seq[idx]
            delta = diff * (seq[(idx - np.arange(n)) % n] + seq[(idx + np.arange(n)) % n])
            sl = slice(1, n // 2 + 1)
            if np.sum(np.abs(acf[sl] + delta[sl]) ** gamma) < np.sum(np.abs(acf[sl]) ** gamma):
                seq[idx] *= -1
                acf += delta
                curr_psl = int(np.max(np.abs(acf[sl])))
                if curr_psl < best_psl:
                    best_psl = curr_psl
                    best_seq = seq.copy()
                    gamma = min(18, gamma + 0.1)
                    if step % 20000 == 0:
                        print(f"  Step {step}: PSL={best_psl}")

        optimized_sequences.append(best_seq)
        optimized_psls.append(best_psl)
        print(f"✓ Weil {seq_idx + 1}: {init_psl} → {best_psl}")

    # FILE 2: OPTIMIZED
    with open(f"Optimized_First50_N{n}_p{p}.txt", "w") as f:
        f.write(f"OPTIMIZED FIRST 50 WEILS | N={n} | p={p}\n\n")
        for i, (init_psl, final_psl, g) in enumerate(zip(initial_psls, optimized_psls, gs)):
            f.write(f"WEIL {i + 1} (g={g}) | Init PSL: {init_psl} | Final PSL: {final_psl} | p={p}\n")
            b_str = "".join(['1' if x > 0 else '0' for x in optimized_sequences[i]])
            for j in range(0, len(b_str), 100):
                f.write(b_str[j:j + 100] + "\n")
            f.write("\n")

    # FILE 3: CROSS CORRELATION MATRIX (INITIAL + FINAL)
    print("\nComputing cross-correlation matrices...")

    # Initial CC matrix
    initial_cc_matrix = compute_cc_matrix(sequences, n)
    # Final CC matrix
    final_cc_matrix = compute_cc_matrix(optimized_sequences, n)

    with open(f"CrossCorrelation_First50_N{n}_p{p}.txt", "w") as f:
        f.write(f"CROSS CORRELATION MATRICES | N={n} | p={p} | First 50 Weils\n")
        f.write("=" * 100 + "\n")

        # INITIAL CC Matrix
        f.write("INITIAL CROSS-CORRELATION MATRIX (Max |CC|)\n")
        f.write("Rows/Cols = Weil 1-50 (g values below)\n\n")
        np.savetxt(f, initial_cc_matrix, fmt='%4d', delimiter=' ')
        f.write("\n" + "=" * 100 + "\n")

        # FINAL CC Matrix
        f.write("FINAL CROSS-CORRELATION MATRIX (after optimization)\n")
        f.write("Rows/Cols = Weil 1-50 (g values below)\n\n")
        np.savetxt(f, final_cc_matrix, fmt='%4d', delimiter=' ')
        f.write("\n" + "=" * 100 + "\n")

        # g values reference
        f.write("COLUMN REFERENCE (g values):\n")
        g_str = " ".join([f"{g:6d}" for g in gs])
        f.write(g_str + "\n")

        # Summary stats
        f.write(f"\nSUMMARY:\n")
        f.write(f"Avg Initial CC: {np.mean(initial_cc_matrix):.1f}\n")
        f.write(f"Avg Final CC:   {np.mean(final_cc_matrix):.1f}\n")
        f.write(f"Improvement:    {np.mean(initial_cc_matrix) - np.mean(final_cc_matrix):.1f}\n")

    print(f"\n✅ 3 FILES SAVED:")
    print(f"Weil_First50_N{n}_p{p}.txt")
    print(f"Optimized_First50_N{n}_p{p}.txt")
    print(f"CrossCorrelation_First50_N{n}_p{p}.txt")

    best_idx = np.argmin(optimized_psls)
    print(f"\n🏆 BEST: Weil {best_idx + 1} (g={gs[best_idx]}): {initial_psls[best_idx]} → {optimized_psls[best_idx]}")


if __name__ == "__main__":
    run_brest_boskovic()