import numpy as np
import os
import time
from datetime import datetime


def get_base_legendre_sequence(n):
    """Generate SINGLE base Legendre sequence χ(i²+1)"""

    def is_prime(num):
        if num < 2: return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0: return False
        return True

    def quadratic_character(x, p):
        """Legendre symbol (x/p)"""
        if x == 0: return 1
        val = pow(x, (p - 1) // 2, p)
        return 1 if val == 1 else -1

    p = n
    while not is_prime(p):
        p += 1

    print(f"Prime p = {p}")

    base_chi = np.zeros(p)
    for i in range(p):
        arg = (i * i + 1) % p
        base_chi[i] = quadratic_character(arg, p)

    print("Base Legendre sequence χ(i²+1) ready")
    return base_chi, p


def generate_pure_weil_shift(base_chi, p, n, k):
    """Pure Weil S_k: χ(k) ⊕ χ(k+1) ⊕ ... ⊕ χ(k+n-1)"""
    xor_seq = np.zeros(n, dtype=int)
    running_xor = 0

    for i in range(n):
        idx = (k + i) % p
        running_xor ^= int(base_chi[idx] > 0)
        xor_seq[i] = running_xor

    return (2 * xor_seq - 1).astype(float)


def compute_psl(seq, n):
    """Peak sidelobe level"""
    autocorr = np.round(np.real(np.fft.ifft(np.fft.fft(seq) * np.conj(np.fft.fft(seq))))).astype(float)
    return int(np.max(np.abs(autocorr[1:n // 2 + 1])))


def compute_balance(seq):
    """|#1s - #0s| (ideal = 0 or 1)"""
    ones = np.sum(seq > 0)
    return int(abs(ones - len(seq) / 2))  # ✅ FIXED: Return int!


def save_single_sequence_to_file(seq, k, init_psl, final_psl, init_bal, final_bal, n, p, filename):
    """Save optimized result"""
    with open(filename, "a", encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'=' * 100}\n")
        f.write(f"PURE WEIL k={k} | N={n} | p={p}\n")
        f.write(f"{timestamp} | PSL: {init_psl}→{final_psl} | Bal: {init_bal}→{final_bal}\n")
        f.write(f"{'=' * 100}\n")
        s_str = "".join(['1' if x > 0 else '0' for x in seq])
        for j in range(0, len(s_str), 100):
            f.write(s_str[j:j + 100] + "\n")
        f.write("\n")


def optimize_single_sequence(seq, n):
    """Brest-Boskovic + Balance optimization"""
    start_time = time.time()

    init_psl = compute_psl(seq, n)
    init_balance = compute_balance(seq)
    best_seq = seq.copy()
    best_psl = init_psl
    best_balance = init_balance
    gamma = 4.0
    balance_weight = 0.1

    f_vec = np.fft.fft(seq)
    acf = np.round(np.real(np.fft.ifft(f_vec * np.conj(f_vec)))).astype(float)

    for step in range(1, 100001):
        idx = np.random.randint(0, n)
        diff = -2 * seq[idx]
        delta = diff * (seq[(idx - np.arange(n)) % n] + seq[(idx + np.arange(n)) % n])
        sl = slice(1, n // 2 + 1)

        old_psl_cost = np.sum(np.abs(acf[sl]) ** gamma)
        old_bal_cost = balance_weight * abs(np.sum(seq > 0) - n / 2)

        new_psl_cost = np.sum(np.abs(acf[sl] + delta[sl]) ** gamma)
        new_ones = np.sum(seq > 0) + (1 if seq[idx] < 0 else -1)
        new_bal_cost = balance_weight * abs(new_ones - n / 2)

        if new_psl_cost + new_bal_cost < old_psl_cost + old_bal_cost:
            seq[idx] *= -1
            acf += delta
            curr_psl = int(np.max(np.abs(acf[sl])))
            curr_balance = compute_balance(seq)

            if (curr_psl < best_psl) or (curr_psl == best_psl and curr_balance < best_balance):
                best_psl = curr_psl
                best_balance = curr_balance
                best_seq = seq.copy()
                gamma = min(18, gamma + 0.1)

    duration = time.time() - start_time
    final_balance = compute_balance(best_seq)
    return best_seq, init_psl, best_psl, init_balance, final_balance, duration


def run_pure_weil_optimization():
    """PURE GLOBAL WEIL: S_k[i] = χ(k+i) ⊕ ... ⊕ χ(k+n-1)"""
    n = int(input("Enter sequence length N: "))

    print("\n🔄 Computing base Legendre sequence...")
    base_chi, p = get_base_legendre_sequence(n)

    max_k = (p - 1) // 2
    print(f"\nGlobal Weil range: k=0 to {max_k}")

    start_k = int(input(f"Start k (0-{max_k}): "))
    stop_k = int(input(f"Stop k (0-{max_k}): "))

    if start_k < 0 or stop_k > max_k or start_k > stop_k:
        print("❌ Invalid range!")
        return

    num_shifts = stop_k - start_k + 1
    filename = f"PureGlobalWeil_N{n}_p{p}_k_{start_k}-{stop_k}.txt"

    with open(filename, "w", encoding='utf-8') as f:
        f.write("PURE GLOBAL WEIL SEQUENCES\n")
        f.write(f"Base: χ(i²+1) mod p={p}\n")
        f.write(f"S_k[i] = χ(k+i) ⊕ χ(k+i+1) ⊕ ... ⊕ χ(k+i+n-1)\n")
        f.write(f"N={n} | k={start_k} to {stop_k} | Shifts: {num_shifts}\n")
        f.write("=" * 100 + "\n")

    print(f"\n🚀 Optimizing {num_shifts} pure Weil shifts")
    print(f"📁 {filename}")

    best_psl = float('inf')
    best_balance = float('inf')
    best_k = None

    for k in range(start_k, stop_k + 1):
        print(f"\n🔄 PURE WEIL k={k:4d}/{stop_k} ({k - start_k + 1}/{num_shifts})")
        print("-" * 60)

        shift_start = time.time()

        seq = generate_pure_weil_shift(base_chi, p, n, k)
        print("  Generating...", end=" → ")

        best_seq, init_psl, final_psl, init_bal, final_bal, duration = optimize_single_sequence(seq, n)

        save_single_sequence_to_file(best_seq, k, init_psl, final_psl, init_bal, final_bal, n, p, filename)

        print(f"PSL {init_psl:4d}→{final_psl:4d} | Bal {init_bal:3d}→{final_bal:3d} | {duration:5.1f}s")

        if (final_psl < best_psl) or (final_psl == best_psl and final_bal < best_balance):
            best_psl = final_psl
            best_balance = final_bal
            best_k = k
            print(f"  🏆 NEW BEST: PSL={final_psl}, Bal={final_bal} at k={k}")

        shift_time = time.time() - shift_start
        print(f"  ✅ k={k} complete: {shift_time:.1f}s")

    print(f"\n{'=' * 80}")
    print(f"🏆 GLOBAL BEST PURE WEIL:")
    print(f"  k = {best_k}")
    print(f"  Final PSL = {best_psl}")
    print(f"  Final Balance = {best_balance}")
    print(f"📁 {filename}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    print("🚀 PURE GLOBAL WEIL OPTIMIZER")
    print("S_k[i] = χ(k+i) ⊕ χ(k+i+1) ⊕ ... ⊕ χ(k+n-1)")
    print("Base: χ(i²+1) mod p")
    print("=" * 50)
    run_pure_weil_optimization()