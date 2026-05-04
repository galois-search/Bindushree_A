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
    """|#1s - #0s|"""
    ones = int(np.sum(seq > 0))
    return abs(ones - len(seq) // 2)


def get_balance_limits(n):
    """Dynamic balance limits: even=0-2, odd=1-3"""
    if n % 2 == 0:
        return 2, "0-2 (even)"  # Perfect=0
    else:
        return 3, "1-3 (odd)"   # Perfect=1


def optimize_balance_first(seq, n, max_flips=10000):
    """Phase 1: Force optimal balance (even:0-2, odd:1-3)"""
    target_ones = n // 2
    max_balance, balance_range = get_balance_limits(n)
    ones = int(np.sum(seq > 0))
    flips = 0

    print(f"    Target: {target_ones}±{max_balance//2} ({balance_range})")

    while compute_balance(seq) > max_balance and flips < max_flips:
        if ones > target_ones:
            candidates = np.where(seq > 0)[0]
            if len(candidates) == 0: break
            idx = np.random.choice(candidates)
            seq[idx] *= -1
            ones -= 1
        else:
            candidates = np.where(seq < 0)[0]
            if len(candidates) == 0: break
            idx = np.random.choice(candidates)
            seq[idx] *= -1
            ones += 1
        flips += 1

    return seq, flips


def optimize_single_sequence_perfect_balance(seq, n):
    """MULTI-STAGE: Smart Balance → PSL → STRICT Protect"""
    start_time = time.time()
    max_balance, balance_range = get_balance_limits(n)

    # PHASE 1: Force optimal balance
    seq_bal, bal_flips = optimize_balance_first(seq.copy(), n)
    balance_after_phase1 = compute_balance(seq_bal)

    # PHASE 2: PSL optimization WITH STRICT BALANCE PROTECTION
    init_psl = compute_psl(seq_bal, n)
    best_seq = seq_bal.copy()
    best_psl = init_psl
    best_balance = compute_balance(best_seq)
    gamma = 4.0

    f_vec = np.fft.fft(seq_bal)
    acf = np.round(np.real(np.fft.ifft(f_vec * np.conj(f_vec)))).astype(float)

    for step in range(1, 100001):
        idx = np.random.randint(0, n)
        diff = -2 * seq_bal[idx]
        delta = diff * (seq_bal[(idx - np.arange(n)) % n] + seq_bal[(idx + np.arange(n)) % n])
        sl = slice(1, n // 2 + 1)

        if np.sum(np.abs(acf[sl] + delta[sl]) ** gamma) < np.sum(np.abs(acf[sl]) ** gamma):
            seq_bal[idx] *= -1
            acf += delta
            curr_psl = int(np.max(np.abs(acf[sl])))
            curr_balance = compute_balance(seq_bal)

            # STRICT: Only accept if PSL improves AND balance stays optimal
            if curr_psl < best_psl and curr_balance <= max_balance:
                best_psl = curr_psl
                best_balance = curr_balance
                best_seq = seq_bal.copy()
                gamma = min(18, gamma + 0.1)

    duration = time.time() - start_time
    init_balance = compute_balance(seq)
    final_balance = compute_balance(best_seq)

    return (best_seq, init_psl, best_psl, init_balance, final_balance,
            bal_flips, balance_after_phase1, duration, max_balance)


def save_single_sequence_to_file(seq, k, init_psl, final_psl, init_bal, final_bal,
                                 bal_flips, bal_phase1, n, p, filename, timestamp_start, max_balance):
    """Enhanced save with all metrics"""
    with open(filename, "a", encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"\n{'=' * 110}\n")
        f.write(f"PURE WEIL k={k} | N={n} | p={p} | Target Bal ≤{max_balance}\n")
        f.write(f"{timestamp} | PSL: {init_psl}→{final_psl} | Bal: {init_bal}→{final_bal} | BalFlips: {bal_flips}\n")
        f.write(f"Phase1 Bal: {bal_phase1} | Duration: {time.time() - timestamp_start:.1f}s\n")
        f.write(f"{'=' * 110}\n")
        s_str = "".join(['1' if x > 0 else '0' for x in seq])
        for j in range(0, len(s_str), 100):
            f.write(s_str[j:j + 100] + "\n")
        f.write("\n")


def run_pure_weil_optimization():
    """PURE GLOBAL WEIL WITH SMART BALANCE"""
    n = int(input("Enter sequence length N: "))
    max_balance, balance_range = get_balance_limits(n)
    print(f"🎯 Balance target: ≤{max_balance} ({balance_range})")

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
    filename = f"PureWeil_Balance_N{n}_p{p}_k_{start_k}-{stop_k}.txt"

    with open(filename, "w", encoding='utf-8') as f:
        f.write("PURE GLOBAL WEIL + SMART BALANCE OPTIMIZATION\n")
        f.write(f"S_k[i] = χ(k+i) ⊕ ... ⊕ χ(k+n-1) | Base: χ(i²+1) mod p={p}\n")
        f.write(f"N={n} | Balance target: ≤{max_balance} ({'even:0-2' if n%2==0 else 'odd:1-3'})\n")
        f.write(f"k={start_k}-{stop_k}\n")
        f.write("=" * 110 + "\n")

    print(f"\n🚀 Smart balance optimization: {num_shifts} shifts (≤{max_balance})")
    print(f"📁 {filename}")

    best_psl = float('inf')
    best_balance = float('inf')
    best_k = None

    for k in range(start_k, stop_k + 1):
        print(f"\n🔄 WEIL k={k:4d}/{stop_k} ({k - start_k + 1}/{num_shifts})")
        print("-" * 70)

        shift_start = time.time()

        # FRESH PURE WEIL
        seq = generate_pure_weil_shift(base_chi, p, n, k)

        # MULTI-STAGE OPTIMIZATION
        best_seq, init_psl, final_psl, init_bal, final_bal, bal_flips, bal_phase1, duration, max_balance = optimize_single_sequence_perfect_balance(
            seq, n)

        # IMMEDIATE SAVE
        save_single_sequence_to_file(best_seq, k, init_psl, final_psl, init_bal, final_bal,
                                     bal_flips, bal_phase1, n, p, filename, shift_start, max_balance)

        print(f"  PSL  {init_psl:4d}→{final_psl:4d} | Bal {init_bal:3d}→{final_bal:3d}")
        print(f"  BalFlips: {bal_flips:4d} | Phase1: {bal_phase1:2d} | {duration:5.1f}s")

        # Global best
        if (final_psl < best_psl) or (final_psl == best_psl and final_bal < best_balance):
            best_psl = final_psl
            best_balance = final_bal
            best_k = k
            print(f"  🏆 NEW BEST: k={k}, PSL={final_psl}, Bal={final_bal}")

        print(f"  ✅ Complete: {time.time() - shift_start:.1f}s")

    print(f"\n{'=' * 90}")
    print(f"🏆 BEST PURE WEIL RESULT:")
    print(f"  k={best_k:4d} | PSL={best_psl:4d} | Balance={best_balance}")
    print(f"📁 {filename}")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    print("🚀 PURE GLOBAL WEIL OPTIMIZER")
    print("• χ(i²+1) base → Fresh XOR windows S_k")
    print("• Smart Balance: even(0-2) odd(1-3) → PSL → Protect")
    print("=" * 55)
    run_pure_weil_optimization()