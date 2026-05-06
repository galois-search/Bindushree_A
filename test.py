import numpy as np
import os
import time
from datetime import datetime


def get_base_legendre_sequence(n):
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
    print(f"✅ Prime p = {p}")

    # ✅ FIXED: PROPER LEGENDRE SEQUENCE χ(i)
    base_chi = np.zeros(p)
    for i in range(p):
        base_chi[i] = quadratic_character(i, p)  # TRUE Legendre: χ(i)
    print("✅ PROPER Legendre χ(i) ready")
    return base_chi, p


def generate_pure_weil_shift(base_chi, p, n, k):
    """✅ FIXED: TRUE WEIL = χ(i) × χ(i+k)"""
    # PROPER WEIL CONSTRUCTION
    chi_shifted = np.roll(base_chi, k)
    weil_seq = base_chi * chi_shifted  # Multiplication = XOR for ±1
    return weil_seq[:n].astype(float)  # Truncate to n


def compute_psl(seq, n):
    autocorr = np.round(np.real(np.fft.ifft(np.fft.fft(seq) * np.conj(np.fft.fft(seq))))).astype(float)
    return int(np.max(np.abs(autocorr[1:n // 2 + 1])))


def compute_balance(seq):
    ones = int(np.sum(seq > 0))
    return abs(ones - (len(seq) - ones))


def get_smart_balance_limits(n):
    """🧠 AUTO balance limits"""
    if n % 2 == 0:
        return [0, 2, 4, 6], 6
    else:
        return [1, 3, 5, 7], 7


def is_valid_balance(balance, n):
    return (balance % 2 == n % 2)


def optimize_balance_auto(seq, n, max_flips=15000):
    """PHASE 1: Force balance"""
    valid_balances, max_target = get_smart_balance_limits(n)
    target_ones = n // 2
    ones = int(np.sum(seq > 0))
    flips = 0

    while (compute_balance(seq) > max_target or not is_valid_balance(compute_balance(seq), n)) and flips < max_flips:
        if ones > target_ones + 3:
            candidates = np.where(seq > 0)[0]
            if len(candidates) > 0:
                idx = np.random.choice(candidates)
                seq[idx] *= -1
                ones -= 1
        elif ones < target_ones - 3:
            candidates = np.where(seq < 0)[0]
            if len(candidates) > 0:
                idx = np.random.choice(candidates)
                seq[idx] *= -1
                ones += 1
        flips += 1

    return seq, flips, compute_balance(seq)


def optimize_psl_protected(seq, n, max_steps=100000):
    """PHASE 2+3: PSL + balance protection"""
    valid_balances, max_target = get_smart_balance_limits(n)

    best_seq = seq.copy()
    best_psl = compute_psl(seq, n)
    best_balance = compute_balance(seq)

    f_vec = np.fft.fft(seq)
    acf = np.round(np.real(np.fft.ifft(f_vec * np.conj(f_vec)))).astype(float)
    gamma = 3.0

    for step in range(max_steps):
        idx = np.random.randint(0, n)
        diff = -2 * seq[idx]
        delta = diff * (seq[(idx - np.arange(n)) % n] + seq[(idx + np.arange(n)) % n])
        sl = slice(1, n // 2 + 1)

        if np.sum(np.abs(acf[sl] + delta[sl]) ** gamma) < np.sum(np.abs(acf[sl]) ** gamma):
            seq[idx] *= -1
            acf += delta

            new_balance = compute_balance(seq)
            if new_balance <= max_target and is_valid_balance(new_balance, n):
                curr_psl = int(np.max(np.abs(acf[sl])))
                if curr_psl < best_psl or (curr_psl == best_psl and new_balance < best_balance):
                    best_psl = curr_psl
                    best_balance = new_balance
                    best_seq = seq.copy()
                gamma = min(20, gamma + 0.01)
            else:
                seq[idx] *= -1  # Revert
                acf -= delta

    return best_seq, best_psl, best_balance


def save_sequence_to_file(filename, seq, k, n, p, init_psl, final_psl, init_bal, final_bal, bal_flips, bal_phase1,
                          duration, max_bal):
    """IMMEDIATE save after each k"""
    valid_balances, _ = get_smart_balance_limits(n)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    with open(filename, "a", encoding='utf-8') as f:
        f.write(f"\n{'=' * 120}\n")
        f.write(f"TRUE WEIL(k={k}) | N={n} | p={p} | {timestamp}\n")
        f.write(f"PSL: {init_psl}→{final_psl} | Bal: {init_bal}→{final_bal} "
                f"| Flips:{bal_flips} | Phase1:{bal_phase1} | {duration:.2f}s\n")
        f.write(f"Valid: {valid_balances} | OK: {'✅' if final_bal in valid_balances else '❌'}\n")
        f.write(f"{'=' * 120}\n")

        s_str = "".join(['1' if x > 0 else '0' for x in seq])
        for j in range(0, len(s_str), 80):
            f.write(s_str[j:j + 80] + "\n")
        f.write("\n")


def optimize_single_sequence(seq, n):
    """3-PHASE: Balance → PSL → Protection"""
    shift_start = time.time()

    init_psl = compute_psl(seq, n)
    init_balance = compute_balance(seq)

    # PHASE 1
    seq_bal, bal_flips, bal_phase1 = optimize_balance_auto(seq.copy(), n)

    # PHASE 2+3
    best_seq, final_psl, final_balance = optimize_psl_protected(seq_bal, n)

    duration = time.time() - shift_start
    valid_balances, max_bal = get_smart_balance_limits(n)

    return (best_seq, init_psl, final_psl, init_balance, final_balance,
            bal_flips, bal_phase1, duration, max_bal)


def run_pure_weil_optimization():
    """MAIN - ONLY Weil generation fixed"""
    n = int(input("Enter N: "))

    print("\n🔄 Computing PROPER Legendre sequence...")
    base_chi, p = get_base_legendre_sequence(n)

    max_k = (p - 1) // 2
    print(f"\n📊 TRUE Weil shifts: k=0 to {max_k}")

    start_k = int(input(f"Start k (0-{max_k}): ") or "0")
    stop_k = int(input(f"Stop k ({start_k}-{max_k}): ") or str(min(max_k, 100)))

    if start_k < 0 or stop_k > max_k or start_k > stop_k:
        print("❌ Invalid range!")
        return

    filename = f"TRUE_Weil_N{n}_p{p}_k{start_k}-{stop_k}.txt"

    # HEADER
    with open(filename, "w", encoding='utf-8') as f:
        f.write("✅ TRUE WEIL OPTIMIZER (χ(i) × χ(i+k))\n")
        f.write(f"N={n} | p={p} | k={start_k}-{stop_k}\n")
        f.write("IMMEDIATE SAVES after each k\n")
        f.write("=" * 120 + "\n")

    print(f"\n🚀 {stop_k - start_k + 1} shifts → {filename}")

    best_psl, best_bal, best_k = float('inf'), float('inf'), None

    for k in range(start_k, stop_k + 1):
        print(f"🔄 k={k}/{stop_k}", end=" ")

        seq = generate_pure_weil_shift(base_chi, p, n, k)
        result = optimize_single_sequence(seq, n)

        (best_seq, init_psl, final_psl, init_bal, final_bal,
         bal_flips, bal_phase1, duration, max_bal) = result

        # IMMEDIATE SAVE
        save_sequence_to_file(filename, best_seq, k, n, p, init_psl, final_psl,
                              init_bal, final_bal, bal_flips, bal_phase1, duration, max_bal)

        print(f"PSL:{init_psl}→{final_psl} Bal:{init_bal}→{final_bal} {duration:.1f}s")

        if final_psl < best_psl or (final_psl == best_psl and final_bal < best_bal):
            best_psl, best_bal, best_k = final_psl, final_bal, k
            print(f"   🏆 NEW BEST!")

    print(f"\n🏆 BEST: k={best_k}, PSL={best_psl}, Bal={best_bal}")
    print(f"📁 {filename}")


if __name__ == "__main__":
    run_pure_weil_optimization()