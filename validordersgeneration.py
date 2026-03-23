from math import gcd

def multiplicative_order(a, n):
    """
    Compute the multiplicative order of a modulo n:
    the smallest k > 0 such that a^k ≡ 1 (mod n).
    Returns None if gcd(a, n) != 1.
    """
    if gcd(a, n) != 1:
        return None
    k = 1
    val = a % n
    while val != 1:
        val = (val * a) % n
        k += 1
        # safety check: if k exceeds n, break
        if k > n*2:
            return None
    return k

def divisors(n):
    """Return all divisors of n."""
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)

def valid_orders(m):
    """
    For given m, compute all divisors of 2^m - 1
    and return those d with ord_d(2) = m.
    """
    group_order = 2**m - 1
    divs = divisors(group_order)
    valid = []
    for d in divs:
        ord_val = multiplicative_order(2, d)
        if ord_val == m:
            valid.append(d)
    return valid

# Example usage:
m = 100
print("Valid orders for degree", m, ":", valid_orders(m))

m = 2
print("Valid orders for degree", m, ":", valid_orders(m))
