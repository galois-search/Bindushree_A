from sympy import symbols, Poly
from sympy.polys.domains import GF

x = symbols('x')


def is_irreducible_poly(p):
    """
    Check irreducibility of polynomial p over GF(2).
    """
    unit, factors = p.factor_list()
    return len(factors) == 1 and factors[0][0] == p


def generate_irreducibles(m):
    """
    Generate ALL irreducible monic polynomials of degree m over GF(2).
    """
    irreducibles = []
    for mask in range(1 << m):  # all possible lower coefficients
        coeffs = [int(b) for b in bin(mask)[2:].zfill(m)]
        coeffs.append(1)  # leading coefficient = 1 (monic)
        p = Poly(coeffs[::-1], x, domain=GF(2))
        if p.degree() == m and is_irreducible_poly(p):
            irreducibles.append(p)
    return irreducibles


def irreducible_polynomials(m, valid_periods):
    """
    For each valid period, return all irreducible polynomials of degree m.
    (Currently just groups them under the given periods.)
    """
    results = {}
    polys = generate_irreducibles(m)
    for period in valid_periods:
        results[period] = polys  # same set for each period
    return results


# -------- Interactive part --------
if __name__ == "__main__":
    m = int(input("Enter degree m: "))
    valid_periods = list(map(int, input("Enter valid periods (comma separated): ").split(",")))

    polys = irreducible_polynomials(m, valid_periods)

    for period, plist in polys.items():
        print(f"\nPeriod {period}:")
        for p in plist:
            print("   ", p)
