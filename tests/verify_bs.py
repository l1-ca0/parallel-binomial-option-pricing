import math

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def black_scholes_put(S, K, r, sigma, T):
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * norm_cdf(-d1)

test_cases = [
    # S, K, r, sigma, T, Expected_in_File
    (100, 100, 0.05, 0.2, 1.0, 5.5731),
    (90, 100, 0.05, 0.2, 1.0, 10.8423),
    (110, 100, 0.05, 0.2, 1.0, 2.0826),
    (100, 100, 0.05, 0.4, 1.0, 10.7994),
    (100, 100, 0.05, 0.1, 1.0, 2.3619),
    (100, 100, 0.05, 0.2, 0.25, 2.8011),
    (100, 100, 0.05, 0.2, 3.0, 9.6489),
    (100, 100, 0.0, 0.2, 1.0, 7.9664),
    (100, 100, 0.1, 0.2, 1.0, 4.0738),
    (80, 100, 0.05, 0.2, 1.0, 19.8564),
    (120, 100, 0.05, 0.2, 1.0, 0.5890),
    (100, 100, 0.05, 0.6, 1.0, 16.2564),
    (100, 100, 0.05, 0.2, 0.0833, 1.1495),
    (100, 100, 0.05, 0.2, 5.0, 12.3842)
]

print(f"{'Case':<5} {'BS_Price':<10} {'File_Val':<10} {'Diff':<10} {'Match?'}")
print("-" * 45)

for i, (S, K, r, sigma, T, expected) in enumerate(test_cases):
    bs_price = black_scholes_put(S, K, r, sigma, T)
    diff = abs(bs_price - expected)
    match = "YES" if diff < 0.01 else "NO"
    print(f"{i+1:<5} {bs_price:<10.4f} {expected:<10.4f} {diff:<10.4f} {match}")
