# Parallel Binomial Option Pricing

Accelerating American option pricing using binomial trees with parallel algorithms on CPU and GPU architectures.

### Binomial Option Pricing Visualization
The Binomial Options Pricing Model (BOPM) values options by constructing a lattice of possible future stock prices. For American option, at every node, the option value V is the maximum of the Exercise Value (E) (payoff from exercising immediately) and the Hold Value (H) (discounted expected value of holding the option).

The animation below demonstrates the pricing of an American Call Option with parameters: **S0 = $100, K = $100, r = 5%, sigma = 20%, T = 4 steps**.

![Binomial Option Pricing Animation](binomial_pricing.gif)


## Build Instructions
```bash
mkdir build && cd build
cmake ..
make
```


## References
See [docs/proposal.pdf](docs/proposal.pdf)