
## Studying GPT for error mitigation

[RydbergGPT](https://arxiv.org/abs/2405.21052) showed interesting results in learning the measurement outcomes. Maybe something similar works for enhancing something else in quantum computing pipelines.

- Fix the noisy quantum computer
- Create a set of circuits with varying measurements
- Compute the exact expectation values
- Compute the noisy expectation values
- Train the model (circuit, noisy expectation value) -> exact expectation value

Forked from [nanoGPT](https://github.com/karpathy/nanoGPT)