MW_H_Total is composed of two correlated components: the intrinsic uncertainty of the patient's features (H_total) and the proximity to the decision boundary (|p - τ|).

H_total is constrained by p. When p is near the boundary (τ=0.11), H_total is forced to be in a narrow range (~0.47-0.53). It cannot be simultaneously:

Near the boundary (p≈τ)
High entropy (H≈1.0)

This is impossible because H=1.0 requires p≈0.5, which is far from τ=0.11.


The only truly orthogonal signal is I_epistemic (but it's not directly observable). H_total is a mixture of aleatoric and epistemic uncertainty, but we can't disentangle them without additional assumptions or data.

We can rely on variance of ensemble predictions (still a proxy for epistemic uncertainty).