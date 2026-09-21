# SVM from Scratch

> A linear support vector machine implemented with NumPy — hinge loss, L2 regularization, and gradient descent, with no `sklearn.svm`.

scikit-learn is used only to load a dataset, split it, and score the result. The
classifier itself is written out: the decision function, the hinge loss, the
subgradient, and the update loop.

## Run it

Open [`SVM.ipynb`](SVM.ipynb) in Jupyter, or
[in Colab](https://colab.research.google.com/github/Fezzaioussama/SVM-from-scratch-using-Python/blob/main/SVM.ipynb).

```bash
pip install numpy matplotlib scikit-learn
```

## What it covers

| Section | Content |
|---|---|
| Define SVM | The model: weights, bias, and the `w·x − b` decision function |
| Hinge loss | `max(0, 1 − yᵢ(w·xᵢ − b))` plus the L2 regularization term |
| Gradient descent | Subgradient updates for `w` and `b`, with the two cases split on whether the margin is satisfied |
| Evaluation | Accuracy against scikit-learn's `accuracy_score`, with the decision boundary plotted |

## The idea

An SVM doesn't just separate the classes — it maximises the **margin**, the
distance from the boundary to the nearest point of either class. A wide margin
generalises better than a boundary that merely happens to split the training
data.

That objective is expressed as hinge loss:

```
L = λ‖w‖² + (1/n) Σ max(0, 1 − yᵢ(w·xᵢ − b))
```

The hinge term charges nothing for points that are correctly classified *and*
outside the margin, and charges linearly for everything else. The `λ‖w‖²` term
pushes the weights down, which widens the margin — so the two terms trade off
directly, and `λ` sets the exchange rate.

The loss isn't differentiable at the hinge, so the update uses a subgradient,
which is why the training loop branches on whether `yᵢ(w·xᵢ − b) ≥ 1`:

- **Margin satisfied** → only the regularization term contributes.
- **Margin violated** → the misclassification term contributes too, pulling the
  boundary toward the offending point.

## Note

This is the **linear, primal** formulation. There's no kernel trick and no dual
solver, so it can only separate linearly separable data. For the nonlinear case,
see [PCA-from-scratch-using-Python](https://github.com/Fezzaioussama/PCA-from-scratch-using-Python),
which covers RBF kernels in the dimensionality-reduction setting.
