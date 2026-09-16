## 1. Checkpoint selection

We considered five saved labels for each objective:

- 25%, 50%, 75%, and 100% of the 100,000-step training budget;
- the checkpoint with minimum pretraining-validation loss (`best_val`).

Across JEPA, Future JEPA, MAE, and Future MAE, this gave **20 candidate
labels representing 17 unique encoder states**. The JEPA, MAE, and Future MAE
`best_val` states were byte-identical at the tensor level to their corresponding
100,000-step milestone states, so each was probed once and retained as an alias.
Future JEPA's `best_val` checkpoint was a distinct state saved at step 98,000.

The official data roles were fixed:

- **Train:** fit probes
- **Validation:** choose probe settings and checkpoint
- **Test:** final score only, after selection was frozen

The test split was not used to choose anything.

### Code provenance

The executed study used the repository's evaluation implementation frozen at
commit `96bc7a92532250f0b0eedae64aed25091162e41a`. The reusable cache, target,
probe, and checkpoint-selection workflow originated in Adil Soubki's commits
`a3c1856` and `185bbe7`. It was not an unchanged copy of that original code:
later commits `6316756`, `a43a8d3`, and `96bc7a9` restricted the MLP to one
deterministic seed, reduced the task grid, fixed probing to transformer block 4,
and corrected the reduced-probe implementation. Thus this analysis used
Adil's repository implementation as its base, with the documented study-specific
changes—not a separate reimplementation.

### Representation extraction

For every checkpoint, the encoder was frozen and given complete, unmasked eight-frame clips. We retained only the output of **transformer block 4**—internally zero-indexed layer `3`.

Two representations were extracted:

- **Pooled:** mean of all encoder tokens

$$
z_{\mathrm{pool}}=\frac{1}{N}\sum_{i=1}^{N}z_i.
$$

- **Token:** representations at 64 deterministic token positions per trajectory.

### Physics targets

Each representation was probed for five quantities:

$$
\omega^2,\qquad
|\nabla b|^2,\qquad
u_yb,\qquad
|\nabla p|,\qquad
|\nabla^2b|,
$$

where

$$
\omega=\partial_xu_y-\partial_yu_x.
$$

These correspond to:

- enstrophy,
- buoyancy-gradient energy,
- convective flux,
- pressure-gradient magnitude,
- buoyancy-Laplacian magnitude.

Pooled targets were spatiotemporal averages. Token targets were averages over the corresponding $2\times16\times16$ spacetime patch.

We used target offsets

$$
\Delta t\in\{0,16,40\},
$$

defined as target-start frame minus context-start frame. Thus each checkpoint had

$$
2\text{ representations}\times3\text{ offsets}\times5\text{ targets}
=30
$$

physics-probing cells.

---

## 2. Probe selection inside each cell

Each cell received both a Ridge probe and a one-hidden-layer MLP. Features were standardized using **training-set statistics only**.

### Ridge

For standardized feature matrix $X$, centered target $y-\bar y$, and $n$ training examples:

$$
\hat w_\alpha=
\left(X^\top X+\alpha nI\right)^{-1}X^\top(y-\bar y).
$$

We searched

$$
\alpha\in
\{10^{-5},10^{-4},10^{-3},10^{-2},10^{-1},1,10,100\}.
$$

The $\alpha$ with the lowest validation VRMSE was retained independently for each target.

### MLP

The nonlinear probe was

$$
\hat y=
W_2\,\mathrm{Dropout}
\left(\mathrm{ReLU}(W_1z+b_1)\right)+b_2,
$$

with:

- 128 hidden units,
- dropout $0.1$,
- Adam,
- learning rate $10^{-2}$,
- weight decay $10^{-4}$,
- full-batch training,
- deterministic probe seed 0.

Validation loss was checked every 20 steps. Training ran for at least 150 and at most 2,000 steps, with patience 100. The best validation step was retained.

For each physics cell, we then selected whichever family—Ridge or MLP—had lower validation VRMSE.

### Selection metric

For predictions $\hat y_i$ and targets $y_i$,

$$
\mathrm{MSE}
=
\frac1N\sum_i(\hat y_i-y_i)^2,
$$

$$
\mathrm{VRMSE}
=
\sqrt{
\frac{\mathrm{MSE}}
{\frac1N\sum_i(y_i-\bar y)^2}
}.
$$

This is related to $R^2$ by

$$
R^2=1-\mathrm{VRMSE}^2.
$$

Lower VRMSE is better.

---

## 3. Combining cells into one checkpoint score

For checkpoint $c$, we first averaged validation VRMSE over the five targets and two representations at each horizon:

$$
H_\delta(c)
=
\frac1{10}
\sum_{\substack{r\in\{\mathrm{pooled,token}\}\\q\in\text{five targets}}}
\mathrm{VRMSE}_{c,r,\delta,q}.
$$

The final checkpoint score gave equal weight to present and future performance:

$$
S(c)
=
\frac12H_0(c)
+
\frac12\left(\frac{H_{16}(c)+H_{40}(c)}2\right).
$$

Equivalently,

$$
S(c)=0.5H_0(c)+0.25H_{16}(c)+0.25H_{40}(c).
$$

For each objective, we selected the checkpoint with minimum $S(c)$. Exact ties would prefer the earlier checkpoint.

### Winners

| Objective | Selected step | $H_0$ | $H_{16}$ | $H_{40}$ | Weighted contributions: $0.5H_0 + 0.25H_{16} + 0.25H_{40}$ | $S(c)$ |
|---|---:|---:|---:|---:|---:|---:|
| JEPA | 25,000 | 0.138208 | 0.355573 | 0.436522 | 0.069104 + 0.088893 + 0.109130 | 0.267128 |
| Future JEPA | 50,000 | 0.180774 | 0.364145 | 0.430322 | 0.090387 + 0.091036 + 0.107580 | 0.289004 |
| MAE | 50,000 | 0.122356 | 0.344792 | 0.466224 | 0.061178 + 0.086198 + 0.116556 | 0.263932 |
| Future MAE | 100,000 | 0.129222 | 0.348714 | 0.470506 | 0.064611 + 0.087178 + 0.117627 | 0.269416 |

### Best-pretraining-validation candidates

| Objective | Best-validation step | Relation to milestone | $H_0$ | $H_{16}$ | $H_{40}$ | $S(c)$ | Selection outcome |
|---|---:|---|---:|---:|---:|---:|---|
| JEPA | 100,000 | Same encoder state as 100% | 0.164039 | 0.359026 | 0.444087 | 0.282798 | 25,000 remained better |
| Future JEPA | 98,000 | Distinct encoder state | 0.184215 | 0.367775 | 0.439731 | 0.293984 | 50,000 remained better |
| MAE | 100,000 | Same encoder state as 100% | 0.129898 | 0.343820 | 0.465221 | 0.267209 | 50,000 remained better |
| Future MAE | 100,000 | Same encoder state as 100% | 0.129222 | 0.348714 | 0.470506 | 0.269416 | Selected state unchanged |

Adding the best-pretraining-validation candidates therefore changed **none**
of the four selected encoder states or any Stage 2 test result.

Regime-parameter probes were **not** included in this checkpoint score.

---

## 4. Final probing after selection

Once these four checkpoints were frozen:

1. We extracted their layer-4 features on the official test split.
2. We reused the exact train-fitted probe, feature normalization, Ridge penalty or MLP stopping point selected on validation.
3. We did **not** refit using validation or test data.
4. Each frozen probe was applied once to the test set.

For each physics cell we reported:

$$
\mathrm{VRMSE},\quad R^2,\quad
r_{\mathrm{Pearson}},\quad
\mathrm{MSE},\quad
\log_{10}(\mathrm{MSE}).
$$

We also evaluated:

- Ridge and MLP separately,
- the validation-selected Ridge/MLP result,
- persistence baselines for future targets,
- regime/time and position controls,
- Ridge augmented with the relevant controls,
- recovery of $\log_{10}\mathrm{Rayleigh}$ and $\log_{10}\mathrm{Prandtl}$.

The regime probes used Ridge and MLP but did not influence checkpoint selection. No noise-corruption sweep was included in this Stage 2 result.

**In one sentence:** we selected checkpoints using balanced validation performance over all 30 layer-4 physics tasks, froze every choice, and then scored those same fitted probes exactly once on the official test split.
