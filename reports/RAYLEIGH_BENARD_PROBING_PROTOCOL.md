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

The checkpoint-selection study used the repository's evaluation implementation
frozen at commit `96bc7a92532250f0b0eedae64aed25091162e41a`. The reusable
cache, target, probe, and checkpoint-selection workflow originated in Adil
Soubki's commits `a3c1856` and `185bbe7`. It was not an unchanged copy of that
original code: later commits `6316756`, `a43a8d3`, and `96bc7a9` restricted the
MLP to one deterministic seed, reduced the task grid, fixed checkpoint
selection to transformer block 4, and corrected the reduced-probe
implementation. Commit `b2b0100` restored probing across all 13 encoder outputs
for the final frozen checkpoints and added the persistence comparison. Thus
this analysis used Adil's repository implementation as its base, with the
documented study-specific changes—not a separate reimplementation.

### Representation extraction

Checkpoint selection retained only the output of **transformer block 4**,
internally zero-indexed layer `3`, for every candidate. After freezing one
checkpoint per objective, the final analysis retained all 12 transformer block
outputs plus the final normalization output.

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

Each cell and encoder output received both a Ridge probe and a one-hidden-layer MLP. Features were standardized using **training-set statistics only**.

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

The penalty and encoder output with the lowest validation VRMSE were retained independently for each target.

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

Validation loss was checked every 20 steps. Training ran for at least 150 and
at most 2,000 steps, with patience 100. The best validation step was retained
at each encoder output. Each family then retained its output with lowest
validation VRMSE. For each physics cell, the selected summary chose whichever
of those independently output-selected Ridge or MLP candidates had lower
validation VRMSE. Test scores never selected an output, stopping point,
penalty, or family.

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

### Scores at every milestone

The selected checkpoint for each objective is marked **Winner**.

| Objective | Step | $H_0$ | $H_{16}$ | $H_{40}$ | Weighted contributions: $0.5H_0 + 0.25H_{16} + 0.25H_{40}$ | $S(c)$ | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| JEPA | 25,000 | 0.138208 | 0.355573 | 0.436522 | 0.069104 + 0.088893 + 0.109130 | 0.267128 | **Winner** |
| JEPA | 50,000 | 0.151653 | 0.359396 | 0.451898 | 0.075826 + 0.089849 + 0.112975 | 0.278650 | |
| JEPA | 75,000 | 0.160473 | 0.361578 | 0.460653 | 0.080236 + 0.090394 + 0.115163 | 0.285794 | |
| JEPA | 100,000 | 0.164039 | 0.359026 | 0.444087 | 0.082019 + 0.089756 + 0.111022 | 0.282798 | |
| Future JEPA | 25,000 | 0.190728 | 0.370519 | 0.423550 | 0.095364 + 0.092630 + 0.105887 | 0.293881 | |
| Future JEPA | 50,000 | 0.180774 | 0.364145 | 0.430322 | 0.090387 + 0.091036 + 0.107580 | 0.289004 | **Winner** |
| Future JEPA | 75,000 | 0.182813 | 0.366525 | 0.439242 | 0.091406 + 0.091631 + 0.109811 | 0.292848 | |
| Future JEPA | 100,000 | 0.188218 | 0.368820 | 0.440077 | 0.094109 + 0.092205 + 0.110019 | 0.296333 | |
| MAE | 25,000 | 0.129035 | 0.349859 | 0.470129 | 0.064518 + 0.087465 + 0.117532 | 0.269515 | |
| MAE | 50,000 | 0.122356 | 0.344792 | 0.466224 | 0.061178 + 0.086198 + 0.116556 | 0.263932 | **Winner** |
| MAE | 75,000 | 0.126783 | 0.341963 | 0.466109 | 0.063391 + 0.085491 + 0.116527 | 0.265409 | |
| MAE | 100,000 | 0.129898 | 0.343820 | 0.465221 | 0.064949 + 0.085955 + 0.116305 | 0.267209 | |
| Future MAE | 25,000 | 0.138356 | 0.349439 | 0.470782 | 0.069178 + 0.087360 + 0.117695 | 0.274233 | |
| Future MAE | 50,000 | 0.131114 | 0.348312 | 0.470056 | 0.065557 + 0.087078 + 0.117514 | 0.270149 | |
| Future MAE | 75,000 | 0.131295 | 0.348342 | 0.469002 | 0.065647 + 0.087085 + 0.117250 | 0.269983 | |
| Future MAE | 100,000 | 0.129222 | 0.348714 | 0.470506 | 0.064611 + 0.087178 + 0.117627 | 0.269416 | **Winner** |

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

1. We fit probes at all 12 transformer-block outputs and the final normalization output.
2. Validation independently selected the output and hyperparameters for Ridge and MLP in each physics cell.
3. Validation selected between the two already output-selected family candidates for the main physics summary.
4. We reused the exact train-fitted probe and training-set normalization on the official test split; there was no validation- or test-set refitting.
5. Depth figures evaluated every saved output, while main and family-specific physics figures used only their validation-selected output.

For each physics cell we reported:

$$
\mathrm{VRMSE},\quad R^2,\quad
r_{\mathrm{Pearson}},\quad
\mathrm{MSE},\quad
\log_{10}(\mathrm{MSE}).
$$

We also evaluated:

- Ridge and MLP separately, each at its own validation-selected output;
- the main validation-selected result, chosen between those two family candidates;
- persistence for future targets by copying the corresponding current target
  (`t+0` persistence is N/A because it is the identity);
- test depth curves for Ridge and MLP at every output, with persistence shown
  as a dashed reference in future pooled panels;
- regime/time and position controls;
- Ridge augmented with the relevant controls;
- recovery of $\log_{10}\mathrm{Rayleigh}$ and $\log_{10}\mathrm{Prandtl}$.

The regime probes used Ridge and MLP but did not influence checkpoint selection. No noise-corruption sweep was included in this Stage 2 result.

**In one sentence:** layer-4 validation performance selected the four checkpoints; after freezing them, validation selected Ridge and MLP settings across all 13 encoder outputs, and test data was used only for final scoring and depth visualization.
