# NFTool — Review Pass / Findings Report

**Scope reviewed:** `NFTool_V3_071125A.py` (1,998 lines, single monolithic
Tkinter + PyTorch script).

> **Important context:** the original prompt describes a FastAPI + Next.js +
> Tauri app under `backend/src/` and `frontend/src/`. **That code does not exist
> in this directory** (`NFTool-old/`). There is no training-engine module, job
> queue, router, schema package, React inspector, or `CLAUDE.md`/`TECHNICAL.md`.
> Findings below are against the code that is actually here. Where the prompt
> names a specific footgun (e.g. `BatchNorm1d` batch-size, multiprocessing job
> queue, WebSocket), I note whether it exists in this version.

Severity: **High** = wrong results / crash / silently trains the wrong model ·
**Medium** = incorrect in an edge case or methodology smell · **Low** =
cosmetic / cleanliness.

---

## 1. Correctness bugs

| # | Sev | Location | Issue |
|---|-----|----------|-------|
| 1 | **High** | `objective` `:1037` + `:1041` | `seed = ... np.random.randint(...)` runs **before** the function-local `import ... numpy as np` on line 1041. Because `np` is assigned (imported) later in the same scope, Python treats `np` as local for the whole function → `UnboundLocalError` whenever the GUI seed field is left blank (`SEED is None`). The "randomize seed" path is broken. |
| 2 | **High** | module level `:1320`–`:1325`, `:1368`–`:1371` | After the CNN search, the "final" model's `config` is built **without `layer_size`** (only `hidden_dim/dropout/lr/optimizer`), then `config.setdefault('layer_size', 32)` forces conv width to 32 — so the final CNN ignores the tuned conv width and trains a different model than Optuna selected. |
| 3 | **High** | inference `:573` (and dead `:128`) | `scaler.fit_transform(...)` is called on **inference/validation** inputs. Scaler must `transform` only; re-fitting at inference is classic scaler drift and produces wrong predictions. |
| 4 | **High** | module level `:1197` vs `:1204`–`:1993` | Training is module-level code that runs as an **import side effect**. `main()` is called at `:1197`, returns for the training path, and execution "falls through" into 800 lines of top-level training. Importing this file launches the GUI and a training run. (A second dead `if __name__ == "__main__"` sits at `:1996`, unreachable after `sys.exit(0)` at `:1993`.) |
| 5 | Med | `train_model` `:957` + `:976` | `history['train'].append(loss_val)` is executed **twice per epoch**, so `history['train']` is twice as long as `val`/`r2`/`mae`. The loss plot at `:1674` plots misaligned train vs val curves. |
| 6 | Med | inference `:582`–`:584` | `scaled_p = preds` is a CUDA/CPU **tensor**, passed straight into `r2_score(scaled_y, scaled_p)` without `.cpu().numpy()`. Fragile / can raise depending on sklearn/torch versions. |
| 7 | Med | EVALUATE `:1399`–`:1407` | `X_test_cnn = preprocess_for_cnn(X_test_np)` is applied **unconditionally**, then fed to `model` even on the NN path, where the net expects 2-D input. It "works" by accident (Linear acts on the last dim) but evaluates a wrong-shaped tensor. `X_test_np` is then reassigned right after, making the first block effectively dead. |
| 8 | Med | `:1235`–`:1237` | `StandardScaler` is `fit_transform`-ed on the **entire dataset before the train/val/test split** → test-set leakage. Metrics are optimistic. |
| 9 | Med | `train_cnn_model` `:814`/`:822` | `best_model_state` is only assigned inside the `if val_loss < best` branch; `model.load_state_dict(best_model_state)` at `:822` raises `UnboundLocalError` if no epoch ever improves (e.g. immediate NaN path on a degenerate trial). |
| 10 | Med | `prompt_and_load` `:136` | `return df_X, df_y, input_size, X, y` references `X` and `y`, which are never defined in that function (it builds `X_scaled`, `y_array`, `X_val`, `y_val`). `NameError` if ever called. Function is currently unused. |
| 11 | Med | `compute_dataset_snr_from_files` `:152` | The error string interpolates `df_X`/`df_y`, undefined in this scope (locals are `X`, `y`). The error handler itself throws. |
| 12 | Low | `:834` `best_r2_so_far` | Module-global `[-inf]` is never reset between the NN and CNN studies; "best so far" logging leaks across runs in a long session. |
| 13 | Low | `:89` | `graph_path = "...model_architecture_{timestamp}.png"` is a plain string, not an f-string — `{timestamp}` is never substituted (later overwritten at `:1812`, so harmless). |

**Prompt-named footguns that do _not_ apply to this version:** there is **no
`BatchNorm1d`** anywhere, and training is **full-batch** (`output = model(X_train)`
over the whole tensor — no `DataLoader`, no `batch_size`, no `drop_last`). The
"`batch_size >= 2`" invariant has nothing to enforce against here. There is also
no multiprocessing job queue and no WebSocket — those belong to the app the
prompt describes, not this script.

---

## 2. Coupling that blocks generalization

Every place that branches on the `model_choice` string (`"NN"` / `"CNN"`). Adding
a third architecture today means editing **all** of these:

| Location | What it hardcodes |
|----------|-------------------|
| `build_model` `:658`, `:669` | `if checkpoint["model_choice"]=="NN": RegressionNet(...) else CNNRegressionNet(...)` |
| `build_model_training` `:689`–`:704` | `if model_choice=="NN" ... elif "CNN" ... else raise` |
| `objective` `:1062`, `:1077`, `:1082` | per-type hyperparameter suggestion + per-type training call |
| `cnn_objective` `:836` | an **entirely separate** Optuna objective for CNN, parallel to `objective` |
| data check `:1247`–`:1249` | `if model_choice=="CNN": require features>=16` |
| optuna block `:1303`, `:1327` | two separate study-driving branches |
| final train `:1368`, `:1382` | two separate training branches |
| best-R² retrain `:1449`, `:1474` | two more |
| unified eval `:1524` | `if CNN: preprocess_for_cnn(...)` |
| config/checkpoint save `:1707`, `:1729`, `:1734`, `:1761` | per-type config keys |
| save-time summary `:1779`, `:1792` | per-type printout |

`preprocess_for_cnn` (`:710`) is the input-shaping concern leaking everywhere it
is conditionally called. **~12 branch sites for what should be one dispatch.**

The hyperparameter ranges are hardcoded in the Tk GUI (`:431`–`:455`) and handed
back as a **25-element positional list** `result[0..22]` (`:292`–`:303`,
unpacked at `:514`–`:531`). Any reordering silently corrupts every downstream
config value. This is the data-driven config the prompt wants moved into the
registry.

---

## 3. Refactor / simplify / dead code

- **Duplicate HTML plot categorization** — the exact same "scan `report_dir`,
  bucket pngs into optuna/intermediate/split/other" loop is written twice
  (`:1824`–`:1851` and `:1866`–`:1899`); the first result is discarded.
- **Two `RegressionNet` training paths** — `train_model` (`:918`) and
  `train_cnn_model` (`:754`) duplicate the whole loop (optimizer build, early
  stop, NaN guard, history) with minor drift between them.
- **Two objectives** — `objective` (`:1033`) already has a CNN branch but the
  CNN run actually uses `cnn_objective` (`:836`); the CNN half of `objective`
  is dead and can diverge.
- **`prompt_and_load` (`:105`) and `to_2d_numpy` (`:138`)** are essentially
  unused; `prompt_and_load` is also broken (finding #10).
- **Redundant in-function imports** — `import torch`, `pandas`, `numpy` re-imported
  inside `main()` (`:499`) and `objective()` (`:1041`); the latter causes bug #1.
- **Magic numbers** — `num_epochs = 200` global (`:71`), CNN conv kernels/strides,
  `>=10`/`>=16` thresholds, default `layer_size=32` fallback — all belong with
  their architecture.
- **Duplicate assignments** — `best_model_path` set twice (`:1723`, `:1727`).

---

## 4. API / UI contract risks (analog)

There is no API or UI in this repo, but the equivalent contract seams are:

- **The 25-tuple GUI return** (`:514`–`:531`) is a positional contract between
  `prompt_initial_settings` and `main`. It is the local equivalent of a
  frontend/backend schema, and it is purely positional — the riskiest possible
  form.
- **`model_choice` magic strings** (`"NN"`/`"CNN"`) must agree across ~12 sites
  plus the saved `checkpoint["model_choice"]` and `config.txt`.
- **`config.txt` mislabels fields** — for NN it writes `hidden_dim: {num_layers}`
  (`:1762`), so the persisted config disagrees with the checkpoint and with what
  `build_model` expects on reload.
- **Optimizer names** from the bitmask map (`:274`) must exactly match
  `getattr(torch.optim, name)`; there's no validation, so a typo fails only at
  trial time.

---

## 5. What the refactor addresses

The registry/factory layer added in `nf/` (see `ARCHITECTURES.md`) directly
removes §2 and §4: one `ArchitectureSpec` per model, a single
architecture-agnostic engine, and a declarative hyperparameter space that
doubles as the `/architectures` schema. It also fixes, in the new code path,
bugs #1 (no shadowed import), #2 (the engine trains exactly the searched
config), #3 (`prepare_inputs` shapes without re-fitting), #5 (single `train`
append), and centralizes the `batch_size >= 2` / `drop_last` invariant
(`resolve_batch_size`) for any future BatchNorm-using architecture. The original
monolith is left intact; the new layer is additive and independently runnable
(`demo.py`).
