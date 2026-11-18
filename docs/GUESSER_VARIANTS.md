# Building a guesser variant

Each of us builds one variant of the **guesser**, the model that turns street-view embeddings into
a location guess, and scores it against the current baseline. Whichever variant wins becomes the
model the agent plays with, and the one the Phase 2 navigation policy is built on.

**Deadlines:** val result by **Sun Nov 29** (M3); test result and model upload by **Wed Dec 2**,
ready for the Dec 4 report.

---

## 1. The spec

### Files you create

| File | Required | What it is |
|---|---|---|
| `models/guesser/<your_variant>.py` | yes | Your model class (the contract below). One file. |
| `scripts/train_<your_variant>.py` | yes | Loads the data, trains, tunes on val, saves the model with `save_model`. |
| `tests/test_<your_variant>.py` | optional | Small tests on random data, if you want them. |

**The only shared file you edit is `docs/RESULTS.md`:** add your val and test rows to its tables,
and 2–3 sentences to its Notes section.

### Files you don't touch

Everything else is shared, and changing it would make your numbers incomparable with everyone
else's:

- `models/guesser/`: `classifier.py`, `combine.py`, `geocells.py`, `evaluation.py`, `registry.py`,
  `random.py`
- `scripts/`: `evaluate.py`, `train_guesser.py`, `play.py`, `data_hub.py` and the rest
- `core/`, `data/`, `env/`, `game/`, `harness/`, `configs/`, `pyproject.toml`, existing tests

If you need a change in one of these (a bug, a missing helper, a new dependency), raise it
separately rather than putting it in your variant PR.

### What your model file must contain

A class decorated with `@register("<kind>")`, where `<kind>` is a unique name for your type of
model. It must implement four methods:

| Method | Input | Output |
|---|---|---|
| `cell_probs(view_embs)` | `(t, 768)` float32 StreetCLIP embeddings (unit length) of the views seen so far, **t from 1 to 8** | A belief: `(K,)` probabilities over your K cells, summing to 1 |
| `guess(probs)` | The output of `cell_probs` | `(lat, lon)` in degrees |
| `save_to(folder)` | A folder path | Writes everything the model needs **into that folder** |
| `load_from(folder)` (classmethod) | The same folder | The same model, giving the same guesses |

**The easy way:** subclass `_GeocellGuesser` (the template in section 3). The base class already
implements all four methods. You only write:
- **`_per_view_probs`:** one embedding to cell probabilities;
- **`fit`:** training;
- **`_arrays` / `_from_arrays`:** the arrays to save and restore.

Override `cell_probs` only if you change how views are combined.

### Constraints

- **Input is fixed:** the StreetCLIP embeddings in `world_v1_sc`. Evaluation feeds your model
  embeddings only, never images, and you don't re-embed.
- **Data use:**
  - Train only on `split == "train"`.
  - Choose settings (hyperparameters, epochs, architecture) on `val`.
  - Run `test` **once**, at the end.
- **1 view and 8 views both have to work.** Evaluation scores every view on its own *and* all 8
  combined, and the game calls your model after each new view.
- **Self-contained at inference:** `load_from(folder)` may only read that folder. For example, a
  nearest-neighbour model must save its index there, not reload the dataset.
- **Size and speed:** keep the model folder under ~500 MB, since it gets uploaded. Scoring val
  (~22,000 guesses) should take minutes, not hours.
- **Dependencies:** numpy, pandas and PyTorch (via `--extra ml`) are available. Ask before adding
  a package, since `pyproject.toml` is shared.
- **Reproducible:** fix your random seeds, and record your settings in `config=` when you call
  `save_model`.
- **Names:**
  - `<kind>` is your variant, e.g. `mlp_head`.
  - The saved model name is `<variant>_<yourname>_v1`.
  - Model names are permanent once pushed, so a retrained model gets `_v2`.

---

## 2. Set up (once, ~5 minutes)

```bash
git checkout main && git pull
uv sync                      # or: uv sync --extra ml   (if you'll use PyTorch)
```

Your `.env` needs your own `HF_TOKEN` (see `docs/DATA.md`, "One-time setup"). Then:

```bash
uv run python scripts/data_hub.py pull world_v1_sc --tables-only   # embeddings, ~400 MB
uv run python scripts/data_hub.py pull-model softmax_k256          # the baseline model
uv run python scripts/evaluate.py --model softmax_k256             # sanity check, ~30 s
```

The last command should print the baseline's val numbers from `docs/RESULTS.md`: 1 view 1156 km /
2286 points, 8 views 1089 km / 2422 points. If it does, you're set.

**If you use PyTorch:** always run with `uv run --extra ml ...`. A plain `uv run` re-syncs the
environment and removes torch.

## 3. Build your model

Make a branch (`git checkout -b guesser/<variant>`), then copy this into
`models/guesser/<your_variant>.py`. As written it's a working copy of the baseline; change the two
lines marked `<-`.

```python
"""My variant: one line on the idea. Owner: <your name>."""

from __future__ import annotations

import numpy as np

from models.guesser.classifier import _GeocellGuesser, _prior, _softmax, train_softmax
from models.guesser.geocells import fit_geocells
from models.guesser.registry import register


@register("my_variant")  # a unique name for your kind of model
class MyGuesser(_GeocellGuesser):
    def __init__(self, centroids, prior, weights, bias, mean, scale):
        super().__init__(centroids, prior)  # sets up the cells and the expected-score guess
        self.weights, self.bias, self.mean, self.scale = weights, bias, mean, scale

    def _per_view_probs(self, view_embs):
        """(t, 768) view embeddings -> (t, K) cell probabilities, each row summing to 1."""
        x = (np.atleast_2d(view_embs) - self.mean) / self.scale
        return _softmax(x @ self.weights + self.bias)  # <- your model goes here

    @classmethod
    def fit(cls, x, lat, lon, *, n_cells=256, l2=0.01, epochs=300):
        """x: (n_views, 768) TRAIN-split view embeddings; lat, lon: (n_views,) their locations."""
        centroids, labels = fit_geocells(lat, lon, n_cells, 50, 0)
        k = len(centroids)
        mean, scale = x.mean(0), x.std(0) + 1e-6
        w, b = train_softmax((x - mean) / scale, labels, k, l2=l2, epochs=epochs)  # <- your training
        return cls(centroids, _prior(labels, k), w, b, mean, scale)

    def _arrays(self):
        """Everything needed to rebuild the model (saved to guesser.npz with the cells)."""
        return {"weights": self.weights, "bias": self.bias, "mean": self.mean, "scale": self.scale}

    @classmethod
    def _from_arrays(cls, a):
        return cls(a["centroids"], a["prior"], a["weights"], a["bias"], a["mean"], a["scale"])
```

**What the base class gives you for free:**
- **Combining views:** `cell_probs` combines the per-view probabilities as independent evidence.
- **The guess:** the point with the highest expected score under your probabilities.
- **Saving and loading:** `save_to` / `load_from` write and read everything returned by `_arrays`.

**Not cell-based?** For retrieval that returns locations directly, for example, skip the base class
and write the four methods from section 1 yourself, still under `@register`.

**PyTorch models:** override `save_to` / `load_from` to write and read a `.pt` file in the folder,
rather than going through `_arrays`.

## 4. Train it, tuning on val

Put this in `scripts/train_<your_variant>.py` and adapt it:

```python
import numpy as np

from data.dataset import load_images, load_views
from data.hub import dataset_dir
from models.guesser.evaluation import predict, summarize, view_matrix
from models.guesser.my_variant import MyGuesser
from models.guesser.registry import save_model

root = dataset_dir("world_v1_sc")
images = load_images(root, check_files=False)
views, emb = load_views(root)
train = images[(images["split"] == "train") & images["is_pano"]]
x = view_matrix(views, emb, train["image_id"].tolist())  # (n_panoramas, 8, 768)
n, h, d = x.shape
x = x.reshape(n * h, d)  # one row per view
lat = np.repeat(train["lat"].to_numpy(), h)
lon = np.repeat(train["lon"].to_numpy(), h)

best, best_score = None, -1
for l2 in [0.001, 0.01, 0.1]:  # try settings, judged on VAL only
    model = MyGuesser.fit(x, lat, lon, l2=l2)
    m = summarize(predict(model, images, views, emb, "val"))
    score = m["1_view"]["overall"]["mean_score"]
    print(f"l2={l2}: val mean score {score:.0f} (1 view)")
    if score > best_score:
        best, best_score = model, score

save_model(best, name="my_variant_yourname_v1", dataset="world_v1_sc",
           description="one line for the leaderboard", config={"l2": 0.01})
```

**The data:**
- 25,928 training panoramas, 8 views each, so about 207,000 training rows of 768 numbers. That's
  about 640 MB in memory.
- Val and test are about 25 cities each that never appear in training.
- `images` has `lat`, `lon`, `country`, `continent`, `region_id` and `split` per panorama.

**Speed:** the baseline trains in about 2 minutes per setting on a laptop; scoring val takes about
30 seconds.

## 5. Evaluate and report

```bash
uv run python scripts/evaluate.py --model my_variant_yourname_v1
```

The report (`runs/eval/<model>/val/report.md`) shows:
- your metrics for 1 view and 8 views, overall and per continent;
- how much looking around helps your model;
- a **paired comparison with the baseline**: both models guess the same panoramas, with a 95%
  interval. **If the interval excludes 0, your model is really better or worse. If it straddles 0,
  the difference may be luck.** Val has only 25 cities, so compare models with this, not by
  eyeballing overall scores.

It ends with a table row. Paste it into the **val** table in `docs/RESULTS.md`.

**When you've finished tuning, and only then**, run the test split once:

```bash
uv run python scripts/evaluate.py --model my_variant_yourname_v1 --split test
```

Paste that row into the **test** table. If you look at test, tweak and look again, the test number
stops meaning anything.

A variant that doesn't beat the baseline is still a result worth recording.

## 6. Share it

```bash
uv run python scripts/data_hub.py push-model my_variant_yourname_v1
```

Then open a PR containing:
- [ ] `models/guesser/<your_variant>.py` and `scripts/train_<your_variant>.py`, and nothing else
      outside `docs/RESULTS.md`
- [ ] your val and test rows in `docs/RESULTS.md`, plus 2–3 sentences in its Notes section: what
      you tried, what worked, what didn't
- [ ] no data or weights in git (they live under `runs/` and `datasets/`, which are ignored)
- [ ] tests and lint pass: `uv run pytest -q -m "not browser and not ml"` and `uv run ruff check .`

Anyone can then reproduce your numbers with `pull-model` + `evaluate.py`, or watch your model play
the game with `scripts/play.py --model <name>` (that needs the full dataset with images).

## Troubleshooting

| Problem | Fix |
|---|---|
| `no model 'softmax_k256' in runs/models/` | `uv run python scripts/data_hub.py pull-model softmax_k256` |
| `unknown kind 'my_variant'` | The `@register` name must match, and your file must be in `models/guesser/` |
| `No module named torch` | `uv sync --extra ml`, then always `uv run --extra ml ...` |
| HF 401 / 403 | Check `HF_TOKEN` in `.env`, and that you've accepted the ResearchGeogussr org invite |
| Download stalls | Re-run the same command: pulls resume where they stopped |
