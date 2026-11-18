# Guesser results

The leaderboard for the guesser (the classifier that turns views into a location). Every variant is
scored by the same script on the same data, so the rows can be compared directly.

**Data:** `world_v1_sc` (312 regions, StreetCLIP embeddings). Splits are by region, so val and test
cities never appear in training.

## Adding your model

1. **Write it.** Add a class in a new module under `models/guesser/` that implements
   `core.interfaces.Guesser` (`cell_probs`, `guess`) plus `save_to(folder)` and a classmethod
   `load_from(folder)`, decorated with `@register("your_kind")`. `CellGuesser` in
   `models/guesser/classifier.py` is a working example.
2. **Train on train, tune on val.** Never pick settings by looking at test. Save the final model
   with `models.guesser.registry.save_model(guesser, name="your_model", dataset="world_v1_sc",
   description="one line")`. It lands in `runs/models/your_model/`.
3. **Evaluate on val:**
   ```bash
   uv run python scripts/data_hub.py pull-model softmax_k256      # the baseline, once
   uv run python scripts/evaluate.py --model your_model
   ```
   The report is in `runs/eval/your_model/val/report.md`. It ends with a table row: paste that
   row into the val table below.
4. **When you're done tuning, evaluate on test once** (`--split test`) and add that row too.
5. **Share it:** `uv run python scripts/data_hub.py push-model your_model`. Then open a PR with
   your code and your rows in this file.

**Reading the last column:** your model minus the baseline, both guessing the same panoramas
(a *paired* comparison), with a 95% interval from resampling whole cities. If the interval excludes
0, the difference is real. If it straddles 0, it could be luck. Comparing overall scores between
rows without this is unreliable: val and test have only about 25 cities each, and city difficulty
varies far more than models do.

## Leaderboard

Columns: median error (km) and mean score (out of 5,000) with 1 view (each view guessed on its own)
and 8 views (all combined), and the 8-view mean score difference from the baseline.

### Val (choose models here)

| model | by | split | 1 view: median km | 1 view: score | 8 views: median km | 8 views: score | vs baseline, 8 views (95% CI) | description |
|---|---|---|---|---|---|---|---|---|
| `softmax_k256` | dhu27 | val | 1156 | 2286 | 1089 | 2422 | (baseline) | Softmax head over 256 geocells (l2=0.01); M2 baseline |
| `nearest_mean_k256` | dhu27 | val | 2295 | 1492 | 1346 | 1982 | -440 (-621 to -252) | M0 placeholder: cosine to each cell mean (256 cells) |

### Test (final numbers, once per model)

| model | by | split | 1 view: median km | 1 view: score | 8 views: median km | 8 views: score | vs baseline, 8 views (95% CI) | description |
|---|---|---|---|---|---|---|---|---|
| `softmax_k256` | dhu27 | test | 786 | 2617 | 587 | 2812 | (baseline) | Softmax head over 256 geocells (l2=0.01); M2 baseline |

## Notes

- **Look-around (decision point G1):** for `softmax_k256`, combining 8 views lowers the median
  error by 25% on test (95% CI 0.3% to 38%) and 6% on val (−0.2% to 20%). Both intervals include
  the suggested 15% threshold, so it's inconclusive. The mean score gain is clearly positive in
  both (+194 test, +137 val). Each report shows this gain for its own model.
- **Weak spots:** Africa and Oceania, where val and test have only 3 cities each and training
  data is thin (Europe is about 46% of regions).
