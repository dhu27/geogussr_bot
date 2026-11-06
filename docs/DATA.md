# Data runbook

**Git holds code, the Hugging Face hub holds data.** Datasets and weights never go in git
(ground rule 3). The team shares them through one **private Hugging Face dataset repo**,
named in `configs/data.yaml` as `hub_repo`.

```
person who downloads                          everyone else
--------------------                          -------------
scripts/download.py  --name pilot_v1
scripts/embed.py     --name pilot_v1
scripts/data_hub.py push pilot_v1   ──HF──▶   scripts/data_hub.py pull pilot_v1
          datasets/pilot_v1/                            datasets/pilot_v1/
```

A dataset **name is its version** (`pilot_v1`, `pilot_v2`, `world_v1`). Once a name is pushed it
is never changed: a fix or a re-download gets a new name, so everyone's results stay
reproducible.

## One-time setup (one person, ~10 minutes)

1. **Create a Hugging Face organization** (free) at <https://huggingface.co/organizations/new>,
   for example `geoguessr-rl-ucla`. An organization is needed because private repos under a
   personal account can't be shared with other people.
2. **Invite every teammate** to it with the `write` role (Organization settings → Members).
3. **Set the repo name** in `configs/data.yaml` and merge it through a PR:
   ```yaml
   hub_repo: geoguessr-rl-ucla/geoguessr-rl-data
   ```
4. **Create the repo** (private, with a dataset card that carries the CC BY-SA license):
   ```bash
   uv run python scripts/data_hub.py create
   ```

## Each teammate (once)

1. Make a Hugging Face account and accept the organization invite.
2. Create a token at <https://huggingface.co/settings/tokens>: **write** if you will push,
   **read** is enough to pull. Put it in your `.env` (never in `.env.example`):
   ```
   HF_TOKEN=hf_...
   ```
   Or run `uv run hf auth login` instead.
3. Check access: `uv run python scripts/data_hub.py list` should print without errors.

## Doing the download (M1)

### 1. Make the downloader real

`data/downloader.py` has two sources behind one entry point: `source="fixtures"` (the default,
copies `tests/fixtures/`; `run_e2e.py` and CI use it, so they never call Mapillary) and
`source="mapillary"`, the real downloader (`scripts/download.py --source mapillary`). The
regions to sample are `pilot.regions` in `configs/data.yaml`: a region is a block of zoom-14
coverage tiles around a centre point (`data/tiles.py`; `data/mvt.py` reads the tile format, so
no extra package is needed). The output contract is:

- `out_dir/images/<image_id>.jpg`: equirectangular panoramas (2:1 aspect; drop anything else).
- `out_dir/images.parquet`: the columns in `core/schemas.py`, written with
  `data.dataset.write_images`, which validates the schema and checks every file exists.
- Assign `split` with `data.splits.assign_splits` (by region and creator, ground rule 2).
- Save `sequence_id` and positions for every image (Phase 2 builds its graph from them).

What we learned about the Mapillary API while building the fixtures (Nov 2026):

- **Box search (`/images?bbox=`) can't do bulk work.** It takes 1–20 s per call and returns
  500s in dense cities, for boxes over about 1 km, or when asked for thumbnail URLs. Building
  44 fixture panoramas took about 8 minutes, and 3 of 25 cities failed completely.
- **Use the coverage vector tiles for sampling** (PLAN.md 7.1). They list image ids with
  `is_pano`, `sequence_id` and `creator_id`, without running a search on the server.
- **Per-image lookups are fast** (`MapillaryClient.image(id)`, about 0.2 s). Use them for
  details and the thumbnail URL. A 1024 px thumbnail is about 70 KB.
- Retry on 5xx errors and timeouts (the client already does), and make the downloader
  resumable: skip images already on disk, like `scripts/make_fixtures.py` does.
- **Resolution:** a 90° view of a 1024 px panorama has only 256 source pixels. Before the full
  download, pull about 300 pilot panoramas at both `thumb_1024_url` and `thumb_2048_url`,
  embed both, and compare. If the cosine similarity is ≥ 0.97 or the guesser's accuracy barely
  changes, keep 1024 (a third of the disk space); otherwise switch to 2048.

### 2. Smoke-test the whole path first (no Mapillary calls)

With the placeholder still in place you can check access to the hub end to end:

```bash
uv run python scripts/download.py --name smoke_test      # copies the fixtures
uv run python scripts/embed.py    --name smoke_test
uv run python scripts/data_hub.py push smoke_test
uv run python scripts/data_hub.py list
```

Then delete `smoke_test/` on the hub (web UI → Files → folder → Delete) so it doesn't clutter the list.

### 3. The real run

First a small trial (a few regions, a few images each) to check your token and the output:

```bash
uv run python scripts/download.py --name trial --source mapillary \
    --regions eu_london,sa_lima --images-per-region 10
```

Look at the printed summary, then delete `datasets/trial/`. Then the real run:

```bash
uv run python scripts/download.py --name pilot_v1 --source mapillary   # -> datasets/pilot_v1/
```

It prints one line per region and a summary (images per continent, regions below
`pilot.min_images_per_region`, failed regions, and the `quality_score` spread). A failed region is
logged and skipped, and the run stops early only if the first three regions all fail (usually a
bad token or no network). If it is interrupted, run the same command again: lookups are cached in
`datasets/pilot_v1/.cache/` and finished images are kept, so it picks up where it stopped (if the
run had already finished, pick a new `--name`). The `.cache/` folder is never uploaded and is safe
to delete after the push. Use the quality summary to tune `min_quality_score` in
`configs/data.yaml`: regions with too few images or missing continents are fixed by editing
`pilot.regions` and re-running under a new name.

```bash
uv sync --extra ml                                       # torch + transformers for StreetCLIP
uv run python scripts/embed.py --name pilot_v1 --encoder streetclip
uv run python scripts/data_hub.py push pilot_v1          # resumable: just re-run if it drops
```

`push` refuses a dataset that fails validation (bad schema, missing images, or a creator or
region in two splits). It uploads images as tar shards of 1000 and writes `MANIFEST.json` last.
`list` marks a dataset complete only once its manifest is there.

### 4. Tell the team

Post the dataset name. Everyone runs:

```bash
uv run python scripts/data_hub.py pull pilot_v1          # -> datasets/pilot_v1/
```

## Reference

| What | Where |
|---|---|
| Local datasets | `datasets/<name>/` (gitignored), or `$GEO_DATA_ROOT/<name>/` |
| Hub repo | `hub_repo` in `configs/data.yaml` |
| Tokens | `.env`: `MAPILLARY_TOKEN`, `HF_TOKEN` |
| Code | `data/hub.py` (push/pull), `data/downloader.py`, `data/mapillary.py`, `env/embed.py` |

Rough sizes: about 70 KB per 1024 px panorama (about 250 KB at 2048), plus about 12 KB of
embeddings per panorama. A 5,000-panorama pilot is about 400 MB.

**Troubleshooting**

- `401` / `404` from the hub: wrong or missing `HF_TOKEN`, a read-only token on push, or you
  haven't accepted the organization invite.
- `already exists ... immutable`: pick a new name (`pilot_v2`).
- `INCOMPLETE` in `list`: a push was interrupted. Re-run the same `push`; it skips finished shards.
- The pull warns about a different view spec: the embeddings were made under an older
  `core/view_spec.py`. Re-embed, or pull a newer dataset.
