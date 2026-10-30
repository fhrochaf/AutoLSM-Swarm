# Adding a dataset

Everything specific to a dataset sits behind one `DatasetSpec`
([`spec.py`](../src/data/spec.py)), which is looked up by `dataset_name`. That spec is
the source for:

- the loading code
- every dataset description the LLMs see: prompts, contract docstrings and the retrieval
  query

Adding a dataset only touches `src/data/`.

## Steps

1. **Create `src/data/datasets/<name>.py`** with:
   - `load_split(split, dataset_dir, max_tiles=None) -> (X, y, ids)`: loads `"train"`,
     `"val"` or `"test"`. For id-indexed data, reuse `split_ids_by_ratio` from
     [`dataset.py`](../src/data/dataset.py).
   - `load_smoke_sample(dataset_dir, ids) -> (X, y)`: loads the few samples used by the
     smoke test.
   - `DESCRIPTION`: a prose description for the LLM. Cover scene size, channels and what
     they mean, whether the data is temporal, and what the labels mean.
   - `SPEC = DatasetSpec(...)`: sets `name`, `description`, `input_shape_doc` /
     `label_shape_doc` (e.g. `"(N,H,W,C)"`), `smoke_tile_ids` and the two loaders.
2. **Check `smoke_tile_ids` by hand.** Pick a few train-split ids that include at least one
   positive and one negative sample.
3. **Register the spec** by adding `SPEC` to `_REGISTRY` in
   [`registry.py`](../src/data/registry.py).
4. **Select the dataset** by setting `dataset_name` and `dataset_dir` in `config.py` or
   `.env`.

[`landslide4sense.py`](../src/data/datasets/landslide4sense.py) is a complete reference
implementation.
