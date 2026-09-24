"""Local CPU smoke test of the exact notebook library code (smoke_lib.py)."""
import os, sys, json, math, random, time
import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "output")  # 1,191 training.jsonl files (see README: data release)
RUN_DIR  = os.path.join(os.path.dirname(__file__), "smoke_run")
os.makedirs(RUN_DIR, exist_ok=True)
for f in os.listdir(RUN_DIR):
    os.remove(os.path.join(RUN_DIR, f))
device = "cpu"
NUM_WORKERS = 0   # macOS spawn can't see exec()-defined classes; Colab uses fork

exec(open(os.path.join(os.path.dirname(__file__), "smoke_lib.py")).read())

# tiny config
CFG = dict(d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.1,
           seq_len=256, batch_size=4, lr=3e-4, weight_decay=0.01,
           epochs=2, warmup=5, grad_clip=1.0,
           val_frac=0.2, val_windows=2, min_freq=2, seed=42)
SEEDS = [42, 7]

t0 = time.time()
vocab = build_vocab(DATA_DIR, min_freq=CFG["min_freq"])
print(f"vocab (full data, normalized): {len(vocab)}  [{time.time()-t0:.0f}s]")
pad_id = vocab[PAD]

pages = load_pages(DATA_DIR, vocab, limit=24)
rng = random.Random(CFG["seed"]); order = list(range(len(pages))); rng.shuffle(order)
n_val = max(1, int(len(pages) * CFG["val_frac"]))
val_pages = [pages[i] for i in order[:n_val]]; train_pages = [pages[i] for i in order[n_val:]]
print(f"train {len(train_pages)} val {len(val_pages)}")

# ── sanity: delta reconstruction & condition transforms ──
p = pages[0]
sm = p["types"] == STROKE_TYPE
recon = torch.cumsum(p["delta"][sm] / DELTA_SCALE, 0)
err = (recon - p["abs"][sm]).abs().max().item()
print(f"delta→abs reconstruction max err: {err:.2e}")
assert err < 1e-4
for cond in CONDITIONS:
    ds = FlatOCTDataset(val_pages, vocab, CFG["seq_len"], cond, deterministic=True, n_windows=2)
    s0, s0b = ds[3], ds[3]
    assert torch.equal(s0["types"], s0b["types"]), "val windows not deterministic"
    b = collate([ds[0], ds[1], ds[3]], pad_id)
    nT = int((b["types"] == WORD_TYPE)[~b["pad"]].sum()); nS = int((b["types"] == STROKE_TYPE)[~b["pad"]].sum())
    extra = ""
    if cond == "mask_strokes":
        assert b["delta_in"][b["types"] == STROKE_TYPE].abs().sum() == 0 and b["delta"][b["types"] == STROKE_TYPE].abs().sum() > 0
        assert nT + nS == int((~b["pad"]).sum()) - 0  # nothing hidden -> all non-batch-pad tokens visible
        extra = " (stroke inputs zeroed, targets intact ✓)"
    if cond == "mask_words":
        real = (b["types"] == WORD_TYPE) & ~b["pad"] & ~torch.isin(b["wids"], ds.special_ids)
        assert (b["wids_in"][real] == vocab[MASK]).all() and (b["wids"][real] != vocab[MASK]).all()
        extra = f" ({int(real.sum())} words masked, targets intact ✓)"
    if cond == "drop_strokes": assert nS == 0
    if cond == "drop_words":  assert nT == sum(int(((b['wids'] == vocab[BOS]) | (b['wids'] == vocab[EOS]))[~b['pad']].sum()) for _ in [0])
    if cond == "nostroke_hidden":
        # same total LENGTH as mask_strokes (types tensor same size), but stroke
        # positions are now inside the pad mask (invisible), not just feature-zeroed
        ms = FlatOCTDataset(val_pages, vocab, CFG["seq_len"], "mask_strokes", deterministic=True, n_windows=2)
        b_ms = collate([ms[0], ms[1], ms[3]], pad_id)
        assert b["types"].shape == b_ms["types"].shape, "nostroke_hidden must match mask_strokes length"
        assert torch.equal(b["types"], b_ms["types"]), "same underlying window as mask_strokes"
        hidden_are_stroke = (b["types"][b["pad"] & ~(b_ms["pad"])] == STROKE_TYPE)
        assert hidden_are_stroke.numel() > 0 and hidden_are_stroke.all(), "only stroke positions should be newly hidden"
        extra = f" ({int((b['pad'] & ~b_ms['pad']).sum())} stroke positions hidden from attention, length unchanged ✓)"
    if cond == "noword_hidden":
        mw = FlatOCTDataset(val_pages, vocab, CFG["seq_len"], "mask_words", deterministic=True, n_windows=2)
        b_mw = collate([mw[0], mw[1], mw[3]], pad_id)
        assert torch.equal(b["types"], b_mw["types"]), "same underlying window as mask_words"
        newly_hidden = b["pad"] & ~(b_mw["pad"])
        assert newly_hidden.any() and (b["types"][newly_hidden] == WORD_TYPE).all()
        extra = f" ({int(newly_hidden.sum())} word positions hidden from attention, length unchanged ✓)"
    print(f"  {cond:16s}: batch words={nT:5d} strokes={nS:5d}{extra}")

# ── regression test for the NaN bug: force windows to start mid-page (no
# BOS), where the first token is often the type being hidden, and run a full
# forward+backward through the model to make sure no all-masked causal row
# survives (this is exactly what a real ~78%-stroke mid-page window hits) ──
for cond in ("nostroke_hidden", "noword_hidden"):
    ds = FlatOCTDataset(train_pages, vocab, CFG["seq_len"], cond, deterministic=True, n_windows=1)
    ds.start_frac = ds.end_frac = 0.0   # force every window to be a uniform mid-page start
    hit_stroke_first = hit_word_first = 0
    m = FlatOCTModel(len(vocab), 16, 1, 2, 16, max(2 * CFG["seq_len"], 2048), 0.0, pad_id).to(device)
    for i in range(len(ds)):
        item = ds[i]
        if item["types"][0].item() == STROKE_TYPE: hit_stroke_first += 1
        if item["types"][0].item() == WORD_TYPE: hit_word_first += 1
        b = collate([item], pad_id)
        b = {k: v.to(device) for k, v in b.items()}
        out = forward_batch(m, b)
        L = m.loss(out, b)
        assert math.isfinite(L["total"].item()), f"{cond} window {i}: NaN/inf loss (type[0]={item['types'][0].item()})"
        L["total"].backward()
        for p in m.parameters():
            assert p.grad is None or torch.isfinite(p.grad).all(), f"{cond} window {i}: non-finite grad"
        m.zero_grad(set_to_none=True)
    print(f"  {cond}: {len(ds)} mid-page windows, all finite "
          f"(first-token stroke={hit_stroke_first}, word={hit_word_first})")

# ── length check across the two three-way chains (the actual point of v2) ──
for chain, key in ((("drop_strokes", "nostroke_hidden", "mask_strokes", "full"), "word"),
                  (("drop_words", "noword_hidden", "mask_words", "full"), "stroke")):
    lens = []
    for cond in chain:
        ds = FlatOCTDataset(val_pages, vocab, CFG["seq_len"], cond, deterministic=True, n_windows=1)
        lens.append(ds[0]["types"].size(0))
    print(f"  chain {chain}: window lengths = {lens}")
    assert lens[1] == lens[2] == lens[3], f"hidden/mask/full must share length: {lens}"
    assert lens[0] <= lens[1], f"drop_* must be no longer than nostroke/noword_hidden: {lens}"

# ── regression: the v2 Colab crash. An all-stroke span becomes EMPTY under
# drop_strokes; hide[0] must not index into it, training must resample it,
# and validation must keep it (paired index) while the metrics skip it. ──
allstroke = max(pages, key=lambda p: int((p["types"] == STROKE_TYPE).sum()))
# fabricate a page that is one word followed by 600 stroke points so any
# mid-page window of 254 tokens is guaranteed all-stroke
fake = {k: v.clone() for k, v in allstroke.items() if torch.is_tensor(v)}
n_pts = 600
fake["types"] = torch.cat([torch.tensor([WORD_TYPE]), torch.full((n_pts,), STROKE_TYPE)])
fake["wids"]  = torch.cat([torch.tensor([vocab["the"]]), torch.zeros(n_pts, dtype=torch.long)])
fake["abs"]   = torch.cat([torch.zeros(1, 2), torch.rand(n_pts, 2)])
fake["delta"] = torch.zeros(n_pts + 1, 2); fake["pens"] = torch.zeros(n_pts + 1, dtype=torch.long)
fake["tag"], fake["topic"] = "fake-allstroke", ""
ds_tr = FlatOCTDataset([fake], vocab, CFG["seq_len"], "drop_strokes")           # training: resamples
ds_tr.start_frac = ds_tr.end_frac = 0.0
ds_va = FlatOCTDataset([fake], vocab, CFG["seq_len"], "drop_strokes", deterministic=True, n_windows=3)
ds_va.start_frac = ds_va.end_frac = 0.0
n_fallback = 0
for i in range(30):
    it = ds_tr[0]
    assert it["types"].numel() > 0, "training must never yield an empty window"
    n_fallback += int(it["wids"][0].item() == vocab[BOS])   # fallback windows start with BOS
empties = sum(1 for i in range(len(ds_va)) if ds_va[i]["types"].numel() == 0)
assert empties == len(ds_va), f"validation must keep empty windows for pairing (got {empties}/{len(ds_va)} empty)"
# an empty row batched with a real one must go through the model in eval without touching the real row's metrics
real = FlatOCTDataset(val_pages, vocab, CFG["seq_len"], "drop_strokes", deterministic=True)[0]
b = collate([ds_va[0], real], pad_id); b = {k: v.to(device) for k, v in b.items()}
mm = FlatOCTModel(len(vocab), 16, 1, 2, 16, max(2 * CFG["seq_len"], 2048), 0.0, pad_id).to(device).eval()
with torch.no_grad():
    o = forward_batch(mm, b)
mets = raw_metrics(o, b)
assert math.isfinite(mets["word_sum"]), "empty row leaked NaN into the real row's metrics"
assert mets["word_n"] > 0, "the real row must still contribute word targets"
print(f"  empty-window regression OK: 30 train draws non-empty ({n_fallback} via BOS fallback), val keeps {empties}/{len(ds_va)} empty windows, metrics finite with an empty row in the batch")

# ── full training loop for every condition ──
results = {c: {} for c in CONDITIONS}
for seed in SEEDS:
    for cond in CONDITIONS:
        results[cond][seed] = run_condition(cond, seed)
        v = results[cond][seed]["best"]["val"]
        for k in ("word", "stroke", "pen", "type"):
            assert v[k] is None or math.isfinite(v[k]), f"{cond} s{seed}: {k} not finite"
print("\nall conditions x seeds trained; metrics finite")

# ── resume path + legacy (untagged seed-42) file compatibility ──
r2 = run_condition("full", SEEDS[0])
assert r2["best"]["epoch"] == results["full"][SEEDS[0]]["best"]["epoch"]
import shutil as _sh
_sh.copy(f"{RUN_DIR}/result_full_s42.json", f"{RUN_DIR}/result_full.json")
_sh.copy(f"{RUN_DIR}/full_s42.best.pt", f"{RUN_DIR}/full.best.pt")
os.remove(f"{RUN_DIR}/result_full_s42.json"); os.remove(f"{RUN_DIR}/full_s42.best.pt")
assert result_path("full", 42).endswith("result_full.json") and ckpt_path("full", 42).endswith("full.best.pt")
r3 = run_condition("full", 42)
assert r3["best"]["epoch"] == r2["best"]["epoch"]
assert result_path("full", 7).endswith("_s7.json"), "legacy fallback must only apply to seed 42"
print("resume-from-json OK (tagged and legacy seed-42 names)")

# ── bootstrap cell (CELL_BOOTSTRAP) ──
def pw_key(cond, seed):
    return f"{cond}_s{seed}"
per_window = {}
for seed in SEEDS:
    for cond in CONDITIONS:
        ck = torch.load(ckpt_path(cond, seed), map_location=device)
        m = FlatOCTModel(len(vocab), CFG["d_model"], CFG["n_layers"], CFG["n_heads"], CFG["d_ff"],
                         max(2 * CFG["seq_len"], 2048), CFG["dropout"], pad_id).to(device)
        m.load_state_dict(ck["model_state"])
        _, va_dl = make_loaders(cond, seed)
        per_window[pw_key(cond, seed)] = evaluate_per_window(m, va_dl)
        assert len(per_window[pw_key(cond, seed)]["word"]) == len(va_dl.dataset) == len(val_pages) * CFG["val_windows"]

n0 = len(per_window[pw_key("full", SEEDS[0])]["word"])
assert all(len(v["word"]) == n0 for v in per_window.values())
print(f"per-window eval OK: {n0} windows per run, {len(per_window)} runs, list-position matched")

bd = bootstrap_delta(per_window[pw_key("mask_strokes", 42)]["word"], per_window[pw_key("full", 42)]["word"], n_boot=200)
assert bd is not None and math.isfinite(bd["point"]) and bd["ci_lo"] <= bd["point"] <= bd["ci_hi"]
print(f"bootstrap_delta sanity: point={bd['point']:+.4f}  CI=[{bd['ci_lo']:+.4f}, {bd['ci_hi']:+.4f}]  n={bd['n_windows']}")

dW_len = bootstrap_delta(per_window[pw_key("drop_strokes", 42)]["word"], per_window[pw_key("nostroke_hidden", 42)]["word"], n_boot=50)
dW_mk  = bootstrap_delta(per_window[pw_key("nostroke_hidden", 42)]["word"], per_window[pw_key("mask_strokes", 42)]["word"], n_boot=50)
dW_ct  = bootstrap_delta(per_window[pw_key("mask_strokes", 42)]["word"], per_window[pw_key("full", 42)]["word"], n_boot=50)
three_way_total = dW_len["point"] + dW_mk["point"] + dW_ct["point"]
naive_total = results["drop_strokes"][42]["best"]["val"]["word"] - results["full"][42]["best"]["val"]["word"]
print(f"3-way sum = {three_way_total:+.4f} vs naive total (token-weighted) = {naive_total:+.4f} "
      f"(won't match exactly -- different weighting/checkpoints -- but same order of magnitude expected)")
assert abs(three_way_total) < 50 and abs(naive_total) < 50  # finite, sane range for this tiny smoke config

# ── generation from the saved full checkpoint ──
ck = torch.load(ckpt_path("full", 42), map_location=device)
model = FlatOCTModel(len(vocab), CFG["d_model"], CFG["n_layers"], CFG["n_heads"], CFG["d_ff"],
                     max(2 * CFG["seq_len"], 2048), CFG["dropout"], pad_id)
model.load_state_dict(ck["model_state"]); model.eval()
id2word = {v: k for k, v in vocab.items()}

exec(open(os.path.join(os.path.dirname(__file__), "gen_lib.py")).read())
page = val_pages[0]
bos = {"types": torch.tensor([WORD_TYPE]), "wids": torch.tensor([vocab[BOS]]),
       "delta": torch.zeros(1, 2), "abs": torch.zeros(1, 2), "pens": torch.tensor([0])}
full = {k: torch.cat([bos[k], page[k]]) for k in bos}
idx = torch.nonzero(full["types"] == WORD_TYPE)[:6, 0]
cold = generate(model, {k: full[k][idx] for k in full}, max_new=60, ctx=256)
warm = generate(model, seed_from_page(full, 120), max_new=60, ctx=256)
print("cold:", stats(cold), "|", words_of(cold, 12))
print("warm:", stats(warm), "|", words_of(warm, 12))
assert all(0 <= t["x"] <= 1 and 0 <= t["y"] <= 1 for t in cold + warm if t["type"] == "stroke")
gt = page_tokens(full, 0, 50)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 3)); draw(ax[0], gt, "#1a3a8c"); draw(ax[1], warm, "#b0202a")
plt.savefig(f"{RUN_DIR}/smoke.png"); print("render OK")
print(f"\nSMOKE TEST PASSED in {time.time()-t0:.0f}s")
