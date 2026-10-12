from biofouling.dataset import group_key, stratified_split


def test_group_key_uses_gbif_record_id():
    assert group_key("4846918689_1134") == "4846918689"
    assert group_key("4846918689_1134_degr") == "4846918689"
    assert group_key("standalone") == "standalone"


def _groups(n, per_group=3):
    return {f"g{i}": [f"g{i}_img{j}" for j in range(per_group)] for i in range(n)}


def test_groups_never_straddle_splits():
    out = stratified_split(_groups(40), val_frac=0.15, test_frac=0.1, seed=42)
    owner = {}
    for split, items in out.items():
        for item in items:
            g = item.split("_")[0]
            assert owner.setdefault(g, split) == split


def test_split_is_complete_and_sized():
    groups = _groups(40)
    out = stratified_split(groups, val_frac=0.15, test_frac=0.1, seed=42)
    assert sum(len(v) for v in out.values()) == 40 * 3
    assert len({i.split("_")[0] for i in out["val"]}) == 6
    assert len({i.split("_")[0] for i in out["test"]}) == 4


def test_split_is_deterministic_for_a_seed():
    groups = _groups(25)
    assert stratified_split(groups, 0.15, 0.1, seed=7) == stratified_split(groups, 0.15, 0.1, seed=7)


def test_tiny_pools_keep_everything_in_train():
    out = stratified_split(_groups(3), val_frac=0.15, test_frac=0.1, seed=0)
    assert out["val"] == [] and out["test"] == []
    assert len(out["train"]) == 9
