from kb.retrieve import RRF_K, reciprocal_rank_fusion


def test_item_found_by_both_lists_ranks_first():
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["c", "d"]])
    assert max(scores, key=scores.get) == "c"  # 2nd-ish in both beats 1st in one
    assert scores["c"] == 1 / (RRF_K + 3) + 1 / (RRF_K + 1)


def test_single_list_keeps_order():
    scores = reciprocal_rank_fusion([["x", "y", "z"]])
    assert sorted(scores, key=scores.get, reverse=True) == ["x", "y", "z"]


def test_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == {}
