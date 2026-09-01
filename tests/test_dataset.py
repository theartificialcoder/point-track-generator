from dtf_eval.dataset import decode_rle


def test_uncompressed_rle_uses_column_major_order() -> None:
    mask = decode_rle({"size": [2, 3], "counts": [1, 2, 3]})

    assert mask.tolist() == [[False, True, False], [True, False, False]]

