"""
Shanten (tiles-away-from-tenpai) calculator.

Shanten 0 = tenpai (one tile away from winning). Lower is closer to winning.
Standard mahjong hand shape = 4 sets + 1 pair. We compute the minimum shanten
across the three recognized hand shapes: standard, chiitoitsu (7 pairs), and
kokushi musou (13 orphans), same as any competitive-rules shanten calculator.

This is pure combinatorics over the 34 tile-type counts of a hand -- no
external data or randomness, so results are deterministic and exact for the
standard-form search (memoized exhaustive search over set/pair/partial-set
decompositions) and closed-form for chiitoitsu/kokushi.
"""

import functools

N_TYPES = 34


@functools.lru_cache(maxsize=None)
def _rec(counts_t, melds, taatsu, pair_used):
    counts_l = list(counts_t)
    idx = -1
    for i in range(N_TYPES):
        if counts_l[i] > 0:
            idx = i
            break
    if idx == -1:
        t_capped = min(taatsu, max(0, 4 - melds))
        has_pair = 1 if pair_used else 0
        return (4 - melds) * 2 - t_capped - has_pair

    best = None
    can_grow = (melds + taatsu) < 4

    # triplet
    if counts_l[idx] >= 3 and can_grow:
        counts_l[idx] -= 3
        best = _min(best, _rec(tuple(counts_l), melds + 1, taatsu, pair_used))
        counts_l[idx] += 3

    # sequence (number suits only, run must start at idx<=idx+2 within the same suit)
    if idx < 27 and idx % 9 <= 6 and can_grow \
            and counts_l[idx] > 0 and counts_l[idx + 1] > 0 and counts_l[idx + 2] > 0:
        counts_l[idx] -= 1; counts_l[idx + 1] -= 1; counts_l[idx + 2] -= 1
        best = _min(best, _rec(tuple(counts_l), melds + 1, taatsu, pair_used))
        counts_l[idx] += 1; counts_l[idx + 1] += 1; counts_l[idx + 2] += 1

    # pair: as the head, or as a partial triplet (taatsu)
    if counts_l[idx] >= 2:
        counts_l[idx] -= 2
        if not pair_used:
            best = _min(best, _rec(tuple(counts_l), melds, taatsu, True))
        if can_grow:
            best = _min(best, _rec(tuple(counts_l), melds, taatsu + 1, pair_used))
        counts_l[idx] += 2

    # ryanmen / penchan (idx, idx+1 partial run)
    if idx < 27 and idx % 9 <= 7 and can_grow and counts_l[idx] > 0 and counts_l[idx + 1] > 0:
        counts_l[idx] -= 1; counts_l[idx + 1] -= 1
        best = _min(best, _rec(tuple(counts_l), melds, taatsu + 1, pair_used))
        counts_l[idx] += 1; counts_l[idx + 1] += 1

    # kanchan (idx, idx+2 partial run)
    if idx < 27 and idx % 9 <= 6 and can_grow and counts_l[idx] > 0 and counts_l[idx + 2] > 0:
        counts_l[idx] -= 1; counts_l[idx + 2] -= 1
        best = _min(best, _rec(tuple(counts_l), melds, taatsu + 1, pair_used))
        counts_l[idx] += 1; counts_l[idx + 2] += 1

    # skip one copy of this tile (isolate it -- contributes to nothing)
    counts_l[idx] -= 1
    best = _min(best, _rec(tuple(counts_l), melds, taatsu, pair_used))
    counts_l[idx] += 1

    return best


def _standard_shanten(counts: tuple) -> int:
    return _rec(tuple(counts), 0, 0, False)


def _min(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return a if a < b else b


def _chiitoitsu_shanten(counts) -> int:
    pairs = sum(1 for c in counts if c >= 2)
    kinds = sum(1 for c in counts if c >= 1)
    return 6 - pairs + max(0, 7 - kinds)


_TERMINAL_HONOR_IDX = [0, 8, 9, 17, 18, 26] + list(range(27, 34))  # 1/9 of each suit + all honors


def _kokushi_shanten(counts) -> int:
    kinds = sum(1 for i in _TERMINAL_HONOR_IDX if counts[i] >= 1)
    has_pair = any(counts[i] >= 2 for i in _TERMINAL_HONOR_IDX)
    return 13 - kinds - (1 if has_pair else 0)


def shanten(counts) -> int:
    """counts: length-34 sequence of tile-type counts (a 13- or 14-tile hand).
    Returns the minimum shanten across standard/chiitoitsu/kokushi shapes.
    -1 means tenpai/complete depending on hand size convention used elsewhere;
    this function just returns the raw combinatorial shanten number."""
    counts = tuple(int(c) for c in counts)
    return min(
        _standard_shanten(counts),
        _chiitoitsu_shanten(counts),
        _kokushi_shanten(counts),
    )


if __name__ == "__main__":
    import time
    # tenpai example: 123456789m + 11p + 22s (13 tiles, waiting on 3s... just a sanity smoke test)
    test_counts = [0] * 34
    for i in range(9):
        test_counts[i] = 1  # 1m-9m one each
    test_counts[9] = 2   # pair of 1p
    test_counts[18] = 2  # pair of 1s
    t0 = time.time()
    s = shanten(test_counts)
    print("shanten:", s, f"({time.time()-t0:.4f}s)")
