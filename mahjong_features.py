"""
Feature engineering for Mahjong discard prediction.

ROUND SEGMENTATION: the source CSV is an arbitrary row-count slice (10,000
rows) of a much longer game log, not an integral number of rounds. A round
boundary is detected whenever Data.remain_tiles jumps UP compared to the
previous row (tiles count down as a round is played, then reset near 68-70
at the start of a new deal). The final round in a raw slice is frequently
truncated mid-round by the arbitrary cutoff -- e.g. in the 10k slice this
skill was built against, the last "round" is only 3 rows long and ends at
remain_tiles=64, nowhere near a natural round end (median round is ~13-14
rows). drop_last_partial_round=True (default) removes it so we don't train
on/evaluate against a fake, artificially-truncated round.

Each row in the source CSV is one decision point: a player with a 14-tile hand
(13 + 1 drawn/claimed) chose one action from Data.valid_actions, recorded at
Data.action_idx. We confirmed that in the overwhelming majority of rows these
are discard actions (type 1) and the chosen tile is always present in the
acting player's hand. We frame the task as 34-way classification: predict the
*tile type* (0-33, collapsing the 4 physical copies of each tile) that gets
discarded, using full game context (self + all three opponents).

Tile encoding (standard Japanese Mahjong, riichi rules):
    ids 0-135  -> 34 tile types x 4 physical copies, id // 4 = type
    types 0-8   : man (1m-9m)
    types 9-17  : pin (1p-9p)
    types 18-26 : sou (1s-9s)
    types 27-30 : winds (E, S, W, N)
    types 31-33 : dragons (haku, hatsu, chun)

IMPORTANT - leakage guard: RoundReward, FinalReward, and every per-player
PointsReward/FinalReward column are outcomes computed *after* the round/game
resolves. They are not known to the player at decision time, so they are
deliberately EXCLUDED from the feature set. Including them would leak the
future into the input and produce an unrealistically "accurate" but useless
model.
"""

import ast
import numpy as np
import pandas as pd

from mahjong_shanten import shanten as calc_shanten

N_TYPES = 34


def tile_to_type(tile_id: int) -> int:
    return tile_id // 4


def dora_type_from_indicator(indicator_type: int) -> int:
    """Given a dora indicator's tile TYPE (0-33), return the actual dora tile type,
    respecting suit wraparound (9->1 within each suit, winds cycle E->S->W->N->E,
    dragons cycle haku->hatsu->chun->haku)."""
    if indicator_type <= 8:  # man 0-8
        return (indicator_type + 1) % 9
    if indicator_type <= 17:  # pin 9-17
        return 9 + (indicator_type - 9 + 1) % 9
    if indicator_type <= 26:  # sou 18-26
        return 18 + (indicator_type - 18 + 1) % 9
    if indicator_type <= 30:  # winds 27-30
        return 27 + (indicator_type - 27 + 1) % 4
    return 31 + (indicator_type - 31 + 1) % 3  # dragons 31-33


def counts_vector(tile_ids) -> np.ndarray:
    """34-dim count vector: how many of each tile TYPE appear in a list of tile ids."""
    v = np.zeros(N_TYPES, dtype=np.float32)
    for t in tile_ids:
        if t is not None and t >= 0:
            v[tile_to_type(t)] += 1.0
    return v


def meld_tile_ids(melds) -> list:
    ids = []
    for m in melds:
        for t in m.get("tiles", []):
            if t is not None and t >= 0:
                ids.append(t)
    return ids


def assign_round_ids(df: pd.DataFrame) -> np.ndarray:
    """Segment rows into rounds using Data.remain_tiles resets (see module
    docstring). Returns an int array of round ids, one per row."""
    rt = df["Data.remain_tiles"].values
    round_ids = np.zeros(len(rt), dtype=np.int64)
    rid = 0
    for i in range(1, len(rt)):
        if rt[i] > rt[i - 1]:
            rid += 1
        round_ids[i] = rid
    return round_ids


def parse_list_cell(cell):
    """CSV stores Python-literal strings for list/dict columns; '[]' -> []"""
    if pd.isna(cell):
        return []
    if isinstance(cell, (list, dict)):
        return cell
    return ast.literal_eval(cell)


MAX_SEQ_LEN = 24          # generous cap; longer histories are truncated to the most recent MAX_SEQ_LEN discards
PAD_TILE_IDX = N_TYPES     # 34 = padding token id for the tile-type embedding (real types are 0-33)


def build_player_scalar_block(row, seat_idx: int) -> np.ndarray:
    """Non-sequential summary features for one seat:
    [points, n_discards, tsumo_giri_ratio, riichi_flag, meld_counts(34), n_melds]
    The raw discard-type counts are DROPPED here -- that signal now lives in
    the ordered discard_seq fed to the GRU instead of being flattened away.
    1 + 3 + 34 + 1 = 39 dims"""
    points = row[f"Data.{seat_idx}.points"] / 25000.0
    discards = parse_list_cell(row[f"Data.{seat_idx}.discards"])
    tsumo_giri = parse_list_cell(row[f"Data.{seat_idx}.tsumo_giri"])
    riichi = row[f"Data.{seat_idx}.riichi"]
    # NOTE: pandas reads this column as numpy.bool_, not Python's built-in bool --
    # `riichi is True` silently and always fails for numpy.bool_ (identity check on a
    # different singleton), which zeroed this feature out entirely in every model
    # trained before this fix. bool(riichi) / == True both work correctly.
    riichi_flag = 1.0 if bool(riichi) else 0.0
    melds = parse_list_cell(row[f"Data.{seat_idx}.melds"])

    n_discards = len(discards) / 24.0
    tg_ratio = (sum(tsumo_giri) / len(tsumo_giri)) if len(tsumo_giri) > 0 else 0.0
    meld_counts = counts_vector(meld_tile_ids(melds))
    n_melds = len(melds) / 4.0

    return np.concatenate([
        [points, n_discards, tg_ratio, riichi_flag],
        meld_counts,
        [n_melds],
    ]).astype(np.float32)  # 39


PLAYER_SCALAR_DIM = 39


def build_player_discard_seq(row, seat_idx: int):
    """Ordered discard history for one seat, most recent MAX_SEQ_LEN tiles.
    Returns (tile_type_seq, tsumogiri_seq, true_len) all padded/truncated to
    MAX_SEQ_LEN. Padded positions use PAD_TILE_IDX (embedding table treats it
    as the zero vector via padding_idx) and tsumogiri=0."""
    discards = parse_list_cell(row[f"Data.{seat_idx}.discards"])
    tsumo_giri = parse_list_cell(row[f"Data.{seat_idx}.tsumo_giri"])

    types = [tile_to_type(t) for t in discards if t is not None and t >= 0]
    tg = [float(x) for x in tsumo_giri] if len(tsumo_giri) == len(discards) else [0.0] * len(types)

    true_len = min(len(types), MAX_SEQ_LEN)
    if len(types) > MAX_SEQ_LEN:
        types = types[-MAX_SEQ_LEN:]   # keep most recent discards -- likeliest to matter for reading current hand shape
        tg = tg[-MAX_SEQ_LEN:]

    seq = np.full(MAX_SEQ_LEN, PAD_TILE_IDX, dtype=np.int64)
    tg_seq = np.zeros(MAX_SEQ_LEN, dtype=np.float32)
    if true_len > 0:
        seq[:true_len] = types
        tg_seq[:true_len] = tg
    return seq, tg_seq, max(true_len, 1)  # length>=1 so GRU packing never sees a zero-length sequence


def build_features_and_label(row):
    """Returns (feature_vector, label, valid_mask) for one row, or None if the
    chosen action isn't a plain discard (skip those rows -- see filtering note
    in build_dataset)."""
    valid_actions = parse_list_cell(row["Data.valid_actions"])
    action_idx = int(row["Data.action_idx"])
    if action_idx >= len(valid_actions):
        return None
    chosen = valid_actions[action_idx]
    if chosen["type"] != 1:
        return None  # not a discard (e.g. chi/pon/kan/riichi-declare/tsumo/ron)
    chosen_tile = [t for t in chosen["tiles"] if t is not None and t >= 0]
    if len(chosen_tile) != 1:
        return None
    label = tile_to_type(chosen_tile[0])

    hand = parse_list_cell(row["Data.hand_tiles"])
    hand_counts = counts_vector(hand)

    # The last entry in hand_tiles is the just-drawn tile: the first len(hand)-1 tiles
    # are always sorted, and the final one is appended unsorted (verified across every
    # hand size observed -- 14/11/8/5/2, the smaller sizes coming from kan calls that
    # reduce the concealed-tile count by 3 each). This is known to the player BEFORE
    # discarding, so it's a legitimate input feature, not a label leak: it doesn't say
    # which tile gets discarded, only which tile was drawn (one candidate among many).
    drawn_tile_type = tile_to_type(hand[-1]) if len(hand) > 0 else -1
    drawn_tile_oh = np.zeros(N_TYPES, dtype=np.float32)
    if 0 <= drawn_tile_type < N_TYPES:
        drawn_tile_oh[drawn_tile_type] = 1.0

    # mask of which tile types are actually offered as valid discards this turn
    valid_mask = np.zeros(N_TYPES, dtype=np.float32)
    for a in valid_actions:
        if a["type"] == 1:
            ts = [t for t in a["tiles"] if t is not None and t >= 0]
            if len(ts) == 1:
                valid_mask[tile_to_type(ts[0])] = 1.0

    round_wind = int(row["Data.round_wind"])          # 0,1,2
    player_wind = int(row["Data.player_wind"])         # 0-3
    position = int(row["Data.position"])               # 0-3, acting seat
    num_honba = row["Data.num_honba"] / 8.0
    num_riichi = row["Data.num_riichi"] / 4.0
    remain_tiles = row["Data.remain_tiles"] / 70.0

    dora_indicators = parse_list_cell(row["Data.dora_indicators"])
    dora_vec = np.zeros(N_TYPES, dtype=np.float32)
    for ind in dora_indicators:
        ind_type = tile_to_type(ind)
        dora_vec[dora_type_from_indicator(ind_type)] += 1.0

    round_wind_oh = np.eye(3, dtype=np.float32)[round_wind]
    player_wind_oh = np.eye(4, dtype=np.float32)[player_wind]  # computed but unused below -- see note near global_feats
    position_oh = np.eye(4, dtype=np.float32)[position]

    # --- tile-efficiency (shanten) features ---
    # current_shanten: how far the 14-tile hand is from tenpai right now (before discarding)
    hand_counts_int = [int(c) for c in hand_counts]
    current_shanten = calc_shanten(hand_counts_int)

    # shanten_after_discard[t]: shanten of the resulting 13-tile hand if we discard tile type t.
    # Only computed for types actually in hand (== the legal discard set); everything else gets a
    # sentinel of current_shanten + 2 (clearly worse than any real option, and irrelevant since
    # valid_mask already zeroes it out downstream -- this just keeps the raw feature well-defined).
    sentinel = float(current_shanten + 2)
    shanten_after_discard = np.full(N_TYPES, sentinel, dtype=np.float32)
    for t in range(N_TYPES):
        if hand_counts_int[t] > 0:
            hand_counts_int[t] -= 1
            shanten_after_discard[t] = float(calc_shanten(hand_counts_int))
            hand_counts_int[t] += 1
    # NOTE: an earlier version also included delta_after_discard = shanten_after_discard -
    # current_shanten as a separate 34-dim block. That's an exact linear function of the two
    # blocks already here (verified to floating-point-exact equality on real data), so it added
    # zero information -- any linear layer can already compute a difference of two inputs it's
    # given. Dropped to save 34 redundant dims.

    shanten_feats = np.concatenate([
        [current_shanten / 8.0],           # 8 is worst-case standard-form shanten
        shanten_after_discard / 8.0,       # 34
    ]).astype(np.float32)  # 1 + 34 = 35

    # NOTE: player_wind_oh was dropped here too -- verified Data.player_wind == Data.position
    # for 100% of rows in this dataset, making it a byte-for-byte duplicate of position_oh.
    global_feats = np.concatenate([
        round_wind_oh, position_oh,
        [num_honba, num_riichi, remain_tiles],
        dora_vec,
        shanten_feats,
        drawn_tile_oh,
    ]).astype(np.float32)  # 3+4+3+34+35+34 = 113

    # relative seating: self, next (position+1), across (+2), previous (+3), mod 4
    seat_order = [(position + k) % 4 for k in range(4)]
    scalar_feats = np.concatenate([build_player_scalar_block(row, s) for s in seat_order])
    # 4 * 39 = 156

    features = np.concatenate([hand_counts, valid_mask, global_feats, scalar_feats])
    # 34 + 34 + 48 + 156 = 272

    seqs, tg_seqs, seq_lens = [], [], []
    for s in seat_order:
        seq, tg_seq, seq_len = build_player_discard_seq(row, s)
        seqs.append(seq)
        tg_seqs.append(tg_seq)
        seq_lens.append(seq_len)
    seqs = np.stack(seqs)          # (4, MAX_SEQ_LEN) int64
    tg_seqs = np.stack(tg_seqs)    # (4, MAX_SEQ_LEN) float32
    seq_lens = np.array(seq_lens, dtype=np.int64)  # (4,)

    return features.astype(np.float32), label, valid_mask, seqs, tg_seqs, seq_lens


FEATURE_DIM = N_TYPES + N_TYPES + 113 + 4 * PLAYER_SCALAR_DIM  # 337


def build_dataset(csv_path: str, verbose: bool = True, drop_last_partial_round: bool = True):
    df = pd.read_csv(csv_path)
    round_ids = assign_round_ids(df)

    if drop_last_partial_round:
        last_round = round_ids.max()
        keep = round_ids < last_round
        n_dropped = (~keep).sum()
        df = df[keep].reset_index(drop=True)
        round_ids = round_ids[keep]
        if verbose:
            print(f"Dropped {n_dropped} row(s) from the final, truncated round "
                  f"({round_ids.max() + 1 if len(round_ids) else 0} complete rounds remain).")

    X, y, masks, seqs, tg_seqs, seq_lens, rounds = [], [], [], [], [], [], []
    skipped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        result = build_features_and_label(row)
        if result is None:
            skipped += 1
            continue
        feats, label, mask, seq, tg_seq, seq_len = result
        X.append(feats)
        y.append(label)
        masks.append(mask)
        seqs.append(seq)
        tg_seqs.append(tg_seq)
        seq_lens.append(seq_len)
        rounds.append(round_ids[i])
    X = np.stack(X)
    y = np.array(y, dtype=np.int64)
    masks = np.stack(masks)
    seqs = np.stack(seqs)
    tg_seqs = np.stack(tg_seqs)
    seq_lens = np.stack(seq_lens)
    rounds = np.array(rounds, dtype=np.int64)
    if verbose:
        print(f"Built dataset: {X.shape[0]} rows kept, {skipped} rows skipped "
              f"(non-discard actions), feature dim = {X.shape[1]}, "
              f"{len(np.unique(rounds))} rounds")
    return X, y, masks, seqs, tg_seqs, seq_lens, rounds


if __name__ == "__main__":
    X, y, masks, seqs, tg_seqs, seq_lens, rounds = build_dataset(
        "/mnt/user-data/uploads/mahjong_ml_slice_10000.csv")
    print("X shape:", X.shape, "y shape:", y.shape, "masks shape:", masks.shape)
    print("seqs shape:", seqs.shape, "tg_seqs shape:", tg_seqs.shape, "seq_lens shape:", seq_lens.shape)
    print("Label distribution (top 10 tile types):")
    vals, counts = np.unique(y, return_counts=True)
    order = np.argsort(-counts)[:10]
    for i in order:
        print(f"  type {vals[i]:>2}: {counts[i]}")
