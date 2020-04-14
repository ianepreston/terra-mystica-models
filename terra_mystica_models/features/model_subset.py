"""Filter down to just the games we care about"""
from terra_mystica_models.features.parse_interim import interim_full_pipe


def get_model_games_df():
    """Filter to the subset of games we care about"""
    idf = interim_full_pipe()
    # ~ inverts the boolean, so turns true to false and vice versa
    mask = (
        (idf["number_of_players"] == 4)
        & (idf["original_map"])
        & (~idf["has_expansion_factions"])
        & (~idf["has_dropped_faction"])
    )
    drop_cols = [
        "number_of_players",
        "map",
        "original_map",
        "has_expansion_factions",
        "has_dropped_faction",
    ]
    for i in range(5, 8):
        for end_str in ["_faction", "_user", "_vp", "_vp_margin"]:
            drop_cols.append(f"player_{i}{end_str}")
    game_df = idf.loc[mask].copy().drop(columns=drop_cols)
    return game_df
