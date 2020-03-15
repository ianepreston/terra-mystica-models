"""Read in the all valid games csv generated from JSON files
clean it up as necessary for actual analysis
"""
from pathlib import Path
import pandas as pd
from terra_mystica_models.data import make_dataset


def read_interim_data():
    """Generate the interim data CSV if necessary and load it to a dataframe

    Do some very basic cleaning. Subsequent transformations can come later
    """
    interim_dir = Path(__file__).resolve().parents[2] / "data" / "interim"
    interim_csv = interim_dir / "all_valid_games.csv"
    if not interim_csv.exists():
        make_dataset.main()
    all_games_df = pd.read_csv(
        interim_csv, parse_dates=["date"], index_col=0, low_memory=False
    )
    # Do some validation
    assert all(all_games_df["player_count_valid"])
    assert not any(all_games_df["has_nofaction_players"])
    all_games_df = all_games_df.drop(
        columns=["player_count_valid", "has_nofaction_players"]
    )
    score_cols = sorted(
        [col for col in all_games_df.columns if col.startswith("score_turn")]
    )
    bonus_cols = sorted([col for col in all_games_df.columns if col.startswith("BON")])
    bonus_cols.pop(1)
    bonus_cols.append("BON10")
    # If it's missing that's because it wasn't included
    for col in bonus_cols:
        all_games_df[col] = all_games_df[col].fillna(False)
    faction_cols = sorted(
        [
            col
            for col in all_games_df.columns
            if col.endswith("faction") and not col.startswith("has_dropped")
        ]
    )
    user_cols = sorted([col for col in all_games_df.columns if col.endswith("user")])
    vp_cols = sorted([col for col in all_games_df.columns if col.endswith("vp")])
    vp_margin_cols = sorted(
        [col for col in all_games_df.columns if col.endswith("vp_margin")]
    )
    category_cols = (
        score_cols + bonus_cols + faction_cols + user_cols + vp_cols + vp_margin_cols
    )
    non_cat_cols = [col for col in all_games_df.columns if col not in category_cols]
    all_games_df = all_games_df.reindex(columns=non_cat_cols + category_cols)
    return all_games_df


def drop_unscored_games(df):
    """Drop games that don't have scoring tiles set

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame of Terra Mystica Games, probably generated by read_interim_data
    """
    score_cols = [col for col in df.columns if col.startswith("score_turn")]
    mask = ~df[score_cols].isna().any(axis="columns")
    return df.loc[mask].copy()


def interim_full_pipe():
    """Combine all the functions above to return one nice dataframe"""
    interim_df = read_interim_data().pipe(drop_unscored_games)
    return interim_df
