"""Generate the data to be used for modeling and its features"""
import pandas as pd

from terra_mystica_models.features.model_subset import get_model_games_df


def _player_n_frame(base_df, n):
    """Create a dataframe of player level results for the nth player in a game frame
    
    This just gets called in player_level_df, doesn't need to be called directly

    Properties
    ----------
    base_df: pd.DataFrame
        game level dataframe
    n: int
        the player number to extract
    
    Returns
    -------
    player_n_df: pd.DataFrame
        The nth player's results
    """
    player_dict = {
        f"player_{n}_faction": "faction",
        f"player_{n}_vp_margin": "vp_margin",
    }
    player_n_df = (
        base_df.rename(columns=player_dict)
        .reindex(columns=["faction", "vp_margin"])
        .assign(player_num=n)
        .reset_index()
        .rename(columns={"index": "game_name"})
    )
    return player_n_df


def player_level_df():
    """Take the game level data and break it out into outcomes per player"""
    score_cols = [f"score_turn_{i}" for i in range(1, 7)]
    bon_cols = [f"BON{i}" for i in range(1, 11)]
    base_df = get_model_games_df()
    game_level_info = (
        base_df.reindex(columns=score_cols + bon_cols)
        .reset_index()
        .rename(columns={"index": "game_name"})
    )
    player_df = pd.concat([_player_n_frame(base_df, i) for i in range(1, 5)])
    recombined_df = player_df.merge(game_level_info, on="game_name")
    # Make sure we didn't drop any players
    assert len(recombined_df) == len(player_df)
    # This was the original transformation, we'll just reverse it
    # Descriptive score card names were good in the dataframe, bad model parameter names
    score_name_dict = {
        "SCORE1": "1 EARTH -> 1 C | SPADE >> 2",
        "SCORE2": "4 EARTH -> 1 SPADE | TOWN >> 5",
        "SCORE3": "4 WATER -> 1 P | D >> 2",
        "SCORE4": "2 FIRE -> 1 W | SA/SH >> 5",
        "SCORE5": "4 FIRE -> 4 PW | D >> 2",
        "SCORE6": "4 WATER -> 1 SPADE | TP >> 3",
        "SCORE7": "2 AIR -> 1 W | SA/SH >> 5",
        "SCORE8": "4 AIR -> 1 SPADE | TP >> 3",
        "SCORE9": "1 CULT_P -> 2 C | TE >> 4",
    }
    easy_score_names = {value: key for key, value in score_name_dict.items()}
    for i in range(1, 7):
        recombined_df[f"score_turn_{i}"] = recombined_df[f"score_turn_{i}"].map(
            easy_score_names
        )
    # Right now score cols are mapped to which turn they're active in
    # add dummy variables indicating whether they're included or not at all
    score_cols = [f"score_turn_{i}" for i in range(1, 7)]
    possible_scores = list(score_name_dict.keys())
    for score in possible_scores:
        recombined_df[score] = (
            recombined_df[score_cols].isin([score]).any(axis="columns")
        )
    # Make dummy variables for the categorical columns
    drop_cols = ["faction", "game_name"]
    drop_cols.extend([f"score_turn_{i}" for i in range(1, 7)])
    dummy_cols = ["faction"]
    dummy_cols.extend([f"score_turn_{i}" for i in range(1, 7)])
    dummy_frames = [
        pd.get_dummies(recombined_df[col], prefix=col) for col in dummy_cols
    ]
    dummy_frames.append(
        pd.get_dummies(
            recombined_df["player_num"], prefix="player_num", drop_first=True
        )
    )
    predict_list = dummy_frames + [recombined_df.drop(columns=drop_cols)]
    predict_df = pd.concat(predict_list, axis="columns")
    # Add all the interaction terms, this is what we're actually interested in
    faction_cols = [col for col in predict_df.columns if col.startswith("faction_")]
    non_interact_cols = ["vp_margin", "player_num"]
    non_interact_cols.extend(faction_cols)
    interact_cols = [col for col in predict_df.columns if col not in non_interact_cols]
    for faction_col in faction_cols:
        for interact_col in interact_cols:
            interact_name = f"{faction_col}_x_{interact_col}"
            predict_df[interact_name] = (
                predict_df[faction_col] * predict_df[interact_col]
            )
    assert len(predict_df) == len(recombined_df)
    return predict_df
