"""Train some models!"""
import re
import statsmodels.api as sm


def simple_model(predict_df):
    """Very straightforward model just to make sure we have things well specified"""
    # Base factions, no interactions
    faction_cols = [
        col
        for col in predict_df.columns
        if col.startswith("faction_") and "_x_" not in col
    ]
    y = predict_df["vp_margin"]
    x_cols = ["player_num"] + faction_cols
    X = sm.add_constant(predict_df[x_cols].drop(columns=["faction_alchemist"]))
    lin_model = sm.OLS(y, X).fit()
    return lin_model


def base_score_model(predict_df):
    """Do I have to account for the scoring tile dummies in my model?"""
    faction_cols = [
        col
        for col in predict_df.columns
        if col.startswith("faction_") and "_x_" not in col
    ]
    bonus_cols = [col for col in predict_df.columns if col.startswith("BON")]
    score_cols = [col for col in predict_df.columns if col.startswith("SCORE")]
    x_cols = ["player_num"] + faction_cols + bonus_cols + score_cols
    y = predict_df["vp_margin"]
    X = sm.add_constant(
        predict_df[x_cols].drop(columns=["BON1", "SCORE2", "faction_alchemist"])
    ).astype(int)
    lin_model = sm.OLS(y, X).fit()
    return lin_model


def interact_model(predict_df):
    """See how score and bonus tiles interact with player choice
    
    At least to start I won't consider which turn a score tile was set, just if it
    was in the game
    """
    faction_cols = [
        col
        for col in predict_df.columns
        if col.startswith("faction_") and "_x_" not in col
    ]
    bonus_regex = r"faction_\w+_x_BON\d+"
    bonus_cols = [
        col
        for col in predict_df.columns
        if re.match(bonus_regex, col) and not col.endswith("BON1")
    ]
    score_regex = r"faction_\w+_x_SCORE\d+"
    score_cols = [
        col
        for col in predict_df.columns
        if re.match(score_regex, col) and not col.endswith("SCORE2")
    ]
    x_cols = ["player_num"] + faction_cols + bonus_cols + score_cols
    y = predict_df["vp_margin"]
    X = predict_df[x_cols].astype(int)
    lin_model = sm.OLS(y, X).fit()
    return lin_model


def score_turn_model(predict_df):
    """See how score and bonus tiles interact with player choice
    
    Now we add in which turn the score tile is active for
    """
    faction_cols = [
        col
        for col in predict_df.columns
        if col.startswith("faction_") and "_x_" not in col
    ]
    bonus_regex = r"faction_\w+_x_BON\d+"
    bonus_cols = [
        col
        for col in predict_df.columns
        if re.match(bonus_regex, col) and not col.endswith("BON1")
    ]
    score_regex = r"faction_\w+_x_score_turn_\d_SCORE\d+"
    score_cols = [
        col
        for col in predict_df.columns
        if re.match(score_regex, col) and not col.endswith("SCORE2")
    ]
    x_cols = ["player_num"] + faction_cols + bonus_cols + score_cols
    y = predict_df["vp_margin"]
    X = predict_df[x_cols].astype(int)
    lin_model = sm.OLS(y, X).fit()
    return lin_model


def faction_level_models(predict_df):
    """Make a model for each faction, return a dictionary of faction models"""
    faction_cols = [
        col
        for col in predict_df.columns
        if col.startswith("faction_") and "_x_" not in col
    ]
    factions = [col.replace("faction_", "") for col in faction_cols]
    model_dict = dict()
    for faction in factions:
        fact_df = predict_df.loc[predict_df[f"faction_{faction}"] == 1].copy()
        bonus_regex = r"BON\d+"
        bonus_cols = [
            col
            for col in fact_df.columns
            if re.match(bonus_regex, col) and not col.endswith("BON1")
        ]
        score_regex = r"score_turn_\d_SCORE\d+"
        score_cols = [
            col
            for col in fact_df.columns
            if re.match(score_regex, col) and not col.endswith("SCORE2")
        ]
        x_cols = ["player_num"] + bonus_cols + score_cols
        y = fact_df["vp_margin"]
        X = fact_df[x_cols].astype(int)
        lin_model = sm.OLS(y, X).fit()
        model_dict[faction] = lin_model
    return model_dict
