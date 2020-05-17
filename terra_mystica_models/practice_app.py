"""
Code to mess around with while I'm building the webapp. 
"""

import pickle
from pathlib import Path
import pandas as pd

def load_model_skl():
    """Deserialize the sklearn implementation of the model"""
    model_folder = Path(__file__).parents[1] / "d6tflow_output" / "TaskSkLearnModel"
    model_file = model_folder / "TaskSkLearnModel__99914b932b-data.pkl"
    with open(model_file, "rb") as pickle_model:
        model = pickle.load(pickle_model)
    return model

def load_model_sm():
    """Deserialize the statsmodels implementation of the model"""
    model_folder = Path(__file__).parents[1] / "d6tflow_output" / "TaskScoreTurnModel"
    model_file = model_folder / "TaskScoreTurnModel__99914b932b-data.pkl"
    with open(model_file, "rb") as pickle_model:
        model = pickle.load(pickle_model)
    return model


def load_input_series():
    """Deserialize a pandas series suitable for feeding to the model"""
    series_folder = Path(__file__).parents[1] / "d6tflow_output" / "TaskScoreTurnModelInputs"
    series_file = series_folder / "TaskScoreTurnModelInputs__99914b932b-data.pkl"
    with open(series_file, "rb") as pickle_series:
        series = pickle.load(pickle_series)
    return series


# I think this is how I want to have the inputs set, as module constants so they don't
# have to be reloaded every time I call a scoring function

_SKLEARN_MODEL = load_model_skl()
_SM_MODEL = load_model_sm()
_INPUT_SERIES = load_input_series()

def generate_input_series(score1, score2, score3, score4, score5, score6, bonuses, faction):
    """Take inputs in the style they'll be received from the webform and turn them into
    a series suitable for producing predictions
    
    Parameters
    ----------
    score{n}: str
        Score tile for the nth turn. Should be something like "SCORE7"
    bonuses: [str]
        The list of bonus tiles for the game, should be something like
        ["BON1", "BON10"...]
    faction: str
        The faction to score for this particular scenario
    """
    input_series = _INPUT_SERIES.copy()
    # Player number doesn't really matter, just put something
    input_series.loc["player_num"] = 2.5
    # Identify faction
    faction_index = f"faction_{faction}"
    if faction_index in input_series.index:
        input_series.loc[faction_index] = 1
    score_seq = [score1, score2, score3, score4, score5, score6]
    for num, score in enumerate(score_seq, 1):
        index = f"faction_{faction}_x_score_turn_{num}_{score}"
        if index in input_series.index:
            input_series.loc[index] = 1
    # Populate the bonus interaction rows
    for bonus in bonuses:
        index = f"faction_{faction}_x_{bonus}"
        if index in input_series.index:
            input_series.loc[index] = 1
    # Get the input in the right shape for prediction
    predict_in = input_series.to_frame().T
    return predict_in
