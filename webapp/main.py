import pickle
from pathlib import Path
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SelectField, SubmitField, SelectMultipleField
from wtforms.widgets import CheckboxInput, ListWidget
from wtforms.validators import DataRequired
import pandas as pd


# https://gist.github.com/juzten/2c7850462210bfa540e3
class MultiCheckboxField(SelectMultipleField):
    widget = ListWidget(prefix_label=False)
    option_widget = CheckboxInput()


_FACTIONS = [
    "alchemists",
    "auren",
    "chaosmagicians",
    "cultists",
    "darklings",
    "dwarves",
    "engineers",
    "fakirs",
    "giants",
    "halflings",
    "mermaids",
    "nomads",
    "swarmlings",
    "witches",
]

_SCORE_NAMES_DICT = {
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

_SCORES = [(key, f"{key} - {value}") for key, value in _SCORE_NAMES_DICT.items()]

_BONUSES = [f"BON{i}" for i in range(1, 11)]


def load_model():
    """Deserialize the predictive model"""
    model_path = Path(__file__).parent / "pickles" / "model.pkl"
    with open(model_path, "rb") as model_pickle:
        model = pickle.load(model_pickle)
    return model


def load_input_series():
    """Deserialize the input series"""
    series_path = Path(__file__).parent / "pickles" / "input_series.pkl"
    with open(series_path, "rb") as series_pickle:
        series = pickle.load(series_pickle)
    return series


_MODEL = load_model()
_INPUT_SERIES = load_input_series()


def generate_input_series(score_seq, bonuses, faction):
    """Take inputs in the style they'll be received from the webform and turn them into
    a series suitable for producing predictions
    
    Parameters
    ----------
    score_seq: [str]
        Sequence of scoring tiles. Should be something like ["SCORE7", "SCORE1"...]
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


def generate_prediction(input_series):
    """Estimate victory point margin for a given scenario and faction"""
    return _MODEL.predict(input_series)[0]


def generate_scenario_df(score1, score2, score3, score4, score5, score6, bonuses):
    """Take inputs in the style they'll be received from the webform and turn them into
    A table of predictions
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
    score_seq = [score1, score2, score3, score4, score5, score6]
    output_series = pd.Series(index=_FACTIONS, data=0)
    for faction in _FACTIONS:
        input_series = generate_input_series(score_seq, bonuses, faction)
        prediction = generate_prediction(input_series)
        output_series.loc[faction] = prediction
    # Do some nice formatting
    pretty_output = (
        output_series.to_frame()
        .rename(columns={0: "Predicted Margin"})
        .sort_values(by="Predicted Margin", ascending=False)
        .style.format("{:.2f}")
        .background_gradient(cmap="viridis")
    )
    return pretty_output


class ScoreForm(FlaskForm):
    score1 = SelectField(label="Score Turn 1", choices=_SCORES)
    score2 = SelectField(label="Score Turn 2", choices=_SCORES)
    score3 = SelectField(label="Score Turn 3", choices=_SCORES)
    score4 = SelectField(label="Score Turn 4", choices=_SCORES)
    score5 = SelectField(label="Score Turn 5", choices=_SCORES)
    score6 = SelectField(label="Score Turn 6", choices=_SCORES)
    bonuses = MultiCheckboxField(
        label="Bonuses", choices=[(bon, bon) for bon in _BONUSES]
    )
    submit = SubmitField("Submit")


app = Flask(__name__)
app.config["SECRET_KEY"] = "this_should_be_an_environment_variable"
bootstrap = Bootstrap(app)


@app.route("/", methods=["GET", "POST"])
def index():
    score1 = None
    score2 = None
    score3 = None
    score4 = None
    score5 = None
    score6 = None
    bonuses = None
    predictions = None
    form = ScoreForm()
    if form.validate_on_submit():
        predictions = generate_scenario_df(
            form.score1.data,
            form.score2.data,
            form.score3.data,
            form.score4.data,
            form.score5.data,
            form.score6.data,
            form.bonuses.data,
        ).render()
    return render_template("index.html", form=form, predictions=predictions)
