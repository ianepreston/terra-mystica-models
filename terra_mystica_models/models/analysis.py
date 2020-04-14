"""Analyse the results of the models"""
import random
from itertools import combinations, permutations, product
from pathlib import Path

import pandas as pd

from terra_mystica_models.models import train_model

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
_SCORES = [f"SCORE{i}" for i in range(1, 10)]
_BONUSES = [f"BON{i}" for i in range(1, 11)]


def _combo_generator(fraction_scenarios, include_factions=True):
    """Create scenarios for the model to predict"""
    # Set a seed so we get the same scenarios every time this is run
    random.seed(42)
    # All possible draws from the scores for 6 turns (can't have SCORE1 in the last turns)
    score_permutes = [
        scores for scores in permutations(_SCORES, 6) if "SCORE1" not in scores[4:]
    ]
    # All possible draws for bonus tiles, order doesn't matter here
    bonus_combos = [bonuses for bonuses in combinations(_BONUSES, 7)]
    # Pick some fraction of all possible scores for inclusion
    # Doing these draws with replacement since that's how it would work in reality
    random_scores = random.choices(
        score_permutes, k=int(len(score_permutes) * fraction_scenarios)
    )
    # Same thing for bonuses
    random_bonuses = random.choices(
        bonus_combos, k=int(len(bonus_combos) * fraction_scenarios)
    )
    # All the combos are the cartesian product of our pool of scores and bonuses
    products = [random_scores, random_bonuses]
    # If Faction is a feature of the model (as opposed to faction specific models)
    # we have to include it in the combos product
    if include_factions:
        products.append(_FACTIONS)
    # Create a generator of all the scenarios we've specified.
    random_combos = (prod for prod in product(*products))
    return random_combos


class ScoreTurnModelAnalysis:
    def __init__(self, fraction_scenarios=0.01):
        """Generate scenario games and see how the big fat model performs
        
        Parameters
        ----------
        fraction_scenarios: float, default 0.01
            There's 15 million possible Terra Mystica setups, we only want to take
            a subsample of those possibilities or else the computer just chugs forever
        """
        self._fraction_scenarios = fraction_scenarios
        self._stm = train_model.load_score_turn_model()
        self._scenario_df = None
        self._clean_df = None

    def _gen_scenario(self, combo):
        """Take some combination of scores, bonuses and faction and put it in a format
        that can be fed into the model
        Parameters
        ----------
        combo: tuple
            A combo of inputs from self.combo_generator
        
        Returns
        -------
        predict_in: pd.DataFrame
            Dataframe with one row with a scenario in the correct input format
        """
        # Start with all input parameters set to 0
        input_series = pd.Series(index=self._stm.params.index, data=0)
        # Player number is 1-4 so just take the average
        # Doesn't really matter since we're just comparing cross faction performance
        input_series.loc["player_num"] = 2.5
        # Guess I could have made the combo a named tuple to make this a bit clearer
        score_seq = combo[0]
        bon_seq = combo[1]
        faction = combo[2]
        # Identify faction
        faction_index = f"faction_{faction}"
        if faction_index in input_series.index:
            input_series.loc[faction_index] = 1
        # Populate the 6 score turn interaction rows
        for num, score in enumerate(score_seq, 1):
            index = f"faction_{faction}_x_score_turn_{num}_{score}"
            if index in input_series.index:
                input_series.loc[index] = 1
        # Populate the bonus interaction rows
        for bonus in bon_seq:
            index = f"faction_{faction}_x_{bonus}"
            if index in input_series.index:
                input_series.loc[index] = 1
        # Get the input in the right shape for prediction
        predict_in = input_series.to_frame().T
        return predict_in

    def _calc_scenario_df(self):
        """Compute a number of scenarios to feed into the predictive model"""
        scenario_df = pd.concat(
            (
                self._gen_scenario(combo)
                for combo in _combo_generator(
                    self._fraction_scenarios, include_factions=True
                )
            ),
            ignore_index=True,
        )
        return scenario_df

    @property
    def scenario_df(self):
        """A number of scenarios to feed into the predictive model"""
        if self._scenario_df is None:
            self._scenario_df = self._calc_scenario_df()
        return self._scenario_df

    def _calc_clean_df(self):
        """Compute a clean dataframe describing the scenarios and their predicted outcomes
        
        The format that comes out of the scenario DF is hard to read, it has a bunch of
        interaction terms and is sparse. Turn each combo into an easier to read frame
        along with it's predicted score for each faction
        """
        scenario_df = self.scenario_df.copy()
        # Create a new dataframe of the correct size
        clean_df = pd.DataFrame(index=scenario_df.index)
        clean_df["predicted_margin"] = self._stm.predict(scenario_df)
        faction_cols = [f"faction_{faction}" for faction in _FACTIONS]
        # Only the interaction columns for the faction played will have 1
        # this will return the column with that 1, which can then be used to identify the
        # faction for that scenario
        clean_df["faction"] = (
            scenario_df[faction_cols].idxmax(axis="columns").str.replace("faction_", "")
        )
        # Use similar logic to the above to ID the scoring tile for each turn
        # Note that SCORE2 is excluded from the interaction terms to avoid
        # multicollinearity, so we also have to check if the max is 0, which would imply
        # that the tile for that turn was SCORE2
        for i in range(1, 7):
            rgx = f"faction_\w+_x_score_turn_{i}_"
            col = f"score_turn_{i}"
            clean_df[col] = (
                scenario_df.filter(regex=rgx)
                .idxmax(axis="columns")
                .str.replace(rgx, "", regex=True)
            )
            score2_mask = scenario_df.filter(regex=rgx).max(axis="columns") == 0
            clean_df.loc[score2_mask, col] = "SCORE2"
        # Same idea for bonus tiles, BON1 is excluded so we have to infer its presence
        # based on the sum of the identified tiles
        for i in range(1, 11):
            col = f"BON{i}"
            rgx = f"faction_\w+_BON{i}"
            clean_df[col] = scenario_df.filter(regex=rgx).max(axis="columns")
        bonus_cols = [f"BON{i}" for i in range(2, 11)]
        clean_df["BON1"] = 0
        clean_df.loc[clean_df[bonus_cols].sum(axis="columns") == 6, "BON1"] = 1
        indexes = [f"score_turn_{i}" for i in range(1, 7)] + [
            f"BON{i}" for i in range(1, 11)
        ]
        # Pivot the results so that each score/bonus combo is a row and each faction's
        # predicted margin is a column
        pivot_result = clean_df.pivot_table(
            values="predicted_margin", columns="faction", index=indexes
        )
        return pivot_result

    @property
    def results_df(self):
        """Show the predicted results for various scenarios for each faction"""
        if self._clean_df is None:
            self._clean_df = self._calc_clean_df()
        return self._clean_df


class FactionModelAnalysis:
    def __init__(self, fraction_scenarios=0.01):
        """"Generate scenario games and see how each faction level model performs
        
        Parameters
        ----------
        fraction_scenarios: float, default 0.01
            There's 15 million possible Terra Mystica setups, we only want to take
            a subsample of those possibilities or else the computer just chugs forever
        
        
        TODO
        ----
        There's a lot of overlap code in this with the ScoreTurnModelAnalysis class
        If I end up making a third model it will probably be worth the effor to write
        up a base class and have them both inherit from it. As it is I'm lazy so I'm
        just going to leave it.
        """
        self._fraction_scenarios = fraction_scenarios
        self._faction_models = train_model.load_faction_models()
        self._scenario_df = None
        self._clean_df = None

    def _gen_scenario(self, combo):
        """Take some combination of scores, bonuses and faction and put it in a format
        that can be fed into the model
        Parameters
        ----------
        combo: tuple
            A combo of inputs from self.combo_generator
        
        Returns
        -------
        predict_in: pd.DataFrame
            Dataframe with one row with a scenario in the correct input format
        
        """
        # Doesn't matter which faction I use for this, they all have the same parameters
        model = self._faction_models["alchemists"]
        input_series = pd.Series(index=model.params.index, data=0)
        input_series.loc["player_num"] = 2.5
        score_seq = combo[0]
        bon_seq = combo[1]
        for num, score in enumerate(score_seq, 1):
            index = f"score_turn_{num}_{score}"
            if index in input_series.index:
                input_series.loc[index] = 1
        for bonus in bon_seq:
            if bonus in input_series.index:
                input_series.loc[bonus] = 1
        predict_in = input_series.to_frame().T
        return predict_in

    def _calc_scenario_df(self):
        """Compute a number of scenarios to feed into the predictive model"""
        scenario_df = pd.concat(
            (
                self._gen_scenario(combo)
                for combo in _combo_generator(
                    self._fraction_scenarios, include_factions=False
                )
            ),
            ignore_index=True,
        )
        return scenario_df

    @property
    def scenario_df(self):
        if self._scenario_df is None:
            self._scenario_df = self._calc_scenario_df()
        return self._scenario_df

    def _calc_clean_df(self):
        """Compute a clean dataframe describing the scenarios and their predicted outcomes"""
        scenario_df = self.scenario_df.copy()
        clean_df = pd.DataFrame(index=scenario_df.index)
        # clean_df["predicted_margin"] = self._stm.predict(scenario_df)
        for i in range(1, 7):
            col = f"score_turn_{i}"
            clean_df[col] = (
                scenario_df.filter(regex=col)
                .idxmax(axis="columns")
                .str.replace(f"{col}_", "")
            )
            score2_mask = scenario_df.filter(regex=col).max(axis="columns") == 0
            clean_df.loc[score2_mask, col] = "SCORE2"
        for i in range(2, 11):
            col = f"BON{i}"
            clean_df[col] = scenario_df[col]
        indexes = [f"score_turn_{i}" for i in range(1, 7)] + [
            f"BON{i}" for i in range(1, 11)
        ]
        bonus_cols = [f"BON{i}" for i in range(2, 11)]
        clean_df["BON1"] = 0
        clean_df.loc[clean_df[bonus_cols].sum(axis="columns") == 6, "BON1"] = 1
        for faction, model in self._faction_models.items():
            clean_df[faction] = model.predict(scenario_df)
        clean_df = clean_df.set_index(indexes).sort_index()
        return clean_df

    @property
    def results_df(self):
        """Show the predicted results for various scenarios for each faction"""
        if self._clean_df is None:
            self._clean_df = self._calc_clean_df()
        return self._clean_df


def save_scenarios(fraction_scenarios=0.01):
    reports_folder = Path(__file__).resolve().parents[2] / "reports"
    stma = ScoreTurnModelAnalysis(fraction_scenarios)
    fma = FactionModelAnalysis(fraction_scenarios)
    fma.results_df.to_csv(reports_folder / "faction_model.csv")
    stma.results_df.to_csv(reports_folder / "one_big_model.csv")
    return True


if __name__ == "__main__":
    save_scenarios(0.03)