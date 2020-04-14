import datetime as dt
import itertools
import json
import re
from pathlib import Path

import d6tflow
import pandas as pd

from terra_mystica_models.data.download_dataset import data_download


class TerraMysticaGame:
    def __init__(self, game):
        """Parse JSON from a Terra Mystica game into something a little more tractable

        This definitely creates some computational overhead, but I'd rather be able to
        iterate quickly on parsing the data and then grab a coffee while it builds
        than prematurely optimize for loading speed.
        
        Parameters
        ----------
        game: dict
            the raw game data from snellman
        """
        self._raw_game = game
        self._users = None
        self._factions = None
        self._faction_selection_order = None
        self._faction_to_player = None

    @property
    def game_name(self):
        """What's the game called?

        Returns
        -------
        game_name: str
            a (unique) name for the game
        """
        return self._raw_game["game"]

    @property
    def game_date(self):
        """When was the game played?

        Returns
        -------
        game_datetime: datetime
            When the game was played (finished)
        """
        return dt.datetime.strptime(self._raw_game["last_update"], "%Y-%m-%d %H:%M:%S")

    @property
    def game_options(self):
        """What options are in place in the game?
        
        Returns
        -------
        options list: list
            text list of options in effect
        """
        return [
            key
            for key in self._raw_game["events"]["global"]
            if key.startswith("option-")
        ]

    @property
    def base_map(self):
        """Where is this played?"""
        return self._raw_game["base_map"]

    @property
    def users(self):
        """Who's playing what faction
        
        Returns
        -------
        player_dict: dict
            Keys for player names, values for their faction
        """
        if self._users is None:
            self._users = self._calc_users()
        return self._users

    def _calc_users(self):
        # Probably a way to do this without dual comprehension but oh well
        faction_user = {x["faction"]: x["player"] for x in self._raw_game["factions"]}
        return {
            f"{self.faction_to_player[key]}_user": value
            for key, value in faction_user.items()
        }

    @property
    def factions(self):
        """What factions are in play
        
        Returns
        -------
        factions: list
            list of factions
        """
        if self._factions is None:
            self._factions = self._calc_factions()
        return self._factions

    def _calc_factions(self):
        return [x["faction"] for x in self._raw_game["factions"]]

    @property
    def faction_selection_order(self):
        """Faction selection order

        Under the key "events"/"faction"/{faction} there's are "order:{N}" keys
        within that there's a turn key which has keys for every turn in which that
        faction played in that order. By finding who has "1" in their "order:{N}" keys
        we can identify the order of play in the first round of the game, and therefore
        the order of faction selection for the game.
        
        Returns
        -------
        {faction: order} or {order:faction}?
        """
        if self._faction_selection_order is None:
            self._faction_selection_order = self._calc_faction_selection_order()
        return self._faction_selection_order

    def _calc_faction_selection_order(self):
        factions = self.factions
        selection_order_dict = dict()
        for i in range(1, len(factions) + 1):
            order_key = f"order:{i}"
            for faction in factions:
                if order_key in self._raw_game["events"]["faction"][faction].keys():
                    if (
                        "1"
                        in self._raw_game["events"]["faction"][faction][order_key][
                            "turn"
                        ].keys()
                    ):
                        selection_order_dict[f"player_{i}_faction"] = faction
        return selection_order_dict

    @property
    def faction_to_player(self):
        """Most stuff in the JSON is recorded by faction but I'd prefer to have it
        mapped to player number for more consistency in my dataframe
        """
        if self._faction_to_player is None:
            self._faction_to_player = self._calc_faction_to_player()
        return self._faction_to_player

    def _calc_faction_to_player(self):
        return {
            value: key.replace("_faction", "")
            for key, value in self.faction_selection_order.items()
        }

    @property
    def scoring_tile_order(self):
        """What scoring tiles were in play and what order were they used in?

        Returns
        -------
        scoring_tile_dict: dict
            key is tile_number, value is round order
        """
        # I'd like to have better names here but right now there's dupes so it'll wait
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
        score_rgx = re.compile(r"SCORE\d+")
        search_keys = self._raw_game["events"]["global"]
        scores = [key for key in search_keys.keys() if score_rgx.search(key)]
        scoring_tile_dict = dict()
        for score in scores:
            turn_list = [key for key in search_keys[score]["round"] if key != "all"]
            assert len(turn_list) == 1
            scoring_tile_dict[f"score_turn_{turn_list[0]}"] = score_name_dict[score]
        return scoring_tile_dict

    @property
    def bonus_tiles(self):
        """What bonus tiles are used in the game?"""
        search_space = self._raw_game["events"]["faction"]["all"].keys()
        bonus_regex = re.compile(r":(BON\d+)")
        return list(
            set(
                bonus_regex.search(x).groups(0)[0]
                for x in search_space
                if bonus_regex.search(x)
            )
        )

    @property
    def victory_points(self):
        """Who won?

        Returns
        -------
        victory_point_dict: dict
            factions as keys, total points as values
        """
        return {
            f"{self.faction_to_player[faction]}_vp": self._raw_game["events"][
                "faction"
            ][faction]["vp"]["round"]["all"]
            for faction in self.factions
        }

    @property
    def victory_points_margin(self):
        """How much did they win by?"""
        vp = self.victory_points
        mean_vp = sum(vp.values()) / len(vp.values())
        return {f"{player}_margin": points - mean_vp for player, points in vp.items()}

    def _check_player_count(self):
        """Make sure game data player count matches my faction count"""
        return self._raw_game["player_count"] == len(self.factions)

    def _check_bonus_tile_count(self):
        """Should be number of players + 3"""
        return len(self.bonus_tiles) == len(self.factions) + 3

    def _is_played_on_original_map(self):
        """At least to start we'll want to restrict analysis to the original map"""
        return self.base_map == "126fe960806d587c78546b30f1a90853b1ada468"

    def _has_nofaction_player(self):
        """Some games have players using "nofaction", want to be able to exclude them"""
        return any([faction.startswith("nofaction") for faction in self.factions])

    def _has_expansion_factions(self):
        """Is anyone playing as an expansion faction?
        
        I could just write in the expansion factions I know about, but I want this to be
        extensible. So instead we'll just check for core factions and nofaction
        (since it's captured elsewhere) and assume any other faction result must be
        an expansion faction
        """
        core_factions = [
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
        check_factions = [
            faction for faction in self.factions if not faction.startswith("nofaction")
        ]
        return any(faction not in core_factions for faction in check_factions)

    def _has_dropped_faction(self):
        """Check if anyone dropped out"""
        return any(key.startswith("drop") for key in self._raw_game["events"]["global"])

    @property
    def is_valid_game(self):
        """Combine our criteria for inclusion in the model"""
        valid_game_criteria = (
            self._check_player_count()
            & self._check_bonus_tile_count()
            & self._is_played_on_original_map()
            & (not self._has_nofaction_player())
            & (not self._has_expansion_factions())
            & (not self._has_dropped_faction())
        )
        return valid_game_criteria

    @property
    def dataframe_row(self):
        """Combine all the game data into a flat dictionary that can be
        parsed into a dataframe row
        """
        df_dict = {
            "date": self.game_date,
            "map": self.base_map,
            "number_of_players": len(self.factions),
            "player_count_valid": self._check_player_count(),
            "original_map": self._is_played_on_original_map(),
            "has_nofaction_players": self._has_nofaction_player(),
            "has_expansion_factions": self._has_expansion_factions(),
            "has_dropped_faction": self._has_dropped_faction(),
        }
        df_dict.update(self.scoring_tile_order)
        df_dict.update({tile: True for tile in self.bonus_tiles})
        df_dict.update(self.users)
        df_dict.update(self.faction_selection_order)
        df_dict.update(self.victory_points)
        df_dict.update(self.victory_points_margin)
        return pd.DataFrame.from_dict(
            data=df_dict, orient="index", columns=[self.game_name]
        ).T


def _games_iterator():
    """Generator for all the games in the JSON in raw

    Yields
    ------
    game: TerraMysticaGame
    """
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw"
    for json_file in raw_dir.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
            for game in data:
                yield {"file": json_file, "game": TerraMysticaGame(game)}


def _games_to_df(limit=None):
    """Parse allll the games

    Parameters
    ----------
    limit: int, default None
        Max number of games to return, defaults to all of them
    
    Returns
    -------
    (game_df, player_df): (pd.DataFrame, pd.DataFrame)
        game and player data
    """
    if limit is None:
        games_it = _games_iterator()
    else:
        games_it = itertools.islice(_games_iterator(), limit)
    return pd.concat(
        [
            game["game"].dataframe_row.assign(file=game["file"].name)
            for game in games_it
            # Nofaction messes things up, they don't have VP stored
            if not game["game"]._has_nofaction_player()
        ]
    )


class TaskGetData(d6tflow.tasks.TaskCSVPandas):
    """
    Download the JSON files if necessary, then convert them into a DataFrame

    TODO
    ----
    add an option to get JSON beyond the cutoff date
    add an option to load a limited number of games
    """

    def run(self):
        # This will only download missing JSON files
        data_download()
        # Read the games into a dataframe
        game_df = _games_to_df()
        # Save the result to CSV
        self.save(game_df)


if __name__ == "__main__":
    d6tflow.run(TaskGetData())
