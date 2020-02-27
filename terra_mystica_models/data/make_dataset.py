# -*- coding: utf-8 -*-
import click
import datetime as dt
import re
import json
import pandas as pd
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


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
    def players(self):
        """Who's playing what faction
        
        Returns
        -------
        player_dict: dict
            Keys for player names, values for their faction
        """
        return {x["faction"]: x["player"] for x in self._raw_game["factions"]}

    @property
    def factions(self):
        """What factions are in play
        
        Returns
        -------
        factions: list
            list of factions
        """
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
                        selection_order_dict[faction] = i
        return selection_order_dict

    @property
    def scoring_tile_order(self):
        """What scoring tiles were in play and what order were they used in?

        Returns
        -------
        scoring_tile_dict: dict
            key is tile_number, value is round order
        """
        score_rgx = re.compile(r"SCORE\d+")
        search_keys = self._raw_game["events"]["global"]
        scores = [key for key in search_keys.keys() if score_rgx.search(key)]
        scoring_tile_dict = dict()
        for score in scores:
            turn_list = [key for key in search_keys[score]["round"] if key != "all"]
            assert len(turn_list) == 1
            scoring_tile_dict[score] = int(turn_list[0])
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
            faction: self._raw_game["events"]["faction"][faction]["vp"]["round"]["all"]
            for faction in self.factions
        }

    @property
    def victory_points_margin(self):
        """How much did they win by?"""
        vp = self.victory_points
        min_vp = min(vp.values())
        return {faction: points - min_vp for faction, points in vp.items()}

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
            yield TerraMysticaGame(game)


def _game_to_dfs(
    game: TerraMysticaGame, game_df: pd.DataFrame, player_df: pd.DataFrame
):
    """Append data from a Terra Mystica game to DataFrames"""
    # only parse valid games
    if not game.is_valid_game:
        return False
    if game.game_name in game_df.index:
        # I've made the assumption that all game IDs are unique, want to fail loudly
        # if that's invalid
        raise IndexError(f"game {game.game_name} already in dataframe")
    index = game.game_name
    game_df.loc[index, "game_date"] = game.game_date
    game_df.loc[index, "map"] = game.base_map
    for option in game.game_options:
        game_df.loc[index, option] = True
    for faction, order in game.faction_selection_order.items():
        game_df.loc[index, f"faction{order}"] = faction
    for tile, order in game.scoring_tile_order.items():
        game_df.loc[index, f"tile_round_{order}"] = tile
    for tile in game.bonus_tiles:
        game_df.loc[index, tile] = True
    for faction, vp in game.victory_points.items():
        game_df.loc[index, f"{faction}_vp"] = vp
    for faction, vp in game.victory_points_margin.items():
        game_df.loc[index, f"{faction}_vp_margin"] = vp

    for faction, player_id in game.players.items():
        if player_id in player_df.index:
            player_df.loc[player_id, "games"] += 1
            player_df.loc[player_id, "sum_vp"] += game.victory_points[faction]
            player_df.loc[player_id, "sum_vp_margin"] += game.victory_points_margin[
                faction
            ]
        else:
            player_df.loc[player_id, "games"] = 1
            player_df.loc[player_id, "sum_vp"] = game.victory_points[faction]
            player_df.loc[player_id, "sum_vp_margin"] = game.victory_points_margin[
                faction
            ]
    return True


def games_to_dfs():
    """Parse allll the games

    Returns
    -------
    (game_df, player_df): (pd.DataFrame, pd.DataFrame)
        game and player data
    """
    game_df = pd.DataFrame()
    player_df = pd.DataFrame()
    # will need to add some cleanup stuff later, but let's just try running this first
    for game in _games_iterator():
        _game_to_dfs(game, game_df, player_df)
    return game_df, player_df


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
