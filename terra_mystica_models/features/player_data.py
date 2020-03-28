"""Take data from a game dataframe and compute player stats"""
import pandas as pd


class TerraMysticaPlayers:
    def __init__(self, game_df):
        """Calculate player statistics from game data"""
        self.game_df = game_df.copy()
        self._mask_infrequent_players()
        self.player_df = self._calc_full_player_df()

    def _find_frequent_players(self):
        """Find players that have enough games to compute credible statistics

        Somewhat arbitrarily setting that at 30 games for now
        """
        user_cols = [col for col in self.game_df.columns if col.endswith("user")]
        player_df = (
            self.game_df.melt(value_vars=user_cols)["value"].dropna().value_counts()
        )
        cutoff_mask = player_df >= 30
        return player_df.loc[cutoff_mask].index

    def _mask_infrequent_players(self):
        """Replace missing or infrequent usernames with a common label"""
        user_cols = [col for col in self.game_df.columns if col.endswith("user")]
        # I bet spaces aren't allowed in real user names
        anon_name = "Other Terra Mystica User"
        # check by columns and index
        assert not (self.game_df[user_cols] == anon_name).any().any()
        rankable_users = self._find_frequent_players()
        for player_num in range(1, 8):
            user_col = f"player_{player_num}_user"
            mask = (~(self.game_df[user_col].isin(rankable_users))) & (
                self.game_df["number_of_players"] >= player_num
            )
            self.game_df.loc[mask, user_col] = anon_name
        # Don't really need to return anything, modifying dataframe in place
        return True

    def _player_num_stats(self, player_num):
        """Calculate the stats for the nth player, will have to aggregate across all"""
        user_col = f"player_{player_num}_user"
        vp_col = f"player_{player_num}_vp"
        margin_col = f"player_{player_num}_vp_margin"
        player_df = (
            self.game_df.loc[self.game_df["number_of_players"] >= player_num]
            .reindex(columns=[user_col, vp_col, margin_col])
            .rename(columns={user_col: "user", vp_col: "vp", margin_col: "vp_margin"})
        )
        return player_df

    def _calc_full_player_df(self):
        """Assemble stats for all players with sufficient number of games"""
        full_player_df = (
            pd.concat(
                [self._player_num_stats(player_num) for player_num in range(1, 8)]
            )
            .reset_index()
            .groupby("user")
            .agg({"index": "count", "vp": "mean", "vp_margin": "mean"})
            .rename(
                columns={
                    "index": "number_of_games",
                    "vp": "mean_vp",
                    "vp_margin": "mean_vp_margin",
                }
            )
        )
        return full_player_df
