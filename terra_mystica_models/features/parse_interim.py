"""Read in the all valid games csv generated from JSON files
clean it up as necessary for actual analysis
"""
import d6tflow
from luigi.util import requires

from terra_mystica_models.data.make_dataset import TaskGetData


@requires(TaskGetData)
class TaskCleanData(d6tflow.tasks.TaskCSVPandas):
    def run(self):
        """Generate the interim data CSV if necessary and load it to a dataframe

        Do some very basic cleaning. Subsequent transformations can come later
        """
        # interim_dir = Path(__file__).resolve().parents[2] / "data" / "interim"
        # interim_csv = interim_dir / "all_valid_games.csv"
        # if not interim_csv.exists():
        #     make_dataset.main()
        # all_games_df = pd.read_csv(
        #     interim_csv, parse_dates=["date"], index_col=0, low_memory=False
        # )
        all_games_df = self.input().load()
        # Do some validation
        assert all(all_games_df["player_count_valid"])
        assert not any(all_games_df["has_nofaction_players"])
        all_games_df = all_games_df.drop(
            columns=["player_count_valid", "has_nofaction_players"]
        )
        score_cols = sorted(
            [col for col in all_games_df.columns if col.startswith("score_turn")]
        )
        bonus_cols = sorted(
            [col for col in all_games_df.columns if col.startswith("BON")]
        )
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
        user_cols = sorted(
            [col for col in all_games_df.columns if col.endswith("user")]
        )
        vp_cols = sorted([col for col in all_games_df.columns if col.endswith("vp")])
        vp_margin_cols = sorted(
            [col for col in all_games_df.columns if col.endswith("vp_margin")]
        )
        category_cols = (
            score_cols
            + bonus_cols
            + faction_cols
            + user_cols
            + vp_cols
            + vp_margin_cols
        )
        non_cat_cols = [col for col in all_games_df.columns if col not in category_cols]
        all_games_df = all_games_df.reindex(columns=non_cat_cols + category_cols)
        # Drop unscored games
        score_cols = [
            col for col in all_games_df.columns if col.startswith("score_turn")
        ]
        mask = ~all_games_df[score_cols].isna().any(axis="columns")
        all_games_df = all_games_df.loc[mask].copy()
        self.save(all_games_df)


if __name__ == "__main__":
    d6tflow.run(TaskCleanData())
