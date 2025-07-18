import helper_functions as hp
import build_model as mb
import modeling_functions as mf
from pathlib import Path
import pandas as pd
import online_predictions as op
import math
import pickle
import os


def main():
    hp.fn_convert_pkl_parquet("./GT_MSA_LBA_Project_Data/V2", "./converted_data")
    hp.fn_feat_eng_pipeline("./converted_data", "./transformed_data")
    hp.fn_train_test_finalization(
        "./transformed_data", "./transformed_data/train", "./transformed_data/test"
    )
    list_high_usage = hp.fn_get_high_usage_players(
        "./converted_data", "./transformed_data"
    )

#     # # Optional. To Build Model if needed. Models will be provided in the folder "models"
#     script_dir = Path(__file__).resolve().parent

# # Navigate up two levels to reach "PracticumSP25"
#     practicum_path = script_dir.parent.parent

#     # Construct the paths to the parquet files
#     train_path = practicum_path / "transformed_data" / "train" / "training_data.parquet"
#     test_path = practicum_path / "transformed_data" / "test" / "testing_data.parquet"
#     df_train = pd.read_parquet(train_path)
#     df_test = pd.read_parquet(test_path)

#     mb.train_and_evaluate(df_train, df_test, "team1_oe")
#     mb.train_and_evaluate(df_train, df_test, "team2_oe")
#     mb.train_and_evaluate(df_train, df_test, "team1_de")
#     mb.train_and_evaluate(df_train, df_test, "team2_de")

    # # get pickle file of entrydicts, evaluate model performance
    print("#" * 90)
    print("Online Prediction Time!\n")

    path_entrydicts = "./entrydicts/items/"

    pkl_games = []
    dir = os.fsencode(path_entrydicts)

    for file in os.listdir(dir):
        filename = os.fsdecode(file)
        file_path = os.path.join(path_entrydicts, filename)
        entrydict = pd.read_pickle(file_path)
        pkl_games = pkl_games + entrydict

    team_season_data = pd.read_parquet("./converted_data/teamsdf.parquet")
    total_games_skipped = 0
    total_games_evaluated = 0

    entry_dict_res = []

    for item in pkl_games:
        if item["team1_gamecount"] < 6 or item["team2_gamecount"] < 6:
            total_games_skipped += 1

        else:
            team_1_missing = None
            team_2_missing = None
            played_date = pd.to_datetime(item["date"])
            home_team = str(item["home_team"])
            season = str(item["season1"])
            team_1 = str(item["team1"])
            team_2 = str(item["team2"])

            season_filter = None

            season_filter = team_season_data.loc[
                (team_season_data["team_id"]
                == team_1) & (team_season_data["season_id"]
                == season)
            ].iloc[0,-1]

            if season_filter is None or season_filter=="":
                print("Error!")
                break

            try:
                team_1_missing = [str(i) for i in item["missing_team1"]]
            except:
                team_1_missing = []

            try:
                team_2_missing = [str(i) for i in item["missing_team2"]]
            except:
                team_2_missing = []

            df_selected_games = op.get_team_filtered_data(
                team_1,
                team_2,
                home_team,
                played_date,
                season_filter,
                team_1_missing,
                team_2_missing,
                list_high_usage,
                "./converted_data",
            )

            _, preds = mf.predict_game_outcomes(df_selected_games)

            res = op.gen_pred_list(team_1, team_2, preds)

            item.update(res)

            entry_dict_res.append(item)

            total_games_evaluated += 1

            if total_games_evaluated % 100 == 0:
                print(f"{total_games_evaluated} Games Evaluated")

    print(f"Total Games Given: {len(pkl_games)}")
    print(f"Total Games Skipped: {total_games_skipped}")
    print(f"Total Games Evaluated: {total_games_evaluated}")

    print(entry_dict_res[0:5])

    with open("./entrydicts/entrydics_res.pickle", "wb") as handle:
        pickle.dump(entry_dict_res, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
