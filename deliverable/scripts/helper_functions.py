from bson import ObjectId
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from pathlib import Path


def fn_repl_str_target(text, targets):
    """
    Replace multiple instances of specified text in a string with empty space.

    :param text: The original string.
    :param targets: A list of substrings to be replaced.
    :return: The modified string with specified substrings replaced by empty space.
    """
    for target in targets:
        text = text.replace(target, '')
    return text


def fn_convert_pkl_parquet(input_path, dest_path):
    """
    This function takes a pkl folder path and converts the file into a parquet file
    inputs
      - input_path: input relative file path of pkl
      - destination_path: destination path folder

    By default it will use the name of the file
    """
    print("#" * 90)
    print("File Conversion Started!\n")
    data_files = os.listdir(input_path)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    else:
        for f in os.listdir(dest_path):
            os.remove(os.path.join(dest_path, f))

    for fname in data_files:
        if (
            ".pptx" not in fname
            and "segment_players" not in fname
            and ".png" not in fname
        ):
            print(f"Creating parquet version of the {fname} file")
            data_file = pd.read_pickle(f"{input_path}/{fname}")
            for column in data_file.columns:
                if data_file[column].apply(lambda x: isinstance(x, ObjectId)).any():
                    data_file[column] = data_file[column].astype(str)
            data_file[data_file.select_dtypes(include="bool").columns] = (
                data_file.select_dtypes(include="bool").astype(int)
            )
            list_remover = ['NCAAM1+ 23-24 ', 'NCAAM1+ 24-25 ', '.pickle', '_V2']
            target_fname=fn_repl_str_target(fname,list_remover)

            f_path = Path(f"{dest_path}/{target_fname}.parquet")

            if 'eventID' in data_file.columns:
                print("Remove unneccessary column eventID")
                data_file = data_file.drop(columns='eventID')
            
            if "23-24" in fname and "teamsdf" in target_fname:
                data_file['season'] = '23-24'
            elif "24-25" in fname and "teamsdf" in target_fname:
                data_file['season'] = '24-25'
            else:
                pass

            if f_path.exists():
                data_file.to_parquet(f_path, engine='fastparquet', append=True)
            else:
                data_file.to_parquet(f_path,engine='fastparquet')
        else:
            continue


def fn_feat_eng_pipeline(input_path, dest_path):
    """
    This function supercedes the notebook and creates a processing pipeline\n
    to create the base data later used to split training and test
    inputs
        - input_path: path location of initial converted parquet files
        - dest_path: path_location to send initial training and test parquet files
    outputs:
    """
    print("#" * 90)
    print("Feature Engineering Started!\n")
    path_game_data = f"{input_path}/gamesdf.parquet"
    path_shot_data = f"{input_path}/shotsdf.parquet"
    path_team_totals = f"{input_path}/segment_stats_teamtotals.parquet"
    path_segment_data = f"{input_path}/segment_information.parquet"

    df_shots = pd.read_parquet(path_shot_data)
    df_games = pd.read_parquet(path_game_data)
    df_segment_team_data = pd.read_parquet(path_team_totals)
    df_segment_data = pd.read_parquet(path_segment_data)

    game_exclude = df_shots[df_shots["xps"] == 0]["game_id"].unique().tolist()

    df_games["date"] = pd.to_datetime(df_games["date"], format="%m/%d/%y")

    df_games = (
        df_games.sort_values(by="date", ignore_index=True)
        .loc[:, ["game_id", "season", "date", "team1", "team2", "home_team"]]
        .loc[~df_games["game_id"].isin(game_exclude)]
    )

    df_games["team1_home_ind"] = (df_games["team1"] == df_games["home_team"]).astype(
        int
    )
    df_games["team2_home_ind"] = (df_games["team2"] == df_games["home_team"]).astype(
        int
    )
    df_games.drop(columns="home_team", inplace=True)

    team_segment_box_score_team1 = df_segment_team_data.groupby(
        ["game_id", "team_id"], as_index=False
    ).agg(
        total_two_makes_team1=("TwoPtMakes", "sum"),
        total_two_misses_team1=("TwoPtMisses", "sum"),
        total_three_makes_team1=("ThreePtMakes", "sum"),
        total_three_misses_team1=("ThreePtMisses", "sum"),
        total_turnovers_team1=("Turnovers", "sum"),
        total_steals_team1=("Steals", "sum"),
        total_assists_team1=("Assists", "sum"),
        total_blocks_team1=("Blocks", "sum"),
        total_oreb_team1=("ORebs", "sum"),
        total_dreb_team1=("DRebs", "sum"),
        total_ft_makes_team1=("FTMakes", "sum"),
        total_ft_misses_team1=("FTMisses", "sum"),
    )

    team_segment_box_score_team2 = df_segment_team_data.groupby(
        ["game_id", "team_id"], as_index=False
    ).agg(
        total_two_makes_team2=("TwoPtMakes", "sum"),
        total_two_misses_team2=("TwoPtMisses", "sum"),
        total_three_makes_team2=("ThreePtMakes", "sum"),
        total_three_misses_team2=("ThreePtMisses", "sum"),
        total_turnovers_team2=("Turnovers", "sum"),
        total_steals_team2=("Steals", "sum"),
        total_assists_team2=("Assists", "sum"),
        total_blocks_team2=("Blocks", "sum"),
        total_oreb_team2=("ORebs", "sum"),
        total_dreb_team2=("DRebs", "sum"),
        total_ft_makes_team2=("FTMakes", "sum"),
        total_ft_misses_team2=("FTMisses", "sum"),
    )

    df_games = (
        df_games.merge(
            team_segment_box_score_team1,
            how="left",
            left_on=["game_id", "team1"],
            right_on=["game_id", "team_id"],
        )
        .drop(columns="team_id")
        .merge(
            team_segment_box_score_team2,
            how="left",
            left_on=["game_id", "team2"],
            right_on=["game_id", "team_id"],
        )
        .drop(columns="team_id")
    )

    df_games["two_pt_perc_team1"] = df_games["total_two_makes_team1"] / (
        df_games["total_two_makes_team1"] + df_games["total_two_misses_team1"]
    )
    df_games["three_pt_perc_team1"] = df_games["total_three_makes_team1"] / (
        df_games["total_three_makes_team1"] + df_games["total_three_misses_team1"]
    )
    df_games["ft_rate_team1"] = df_games["total_ft_makes_team1"] / (
        df_games["total_ft_makes_team1"] + df_games["total_ft_misses_team2"]
    )
    df_games["total_points_team1"] = (
        df_games["total_ft_makes_team1"]
        + 2 * (df_games["total_two_makes_team1"])
        + 3 * (df_games["total_three_makes_team1"])
    )

    df_games["two_pt_perc_team2"] = df_games["total_two_makes_team2"] / (
        df_games["total_two_makes_team2"] + df_games["total_two_misses_team2"]
    )
    df_games["three_pt_perc_team2"] = df_games["total_three_makes_team2"] / (
        df_games["total_three_makes_team2"] + df_games["total_three_misses_team2"]
    )
    df_games["ft_rate_team2"] = df_games["total_ft_makes_team2"] / (
        df_games["total_ft_makes_team2"] + df_games["total_ft_misses_team2"]
    )
    df_games["total_points_team2"] = (
        df_games["total_ft_makes_team2"]
        + 2 * (df_games["total_two_makes_team2"])
        + 3 * (df_games["total_three_makes_team2"])
    )

    df_games.drop(
        columns=[
            "total_two_makes_team1",
            "total_two_makes_team2",
            "total_two_misses_team1",
            "total_two_misses_team2",
            "total_ft_makes_team1",
            "total_ft_makes_team2",
            "total_ft_misses_team1",
            "total_ft_misses_team2",
        ],
        inplace=True,
    )

    team1_game_shot_data = df_shots.groupby(["game_id", "team_id"], as_index=False).agg(
        average_shot_opp_team1=("opportunity", "mean"),
        average_xps_team1=("xps", "mean"),
        total_assisted_team1=("Assisted", "sum"),
        total_paint_team1=("Paint", "sum"),
        average_distace_team1=("Distance", "mean"),
    )

    team2_game_shot_data = df_shots.groupby(["game_id", "team_id"], as_index=False).agg(
        average_shot_opp_team2=("opportunity", "mean"),
        average_xps_team2=("xps", "mean"),
        total_assisted_team2=("Assisted", "sum"),
        total_paint_team2=("Paint", "sum"),
        average_distace_team2=("Distance", "mean"),
    )

    df_games = (
        df_games.merge(
            team1_game_shot_data,
            how="left",
            left_on=["game_id", "team1"],
            right_on=["game_id", "team_id"],
        )
        .drop(columns="team_id")
        .merge(
            team2_game_shot_data,
            how="left",
            left_on=["game_id", "team2"],
            right_on=["game_id", "team_id"],
        )
        .drop(columns="team_id")
    )

    df_segment_stats = df_segment_data.groupby(["game_id"], as_index=False).agg(
        total_possessions_team1=("possessions_team1", "sum"),
        total_possessions_team2=("possessions_team2", "sum"),
        total_game_duration=("game_duration", "sum"),
    )

    df_games = df_games.merge(df_segment_stats, on="game_id", how="left")

    df_games["team1_oe"] = (
        df_games["total_points_team1"] / df_games["total_possessions_team1"]
    ) * 100

    df_games["team1_de"] = (
        df_games["total_points_team2"] / df_games["total_possessions_team2"]
    ) * 100


    df_games["team2_oe"] = (
        df_games["total_points_team2"] / df_games["total_possessions_team2"]
    ) * 100

    df_games["team2_de"] = (
        df_games["total_points_team1"] / df_games["total_possessions_team1"]
    ) * 100

    df_games["team1_team2_spread"] = (
        df_games["total_points_team1"] - df_games["total_points_team2"]
    )

    df_games.drop(columns=["total_points_team1", "total_points_team2"], inplace=True)

    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    else:
        if os.path.exists(f"{dest_path}/base_data.parquet"):
            os.remove(f"{dest_path}/base_data.parquet")

    print(df_games.head(10))

    df_games.to_parquet(f"{dest_path}/base_data.parquet")

    print("Data Transformed!")
    print("#" * 90)


def fn_train_test_finalization(input_path, train_path, test_path):
    """
    This function supercedes the notebook and creates\n
    to create the datasets needed for training and testing
    inputs
        - input_path: path location of initial base_file
        - train_path: train path to output_data
        - test_path: test path to output data
    outputs:
    """
    print("#" * 90)
    print("Creating Training and Test Splits")

    path_base_data = f"{input_path}/base_data.parquet"

    df_base_data = pd.read_parquet(path_base_data)

    print("Iterate through seasons")

    list_seasons = set(df_base_data["season"].tolist())

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    else:
        for f in os.listdir(train_path):
            os.remove(os.path.join(train_path, f))

    if not os.path.exists(test_path):
        os.makedirs(test_path)
    else:
        for f in os.listdir(train_path):
            os.remove(os.path.join(train_path, f))

    for season in list_seasons:

        df_season_data = df_base_data[df_base_data['season'] == season].sort_values(by="date", ignore_index=True)

        df_season_data.drop(columns='date', inplace=True)

        df_train, df_test = train_test_split(df_season_data, test_size=0.2, shuffle=False)

        print(f"Total Training Data Size for {season}: {len(df_train)}")

        print(df_train.head())

        print(f"Total Testing Data Size for {season}: {len(df_test)}")
        print(df_test.head())

        path_train = Path(f"{train_path}/training_data.parquet")
        path_test = Path(f"{test_path}/testing_data.parquet")

        if path_train.exists() and path_test.exists():
            df_train.to_parquet(path_train, engine='fastparquet', append=True)
            df_test.to_parquet(path_test, engine = 'fastparquet', append=True)
        else:
            df_train.to_parquet(path_train, engine='fastparquet')
            df_test.to_parquet(path_test, engine = 'fastparquet')


def fn_is_highusage_insegment(s, list_high_usage):
    segment = s.loc["segment_id"]
    player_id = s.loc["player_id"]
    is_player_in_list = player_id in list_high_usage
    if is_player_in_list:
        return 1

    return 0


def fn_get_high_usage_players(input_path, dest_path):
    """
    This function takes a pkl folder path and determines the high usage players in training
      - input_path: Location of raw files
      - dest_path: input relative file path of pkl

    Returns
      - A modified version of training and test data with a % high usage player played

    By default it will use the name of the file
    """

    print("#"*90)
    print("Get the High Usage Players Missing Data (Based on Training Data)")

    path_train_data = f"{dest_path}/train/training_data.parquet"
    path_test_data = f"{dest_path}/test/testing_data.parquet"
    path_player_totals = (
        f"{input_path}/segment_stats_playertotals.parquet"
    )
    path_segment_info = f"{input_path}/segment_information.parquet"
    path_player_info = f"{input_path}/player_ids.parquet"

    df_train = pd.read_parquet(path_train_data)
    df_test = pd.read_parquet(path_test_data)
    df_player_totals = pd.read_parquet(path_player_totals)
    df_segment_info = pd.read_parquet(path_segment_info)

    list_train_games = df_train["game_id"].tolist()
    list_test_games = df_test["game_id"].tolist()

    df_player_totals_train = df_player_totals[
        df_player_totals["game_id"].isin(list_train_games)
    ].copy()

    df_player_agg_per_game = df_player_totals_train.groupby(
        ["game_id", "player_id"], as_index=False
    ).agg(
        total_usage=("usage", "sum"),
        total_two_pts=("TwoPtMakes", "sum"),
        total_three_pts=("ThreePtMakes", "sum"),
        total_ft_makes=("FTMakes", "sum"),
        total_o_rebounds=("ORebs", "sum"),
        total_d_rebounds=("DRebs", "sum"),
        total_assists=("Assists", "sum"),
    )

    df_player_agg_per_game["total_points"] = (
        df_player_agg_per_game["total_two_pts"] * 2
        + df_player_agg_per_game["total_three_pts"] * 3
        + df_player_agg_per_game["total_ft_makes"]
    )

    df_player_agg_per_game["total_rebounds"] = (
        df_player_agg_per_game["total_o_rebounds"]
        + df_player_agg_per_game["total_d_rebounds"]
    )

    df_player_agg_per_game.drop(
        columns=[
            "total_two_pts",
            "total_three_pts",
            "total_ft_makes",
            "total_o_rebounds",
            "total_d_rebounds",
        ],
        inplace=True,
    )

    df_player_averages_per_game = df_player_agg_per_game.groupby(
        "player_id", as_index=False
    ).agg(
        ppg=("total_points", "mean"),
        rpg=("total_rebounds", "mean"),
        apg=("total_assists", "mean"),
        upg=("total_usage", "mean"),
    )

    df_high_usage_player = df_player_averages_per_game[
        df_player_averages_per_game["upg"] > 20
    ]["player_id"].reset_index(drop=True)

    df_player_totals_train["high_usage_player_in_segment"] = df_player_totals_train[
        "player_id"
    ].isin(df_high_usage_player.tolist())

    team1_segment_info = (
        df_player_totals_train.groupby(["team_id", "segment_id"], as_index=False)
        .agg(high_usage_in_segment=("high_usage_player_in_segment", "max"))
        .rename(
            columns={
                "segment_id": "segment_id1",
                "high_usage_in_segment": "team1_high_usage_missing",
            }
        )
    )

    team2_segment_info = (
        df_player_totals_train.groupby(["team_id", "segment_id"], as_index=False)
        .agg(high_usage_in_segment=("high_usage_player_in_segment", "max"))
        .rename(
            columns={
                "segment_id": "segment_id2",
                "high_usage_in_segment": "team2_high_usage_missing",
            }
        )
    )

    train_segment_stats = (
        df_segment_info[df_segment_info["game_id"].isin(list_train_games)]
        .merge(
            team1_segment_info,
            left_on=["teamid1", "segment_id"],
            right_on=["team_id", "segment_id1"],
            how="left",
        )
        .drop(columns=["team_id", "segment_id1"])
        .merge(
            team2_segment_info,
            left_on=["teamid2", "segment_id"],
            right_on=["team_id", "segment_id2"],
            how="left",
        )
        .drop(columns=["team_id", "segment_id2"])
    )

    train_segment_totals = train_segment_stats.groupby(["game_id"], as_index=False).agg(
        total_possessions_team1=("possessions_team1", "sum"),
        total_possessions_team2=("possessions_team2", "sum"),
        total_game_duration=("game_duration", "sum"),
    )

    train_segment_team1_high_usage = (
        train_segment_stats[train_segment_stats["team1_high_usage_missing"] == 1]
        .groupby(["game_id"], as_index=False)
        .agg(total_missing_duration_team1=("game_duration", "sum"))
    )

    train_segment_team2_high_usage = (
        train_segment_stats[train_segment_stats["team2_high_usage_missing"] == 1]
        .groupby(["game_id"], as_index=False)
        .agg(total_missing_duration_team2=("game_duration", "sum"))
    )

    df_train_segment_info = (
        train_segment_totals.merge(
            train_segment_team1_high_usage, on="game_id", how="left"
        )
        .merge(train_segment_team2_high_usage, on="game_id", how="left")
        .fillna(0.0)
    )

    df_train_segment_info["prop_team1_highusage_missing"] = (
        df_train_segment_info["total_missing_duration_team1"]
        / df_train_segment_info["total_game_duration"]
    )
    df_train_segment_info["prop_team2_highusage_missing"] = (
        df_train_segment_info["total_missing_duration_team2"]
        / df_train_segment_info["total_game_duration"]
    )

    df_train_segment_info.drop(
        columns=[
            "total_game_duration",
            "total_missing_duration_team1",
            "total_possessions_team1",
            "total_possessions_team2",
            "total_missing_duration_team2",
        ],
        inplace=True,
    )

    df_train_final = df_train.merge(df_train_segment_info, on="game_id", how="left")

    df_player_totals_test = df_player_totals[
        df_player_totals["game_id"].isin(list_test_games)
    ].copy()

    df_player_totals_test["high_usage_player_in_segment"] = df_player_totals_train[
        "player_id"
    ].isin(df_high_usage_player.tolist())

    team1_segment_info = (
        df_player_totals_test.groupby(["team_id", "segment_id"], as_index=False)
        .agg(high_usage_in_segment=("high_usage_player_in_segment", "max"))
        .rename(
            columns={
                "segment_id": "segment_id1",
                "high_usage_in_segment": "team1_high_usage_missing",
            }
        )
    )

    team2_segment_info = (
        df_player_totals_test.groupby(["team_id", "segment_id"], as_index=False)
        .agg(high_usage_in_segment=("high_usage_player_in_segment", "max"))
        .rename(
            columns={
                "segment_id": "segment_id2",
                "high_usage_in_segment": "team2_high_usage_missing",
            }
        )
    )

    test_segment_stats = (
        df_segment_info[df_segment_info["game_id"].isin(list_test_games)]
        .merge(
            team1_segment_info,
            left_on=["teamid1", "segment_id"],
            right_on=["team_id", "segment_id1"],
            how="left",
        )
        .drop(columns=["team_id", "segment_id1"])
        .merge(
            team2_segment_info,
            left_on=["teamid2", "segment_id"],
            right_on=["team_id", "segment_id2"],
            how="left",
        )
        .drop(columns=["team_id", "segment_id2"])
    )

    test_segment_totals = test_segment_stats.groupby(["game_id"], as_index=False).agg(
        total_possessions_team1=("possessions_team1", "sum"),
        total_possessions_team2=("possessions_team2", "sum"),
        total_game_duration=("game_duration", "sum"),
    )

    test_segment_team1_high_usage = (
        test_segment_stats[test_segment_stats["team1_high_usage_missing"] == 1]
        .groupby(["game_id"], as_index=False)
        .agg(total_missing_duration_team1=("game_duration", "sum"))
    )

    test_segment_team2_high_usage = (
        test_segment_stats[test_segment_stats["team2_high_usage_missing"] == 1]
        .groupby(["game_id"], as_index=False)
        .agg(total_missing_duration_team2=("game_duration", "sum"))
    )

    df_test_segment_info = (
        test_segment_totals.merge(
            test_segment_team1_high_usage, on="game_id", how="left"
        )
        .merge(test_segment_team2_high_usage, on="game_id", how="left")
        .fillna(0.0)
    )

    df_test_segment_info["prop_team1_highusage_missing"] = 1 - (
        df_test_segment_info["total_missing_duration_team1"]
        / df_test_segment_info["total_game_duration"]
    )
    df_test_segment_info["prop_team2_highusage_missing"] = 1 - (
        df_test_segment_info["total_missing_duration_team2"]
        / df_test_segment_info["total_game_duration"]
    )

    df_test_segment_info.drop(
        columns=[
            "total_game_duration",
            "total_missing_duration_team1",
            "total_possessions_team1",
            "total_possessions_team2",
            "total_missing_duration_team2",
        ],
        inplace=True,
    )

    df_test_final = df_test.merge(df_test_segment_info, on="game_id", how="left")

    df_train_final.to_parquet(path_train_data)
    df_test_final.to_parquet(path_test_data)

    print(df_high_usage_player.head())

    return(df_high_usage_player.tolist())