import pandas as pd
import numpy as np
from scipy.stats import norm


def get_team_filtered_data(
    team_id1,
    team_id2,
    home_team,
    date,
    season_filter,
    missing_players1,
    missing_players2,
    list_high_usage,
    path_converted,
):
    """
    This function returns the filtered data that will be used to feed the model

    It performs the followings
    - Finds team games before the following date filter
    - Filters out any players that are missing in the data
    - Reevaluate features

    Returns:
    - All Relevant Data for the team for model prediction as a pandas dataframe
    """
    path_segment_player = (
        f"{path_converted}/segment_stats_playertotals.parquet"
    )
    path_game_data = f"{path_converted}/gamesdf.parquet"
    path_segment_info = f"{path_converted}/segment_information.parquet"
    df_segment_player = pd.read_parquet(path_segment_player)
    df_games = pd.read_parquet(path_game_data)
    df_segment_info = pd.read_parquet(path_segment_info)

    # Get Games To Review

    df_games["date"] = pd.to_datetime(df_games["date"], format="%m/%d/%y")

    mask_date_filter = df_games["date"] < date

    mask_season_filter = df_games['season'] == season_filter

    list_team1_team_2 = [team_id1, team_id2]

    mask_team_filter = df_games["team1"].isin(list_team1_team_2) | df_games[
        "team2"
    ].isin(list_team1_team_2)

    df_games_filtered = df_games[mask_date_filter & (mask_team_filter & mask_season_filter)].reset_index(
        drop=True
    )

    list_selected_games = df_games_filtered["game_id"].tolist()

    # Filter Player Segment Data

    df_filtered_player_segments = df_segment_player[
        df_segment_player["game_id"].isin(list_selected_games)
    ]

    mask_filter_players = df_filtered_player_segments["player_id"].isin(
        missing_players1
    ) | df_filtered_player_segments["player_id"].isin(missing_players2)

    list_exclude_segments = set(
        df_filtered_player_segments[mask_filter_players]["segment_id"].tolist()
    )

    # Now Filter The Segments With Missing Players
    mask_filter_segments = df_filtered_player_segments["segment_id"].isin(
        list_exclude_segments
    )

    df_feat_start = df_filtered_player_segments[~mask_filter_segments]

    df_filtered_seg_info = df_segment_info[
        ~df_segment_info["segment_id"].isin(list_exclude_segments)
    ]

    df_filtered_seg_info = df_filtered_seg_info[
        df_filtered_seg_info["game_id"].isin(list_selected_games)
    ]

    df_games_filtered["team1_home_ind"] = (
        df_games_filtered["team1"] == home_team
    ).astype(int)
    df_games_filtered["team2_home_ind"] = (
        df_games_filtered["team2"] == home_team
    ).astype(int)
    df_games_filtered.drop(columns="home_team", inplace=True)

    df_team1_agg = df_feat_start.groupby(["game_id", "team_id"], as_index=False).agg(
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

    df_team2_agg = df_feat_start.groupby(["game_id", "team_id"], as_index=False).agg(
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

    df_games_filtered = (
        df_games_filtered.merge(
            df_team1_agg,
            how="left",
            left_on=["game_id", "team1"],
            right_on=["game_id", "team_id"],
        )
        .drop(columns="team_id")
        .merge(
            df_team2_agg,
            how="left",
            left_on=["game_id", "team2"],
            right_on=["game_id", "team_id"],
        )
        .drop(columns="team_id")
    )

    df_games_filtered["two_pt_perc_team1"] = df_games_filtered[
        "total_two_makes_team1"
    ] / (
        df_games_filtered["total_two_makes_team1"]
        + df_games_filtered["total_two_misses_team1"]
    )
    df_games_filtered["three_pt_perc_team1"] = df_games_filtered[
        "total_three_makes_team1"
    ] / (
        df_games_filtered["total_three_makes_team1"]
        + df_games_filtered["total_three_misses_team1"]
    )
    df_games_filtered["ft_rate_team1"] = df_games_filtered["total_ft_makes_team1"] / (
        df_games_filtered["total_ft_makes_team1"]
        + df_games_filtered["total_ft_misses_team2"]
    )
    df_games_filtered["total_points_team1"] = (
        df_games_filtered["total_ft_makes_team1"]
        + 2 * (df_games_filtered["total_two_makes_team1"])
        + 3 * (df_games_filtered["total_three_makes_team1"])
    )

    df_games_filtered["two_pt_perc_team2"] = df_games_filtered[
        "total_two_makes_team2"
    ] / (
        df_games_filtered["total_two_makes_team2"]
        + df_games_filtered["total_two_misses_team2"]
    )
    df_games_filtered["three_pt_perc_team2"] = df_games_filtered[
        "total_three_makes_team2"
    ] / (
        df_games_filtered["total_three_makes_team2"]
        + df_games_filtered["total_three_misses_team2"]
    )
    df_games_filtered["ft_rate_team2"] = df_games_filtered["total_ft_makes_team2"] / (
        df_games_filtered["total_ft_makes_team2"]
        + df_games_filtered["total_ft_misses_team2"]
    )
    df_games_filtered["total_points_team2"] = (
        df_games_filtered["total_ft_makes_team2"]
        + 2 * (df_games_filtered["total_two_makes_team2"])
        + 3 * (df_games_filtered["total_three_makes_team2"])
    )

    df_games_filtered.drop(
        columns=[
            "total_two_makes_team1",
            "total_two_makes_team2",
            "total_two_misses_team1",
            "total_two_misses_team2",
            "total_ft_makes_team1",
            "total_ft_makes_team2",
            "total_ft_misses_team1",
            "total_ft_misses_team2",
            "season",
            "league",
            "date",
            "winning_team",
            "validteamcount",
        ],
        inplace=True,
    )

    ## Get The Advanced Metrics
    path_shot_data = f"{path_converted}/shotsdf.parquet"
    df_shot_data = pd.read_parquet(path_shot_data)

    df_filtered_shots = df_shot_data.loc[
        df_shot_data["game_id"].isin(list_selected_games)
    ].loc[~df_shot_data["segment_id"].isin(list_exclude_segments)]

    team1_game_shot_data = df_filtered_shots.groupby(
        ["game_id", "team_id"], as_index=False
    ).agg(
        average_shot_opp_team1=("opportunity", "mean"),
        average_xps_team1=("xps", "mean"),
        total_assisted_team1=("Assisted", "sum"),
        total_paint_team1=("Paint", "sum"),
        average_distace_team1=("Distance", "mean"),
    )

    team2_game_shot_data = df_filtered_shots.groupby(
        ["game_id", "team_id"], as_index=False
    ).agg(
        average_shot_opp_team2=("opportunity", "mean"),
        average_xps_team2=("xps", "mean"),
        total_assisted_team2=("Assisted", "sum"),
        total_paint_team2=("Paint", "sum"),
        average_distace_team2=("Distance", "mean"),
    )

    df_games_filtered = (
        df_games_filtered.merge(
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

    df_games_filtered.dropna(axis=0, how="any", inplace=True)

    df_segment_stats = df_filtered_seg_info.groupby(["game_id"], as_index=False).agg(
        total_possessions_team1=("possessions_team1", "sum"),
        total_possessions_team2=("possessions_team2", "sum"),
        total_game_duration=("game_duration", "sum"),
    )

    df_games_filtered = df_games_filtered.merge(
        df_segment_stats, on="game_id", how="left"
    )

    df_games_filtered["team1_oe"] = (
        df_games_filtered["total_points_team1"]
        / df_games_filtered["total_possessions_team1"]
    ) * 100

    df_games_filtered["team1_de"] = (
        df_games_filtered["total_points_team2"]
        / df_games_filtered["total_possessions_team2"]
    ) * 100

    df_games_filtered["team2_oe"] = (
        df_games_filtered["total_points_team2"]
        / df_games_filtered["total_possessions_team2"]
    ) * 100

    df_games_filtered["team2_de"] = (
        df_games_filtered["total_points_team1"]
        / df_games_filtered["total_possessions_team1"]
    ) * 100

    df_games_filtered["team1_team2_spread"] = (
        df_games_filtered["total_points_team1"]
        - df_games_filtered["total_points_team2"]
    )

    df_games_filtered.drop(
        columns=["total_points_team1", "total_points_team2"], inplace=True
    )

    ## Start High Usage Set Up

    df_high_usage_start = df_filtered_player_segments.copy()

    df_high_usage_start["high_usage_player_in_segment"] = df_high_usage_start[
        "player_id"
    ].isin(list_high_usage)

    team1_segment_info = (
        df_high_usage_start.groupby(["team_id", "segment_id"], as_index=False)
        .agg(high_usage_in_segment=("high_usage_player_in_segment", "max"))
        .rename(
            columns={
                "segment_id": "segment_id1",
                "high_usage_in_segment": "team1_high_usage_missing",
            }
        )
    )

    team2_segment_info = (
        df_high_usage_start.groupby(["team_id", "segment_id"], as_index=False)
        .agg(high_usage_in_segment=("high_usage_player_in_segment", "max"))
        .rename(
            columns={
                "segment_id": "segment_id2",
                "high_usage_in_segment": "team2_high_usage_missing",
            }
        )
    )

    df_filtered_seg_info = (
        df_filtered_seg_info.merge(
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

    df_seg_total = df_filtered_seg_info.groupby(["game_id"], as_index=False).agg(
        total_game_duration=("game_duration", "sum"),
    )

    df_team1_high_usage = (
        df_filtered_seg_info[df_filtered_seg_info["team1_high_usage_missing"] == 1]
        .groupby(["game_id"], as_index=False)
        .agg(total_missing_duration_team1=("game_duration", "sum"))
    )

    df_team2_high_usage = (
        df_filtered_seg_info[df_filtered_seg_info["team2_high_usage_missing"] == 1]
        .groupby(["game_id"], as_index=False)
        .agg(total_missing_duration_team2=("game_duration", "sum"))
    )

    df_missing_high_usage = (
        df_seg_total.merge(df_team1_high_usage, on="game_id", how="left")
        .merge(df_team2_high_usage, on="game_id", how="left")
        .fillna(0.0)
    )

    df_missing_high_usage["prop_team1_highusage_missing"] = (
        df_missing_high_usage["total_missing_duration_team1"]
        / df_missing_high_usage["total_game_duration"]
    )
    df_missing_high_usage["prop_team2_highusage_missing"] = (
        df_missing_high_usage["total_missing_duration_team2"]
        / df_missing_high_usage["total_game_duration"]
    )

    df_missing_high_usage.drop(
        columns=[
            "total_game_duration",
            "total_missing_duration_team1",
            "total_missing_duration_team2",
        ],
        inplace=True,
    )

    df_games_filtered_final = df_games_filtered.merge(
        df_missing_high_usage, on="game_id", how="left"
    )

    return df_games_filtered_final


def gen_pred_list(team1, team2, preds):
    """
    This function returns the following prediction results back for entrydict

    Input: team1 name, team2 name, predictions from model

    Output: A dict for each team containing
    - The mean oe, the mean de, stddev oe, stddev de, spread mean, spread stddev, average spread prob
    """
    pred_dicts_prelim = {
        "team1": {"oe_list": [], "de_list": [], "score_list": []},
        "team2": {"oe_list": [], "de_list": [], "score_list": []},
    }

    for k, v in preds.items():
        team_1_val = v["team1"]
        team_2_val = v["team2"]

        if team_1_val == team1:
            pred_dicts_prelim["team1"]["oe_list"].append(v["pred_team1_oe"])
            pred_dicts_prelim["team1"]["de_list"].append(v["pred_team1_de"])
            pred_dicts_prelim["team1"]["score_list"].append(v["expected_team1_score"])

        elif team_2_val == team1:
            pred_dicts_prelim["team1"]["oe_list"].append(v["pred_team2_oe"])
            pred_dicts_prelim["team1"]["de_list"].append(v["pred_team2_de"])
            pred_dicts_prelim["team1"]["score_list"].append(v["expected_team2_score"])
        else:
            continue

    for k, v in preds.items():
        team_1_val = v["team1"]
        team_2_val = v["team2"]

        if team_1_val == team2:
            pred_dicts_prelim["team2"]["oe_list"].append(v["pred_team1_oe"])
            pred_dicts_prelim["team2"]["de_list"].append(v["pred_team1_de"])
            pred_dicts_prelim["team2"]["score_list"].append(v["expected_team1_score"])

        elif team_2_val == team2:
            pred_dicts_prelim["team2"]["oe_list"].append(v["pred_team2_oe"])
            pred_dicts_prelim["team2"]["de_list"].append(v["pred_team2_de"])
            pred_dicts_prelim["team2"]["score_list"].append(v["expected_team2_score"])
        else:
            continue

    pred_dicts_intermed = {
        "team1_pred": {
            "mean_oe": np.mean(pred_dicts_prelim["team1"]["oe_list"]),
            "stddev_oe": np.std(pred_dicts_prelim["team1"]["oe_list"]),
            "mean_de": np.mean(pred_dicts_prelim["team1"]["de_list"]),
            "stddev_de": np.std(pred_dicts_prelim["team1"]["de_list"]),
            "mean_score": np.mean(pred_dicts_prelim["team1"]["score_list"]),
            "stddev_score": np.std(pred_dicts_prelim["team1"]["score_list"]),
            "n": len(pred_dicts_prelim["team1"]["oe_list"]),
        },
        "team2_pred": {
            "mean_oe": np.mean(pred_dicts_prelim["team2"]["oe_list"]),
            "stddev_oe": np.std(pred_dicts_prelim["team2"]["oe_list"]),
            "mean_de": np.mean(pred_dicts_prelim["team2"]["de_list"]),
            "stddev_de": np.std(pred_dicts_prelim["team2"]["de_list"]),
            "mean_score": np.mean(pred_dicts_prelim["team2"]["score_list"]),
            "stddev_score": np.std(pred_dicts_prelim["team2"]["score_list"]),
            "n": len(pred_dicts_prelim["team1"]["oe_list"]),
        },
    }

    pred_dicts_intermed2 = {
        "team1_pred": {
            "mean_oe": np.round(pred_dicts_intermed["team1_pred"]["mean_oe"], 2).item(),
            "stddev_oe": np.round(pred_dicts_intermed["team1_pred"]["stddev_oe"], 2).item(),
            "mean_de": np.round(pred_dicts_intermed["team1_pred"]["mean_de"], 2).item(),
            "stddev_de": np.round(pred_dicts_intermed["team1_pred"]["stddev_de"], 2).item(),
            "mean_score": np.round(pred_dicts_intermed["team1_pred"]["mean_score"], 2).item(),
            "stddev_score": np.round(pred_dicts_intermed["team1_pred"]["stddev_score"], 2).item(),
            "n": pred_dicts_intermed["team1_pred"]["n"],
        },
        "team2_pred": {
            "mean_oe": np.round(pred_dicts_intermed["team2_pred"]["mean_oe"], 2).item(),
            "stddev_oe": np.round(pred_dicts_intermed["team2_pred"]["stddev_oe"], 2).item(),
            "mean_de": np.round(pred_dicts_intermed["team2_pred"]["mean_de"], 2).item(),
            "stddev_de": np.round(pred_dicts_intermed["team2_pred"]["stddev_de"], 2).item(),
            "mean_score": np.round(pred_dicts_intermed["team2_pred"]["mean_score"], 2).item(),
            "stddev_score": np.round(
                pred_dicts_intermed["team2_pred"]["stddev_score"], 2
            ).item(),
            "n": pred_dicts_intermed["team2_pred"]["n"],
        },
        "team_1_team2_expected_difference": {
            "mean": (np.round(pred_dicts_intermed["team1_pred"]["mean_score"]
            - pred_dicts_intermed["team2_pred"]["mean_score"],2)).item(),
            "stddev": np.round(np.sqrt(
                (
                    pred_dicts_intermed["team1_pred"]["stddev_score"] ** 2
                    / pred_dicts_intermed["team1_pred"]["n"]
                )
                + (
                    pred_dicts_intermed["team2_pred"]["stddev_score"] ** 2
                    / pred_dicts_intermed["team2_pred"]["n"]
                )
            ),2).item(),
        },
    }

    pred_dict_final = {
        't1_minus_t2_exp_score_difference_mean': pred_dicts_intermed2['team_1_team2_expected_difference']['mean'],
        't1_minus_t2_exp_score_difference_stddev': pred_dicts_intermed2['team_1_team2_expected_difference']['stddev']
    }

    return pred_dict_final