from nba_api.stats.endpoints import playergamelog
import pandas as pd
import numpy as np
from scipy.stats import norm

# Pull game logs across multiple seasons and combine into one dataframe
def get_logs_multiyear(player_id, seasons=["2025-26", "2024-25", "2023-24"], verbose=True):
    dfs = []
    for season in seasons:
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
        df = gl.get_data_frames()[0]
        if verbose:
            print(f"Games found: {len(df)}")
        if not df.empty:
            dfs.append(df)
    return pd.concat(dfs).sort_values("GAME_DATE").reset_index(drop=True)

# Filter game log to only games against a specific opponent
def get_h2h(df, opponent_abbr):
    return df[df["MATCHUP"].str.contains(opponent_abbr)].reset_index(drop=True)

# Project minutes using a 65/35 blend of recent (last 5) and season average
def project_minutes(df):
    mins = df["MIN"].astype(float).values
    recent = np.mean(mins[-5:])
    season = np.mean(mins)
    return 0.65 * recent + 0.35 * season

# Calculate total stat per total minutes played — rate based mean
def per_minute(df, col):
    return df[col].astype(float).sum() / df["MIN"].astype(float).sum()

# Project a stat using per-minute rate scaled to projected minutes (mean)
def project_stat(df, stat, projected_minutes):
    ppm = per_minute(df, stat)
    return ppm * projected_minutes

# Project a stat using median per-minute rate — more resistant to outliers than mean
def project_stat_median(df, stat, projected_minutes):
    per_min_values = df[stat].astype(float).values / df["MIN"].astype(float).values
    median_ppm = np.median(per_min_values)
    return median_ppm * projected_minutes

# Classify the edge between projection and line
def get_edge(projection, line):
    diff = projection - line
    if diff >= 2.5:
        return "STRONG OVER"
    elif diff >= 1.5:
        return "LEAN OVER"
    elif diff <= -2.5:
        return "STRONG UNDER"
    elif diff <= -1.5:
        return "LEAN UNDER"
    else:
        return "NO BET"

# Estimate std dev using last 10 games only — captures current volatility not historical noise
def estimate_std(df, stat, projected_minutes):
    values = df[stat].astype(float).values[-10:]
    mins = df["MIN"].astype(float).values[-10:]
    per_minute = values / mins
    ppm_std = np.std(per_minute)
    return ppm_std * projected_minutes

# Probability of going over the line using normal distribution
def probability_over(projection, std, line):
    return 1 - norm.cdf(line, loc=projection, scale=std)

# Probability of going under the line using normal distribution
def probability_under(projection, std, line):
    return norm.cdf(line, loc=projection, scale=std)

# Add combined stat columns to match prop types
# TO DO: Add any additional combos as needed
def add_combo_stats(df):
    df["PTS_AST"] = df["PTS"].astype(float) + df["AST"].astype(float)
    df["PTS_REB"] = df["PTS"].astype(float) + df["REB"].astype(float)
    df["PTS_REB_AST"] = df["PTS"].astype(float) + df["REB"].astype(float) + df["AST"].astype(float)
    df["AST_REB"] = df["AST"].astype(float) + df["REB"].astype(float)
    df["STL_BLK"] = df["STL"].astype(float) + df["BLK"].astype(float)
    return df

# Remove games where player played fewer than 20 minutes — DNPs, blowouts, injury games
def clean_data(df):
    return df[df["MIN"].astype(float) > 20]

# Walk forward backtest — for each game, use all prior games to project
# then compare projection direction to what actually happened
# window = minimum number of games needed before starting predictions
# returns hit rates by edge type for use in the verdict
def get_backtest_hit_rates(player_id, stat, line, window=50):
    df = get_logs_multiyear(player_id, verbose=False)
    df = clean_data(df).reset_index(drop=True)
    df = add_combo_stats(df)

    results = []

    for i in range(window, len(df)):
        # everything before this game
        past_games = df.iloc[:i]
        # what actually happened
        actual = float(df.iloc[i][stat])

        mins_proj = project_minutes(past_games)
        projection = project_stat(past_games, stat, mins_proj)
        projection_median = project_stat_median(past_games, stat, mins_proj)
        std = estimate_std(past_games, stat, mins_proj)
        edge = get_edge(projection, line)
        prob_over = probability_over(projection, std, line)
        prob_under = probability_under(projection, std, line)

        results.append({
            "game_num": i,
            "projection_mean": round(projection, 2),
            "projection_median": round(projection_median, 2),
            "actual": actual,
            "edge": edge,
            "prob_over": round(prob_over * 100, 1),
            "prob_under": round(prob_under * 100, 1),
            # hit = True if projection direction matches actual outcome
            "hit": (actual > line) if projection > line else (actual < line)
        })

    results_df = pd.DataFrame(results)
    # only evaluate games where model signalled a bet
    bets = results_df[results_df["edge"] != "NO BET"]

    print(f"\n========== BACKTEST: {stat} line {line} ==========")
    print(f"Total games evaluated:  {len(results_df)}")
    print(f"Total bets signalled:   {len(bets)}")
    print(f"NO BET filtered:        {len(results_df) - len(bets)}")
    print(f"\nOverall hit rate:       {bets['hit'].mean()*100:.1f}%")

    print(f"\n---------- By edge type ----------")
    summary = bets.groupby("edge")["hit"].agg(
        Count="count",
        Hits="sum",
        Hit_Rate=lambda x: round(x.mean() * 100, 1)
    ).reset_index()
    print(summary.to_string(index=False))

    print(f"\n---------- Mean vs Median ----------")
    print(f"Avg mean projection:    {results_df['projection_mean'].mean():.2f}")
    print(f"Avg median projection:  {results_df['projection_median'].mean():.2f}")
    print(f"Avg actual:             {results_df['actual'].mean():.2f}")

    # return hit rates by edge type for verdict system
    hit_rates = bets.groupby("edge")["hit"].mean().to_dict()
    return hit_rates

# Final verdict combining edge signal and backtest hit rate
# only recommends a bet if backtest confirms edge at min_hit_rate or above
# default threshold is 55% — adjust if needed
def get_verdict(edge, hit_rates, min_hit_rate=0.55):
    if edge == "NO BET":
        return "NO BET"
    hit_rate = hit_rates.get(edge, 0)
    direction = edge.split()[1]
    if hit_rate >= min_hit_rate:
        return f"BET {direction} — backtest confirms ({hit_rate*100:.1f}% hit rate)"
    else:
        return f"NO BET — edge exists but backtest doesn't confirm ({hit_rate*100:.1f}% hit rate)"

def run_model(player_id, stat="PTS", line=None, opponent=None, backtest=True):
    df = get_logs_multiyear(player_id)
    df = clean_data(df)
    df = add_combo_stats(df)

    # Season projection
    mins_proj = project_minutes(df)
    projection = project_stat(df, stat, mins_proj)
    projection_median = project_stat_median(df, stat, mins_proj)
    std = estimate_std(df, stat, mins_proj)
    prob_over = probability_over(projection, std, line)
    prob_under = probability_under(projection, std, line)
    edge = get_edge(projection, line)

    print("\n==========")
    print(f"Projected {stat}:      {projection:.2f}")
    print(f"Std Dev:              {std:.2f}")
    print(f"Market Line:          {line}")
    print(f"Over Probability:     {prob_over*100:.1f}%")
    print(f"Under Probability:    {prob_under*100:.1f}%")
    print(f"Edge:                 {edge}")
    print("==========\n")

    h2h_edge = None
    h2h_proj = None
    h2h_proj_median = None

    if opponent:
        h2h_df = get_h2h(df, opponent)

        if len(h2h_df) < 3:
            print(f"\n========== H2H vs {opponent} ==========")
            print(f"Only {len(h2h_df)} game(s) found — not enough data for a reliable H2H projection.")
        else:
            # H2H projection against specific opponent
            h2h_mins = project_minutes(h2h_df)
            h2h_proj = project_stat(h2h_df, stat, h2h_mins)
            h2h_proj_median = project_stat_median(h2h_df, stat, h2h_mins)
            h2h_std = estimate_std(h2h_df, stat, h2h_mins)
            h2h_over = probability_over(h2h_proj, h2h_std, line)
            h2h_under = probability_under(h2h_proj, h2h_std, line)
            h2h_edge = get_edge(h2h_proj, line)

            print(f"\n========== H2H vs {opponent} ==========")
            print(f"Projected {stat}:      {h2h_proj:.2f}")
            print(f"Std Dev:              {h2h_std:.2f}")
            print(f"Market Line:          {line}")
            print(f"Over Probability:     {h2h_over*100:.1f}%")
            print(f"Under Probability:    {h2h_under*100:.1f}%")
            print(f"Edge:                 {h2h_edge}")
            print("==========\n")

    print("\n========== SUMMARY ==========")
    print(f"Projected {stat} (mean):          {projection:.2f}")
    print(f"Projected {stat} (median):        {projection_median:.2f}")
    if h2h_proj is not None:
        print(f"H2H Projected {stat} (mean):      {h2h_proj:.2f}")
        print(f"H2H Projected {stat} (median):    {h2h_proj_median:.2f}")

    if backtest:
        hit_rates = get_backtest_hit_rates(player_id, stat, line)
        verdict = get_verdict(edge, hit_rates)
        print(f"\n========== VERDICT ==========")
        print(f"Season edge:  {edge}")
        if h2h_edge:
            print(f"H2H edge:     {h2h_edge}")
        print(f"Verdict:      {verdict}")
        print("==========\n")

run_model(
    player_id=,   # Put player ID here
    stat="", # Stat to project (PTS, REB, AST, FG3M, PTS_REB_AST, etc.)
    line=,          # Prop line
    opponent="",     # Opponent abbreviation
    backtest=True       # Set False to skip backtest and get quick projection only
)