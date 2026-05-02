# NBA Player Prop Model

## Project Overview
A sports analytics model that projects NBA player performance against prop lines 
using historical game log data. The model combines statistical projection methods, 
head-to-head opponent analysis, and a walk-forward backtesting framework to 
generate data-driven over/under verdicts.

## Tools and Technologies
### Python
- nba_api — game log data across multiple seasons
- pandas — data manipulation and cleaning
- numpy — statistical calculations
- scipy — normal distribution probability modeling

## How It Works
1. Pulls 3 seasons of game logs via the NBA API
2. Projects the stat using per-minute rate scaled to projected minutes
3. Runs head-to-head analysis against the specific opponent
4. Backtests the edge signal against historical outcomes
5. Generates a verdict only when the backtest confirms the edge

## Supported Stats
PTS, REB, AST, STL, BLK, FG3M, PTS+REB, PTS+AST, PTS+REB+AST, AST+REB, STL+BLK

## Usage
```python
run_model(
    player_id=201939,   # NBA API player ID
    stat="PTS",         # Stat to project
    line=26.5,          # Prop line
    opponent="LAC",     # Opponent abbreviation
    backtest=True       # Run backtest and generate verdict
)
```

## Example Output
Season projects Curry at 24.99 PTS against a 26.5 line. Backtest confirmed 
LEAN UNDER at 44.4% hit rate, verdict returned NO BET as it did not meet the 
55% confidence threshold.