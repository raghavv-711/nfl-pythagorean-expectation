# Pythagorean Expectation for the NFL

Fits the Pythagorean exponent `k` for NFL teams, then uses it to track expected wins during the season and to give a win probability for each team's next game.

**Win% = PF^k / (PF^k + PA^k)**

Data: every regular-season game from 1999 to the present via [`nfl_data_py`](https://github.com/nflverse/nfl_data_py) (nflverse). Ties count as half a win.

## Method

- **Exponent:** binomial GLM with a logit link, `logit(p) = k · log(PF/PA)`, fit on 861 team-seasons (1999–2025), weighted by games played. No intercept is used because it isn't needed (see results).
- **Weekly tracker:** cumulative PF/PA through each week of the in-progress season, with expected wins computed using `k` fit on completed seasons only (no look-ahead).
- **Offense/defense ratings:** each team gets an offense rating (points scored per game above average) and a defense rating (points allowed per game vs average), each shrunk toward average with its own constant (offense 4 phantom games, defense 8), chosen by score error on 1999–2019. Expected score is `home pts = c0 + HFA + c_off·off_home + c_def·def_away` (and symmetrically for the away team), with coefficients fit by OLS on team-game scores. Win probability is `Φ(expected margin / 13.8)`.
- **Full-season predictions:** every game gets an expected score and a home-win probability from the offense/defense model (played games use pre-game ratings; unplayed games use ratings as of the latest completed week). The remaining games are then simulated 20,000 times, **week by week**: before each week, every team's offense and defense are recomputed from its points for/against so far in that simulated season, and each team's score is drawn as `Normal(expected points, 9.8)`. A static-ratings run is kept for comparison. Output: projected records, win-total ranges and odds of 10+ wins or the league's best record.
- **Playoff seeding:** the same 20,000 week-by-week simulated seasons are seeded per conference (4 division winners + 3 wild cards). Tiebreakers are simplified: wins, then division record (division winners) or conference record (wild cards), then a coin flip. Head-to-head, common opponents and strength of victory are not modeled.
- **Baseline win probability:** the earlier single-number Pythagorean rating, `k · log(PF/PA)` shrunk with 8 phantom games and mapped to a win probability by logistic regression, is kept as a comparison model.

## Results

| | |
|---|---|
| Fitted `k` | **2.67** (95% bootstrap CI 2.57–2.77) |
| Commonly cited NFL `k` | 2.37 (outside the CI) |
| Intercept α | 0.013 (p = 0.48), so the plain formula holds |

**Expected wins, test on 2020–2025 (fit on 1999–2019):**

| Model | MAE (wins) | RMSE (wins) |
|---|---|---|
| Fitted k = 2.62 | 1.11 | 1.41 |
| Benchmark k = 2.37 | 1.16 | 1.46 |
| MLB exponent (1.83) | 1.36 | 1.67 |
| Everyone at .500 | 2.66 | 3.19 |

**Exponent by era:** 2.55 (1999–2008), 2.69 (2009–2016), 2.79 (2017–2025). The 95% intervals overlap, so the upward trend isn't clearly significant.

**Luck vs. skill:** a team's gap between actual and expected wins in one season has almost no correlation with the next (0.003). Expected win% predicts next-season win% slightly better than actual win% does (0.363 vs 0.329).

**Biggest over-performers:** 2024 Chiefs (15 wins vs 10.4 expected), 2022 Vikings (13 vs 8.4), 2012 Colts (11 vs 7.1).

**Game win probabilities, test on 2020–2025:**

| Model | Log-loss | Brier | Accuracy (non-ties) |
|---|---|---|---|
| Offense/defense model | 0.654 | 0.2306 | 59.6% |
| Pythagorean model | 0.655 | 0.2310 | 60.8% |
| Home-field only | 0.693 | 0.249 | 53.4% |
| Coin flip | 0.693 | 0.249 | n/a |

The offense/defense model is marginally better on log-loss and Brier score and marginally worse on accuracy. The two are statistically hard to tell apart, so the main gain is interpretability (separate offense and defense ratings, expected scores) and a simulation whose offense and defense evolve independently.

## Caveats

- Early-season numbers rest on very few games. Treat projected wins and win probabilities as rough until about week 8.
- The home-field term is fit on 1999–2019 (about 58% for evenly matched teams), which may overstate the recent home edge.
- The model ignores injuries, quarterback changes and rest.
- Week-by-week updating widens the projected win ranges (average SD of season wins 2.3 vs 1.8 with static ratings), because simulated results feed back into ratings. Simulated scores are independent normals around each team's expected points, with no correlation between the two teams' scores.

## How to run

Requires Python 3.9+ and an internet connection (data is downloaded on first run).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy statsmodels matplotlib nfl_data_py jupyter
jupyter notebook "NFL Pythagorean Expectation.ipynb"
```

Run all cells. To refresh during the season, re-run the notebook: it detects the in-progress season and the latest completed week automatically.

To run headless and save the outputs into the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace "NFL Pythagorean Expectation.ipynb"
```

## Files

| File | What it is |
|---|---|
| `NFL Pythagorean Expectation.ipynb` | The analysis (fit, evaluation, tracker, win probabilities) |
| `build_nb.py` | Script that generates the notebook (`pip install nbformat`); edit it and run `python build_nb.py` to regenerate |
| `expected_wins_<season>.csv` | Weekly cumulative expected wins per team (created by the notebook) |
| `next_game_win_prob_<season>.csv` | Next-game win probability, predicted score and offense/defense ratings (created by the notebook) |
| `season_predictions_<season>.csv` | Every regular-season game: expected score, home-win probability, pick, and result if played |
| `playoff_odds_<season>.csv` | Per-team odds of making the playoffs, winning the division and landing each seed |
| `projected_records_<season>.csv` | Simulated final wins per team with percentiles and threshold odds |

## Notebook sections

1. Team-season data
2. Fit `k`, intercept test, bootstrap CI
3. Out-of-sample accuracy
4. Fit quality plots
5. Over- and under-performers
6. Does over-performance persist?
7. Exponent by era
8. Last completed season
9. Weekly expected-wins tracker
10. Next-game win probability (Pythagorean baseline)
10b. Team offense and defense ratings
11. Full-season matchup predictions and projected records
12. Playoff seeding simulation
