# NFL Pythagorean Expectation

I wanted to see if the "Pythagorean expectation" idea from baseball works for the NFL, and then keep going and use it to predict this season. This repo is the result: a notebook that fits the formula, checks how good it is, and then builds game predictions, projected records and playoff odds for the 2026 season.

Everything is in one notebook: `NFL Pythagorean Expectation.ipynb`.

## The basic idea

The Pythagorean expectation says you can guess a team's win percentage from how many points it scores and allows:

```
win % = PF^k / (PF^k + PA^k)
```

`PF` is points for, `PA` is points against, and `k` is a number that depends on the sport. If a team scores and allows the same amount, it comes out to 50%. If it scores a lot more than it allows, the number climbs toward 100%. The exponent `k` controls how fast that happens.

Baseball's `k` is about 1.83. People usually say the NFL's is about 2.37. I fit it myself using every regular-season game from 1999 to 2025 (data from `nfl_data_py`) and got about **2.67**.

## How I fit `k`, in plain words

I took every team-season (861 of them) and wrote down two things: how the points for vs. points against ratio looked, and what the team's win percentage actually was. Then I let a logistic regression find the `k` that makes the formula match reality as closely as possible. Ties count as half a win.

I also tried adding a fudge factor (an intercept) in case teams with equal points for and against don't land at exactly 50%. It came out basically zero (0.013, p = 0.48), so the plain formula is fine.

## What I found

- **Fitted `k` = 2.67**, with a 95% range of 2.57 to 2.77. The usual 2.37 is outside that range, so the modern NFL seems to be a bit more sensitive to point differential.
- **It predicts wins well.** Fit on 1999–2019 and tested on 2020–2025, it's off by about 1.1 wins per team on average. Guessing every team goes 8.5–8.5 is off by about 2.7.

| Model | Avg. miss (wins) |
|---|---|
| My fitted `k` | 1.11 |
| The usual 2.37 | 1.16 |
| Baseball's 1.83 | 1.36 |
| Everybody at .500 | 2.66 |

- **Luck matters a lot.** How much a team over- or under-performs its point differential one year tells you almost nothing about the next year (correlation 0.003). Expected win% also predicts next year's record slightly better than actual win% does (0.363 vs 0.329).
- **Biggest lucky teams:** 2024 Chiefs (15 wins vs. about 10.4 expected), 2022 Vikings (13 vs. 8.4), 2012 Colts (11 vs. 7.1).
- `k` looks like it drifts up over time (2.55 in 1999–2008, 2.69 in 2009–2016, 2.79 in 2017–2025), but the error bars overlap, so I wouldn't say it's proven.

## Predicting games

Season records are one thing, but I also wanted to predict individual games. Here's how it works, step by step.

**1. Give every team an offense score and a defense score.**
Offense = how many points per game they score compared to the league average. Defense = how many points per game they allow compared to average. Early in the season these are noisy (one 45-point game can make an offense look amazing), so I pull them back toward average by pretending each team has some extra "average" games on top of their real ones. For offense I add 4 pretend games, for defense 8. Defense gets more because it's noisier week to week. I picked those numbers by seeing which ones predicted scores best on old seasons.

**2. Turn those into an expected score for each game.**
A team's expected points depend on its own offense, the other team's defense, and home-field advantage (worth about 2.5 points). So a good offense against a bad defense at home gets a big number.

**3. Turn the expected scores into a win probability.**
Take the expected margin (home points minus away points). Real NFL margins scatter around the prediction by about 14 points, so I use a bell curve to convert "we expect to win by 3" into a percentage.

**How good is it?** On the 2020–2025 games it picks the winner about 60% of the time, compared to 53% for "always pick the home team." It has a tiny edge over the simpler version that only uses one rating per team (`k * log(PF/PA)`) on log-loss and Brier score, and it's a tiny bit worse on plain accuracy, so honestly they're about even. The real upside of the offense/defense version is that you get separate ratings and an actual predicted score.

| Model | Log-loss (lower is better) | Accuracy |
|---|---|---|
| Offense/defense model | 0.654 | 59.6% |
| One-rating Pythagorean model | 0.655 | 60.8% |
| Always pick home team | 0.693 | 53.4% |
| Coin flip | 0.693 | n/a |

## Simulating the rest of the season

To get projected records and playoff odds, I play out the rest of the schedule 20,000 times.

- Each game gets a random score based on the expected points from above (with about 10 points of randomness per team).
- The simulation goes **week by week**. After each simulated week, I recalculate every team's offense and defense using the simulated results, so a team that gets hot in a simulation carries that forward.
- After all 18 weeks, I count up wins in each of the 20,000 seasons and average them.

For playoffs I seed 7 teams per conference in each simulated season: 4 division winners, then 3 wild cards. I simplified the tiebreakers: wins first, then division record (for division titles) or conference record (for wild cards), then a coin flip. The real NFL tiebreakers (head-to-head, strength of victory, etc.) aren't in there, so close races are approximate.

I also kept a version where ratings never update during the simulation. It gives similar averages but tighter ranges. The week-by-week version is wider because simulated results feed back into the ratings.

## Things to keep in mind

- Early in the season everything is based on very few games. Take the numbers with a grain of salt until around week 8.
- The home-field number comes from 1999–2019. Home teams have won a bit less often recently, so it might be a little high.
- It knows nothing about injuries, quarterback changes, weather or rest.
- The two teams' scores in a simulated game are random independently of each other, which isn't quite realistic.

## How to run it

You need Python 3.9+ and internet (it downloads the data).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy statsmodels matplotlib scipy nfl_data_py jupyter
jupyter notebook "NFL Pythagorean Expectation.ipynb"
```

Run all the cells. If you rerun it during the season, it finds the current season and the latest completed week on its own. To run it from the terminal and save the results into the notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace "NFL Pythagorean Expectation.ipynb"
```

## What's in the repo

| File | What it is |
|---|---|
| `NFL Pythagorean Expectation.ipynb` | The whole analysis |
| `build_nb.py` | Script I use to generate the notebook (needs `nbformat`) |
| `expected_wins_2026.csv` | Expected wins for each team after each week |
| `next_game_win_prob_2026.csv` | Each team's next game: win probability, predicted score, offense/defense ratings |
| `season_predictions_2026.csv` | Every regular-season game with a predicted score and win probability |
| `projected_records_2026.csv` | Projected final wins, ranges and odds from the simulation |
| `playoff_odds_2026.csv` | Odds of making the playoffs, winning the division and getting each seed |

## What's in the notebook

1. Building the team-season data
2. Fitting `k` (plus the intercept test and confidence range)
3. Testing accuracy on later seasons
4. Fit plots
5. Biggest over- and under-performers
6. Does over-performance carry over to next year?
7. Does `k` change over time?
8. Last completed season
9. Weekly expected-wins tracker
10. Baseline win probability (one rating per team)
10b. Offense and defense ratings
11. Full-season game predictions and projected records
12. Playoff seeding simulation
