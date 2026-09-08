# 🏏 Cricket Analytics: Match Outcome Prediction & Feature Engineering

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/Model-CatBoost-yellow.svg)](https://catboost.ai/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end Machine Learning and Predictive Analytics solution built for the **American Express (Amex) Campus Superbowl Cricket Analytics Competition**. 

This repository implements advanced statistical feature extraction from historical cricket match logs, player-level performance metrics, venue pitch dynamics, and toss conditions to build high-accuracy predictive models for cricket match outcomes.

---

## 📌 Project Overview

Predicting cricket match outcomes is challenging due to the dynamic nature of the sport, player form, pitch conditions, and match situation factors. This project addresses the challenge by:
1. Parsing granular **player-level (batsmen & bowler scorecards)** and **match-level historical data**.
2. Aggregating dynamic rolling performance windows (last *N* matches) for individual players and team rosters.
3. Modeling contextual effects such as toss decisions (bat vs. field), day/night lighting, ground statistics, and head-to-head match history.
4. Training regularized ensemble models (**CatBoostClassifier**, LightGBM, XGBoost) to predict match winners accurately while avoiding overfitting.

---

## 🛠️ Key Features & Feature Engineering Pipeline

The core strength of this repository lies in its feature engineering pipeline:

### 1. 🧢 Player-Level Form & Historical Metrics (Rolling Last $N$ Games)
* **Batting Aggregates**:
  * `nostrikerateLastn`: Average strike rate of team roster over the last $N$ matches.
  * `nosixfoursLastn`: Ratio of boundaries (4s & 6s) hit by roster players.
* **Bowling Aggregates**:
  * `nowicketsLastn`: Average wickets taken by team bowlers over the last $N$ matches.
  * `noeconomyLastn`: Average economy rate of bowlers over recent matches.
  * `nomaidenLastn`: Frequency of maiden overs bowled by roster bowlers.
* **Discipline & Control**:
  * `nodotdiffLastn`: Differential in dot balls delivered between competing teams.
  * `nowidesnoballsdiffLastn`: Differential in extra runs (wides and no-balls) conceded.
* **Star Factor**:
  * `player_of_match_advantage`: Comparative count of Player of the Match awards in current team rosters.

### 2. 🏟️ Ground & Environmental Context
* `winpLastnground`: Team win percentage at a specific stadium/ground over recent matches.
* `winpLastnlighting`: Performance metrics under day, night, or day/night lighting conditions.
* `winfortosswinnerlastn`: Historical win frequency of toss winners at a specific venue.
* `win_when_bat_or_field_lastn`: Ground-specific win rate based on toss decision (batting vs. fielding first).
* `avgWicketsGround` / `avgBallsGround`: Historical ground pitch aggressiveness and scoring profile.

### 3. ⚔️ Head-to-Head & Team Dynamics
* `team1_winp_team2_last15`: Head-to-head win percentage between the two competing teams in their last 15 meetings.
* `team_winp_last5`: Recent win momentum across all grounds.
* `team1only_avg_runs_last15`: Average runs scored by Team 1 in recent fixtures.

---

## 📊 Dataset Architecture

The project utilizes five structured CSV datasets containing historical fixture data and granular scorecards:

| Dataset File | Description | Key Columns |
| :--- | :--- | :--- |
| `663e2b6d54457_train_data_with_samplefeatures.csv` | Primary training dataset (948 matches) | `match id`, `team1`, `team2`, `winner`, `toss decision`, `venue`, `ground_id`, baseline features |
| `6644a1e287df6_test_data_with_samplefeatures.csv` | Primary test dataset (271 matches) | Same as train dataset (excluding target variable) |
| `664389efa0868_match_level_scorecard.csv` | Match-level metadata & statistics | `inning1_runs`, `inning2_runs`, `inning1_wickets`, `lighting`, `series_name` |
| `663e2b548c98c_batsman_level_scorecard.csv` | Granular batsman performance logs | `batsman_id`, `runs`, `balls_faced`, `strike_rate`, `Fours`, `Sixes`, `match_dt` |
| `663e2b2c60743_bowler_level_scorecard.csv` | Granular bowler performance logs | `bowler_id`, `runs`, `wicket_count`, `economy`, `maiden_overs`, `match_dt` |

---

## 🤖 Modeling & Performance

Multiple machine learning algorithms were benchmarked during the competition. The final production pipeline employs a heavily regularized **CatBoostClassifier** to manage categorical card features and high-dimensional engineered ratios:

### Model Configuration (CatBoost)
* **Estimators (`n_estimators`)**: 300
* **Max Depth (`max_depth`)**: 3
* **Learning Rate (`learning_rate`)**: 0.01
* **L2 Regularization (`reg_lambda`)**: 1.0
* **Early Stopping**: 10 rounds

### Benchmark Results
* **Training Accuracy**: `65.96%`
* **Test Accuracy**: `70.00%`
* **Precision / Recall**: Balanced macro & weighted F1-scores (~0.70 across classes)

---

## 📁 Repository Structure

```
Cricket-Analytics/
├── amexCodefinal_best_real.ipynb  # Primary notebook containing production pipeline & final CatBoost model
├── amex (Experimental).ipynb      # Research notebook with EDA, feature extraction experiments & model benchmarks
├── Round 1 Results.png            # Leaderboard performance proof / visual metric result
├── README.md                      # Project documentation
└── *.csv                          # Train/Test datasets and player/match scorecards (git untracked by default)
```

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.8+ installed along with Jupyter Notebook or JupyterLab.

```bash
pip install pandas numpy scikit-learn catboost lightgbm xgboost matplotlib seaborn ipykernel
```

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nikhil-264/Cricket-Analytics.git
   cd Cricket-Analytics
   ```

2. **Run the Notebooks**:
   Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
   Open `amexCodefinal_best_real.ipynb` and execute the cells sequentially to reproduce feature engineering, model training, and evaluation.

---

## 📈 Future Scope & Improvements
- **Elo Rating Integration**: Incorporating dynamic team and individual player Elo ratings updated per match.
- **Weather & Pitch Analytics**: Utilizing external weather data (humidity, temperature) for dew factor predictions.
- **Deep Learning / Graph Neural Networks**: Modeling team roster interactions using GNNs based on player synergy.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

**Author**: Nikhil ([@Nikhil-264](https://github.com/Nikhil-264))
