# Premier League Match Outcome Prediction

This project predicts the outcome of Premier League matches  
(home win / draw / away win) using historical data and simple
form-based features.

It was developed as part of a programming / research project:
the goal is not only to get good predictions, but also to build a
clean, reproducible pipeline and to analyse its limits
(e.g. difficulty of predicting draws).

---

## 1. Research Goals

We focus on two main questions:

1. **How well can simple, interpretable features (team form, goal difference, etc.) predict match outcomes compared to a naïve “always home win” baseline?**

2. **Can these models generate realistic league tables when applied to:**
   - the **full 2024/2025 Premier League season**, and  
   - the **first 14 matchdays of the 2025/2026 season**?

We also look specifically at how the models handle **draws**, which are
less frequent and harder to predict.

---

## 2. Data

All raw CSVs are stored in `data/raw/`.

- Historical seasons from **2018/2019 to 2024/2025**, downloaded from  
  [football-data.co.uk](https://www.football-data.co.uk/):
  - `PL_2018_2019_data.csv`
  - `PL_2019_2020_data.csv`
  - `PL_2020_2021_data.csv`
  - `PL_2021_2022_data.csv`
  - `PL_2022_2023_data.csv`
  - `PL_2023_2024_data.csv`
  - `PL_2024_2025_data.csv` (used as held-out **test season**)

- A special file for the **2025/2026 season**:
  - `epl-2025-GMTStandardTime.csv`  
    (contains at least: Match Number, Round Number, Date, Location,
    Home Team, Away Team, Result.  
    We only need the teams + date to build features, and we use the
    actual result to evaluate the first 14 games.)

### 2.1 Feature Construction

From each match we keep:

- Date  
- Home team name  
- Away team name  
- Full-time result (H/D/A) for historical seasons only

Then we build **rolling “form” features** for each team based on the last
5 games:

- `home_form` / `away_form`  
  → average points over last 5 matches (3 for win, 1 for draw, 0 for loss)

- `home_gd_form` / `away_gd_form`  
  → average goal difference over last 5 matches

Optionally, the feature set can be extended with:

- `rel_form = home_form - away_form`  
- `rel_gd_form = home_gd_form - away_gd_form`  
- Normalised matchday index (early vs late season)

The **target label** is encoded as:

- `0` = away win  
- `1` = draw  
- `2` = home win  

---

## 3. Project Structure

```text
Project_Premier_League_Prediction/
│
├─ data/
│   └─ raw/
│       ├─ PL_2018_2019_data.csv
│       ├─ ...
│       ├─ PL_2024_2025_data.csv
│       └─ epl-2025-GMTStandardTime.csv
│
├─ results/              # Generated automatically by main.py
│   ├─ model_accuracies_2024_2025.png
│   ├─ table_actual_2024_2025.csv
│   ├─ table_predicted_2024_2025.csv
│   ├─ table_actual_2025_2026_first14.csv
│   └─ table_predicted_2025_2026_first14.csv
│
├─ src/
│   ├─ __init__.py
│   ├─ data_loader.py     # data loading + feature creation
│   ├─ models.py          # model definitions and training helpers
│   └─ evaluation.py      # metrics, league table building, plots
│
├─ tests/                 # (optional) small unit tests
│   ├─ test_data_loader.py
│   └─ test_league_table.py
│
├─ main.py                # main entry point (python main.py)
├─ environment.yml        # conda environment
└─ README.md
