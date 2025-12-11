# Premier League Match Outcome Prediction

This project predicts the outcome of Premier League matches (home win / draw / away win) using historical match data from **football-data.co.uk**.

We:

* Build simple but meaningful **features of team form** (rolling averages of points and goal difference).
* Train several ML models (Random Forest, KNN, Logistic Regression, Gradient Boosting).
* Compare them to a naïve **baseline** that always predicts “home win”.
* Evaluate them on the **2024/2025 season**.
* Use the best model to:

  * Rebuild the league table for 2024/2025 (actual vs predicted).
  * Predict and rank the **first 14 games of the 2025/2026 season**.

All core logic is in `src/`, and the main entry point is `main.py`.

---

## 1. Project Structure

```
Project_Premier_League_Prediction/
│
├─ data/
│   └─ raw/
│       ├─ PL_2018_2019_data.csv
│       ├─ PL_2019_2020_data.csv
│       ├─ PL_2020_2021_data.csv
│       ├─ PL_2021_2022_data.csv
│       ├─ PL_2022_2023_data.csv
│       ├─ PL_2023_2024_data.csv
│       ├─ PL_2024_2025_data.csv          # test season (380 matches)
│       └─ epl-2025-GMTStandardTime.csv   # first 14 games of 2025/26
│
├─ src/
│   ├─ data_loader.py      # loading data & feature engineering
│   ├─ models.py           # model definitions & training helpers
│   └─ evaluation.py       # evaluation metrics, league tables & plots
│
├─ results/                 # created automatically (tables, plots)
│
├─ notebook/
│   └─ premier_league_analysis.ipynb   # optional exploratory notebook
│
├─ environment.yml          # conda environment definition
│
└─ main.py                  # main script the grader will run
```

---

## 2. Environment Setup

You need Conda (Anaconda or Miniconda).

From the project root:

```
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate it
conda activate premier-league-project
```

The environment includes:

* Python 3.x
* numpy, pandas
* scikit-learn
* matplotlib

---

## 3. Data

Required files:

* `PL_2018_2019_data.csv`
* `PL_2019_2020_data.csv`
* `PL_2020_2021_data.csv`
* `PL_2021_2022_data.csv`
* `PL_2022_2023_data.csv`
* `PL_2023_2024_data.csv`
* `PL_2024_2025_data.csv` (full 2024/25 season)
* `epl-2025-GMTStandardTime.csv` (first 14 rounds of 2025/26)

All historical season files come from **football-data.co.uk**.
The 2025/26 file is a custom CSV that includes at least: Date, Home Team, Away Team, Result.

---

## 4. How to Run the Project

From the project root, with the conda environment activated:

```
python main.py
```

The script will:

### 1. Load and preprocess data

* Build rolling features of team form (average points and goal difference over the last matches).

### 2. Split the dataset

* Train: 2018/2019 → 2023/2024
* Test: 2024/2025 full season

### 3. Train models

* Baseline (always predict “home win”)
* Random Forest
* K-Nearest Neighbors (KNN)
* Logistic Regression
* Gradient Boosting

### 4. Evaluate them on the 2024/2025 season

* Print accuracy, classification report, and confusion matrix
* Save a bar plot of model accuracies to:

  `results/model_accuracies_2024_2025.png`

### 5. Build league tables for 2024/2025

* Actual table from true results
* Predicted table from best model

Saved to:

* `results/table_actual_2024_2025.csv`
* `results/table_predicted_2024_2025.csv`

### 6. Predict the first 14 games of 2025/2026

* Recompute team form with all historical data
* Predict the first 14 rounds with the best model
* Build and save actual vs predicted league tables:

  * `results/table_actual_2025_2026_first14.csv`
  * `results/table_predicted_2025_2026_first14.csv`

All key outputs and tables are printed in the terminal and saved in `results/`.

---

## 5. Limitations & Possible Extensions

### Current limitations

* Uses only form-based features (rolling team performance).
* No player-level data, no betting odds, no xG.
* Draws are harder to predict than wins — which is expected in football.

### Possible extensions

* Add richer features (Elo ratings, rest days, budgets, odds).
* Try different algorithms (XGBoost, Poisson/Dixon–Coles models).
* Simulate entire seasons to estimate title / relegation probabilities.
* Add unit tests for data loading and feature engineering.
