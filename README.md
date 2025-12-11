# Premier League Match Outcome Prediction

This project predicts the outcome of Premier League matches (home win / draw / away win) using historical match data from **football-data.co.uk**.  

We:

- Build simple but meaningful **features of team form** (rolling averages of points and goal difference).
- Train several ML models (Random Forest, KNN, Logistic Regression, Gradient Boosting).
- Compare them to a naïve **baseline** that always predicts “home win”.
- Evaluate them on the **2024/2025 season**.
- Use the best model to:
  - Rebuild the league table for 2024/2025 (actual vs predicted).
  - Predict and rank the **first 14 games of the 2025/2026 season**.

All core logic is in `src/`, and the main entry point is `main.py`.

---

## 1. Project structure

```text
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
│       ├─ PL_2024_2025_data.csv   # test season (380 matches)
│       └─ epl-2025-GMTStandardTime.csv  # first 14 games of 2025/26
│
├─ src/
│   ├─ data_loader.py      # loading data & feature engineering
│   ├─ models.py           # model definitions & training helpers
│   └─ evaluation.py       # evaluation metrics, league tables & plots
│
├─ results/                # created automatically (tables, plots)
├─ notebook/
│   └─ premier_league_analysis.ipynb   # optional analysis notebook
│
├─ environment.yml         # conda environment
└─ main.py                 # main script the grader will run

 ## 2. Environment setup

You need conda (Anaconda or Miniconda).
From the project root:

# 1. Create the environment
conda env create -f environment.yml

# 2. Activate it
conda activate premier-league-project

The environment includes:

-Python 3.x
-numpy, pandas
-scikit-learn
-matplotlib

## 3. Data

Required files:

-PL_2018_2019_data.csv
-PL_2019_2020_data.csv
-PL_2020_2021_data.csv
-PL_2021_2022_data.csv
-PL_2022_2023_data.csv
-PL_2023_2024_data.csv
-PL_2024_2025_data.csv (full 2024/25 season)
-epl-2025-GMTStandardTime.csv (first 14 rounds of 2025/26)

All historical season files come from **football-data.co.uk**
The 2025/26 file is a custom CSV with at least: Date, Home Team, Away Team, Result.

##4. How to run the project

From the project root, with the conda environment activated:
python main.py


The script will:

Load and preprocess data
Build rolling features of team form (average points and goal difference over the last matches).

Split data into:
Train: 2018/2019–2023/2024
Test: full 2024/2025 season.
Train models

-Baseline: always predict “home win”.
-Random Forest
-K-Nearest Neighbors (KNN)
-Logistic Regression
-Gradient Boosting

Evaluate on the 2024/2025 season
Print accuracy, classification report and confusion matrix for each model.
Select the best model (highest accuracy on 2024/25).

Save a bar plot of model accuracies to:

-results/model_accuracies_2024_2025.png


Build league tables for 2024/2025
Construct the actual table from true results.
Construct the predicted table from best model predictions.
Save both tables to:

-results/table_actual_2024_2025.csv
-results/table_predicted_2024_2025.csv


Predict first 14 games of 2025/2026:

Recalculate team form including all previous seasons.
Predict the outcome of the first 14 rounds with the best model.

Build and save two league tables (actual vs predicted over these 14 games):

-results/table_actual_2025_2026_first14.csv
-results/table_predicted_2025_2026_first14.csv


All key intermediate outputs and tables are printed in the terminal and saved under results/.


5. Limitations & possible extensions 

-Current version:

Uses only simple form-based features (rolling averages of team performance).
Works at match-level (no player data, no betting odds, no xG).
Draws remain harder to predict than wins, which is expected in football.

-Possible extensions:

Add richer features (Elo ratings, rest days, budget, betting odds).
Try other models (XGBoost, probabilistic Poisson / Dixon–Coles models).
Simulate full seasons many times to estimate probabilities of title qualification / relegation.
Add unit tests for data loading and feature construction.


