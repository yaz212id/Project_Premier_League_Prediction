# Premier League Match Outcome Prediction

Predict the result of Premier League matches (home win / draw / away win) using a simple machine-learning pipeline.

The project was built for the **Advanced Programming / Data Science** course and follows the recommended structure:
- clear prediction task  
- multiple models (baseline, linear, non-linear)  
- reproducible environment (`environment.yml`)  
- clean separation between code, data and results.

---

## 1. Project Overview

**Research question**

> Given basic pre–match statistics (recent form and goal difference for both teams), can we predict the outcome of a Premier League game better than a naive baseline?

**Target**

- 3-class classification:  
  - `0` = away win  
  - `1` = draw  
  - `2` = home win  

**Features (per match, simplified)**

- `home_form`, `away_form` – average points over the last *N* games  
- `home_gd_form`, `away_gd_form` – average goal difference over the last *N* games  
- (plus standard columns like date, teams, full-time result, etc. in the raw CSVs)

**Models compared**

1. **Baseline** – always predict “home win” (very naive, used as reference).
2. **Logistic Regression** – linear model for multi-class classification.
3. **Random Forest** – non-linear tree-based model to capture more complex patterns.

The goal is to see whether the non-linear model (Random Forest) improves performance compared to Logistic Regression and to the naive baseline.

**Data**

- Historical Premier League results (several seasons, e.g. 2018–2024).  
- Source: standard football results datasets (e.g. Football-Data.co.uk / similar CSVs).  
- CSV files are stored in: `data/raw/`.

**Train / test setup**

- Multiple past seasons are used for **training**.
- The most recent season is kept as a **test set** (around 380 matches).
- This mimics a realistic “train on past, predict the future season” scenario.

---

## 2. Installation Instructions

These steps assume you have **conda** installed (Anaconda / Miniconda).

## 3.Run the main script

conda activate premier-league-env
python main.py






