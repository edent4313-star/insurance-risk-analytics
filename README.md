# insurance-risk-analytics
# Insurance Risk Analytics

## Project Overview

This project analyzes historical insurance claims data
to identify low-risk customer segments and support
risk-based pricing strategies.

## Objectives

- Perform exploratory data analysis
- Evaluate profitability and loss ratios
- Detect risk patterns across provinces and vehicles
- Build predictive insurance models

## Technologies

- Python
- Pandas
- Seaborn
- Scikit-learn
- DVC
- GitHub Actions

## Project Structure

- notebooks/
- src/
- tests/
- reports/

  ## Reproducing the Project

### Clone Repository
git clone <repo-url>

### Create Environment
python -m venv .venv

### Install Dependencies
pip install -r requirements.txt

### Pull DVC Data
dvc pull

### Run Pipeline
dvc repro

### Run Tests
pytest
pytest
