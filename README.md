# Epigenetic Age Prediction

This is my Assignment 1 for the Machine Learning in Computational Biology course 2025-2026.

## What is this project about?
We used DNA methylation data to predict how old someone is. DNA methylation changes as we age so we can use it as a biological clock. The dataset comes from real blood samples of people aged 19 to 101.

## What did I build?
A full machine learning pipeline that cleans and preprocesses the data, tests different models to predict age from DNA data, selects the most useful features (CpG sites), tunes the models to get the best performance, and as a bonus also predicts sex from the same data.

## Best result
My best model (ElasticNet) predicted age with an average error of about 4.9 years on unseen data.

## Files
- src/functions.py — all the functions I wrote
- notebooks/data_exploration.ipynb — data loading and exploration
- notebooks/model_analysis.ipynb — model training and evaluation
- models/best_model.pkl — the saved best model

## How to run
Put the data files in the data folder and run the notebooks in order.

## Libraries used
numpy, pandas, scikit-learn, scipy, mrmr-selection, optuna, matplotlib
