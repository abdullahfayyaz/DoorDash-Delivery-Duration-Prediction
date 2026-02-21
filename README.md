# DoorDash Delivery Duration Prediction

## Overview
When a consumer places an order on DoorDash, showing an accurate expected time of delivery is critical for user experience. This project builds a machine learning pipeline to predict the total delivery duration (in seconds) from the moment an order is submitted to the moment it is delivered. 

## Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost, LightGBM, Keras (Deep Learning)
* **Statistical Analysis:** Statsmodels (Variance Inflation Factor)
* **Visualization:** Matplotlib, Seaborn

## Methodology
1. **Feature Engineering:** Extracted temporal features and engineered new metrics such as `percent_distinct_item_of_total`, `avg_price_per_item`, and `busy_dashers_ratio`.
2. **Multicollinearity Elimination:** Applied Variance Inflation Factor (VIF) analysis to aggressively filter redundant features, reducing the feature space to the top 40 most predictive variables.
3. **Decomposed Target Strategy:** Instead of predicting the total end-to-end time directly, the model isolates and predicts **food preparation time** (the most variable factor), and algebraically combines it with estimated driving and order-placement times.
4. **Stacked Regression:** Benchmarked 6 algorithms (including Random Forest, MLP Neural Networks, and Ridge Regression). Built a two-stage stacked regression pipeline using XGBoost/LightGBM for the prep-time inference, smoothed with a Linear Regression meta-learner.

## Key Results
* **51% Error Reduction:** The decomposed stacked approach reduced the overall model RMSE from 2035 seconds to 986 seconds.
* **Algorithm Performance:** The stacked Linear Regression approach outperformed complex standalone architectures (like XGBoost and Deep Learning) by up to 28% on the final inference layer.

## How to Run
1. Clone the repository.
2. Ensure `historical_data.csv` is in the root directory.
3. Run the Jupyter Notebook to view the EDA, VIF analysis, and model training steps.
