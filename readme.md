## My submission to [kaggle housing competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview)
Final score was 0.11717, which is a top 3% submission.
See [a summary](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/discussion/170472) I posted after the competition

### Usage
Datasets are original datasets
Output are outputs, mainly train/test dataframes and models

The numbered notebooks are the core steps in the pipeline. 1,2 and 4 are essential. 
1. EDA
2. Feature Engineering, coming up with additional features
3. Feature Selection, trimming the amount of features down somewhat with Boruta algo
4. Model Selection, trying out different models, see what works best
5. Model Interpretation, with the awesome shap package. One workbook for a catboost model, one for a lasso model. Neither is superior
9. Tried to apply fastai deeplearning to the dataset, didn't give a good result

### Thanks
Thanks to Kaggle for making such a great platform for competitive coding

### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
You can use this under the MIT license