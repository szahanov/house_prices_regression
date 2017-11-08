
# Predicting House Prices with Gradient Boosting #


## Links ##

* [R code](https://github.com/szahanov/house_prices_regression/blob/master/house_prices_regression.R)

* [Training Data](https://github.com/szahanov/house_prices_regression/blob/master/train.csv)

* [Test Data](https://github.com/szahanov/house_prices_regression/blob/master/test.csv)

* [Charts](https://github.com/szahanov/house_prices_regression/blob/master/charts/)

* [Original data source from kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Goal ##

For this exercise we use the xgboost package, also known as extreme gradient boosting, to predict house sale prices using the data we are given.


## Data Visualization ##

#### Feature Selection ####

One way that we selected our variables is by generating an importance matrix for our model:

![Importance Matrix](https://raw.githubusercontent.com/szahanov/house_prices_regression/master/charts/importance_matrix.png "Importance Matrix")

OverallQual seemed to be an overwhelmingly useful variable so here is another vesion excluding it:

![Importance Matrix excluding OveralQual](https://raw.githubusercontent.com/szahanov/house_prices_regression/master/charts/importance_matrix_without_overallqual.png "Importance Matrix excluding OveralQual")

We exclude variables from the lower end of the importance vector until the metrics we use to test our model stop improving

The important matrix will change depending on what variables are included in the model, so this approach could be refined further


#### Judgement ####

Some variables are made up of others. These will correlate heavily and it might not make sense to include both.

* TotalBsmtSF (total basement square footage) is composed of BsmtFinSF1, BsmtFinSF2, and BsmtUnfSF
* GrLivArea (above ground living area) is composed of 1stFlrSF, 2ndFlrSF,and LowQualFinSF
* TotRmsAbvGrd (total rooms above ground) contains BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, and KitchenAbvGr
* GarageCars and GarageArea will be correlated; one is the size of the garage in car capacity and the other is in square feet



## Results ##

Using xgboost we create a model that closely describes our training data, here is the comparison of the model vs the training data:

Here our model closely matches the Sale Price vector:

![Predictions vs Sale Prices](https://raw.githubusercontent.com/szahanov/house_prices_regression/master/charts/xgbplot1.png "Predictions vs Sale Prices")

Here we have predictions indexed to the actual results ((Prediction - Sale Price)/(Sale Price)), the differences look random and averaged around 0. The variance of the error is slightly skewed but it's not noticeable just by looking at the plot:

![(Prediction - Sale Price)/(Sale Price)](https://raw.githubusercontent.com/szahanov/house_prices_regression/master/charts/xgbplot3.png "(Prediction - Sale Price)/(Sale Price)")

Performance Metrics

>   Mean squared error  
>       775421  
>   Square root of MSE  
>       880.58

Given that the average house price is $180,000, this is a good mean squared error

>   R squared  
>       0.9998803  

These are good results for our model. Naturally there is a chance of overfitting and this model likely does overfit a bit, with more analysis we could remove more variables and adjust our model further. In real life I would not buy a house using raw machine learning results, results would need to be smoothed to account for outliers. For large and uncommon purchases it could make more sense to use simpler averages or a model that's more transparent like a GLM, but as the number of decisions required increases the case for machine learning gets stronger.

Great success


