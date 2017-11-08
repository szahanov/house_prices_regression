
#### Table of Contents
#### Section 1 - Load the data, clean the data
#### Section 2 - Create a preliminary model
#### Section 3 - Analyse the results
#### Section 4 - Variable Selection
#### Section 5 - Apply the model on the testing data



#### Section 1 - Load the data, clean the data


# Set the working directory
setwd("C:/R/House Prices Regression Techniques")


## Install and load packages
# install.packages('e1071', dependencies=TRUE) # If the package has not already been installed
# install.packages('caret', dependencies=TRUE) # If the package has not already been installed
# install.packages('xgboost') # Make sure you are using a recent enough version of R for this to work
library(e1071)
library(caret)
library(xgboost)


# Import the dataset, fields are already named
# dataset: https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
dataset <- read.csv("train.csv")
dataset = dataset[1:1000,]
train = dataset[1:1000,]
test = dataset[1001:1460,]


# Take a look at the data
summary(train)


# Need to convert some columns to numeric;
training_clean <- transform(train,SalePrice = as.numeric(SalePrice),MSSubClass = as.numeric(MSSubClass),LotArea = as.numeric(LotArea),OverallQual = as.numeric(OverallQual),OverallCond = as.numeric(OverallCond),YearBuilt = as.numeric(YearBuilt),YearRemodAdd = as.numeric(YearRemodAdd),BsmtFinSF1 = as.numeric(BsmtFinSF1),BsmtUnfSF = as.numeric(BsmtUnfSF),TotalBsmtSF = as.numeric(TotalBsmtSF),X1stFlrSF = as.numeric(X1stFlrSF),X2ndFlrSF = as.numeric(X2ndFlrSF),BsmtFullBath = as.numeric(BsmtFullBath),BsmtHalfBath = as.numeric(BsmtHalfBath),FullBath = as.numeric(FullBath),HalfBath = as.numeric(HalfBath),BedroomAbvGr = as.numeric(BedroomAbvGr),KitchenAbvGr = as.numeric(KitchenAbvGr),GarageCars = as.numeric(GarageCars),WoodDeckSF = as.numeric(WoodDeckSF),OpenPorchSF = as.numeric(OpenPorchSF),EnclosedPorch = as.numeric(EnclosedPorch),X3SsnPorch = as.numeric(X3SsnPorch),ScreenPorch = as.numeric(ScreenPorch),PoolArea = as.numeric(PoolArea),YrSold = as.numeric(YrSold))


# For now we are starting with all variables. We will analyse the results and then remove variables to simplify and improve the model.
dataset_clean <- training_clean


#### Section 2 - Create a preliminary model


# Gradient Boosting Model:
output_vector <- dataset_clean[,81] # the 81st column is SalePrice
xgb <- xgboost(data = data.matrix(dataset_clean[,2:80]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100) # nrounds: the number of decision trees in the final model
xgb_predictions <- predict(xgb, data.matrix(dataset_clean[,2:80])) # predict values in training set


png(filename="xgbplot1.png")
plot(xgb_predictions,dataset_clean$SalePrice)
dev.off()
# plot looks good



#### Section 3 - Analyse the results


# Mean Square Error
mse_test_value <- mean((xgb_predictions - dataset_clean$SalePrice)^2)
# sqrt(mse_test_value) = 680 where the average SalePrice is $180,000, not bad


# R squared
r2 <- cor(dataset_clean$SalePrice,xgb_predictions)^2
# r2 = 0.9999289, looks good if we're not overfitting


# check average
mean_diff <- mean(xgb_predictions - dataset_clean$SalePrice)
# mean_diff = -0.5406211 where the average SalePrice is $180,000, no immediately obvious bias


# some spot checks
i=5;(xgb_predictions[i] - dataset_clean$SalePrice[i])/dataset_clean$SalePrice[i]
i=20;(xgb_predictions[i] - dataset_clean$SalePrice[i])/dataset_clean$SalePrice[i]
i=35;(xgb_predictions[i] - dataset_clean$SalePrice[i])/dataset_clean$SalePrice[i]
# spot checked predictions are all within 1% of the actual sale price


# reasonableness plots
png(filename="xgbplot2.png")
plot(xgb_predictions/max(dataset_clean$SalePrice),dataset_clean$SalePrice/max(dataset_clean$SalePrice))
dev.off()
# looks to be in order
png(filename="xgbplot3.png")
plot((xgb_predictions - dataset_clean$SalePrice)/dataset_clean$SalePrice)
dev.off()


# When testing the model it looked like the sale price predictions tended to be lower than the real values
# There were more values that were way above the correct value than way below
# Here we can still sort of see this pattern, the outliers seem to be too high more often than too low, while the rest of the predictions seem to be biased a little bit below the true sale price

# In other words this model tends to predict sale prices a little bit lower than they should be, however every now and then it will overestimate the value of a house by about 5%.
# When the model is wrong it tends to way overvalue a house rather than way undervalue it, which can be unsafe if you're buying. When the model is working well it very sligtly underestimes the value of a house, which is conservative and safe if you're buying.


#### Section 4 - Variable Selection
# Look at variable importance using the feature importance matrix and judgement


# Compute a feature importance matrix
names <- dimnames(data.matrix(dataset_clean[,2:80]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb)
png(filename="importance_matrix.png")
xgb.plot.importance(importance_matrix[1:30,])
dev.off()


# Variable OverallQual seemed to be the most useful by far so let's see a version of the importance matrix without it
xgb_xOverallQual <- xgboost(data = data.matrix(dataset_clean[,c(2:17,19:80)]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
xgb_xOverallQual_predictions <- predict(xgb_xOverallQual, data.matrix(dataset_clean[,c(2:17,19:80)]))
names_xOverallQual <- dimnames(data.matrix(dataset_clean[,c(2:17,19:80)]))[[2]]
importance_matrix_xOverallQual <- xgb.importance(names_xOverallQual, model = xgb_xOverallQual)
png(filename="importance_matrix_without_overallqual.png")
xgb.plot.importance(importance_matrix_xOverallQual[1:20,])
dev.off()
mse_test_value2 <- mean((xgb_xOverallQual_predictions - dataset_clean$SalePrice)^2)


# These are the variables in order of importance:
# "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "TotalBsmtSF", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "GarageArea", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "LandContour", "BsmtFinSF2", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "RoofStyle", "Condition2", "SaleType", "GarageQual", "Fence", "MasVnrType", "HouseStyle", "RoofMatl", "Exterior2nd", "Alley", "BsmtCond", "HeatingQC", "HalfBath", "BldgType", "Electrical", "BsmtFinType2", "MiscVal", "BsmtHalfBath", "GarageCond", "ExterCond", "X3SsnPorch", "MiscFeature", "Foundation", "LandSlope", "LowQualFinSF", "", "Utilities", "KitchenAbvGr", "PoolArea"


## Notes on selecting our variables and what we're excluding:
# 1) generated importance matrices and removed the least important variables until the MSE stopped improving
# 2) removed variables that intuitively made sense and that simplified data collection, for example we don't need both the total basement square footage and the various squar footages of different parts of the basement
#   removed TotalBsmtSF because we have BsmtFinSF1 and BsmtFinSF2, although we don't have BsmtUnfSF; didn't remove GrLivArea even though we have 1stFlrSF and 2ndFlrSF; removed GarageArea because we have GarageCars

## Otherwise these are some variables we look out for for correlation:
# 	TotalBsmtSF with BsmtFinSF1, BsmtFinSF2, and BsmtUnfSF
# 	GrLivArea with 1stFlrSF, 2ndFlrSF,and LowQualFinSF
# 	TotRmsAbvGrd with BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, and KitchenAbvGr
# 	GarageCars and GarageArea
#   The important matrix will change depending on what variables are included in the model, this model isn't intended to be perfect


# Final Variable Selection
dataset_clean <- training_clean[,c("SalePrice", "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "BsmtFinSF2", "LandContour", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "Condition2", "RoofStyle", "SaleType", "GarageQual", "HouseStyle", "Fence", "MasVnrType", "RoofMatl", "Exterior2nd", "Alley")]
colnames(dataset_clean) <- c("SalePrice", "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "BsmtFinSF2", "LandContour", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "Condition2", "RoofStyle", "SaleType", "GarageQual", "HouseStyle", "Fence", "MasVnrType", "RoofMatl", "Exterior2nd", "Alley")


# Reselected Gradient Boosting Model:
output_vector <- dataset_clean[,1]
xgb <- xgboost(data = data.matrix(dataset_clean[,-1]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
xgb_predictions <- predict(xgb, data.matrix(dataset_clean[,-1]))
mse_reselected <- mean((xgb_predictions - dataset_clean$SalePrice)^2)
# > mse_reselected
# [1] 515102


#### Section 5 - Run the model with the testing data


# Load the test data
# dataset_test <- read.csv("test.csv")
test <- read.csv("train.csv")
test = test[1001:1460,] # using this so we can see the SalePrice, test.csv doesn't have it


# Clean the test data
test_clean <- transform(test,SalePrice = as.numeric(SalePrice),MSSubClass = as.numeric(MSSubClass),LotArea = as.numeric(LotArea),OverallQual = as.numeric(OverallQual),OverallCond = as.numeric(OverallCond),YearBuilt = as.numeric(YearBuilt),YearRemodAdd = as.numeric(YearRemodAdd),BsmtFinSF1 = as.numeric(BsmtFinSF1),BsmtUnfSF = as.numeric(BsmtUnfSF),TotalBsmtSF = as.numeric(TotalBsmtSF),X1stFlrSF = as.numeric(X1stFlrSF),X2ndFlrSF = as.numeric(X2ndFlrSF),BsmtFullBath = as.numeric(BsmtFullBath),BsmtHalfBath = as.numeric(BsmtHalfBath),FullBath = as.numeric(FullBath),HalfBath = as.numeric(HalfBath),BedroomAbvGr = as.numeric(BedroomAbvGr),KitchenAbvGr = as.numeric(KitchenAbvGr),GarageCars = as.numeric(GarageCars),WoodDeckSF = as.numeric(WoodDeckSF),OpenPorchSF = as.numeric(OpenPorchSF),EnclosedPorch = as.numeric(EnclosedPorch),X3SsnPorch = as.numeric(X3SsnPorch),ScreenPorch = as.numeric(ScreenPorch),PoolArea = as.numeric(PoolArea),YrSold = as.numeric(YrSold))


# these are the variables we selected before:
test_clean <- test_clean[,c("SalePrice", "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "BsmtFinSF2", "LandContour", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "Condition2", "RoofStyle", "SaleType", "GarageQual", "HouseStyle", "Fence", "MasVnrType", "RoofMatl", "Exterior2nd", "Alley")]
colnames(test_clean) <- c("SalePrice", "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "BsmtFinSF2", "LandContour", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "Condition2", "RoofStyle", "SaleType", "GarageQual", "HouseStyle", "Fence", "MasVnrType", "RoofMatl", "Exterior2nd", "Alley")


# Using our earlier model with our cleaned test data
xgb_test_predictions <- predict(xgb, data.matrix(test_clean[,-1]))






