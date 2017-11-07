
#### Section 1 - Load the data, clean the data


# Set the working directory
setwd("C:/R/House Prices Regression Techniques")

# if using linux and some of the below libraries are having trouble installing:
# sudo apt-get install g++
# sudo apt-get install libnlopt-dev
# install.packages()

# Install and load packages
# install.packages('e1071', dependencies=TRUE) # If the package has not already been installed
library(e1071)	# load package
# install.packages('caret', dependencies=TRUE) # If the package has not already been installed
library(caret)	# load package
# install.packages("xgboost") # Make sure you are using a recent enough version of R for this to work
library(xgboost)


# Import the dataset, fields are already named
# dataset: https://archive.ics.uci.edu/ml/datasets/Image+Segmentation
dataset<- read.csv("train.csv")
dataset = dataset[1:1000,]
train = dataset[1:1000,]
test = dataset[1001:1460,]


# Take a look at the data
summary(train)


# Some notes on the variables:
# 	BsmtFinSF1, BsmtUnfSF, TotalBsmtSF: Unfinished Basement Square Footage was shown to correlate well with sale price but Total Basement Square Footage makes more sense to me so we will use the total, rather than splitting between finished and unfinished square footage
# 	GrLivArea: We are not using living area because we are using the other square footages
# 	BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd: We are not using total rooms above ground because we are using the rooms separately
# 	GarageCars, GarageArea: We are not using garage area because we are using garage cars

# Need to convert some columns to numeric;
training_clean <- transform(train,SalePrice = as.numeric(SalePrice),MSSubClass = as.numeric(MSSubClass),LotArea = as.numeric(LotArea),OverallQual = as.numeric(OverallQual),OverallCond = as.numeric(OverallCond),YearBuilt = as.numeric(YearBuilt),YearRemodAdd = as.numeric(YearRemodAdd),BsmtFinSF1 = as.numeric(BsmtFinSF1),BsmtUnfSF = as.numeric(BsmtUnfSF),TotalBsmtSF = as.numeric(TotalBsmtSF),X1stFlrSF = as.numeric(X1stFlrSF),X2ndFlrSF = as.numeric(X2ndFlrSF),BsmtFullBath = as.numeric(BsmtFullBath),BsmtHalfBath = as.numeric(BsmtHalfBath),FullBath = as.numeric(FullBath),HalfBath = as.numeric(HalfBath),BedroomAbvGr = as.numeric(BedroomAbvGr),KitchenAbvGr = as.numeric(KitchenAbvGr),GarageCars = as.numeric(GarageCars),WoodDeckSF = as.numeric(WoodDeckSF),OpenPorchSF = as.numeric(OpenPorchSF),EnclosedPorch = as.numeric(EnclosedPorch),X3SsnPorch = as.numeric(X3SsnPorch),ScreenPorch = as.numeric(ScreenPorch),PoolArea = as.numeric(PoolArea),YrSold = as.numeric(YrSold))


# These are the variables we've selected. These are selected through a combination of looking at the feature importance matrix (in section 3 below) and through judgement. Some notes on the judgemental selection can be found above.
dataset_clean <- training_clean[,c("SalePrice",	"MSSubClass",	"LotArea",	"OverallQual",	"OverallCond",
"YearBuilt",	"YearRemodAdd",	"BsmtFinSF1",	"BsmtUnfSF",	"TotalBsmtSF",	"X1stFlrSF",	"X2ndFlrSF",	"BsmtFullBath",	"BsmtHalfBath",	"FullBath",	"HalfBath",	"BedroomAbvGr",	"KitchenAbvGr",	"GarageCars",	"WoodDeckSF",	"OpenPorchSF",	"EnclosedPorch",	"X3SsnPorch",	"ScreenPorch",	"PoolArea",	"YrSold")]
colnames(dataset_clean) <- c("SalePrice",	"MSSubClass",	"LotArea",	"OverallQual",	"OverallCond",	"YearBuilt",	"YearRemodAdd",	"BsmtFinSF1",	"BsmtUnfSF",	"TotalBsmtSF",	"X1stFlrSF",	"X2ndFlrSF",	"BsmtFullBath",	"BsmtHalfBath",	"FullBath",	"HalfBath",	"BedroomAbvGr",	"KitchenAbvGr",	"GarageCars",	"WoodDeckSF",	"OpenPorchSF",	"EnclosedPorch",	"X3SsnPorch",	"ScreenPorch",	"PoolArea",	"YrSold")


#### Section 2 - Create the model


# Gradient Boosting Model:
output_vector <- dataset_clean[,1]
xgb <- xgboost(data = data.matrix(dataset_clean[,-1]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
# nrounds: the number of decision trees in the final model

# predict values in training set
xgb_predictions <- predict(xgb, data.matrix(dataset_clean[,-1]))


plot(xgb_predictions,dataset_clean$SalePrice)
# plot looks good






#### Section 4 - Analyse the results


# Mean Square Error
mse_test_value <- mean((xgb_predictions - dataset_clean$SalePrice)^2)
# sqrt(mse_test_value) = 1217.37 where the average SalePrice is $180,000, not bad

# R squared
r2 <- cor(dataset_clean$SalePrice,xgb_predictions)^2
# r2 = 0.9997722, looks good if we're not overfitting

# check average
mean_diff <- mean(xgb_predictions - dataset_clean$SalePrice)
# mean_diff = 0.8965117 where the average SalePrice is $180,000, no immediately obvious bias

# some spot checks
i=5;(xgb_predictions[i] - dataset_clean$SalePrice[i])/dataset_clean$SalePrice[i]
i=20;(xgb_predictions[i] - dataset_clean$SalePrice[i])/dataset_clean$SalePrice[i]
i=35;(xgb_predictions[i] - dataset_clean$SalePrice[i])/dataset_clean$SalePrice[i]
# spot checked predictions are all within 1% of the actual sale price

# reasonableness plots
plot(xgb_predictions/max(dataset_clean$SalePrice),dataset_clean$SalePrice/max(dataset_clean$SalePrice))
# looks to be in order
plot((xgb_predictions - dataset_clean$SalePrice)/dataset_clean$SalePrice)

# When testing the model it looked like the sale price predictions tended to be lower than the real values
# There were more values that were way above the correct value than way below
# Here we can still sort of see this pattern, the outliers seem to be too high more often than too low, while the rest of the predictions seem to be biased a little bit below the true sale price

# In other words this model tends to predict sale prices a little bit lower than they should be, however every now and then it will overestimate the value of a house by about 5%.
# When the model is wrong it tends to way overvalue a house rather than way undervalue it, which can be unsafe if you're buying. When the model is working well it very sligtly underestimes the value of a house, which is conservative and safe if you're buying.


#### Section 4 - Variable reselection
# Look at variable importance using the feature importance matrix


# Gradient Boosting Model:
X <- training_clean
output_vector <- X[,81] # SalePrice is the 81st column
xgb_original <- xgboost(data = data.matrix(X[,2:80]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100) # Column 1 is the id and is irrelevant
xgb_original_predictions <- predict(xgb_original, data.matrix(X[,2:80]))


# Compute feature importance matrix
names <- dimnames(data.matrix(X[,2:80]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb_original)
xgb.plot.importance(importance_matrix[1:30,])



# Variable OverallQual seemed to be overwhelmingly important so let's see a version without it
xgb_original <- xgboost(data = data.matrix(X[,c(2:17,19:80)]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
xgb_original_predictions <- predict(xgb_original, data.matrix(X[,c(2:17,19:80)]))
names <- dimnames(data.matrix(X[,c(2:17,19:80)]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb_original)
xgb.plot.importance(importance_matrix[1:20,])
mse_test_value2 <- mean((xgb_original_predictions - dataset_clean$SalePrice)^2)



# Gradient Boosting Model with all variables:
output_vector <- dataset_clean[,1]
xgb_allvars <- xgboost(data = data.matrix(training_clean[,2:80]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
xgb_allvars_predictions <- predict(xgb_allvars, data.matrix(training_clean[,2:80]))
mse_allvars <- mean((xgb_allvars_predictions - dataset_clean$SalePrice)^2)
names <- dimnames(data.matrix(training_clean[,2:80]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb_allvars)
xgb.plot.importance(importance_matrix[1:20,])

# Variables in order of importance:
# "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "TotalBsmtSF", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "GarageArea", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "LandContour", "BsmtFinSF2", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "RoofStyle", "Condition2", "SaleType", "GarageQual", "Fence", "MasVnrType", "HouseStyle", "RoofMatl", "Exterior2nd", "Alley", "BsmtCond", "HeatingQC", "HalfBath", "BldgType", "Electrical", "BsmtFinType2", "MiscVal", "BsmtHalfBath", "GarageCond", "ExterCond", "X3SsnPorch", "MiscFeature", "Foundation", "LandSlope", "LowQualFinSF", "", "Utilities", "KitchenAbvGr", "PoolArea"

# Saved after feature selection:
# "SalePrice",	"OverallQual", "GrLivArea", "GarageCars", "FullBath", "TotalBsmtSF", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "GarageArea", "YearRemodAdd", "LotFrontage", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional"
# > mse_reselected
# [1] 427093.6


# Second try:
# "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "TotalBsmtSF", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "GarageArea", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "BsmtFinSF2", "LandContour", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "Condition2", "RoofStyle", "SaleType", "GarageQual", "HouseStyle", "Fence", "MasVnrType", "RoofMatl", "Exterior2nd", "Alley", "BsmtCond", "HalfBath", "HeatingQC", "BldgType", "Electrical", "BsmtFinType2", "MiscVal", "BsmtHalfBath", "GarageCond", "X3SsnPorch", "ExterCond", "MiscFeature", "LowQualFinSF", "Utilities", "Foundation", "LandSlope", "KitchenAbvGr", "PoolArea", "Heating"



# Reselect our variables, taking out some of them:
# 1) generated importance matrices and removed the least important variables until the MSE stopped improving
# 2) removed variables that intuitively made sense and that simplified data collection, for example we don't need both the total basement square footage and the various squar footages of different parts of the basement
dataset_clean <- training_clean[,c("SalePrice", "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "TotalBsmtSF", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "GarageArea", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "BsmtFinSF2", "LandContour", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "Condition2", "RoofStyle", "SaleType", "GarageQual", "HouseStyle", "Fence", "MasVnrType", "RoofMatl", "Exterior2nd", "Alley")]
colnames(dataset_clean) <- c("SalePrice", "OverallQual", "GrLivArea", "GarageCars", "FullBath", "BsmtFinSF1", "TotalBsmtSF", "LotArea", "TotRmsAbvGrd", "X1stFlrSF", "GarageArea", "YearRemodAdd", "LotFrontage", "BsmtFinType1", "GarageFinish", "Fireplaces", "MoSold", "Neighborhood", "X2ndFlrSF", "YearBuilt", "SaleCondition", "BsmtUnfSF", "OverallCond", "GarageYrBlt", "CentralAir", "BsmtQual", "MasVnrArea", "WoodDeckSF", "GarageType", "MSZoning", "BedroomAbvGr", "Exterior1st", "MSSubClass", "Condition1", "KitchenQual", "BsmtExposure", "ScreenPorch", "OpenPorchSF", "Functional", "EnclosedPorch", "LotConfig", "PavedDrive", "BsmtFinSF2", "LandContour", "LotShape", "BsmtFullBath", "ExterQual", "YrSold", "FireplaceQu", "Condition2", "RoofStyle", "SaleType", "GarageQual", "HouseStyle", "Fence", "MasVnrType", "RoofMatl", "Exterior2nd", "Alley")

# Reselected Gradient Boosting Model:
output_vector <- dataset_clean[,1]
xgb <- xgboost(data = data.matrix(dataset_clean[,-1]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
xgb_predictions <- predict(xgb, data.matrix(dataset_clean[,-1]))
mse_reselected <- mean((xgb_predictions - dataset_clean$SalePrice)^2)
names <- dimnames(data.matrix(dataset_clean[,-1]))[[2]]
importance_matrix <- xgb.importance(names, model = xgb)




## Remove variables
# xgb3 <- xgboost(data = data.matrix(X[,c(2:39,41:80)]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
# xgb3_predictions <- predict(xgb3, data.matrix(X[,c(2:39,41:80)]))
# names <- dimnames(data.matrix(X[,c(2:39,41:80)]))[[2]]
# importance_matrix <- xgb.importance(names, model = xgb3)
# xgb.plot.importance(importance_matrix[1:20,])
# mse_test_value3 <- mean((xgb3_predictions - dataset_clean$SalePrice)^2)



# > mse_test_value
# [1] 1481990
# > mse_test_value2
# [1] 461194.3
# > mse_test_value3
# [1] 462714.4
# importance_matrix[1:40,1]




# Final variable selection:

# overallqual, garagecars, x1st, x2nd, bsmtfin, fullbath, lotarea, yearremodeled, openporch, yearbuilt, overallcond, halfbath, bsmtunf, mssubclass, wooddeck

# unfinished basement space correlated well, using total basement sf instead because it makes more sense
# halfbath correlated decently well too but it doesn't make as much sense as a total bathroom count; actually since both total bathroom and halfbath correlated then maybe it makes sense to keep them both






#### Section 2 - Test the model

# test data
# dataset_test <- read.csv("test.csv")
dataset_test <- read.csv("train.csv")
dataset_test = dataset_test[1001:1460,]

# convert data to numeric and clean data
dataset_test_clean <- transform(dataset_test,
	SalePrice = as.numeric(SalePrice),
	MSSubClass = as.numeric(MSSubClass),
	LotArea = as.numeric(LotArea),
	OverallQual = as.numeric(OverallQual),
	OverallCond = as.numeric(OverallCond),
	YearBuilt = as.numeric(YearBuilt),
	YearRemodAdd = as.numeric(YearRemodAdd),
	BsmtFinSF1 = as.numeric(BsmtFinSF1),
	BsmtUnfSF = as.numeric(BsmtUnfSF),
	TotalBsmtSF = as.numeric(TotalBsmtSF),
	X1stFlrSF = as.numeric(X1stFlrSF),
	X2ndFlrSF = as.numeric(X2ndFlrSF),
	BsmtFullBath = as.numeric(BsmtFullBath),
	BsmtHalfBath = as.numeric(BsmtHalfBath),
	FullBath = as.numeric(FullBath),
	HalfBath = as.numeric(HalfBath),
	BedroomAbvGr = as.numeric(BedroomAbvGr),
	KitchenAbvGr = as.numeric(KitchenAbvGr),
	GarageCars = as.numeric(GarageCars),
	WoodDeckSF = as.numeric(WoodDeckSF),
	OpenPorchSF = as.numeric(OpenPorchSF),
	EnclosedPorch = as.numeric(EnclosedPorch),
	X3SsnPorch = as.numeric(X3SsnPorch),
	ScreenPorch = as.numeric(ScreenPorch),
	PoolArea = as.numeric(PoolArea),
	YrSold = as.numeric(YrSold))


# these are the variables we chose to keep based on our earlier work
dataset_test_clean <- dataset_test_clean[,c("SalePrice",	"MSSubClass",	"LotArea",	"OverallQual",	"OverallCond",
"YearBuilt",	"YearRemodAdd",	"BsmtFinSF1",	"BsmtUnfSF",	"TotalBsmtSF",	"X1stFlrSF",	"X2ndFlrSF",	"BsmtFullBath",	"BsmtHalfBath",	"FullBath",	"HalfBath",	"BedroomAbvGr",	"KitchenAbvGr",	"GarageCars",	"WoodDeckSF",	"OpenPorchSF",	"EnclosedPorch",	"X3SsnPorch",	"ScreenPorch",	"PoolArea",	"YrSold")]
colnames(dataset_test_clean) <- c("SalePrice",	"MSSubClass",	"LotArea",	"OverallQual",	"OverallCond",	"YearBuilt",	"YearRemodAdd",	"BsmtFinSF1",	"BsmtUnfSF",	"TotalBsmtSF",	"X1stFlrSF",	"X2ndFlrSF",	"BsmtFullBath",	"BsmtHalfBath",	"FullBath",	"HalfBath",	"BedroomAbvGr",	"KitchenAbvGr",	"GarageCars",	"WoodDeckSF",	"OpenPorchSF",	"EnclosedPorch",	"X3SsnPorch",	"ScreenPorch",	"PoolArea",	"YrSold")


# gradient boosting model
output_vector <- dataset_test_clean[,1]
xgb <- xgboost(data = data.matrix(dataset_test_clean[,-1]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)
xgb_predictions <- predict(xgb, data.matrix(dataset_test_clean[,-1]))
plot(xgb_predictions,dataset_test_clean$SalePrice)




