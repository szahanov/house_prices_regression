
#### Section 1 - Clean the data, load the data

# Set the working directory
setwd("C:/R/House Prices Regression Techniques")

# Dataset: https://archive.ics.uci.edu/ml/datasets/Image+Segmentation

# Import the dataset, fields are already named
# image <- read.csv(url("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv"), header = FALSE)
dataset<- read.csv("train.csv")
dataset = dataset_test[1:1000,]


# Requires these packages, make sure to install them and load them
# install.packages('caret', dependencies=FALSE) # If the package has not already been installed
library(caret)	# load package
# install.packages('e1071', dependencies=FALSE) # If the package has not already been installed
library(e1071)	# load package
# install.packages("xgboost") # Make sure you are using a recent enough version of R for this to work
library(xgboost)



# Take a look at the data
summary(dataset)

# Some notes on the variables:
# BsmtFinSF1, BsmtUnfSF, TotalBsmtSF # Unfinished Basement Square Footage was shown to correlate well with sale price but Total Basement Square Footage makes more sense to me so we will use the total, rather than splitting between finished and unfinished square footage
# GrLivArea # We are not using living area because we are using the other square footages
#BsmtFullBath, BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd # We are not using total rooms above ground because we are using the rooms separately
# GarageCars, GarageArea # We are not using garage area because we are using garage cars

# Need to convert some columns to numeric;
dataset <- transform(dataset,
	SalePrice = as.numeric(SalePrice),
	MSSubClass = as.numeric(MSSubClass),
	LotArea = as.numeric(LotArea),
	OverallQual = as.numeric(OverallQual),
	OverallCond = as.numeric(OverallCond),
	YearBuilt = as.numeric(YearBuilt),
	YearRemodAdd = as.numeric(YearRemodAdd),
	BsmtFinSF1 = as.numeric(BsmtFinSF1),
	BsmtUnfSF = as.numeric(BsmtUnfSF),
	TotalBsmtSF = as.numeric(TotalBsmtSF),	# Added back to replace the above two
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


# These are the variables we've selected. These are selected through a combination of looking at the feature importance matrix (in section 3 below) and through judgement. Some notes on the judgemental selection can be found above.
dataset1 <- dataset[,c("SalePrice",	"MSSubClass",	"LotArea",	"OverallQual",	"OverallCond",
"YearBuilt",	"YearRemodAdd",	"BsmtFinSF1",	"BsmtUnfSF",	"TotalBsmtSF",	"X1stFlrSF",	"X2ndFlrSF",	"BsmtFullBath",	"BsmtHalfBath",	"FullBath",	"HalfBath",	"BedroomAbvGr",	"KitchenAbvGr",	"GarageCars",	"WoodDeckSF",	"OpenPorchSF",	"EnclosedPorch",	"X3SsnPorch",	"ScreenPorch",	"PoolArea",	"YrSold")]
colnames(dataset1) <- c("SalePrice",	"MSSubClass",	"LotArea",	"OverallQual",	"OverallCond",	"YearBuilt",	"YearRemodAdd",	"BsmtFinSF1",	"BsmtUnfSF",	"TotalBsmtSF",	"X1stFlrSF",	"X2ndFlrSF",	"BsmtFullBath",	"BsmtHalfBath",	"FullBath",	"HalfBath",	"BedroomAbvGr",	"KitchenAbvGr",	"GarageCars",	"WoodDeckSF",	"OpenPorchSF",	"EnclosedPorch",	"X3SsnPorch",	"ScreenPorch",	"PoolArea",	"YrSold")


# https://www.analyticsvidhya.com/blog/2016/01/xgboost-algorithm-easy-steps/

# xgb <- xgboost(data = data.matrix(dataset[,-1]),  label = y,  eta = 0.1, max_depth = 15,  nround=25,  subsample = 0.5, colsample_bytree = 0.5, seed = 1, eval_metric = "merror", objective = "multi:softprob", num_class = 12, nthread = 3)


output_vector <- dataset1[,1]

# xgb <- xgboost(data = data.matrix(dataset1[,-1]),  label = output_vector,  eta = 0.1, max_depth = 15,  nround=25)
xgb <- xgboost(data = data.matrix(dataset1[,-1]),  label = output_vector,  eta = 0.3, max_depth = 6,  nround=100)

# predict values in test set
xgb_predictions <- predict(xgb, data.matrix(dataset1[,-1]))

plot(xgb_predictions,dataset1$SalePrice)

# MSE
sum((dataset1$SalePrice - xgb_predictions)^2)
# AVG - test mean square error
mse_test_value <- mean((dataset1$SalePrice - xgb_predictions)^2)
# R squared
r2 <- cor(dataset1$SalePrice,xgb_predictions)^2
mean_diff <- mean(xgb_predictions - dataset1$SalePrice)

# some spot checks
i=5;(xgb_predictions[i] - dataset1$SalePrice[i])/dataset1$SalePrice[i]
i=20;(xgb_predictions[i] - dataset1$SalePrice[i])/dataset1$SalePrice[i]
i=35;(xgb_predictions[i] - dataset1$SalePrice[i])/dataset1$SalePrice[i]
plot(xgb_predictions/max(dataset1$SalePrice),dataset1$SalePrice/max(dataset1$SalePrice))
plot((xgb_predictions - dataset1$SalePrice)/dataset1$SalePrice)


# nrounds: the number of decision trees in the final model
# objective: the training objective to use, where “binary:logistic” means a binary classifier.




# Look at variable importance
X <- dataset1
# Lets start with finding what the actual tree looks like
model <- xgb.dump(xgb, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model
# Get the feature real names
names <- dimnames(data.matrix(X[,-1]))[[2]]
# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:20,])
#In case last step does not work for you because of a version issue, you can try following :
barplot(importance_matrix[,1])


# results: overallqual, garagecars, x1st, x2nd, bsmtfin, fullbath, lotarea, yearremodeled, openporch, yearbuilt, overallcond, halfbath, bsmtunf, mssubclass, wooddeck

# unfinished basement space correlated well, using total basement sf instead because it makes more sense
# halfbath correlated decently well too but it doesn't make as much sense as a total bathroom count; actually since both total bathroom and halfbath correlated then maybe it makes sense to keep them both


# Second try
dataset2 <- dataset[,c("SalePrice", "MSSubClass", "LotArea", "OverallQual", "OverallCond",
"YearBuilt", "YearRemodAdd", "TotalBsmtSF", "X1stFlrSF","X2ndFlrSF","FullBath",	"HalfBath", "GarageCars", "WoodDeckSF",	"OpenPorchSF")]
colnames(dataset2) <- c("SalePrice", "MSSubClass", "LotArea", "OverallQual", "OverallCond",
"YearBuilt", "YearRemodAdd", "TotalBsmtSF", "X1stFlrSF","X2ndFlrSF","FullBath",	"HalfBath", "GarageCars", "WoodDeckSF",	"OpenPorchSF")

# Second try xgboost model

output_vector2 <- dataset2[,1]

xgb2 <- xgboost(data = data.matrix(dataset2[,-1]),  label = output_vector,  eta = 0.1, max_depth = 15,  nround=25)

# predict values in test set
xgb_predictions2 <- predict(xgb2, data.matrix(dataset2[,-1]))

plot(xgb_predictions2,dataset2$SalePrice)

# MSE
sum((dataset2$SalePrice - xgb_predictions2)^2)
# AVG - test mean square error
mse_test_value2 <- mean((dataset2$SalePrice - xgb_predictions2)^2)

# some spot checks
i=5;(xgb_predictions2[i] - dataset2$SalePrice[i])/dataset2$SalePrice[i]
i=20;(xgb_predictions2[i] - dataset2$SalePrice[i])/dataset2$SalePrice[i]
i=35;(xgb_predictions2[i] - dataset2$SalePrice[i])/dataset2$SalePrice[i]
plot((xgb_predictions2 - dataset2$SalePrice)/dataset2$SalePrice)


# /Second try xgboost model

# Second try was worse



# Second try look at variable importance

X <- dataset2
# Lets start with finding what the actual tree looks like
model <- xgb.dump(xgb, with.stats = T)
model[1:10] #This statement prints top 10 nodes of the model
# Get the feature real names
names <- dimnames(data.matrix(X[,-1]))[[2]]
# Compute feature importance matrix
importance_matrix <- xgb.importance(names, model = xgb)
# Nice graph
xgb.plot.importance(importance_matrix[1:20,])
#In case last step does not work for you because of a version issue, you can try following :
barplot(importance_matrix[,1])

# /Second try look at variable importance


# Check for overfitting and do general testing
# https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/


## Random Forest model for comparison

# better row and column names: https://stackoverflow.com/questions/16032778/how-to-set-unique-row-and-column-names-of-a-matrix-when-its-dimension-is-unknown
controlDTree <- trainControl(method="cv", 5)
modelDTree <- train(SalePrice ~ ., data=dataset, method="rf", trControl=controlDTree, ntree=150)
modelDTree




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




