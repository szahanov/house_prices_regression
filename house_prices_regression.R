
#### Section 1 - Clean the data, load the data

# Set the working directory
setwd("C:/R/House Prices Regression Techniques")

# Dataset: https://archive.ics.uci.edu/ml/datasets/Image+Segmentation

# Import the dataset, fields are already named
# image <- read.csv(url("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv"), header = FALSE)
dataset<- read.csv("train.csv")
#image <- dataset

# Import the test dataset
# image_test <- read.csv(url("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/test.csv"), header = FALSE)
dataset_test<- read.csv("test.csv")

# Requires these packages, make sure to install them and load them
# install.packages('caret', dependencies=FALSE) # If the package has not already been installed
library(caret)	# load package
# install.packages('e1071', dependencies=FALSE) # If the package has not already been installed
library(e1071)	# load package


# install.packages("xgboost",dependencies=FALSE) # If the package has not already been installed
install.packages("drat")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
library(xgboost)

# Take a look at the data
summary(dataset)

# Change column names
# names(df)[names(df) == 'old.var.name'] <- 'new.var.name'

# Convert columns to numeric
#MSSubClass
#LotArea
#OverallQual
#OverallCond
#YearBuilt
#YearRemodAdd
#BsmtFinSF1
#BsmtUnfSF
# TotalBsmtSF # Not using total basement square footage because we are including finished and unfinished basement square footage
# GrLivArea # Not using living area because we are using the other square footages
#BsmtFullBath
#BsmtHalfBath
#FullBath
#HalfBath
#BedroomAbvGr
#KitchenAbvGr
# TotRmsAbvGrd # Not using total rooms above ground because we are using the rooms separately
#GarageCars
# GarageArea # Not using garage area because we are using garage cars
#WoodDeckSF
#OpenPorchSF
#EnclosedPorch
#3SsnPorch
#ScreenPorch
#PoolArea
#YrSold



dataset <- transform(dataset, X1stFlrSF = as.numeric(X1stFlrSF))
dataset <- transform(dataset, X2ndFlrSF = as.numeric(X2ndFlrSF))

dataset <- transform(dataset,
	MSSubClass = as.numeric(MSSubClass),
	LotArea = as.numeric(LotArea),
	OverallQual = as.numeric(OverallQual),
	OverallCond = as.numeric(OverallCond),
	YearBuilt = as.numeric(YearBuilt),
	YearRemodAdd = as.numeric(YearRemodAdd),
	BsmtFinSF1 = as.numeric(BsmtFinSF1),
	BsmtUnfSF = as.numeric(BsmtUnfSF),
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
dataset <- transform(dataset, SalePrice = as.numeric(SalePrice))


# Split the dependent and independent variables for our case
#x <- dataset[,1:80]
x <- dataset[,44:45] # testing with just some columns
y <- dataset[,81]


# Convert columns to a usable data type
x <- sapply(x, as.numeric)
# removes columns containing NA
# x <- x[, unlist(lapply(x, function(i) !any(is.na(i))))]
# https://rstudio-pubs-static.s3.amazonaws.com/95110_608a586066a1493da1e790defb8c595b.html

# datasubset <- matrix(c(dataset$SalePrice,dataset$X1stFlrSF,dataset$X2ndFlrSF),nrow=length(dataset$SalePrice))
#datasubset <- dataset[,c("SalePrice","MSZoning","X1stFlrSF","X2ndFlrSF")]
#colnames(datasubset) <- c("SalePrice","MSZoning","X1stFlrSF","X2ndFlrSF")
datasubset <- dataset[,c("SalePrice",
	"MSSubClass",
	"LotArea",
	"OverallQual",
	"OverallCond",
	"YearBuilt",
	"YearRemodAdd",
	"BsmtFinSF1",
	"BsmtUnfSF",
	"X1stFlrSF",
	"X2ndFlrSF",
	"BsmtFullBath",
	"BsmtHalfBath",
	"FullBath",
	"HalfBath",
	"BedroomAbvGr",
	"KitchenAbvGr",
	"GarageCars",
	"WoodDeckSF",
	"OpenPorchSF",
	"EnclosedPorch",
	"X3SsnPorch",
	"ScreenPorch",
	"PoolArea",
	"YrSold")]
colnames(datasubset) <- c("SalePrice",
	"MSSubClass",
	"LotArea",
	"OverallQual",
	"OverallCond",
	"YearBuilt",
	"YearRemodAdd",
	"BsmtFinSF1",
	"BsmtUnfSF",
	"X1stFlrSF",
	"X2ndFlrSF",
	"BsmtFullBath",
	"BsmtHalfBath",
	"FullBath",
	"HalfBath",
	"BedroomAbvGr",
	"KitchenAbvGr",
	"GarageCars",
	"WoodDeckSF",
	"OpenPorchSF",
	"EnclosedPorch",
	"X3SsnPorch",
	"ScreenPorch",
	"PoolArea",
	"YrSold")

# better row and column names: https://stackoverflow.com/questions/16032778/how-to-set-unique-row-and-column-names-of-a-matrix-when-its-dimension-is-unknown
controlDTree <- trainControl(method="cv", 5)
modelDTree <- train(SalePrice ~ ., data=datasubset, method="rf", trControl=controlDTree, ntree=150)
modelDTree

## Test the model

#predictions <- predict(modelDTree, datasubset[,2:4])
predictions <- predict(modelDTree, datasubset[,2:25])
#confusionMatrix(predictions, datasubset$SalePrice) #not for regression?
plot(predictions,datasubset$SalePrice)


# MSE
sum((datasubset$SalePrice - predictions)^2)
# AVG - test mean square error
mse_test_value <- mean((datasubset$SalePrice - predictions)^2)
# https://tomaztsql.wordpress.com/2016/01/11/playing-with-regression-prediction-and-mse-measure/

# some spot checks
i=5;(datasubset$SalePrice[i] - predictions[i])/datasubset$SalePrice[i]
i=20;(datasubset$SalePrice[i] - predictions[i])/datasubset$SalePrice[i]
i=35;(datasubset$SalePrice[i] - predictions[i])/datasubset$SalePrice[i]
plot((datasubset$SalePrice - predictions)/datasubset$SalePrice)


#### Section 2 - Look at the data


# http://r-statistics.co/Linear-Regression.html
cor(dataset$SalePrice, dataset$X1stFlrSF) # calculate correlation
linearMod <- lm(SalePrice ~ X1stFlrSF, data=dataset) # build linear regression model 
print(linearMod)
plot(linearMod)


# Boxplot
par(mfrow=c(1,6))
  for(i in 1:6) {
  boxplot(x[,i], main=names(image)[i])
  # boxplot(as.numeric(as.character(x[,i])), main=names(image)[i])
}
# not all data here is numeric

# Bar plot
plot(y)

## Do these not work if it's a regression project, rather than classification?

# solution to some graphics error, now defunct
# http://stackoverflow.com/questions/20155581/persistent-invalid-graphics-state-error-when-using-ggplot2
# dev.off()

# Scatterplot
#featurePlot(x=x, y=y, plot="ellipse")

# Box and whisker plots
#featurePlot(x=x, y=y, plot="box")

# Density plot
#scales <- list(x=list(relation="free"), y=list(relation="free"))
#featurePlot(x=x, y=y, plot="density", scales=scales)

#### Section 3 - Try three different models

# settings for the train function
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Random Forest
set.seed(7)
fit.rf <- train(SalePrice~., data=dataset, method="rf", metric=metric, trControl=control)

# K Nearest Neighbour
set.seed(7)
fit.knn <- train(SalePrice~., data=dataset, method="knn", metric=metric, trControl=control)

# SVM
set.seed(7)
fit.svm <- train(SalePrice~., data=dataset, method="svmRadial", metric=metric, trControl=control)

# summarize results
results <- resamples(list(rf=fit.rf, knn=fit.knn, svm=fit.svm))
summary(results)

#### Section 4 - Display results

# compare the different models
# random forest has the highest accuracy and highest kappa coefficient
dotplot(results)

# summarize the random forest results
print(fit.rf)

# test the model
predictions <- predict(fit.rf, image_test[,2:20])
confusionMatrix(predictions, image_test$ImageBackground)

# looking good














#### House Price Regression with a Generalized Linear Model

setwd("C:/R/House Prices Regression Techniques")

# Dataset: https://archive.ics.uci.edu/ml/datasets/Image+Segmentation

# Import the dataset, fields are already named
# image <- read.csv(url("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv"), header = FALSE)
dataset<- read.csv("train.csv")
#image <- dataset

# Import the test dataset
# image_test <- read.csv(url("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/test.csv"), header = FALSE)
dataset_test<- read.csv("test.csv")

# Requires these packages, make sure to install them and load them
# install.packages('caret', dependencies=FALSE) # If the package has not already been installed
library(caret)	# load package
# install.packages('e1071', dependencies=FALSE) # If the package has not already been installed
library(e1071)	# load package

# Take a look at the data
summary(dataset)

# Change column names
# names(df)[names(df) == 'old.var.name'] <- 'new.var.name'

# Convert columns to numeric
dataset <- transform(dataset, X1stFlrSF = as.numeric(X1stFlrSF))
dataset <- transform(dataset, X2ndFlrSF = as.numeric(X2ndFlrSF))
dataset <- transform(dataset, SalePrice = as.numeric(SalePrice))

# datasubset <- matrix(c(dataset$SalePrice,dataset$X1stFlrSF,dataset$X2ndFlrSF),nrow=length(dataset$SalePrice))
datasubset <- dataset[,c("SalePrice","MSZoning","X1stFlrSF","X2ndFlrSF")]
colnames(datasubset) <- c("SalePrice","MSZoning","X1stFlrSF","X2ndFlrSF")
# better row and column names: https://stackoverflow.com/questions/16032778/how-to-set-unique-row-and-column-names-of-a-matrix-when-its-dimension-is-unknown
controlDTree <- trainControl(method="cv", 5)
modelDTree <- train(SalePrice ~ ., data=datasubset, method="rf", trControl=controlDTree, ntree=150)
modelDTree

## Test the model

predictions <- predict(modelDTree, datasubset[,2:4])
#confusionMatrix(predictions, datasubset$SalePrice) #not for regression?
plot(predictions,datasubset$SalePrice)



#### Section 2 - Look at the data


# http://r-statistics.co/Linear-Regression.html
cor(dataset$SalePrice, dataset$X1stFlrSF) # calculate correlation
linearMod <- lm(SalePrice ~ X1stFlrSF, data=dataset) # build linear regression model 
print(linearMod)
plot(linearMod)



