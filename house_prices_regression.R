
#### Section 1 - Clean the data, load the data

# Dataset: https://archive.ics.uci.edu/ml/datasets/Image+Segmentation

# Import the dataset, fields are already named
# image <- read.csv(url("https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/train.csv"), header = FALSE)
dataset<- read.csv("train.csv")
image <- dataset

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

# Split the dependent and independent variables for our case
x <- dataset[,1:80]
y <- dataset[,81]


#### Section 2 - Look at the data


# Boxplot
par(mfrow=c(1,6))
  for(i in 1:6) {
  boxplot(x[,i], main=names(image)[i])
  # boxplot(as.numeric(as.character(x[,i])), main=names(image)[i])
}

# Bar plot
plot(y)

# solution to some graphics error, now defunct
# http://stackoverflow.com/questions/20155581/persistent-invalid-graphics-state-error-when-using-ggplot2
# dev.off()

# Scatterplot
featurePlot(x=x, y=y, plot="ellipse")

# Box and whisker plots
featurePlot(x=x, y=y, plot="box")

# Density plot
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)

#### Section 3 - Try three different models

# settings for the train function
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# Random Forest
set.seed(7)
fit.rf <- train(ImageBackground~., data=dataset, method="rf", metric=metric, trControl=control)

# K Nearest Neighbour
set.seed(7)
fit.knn <- train(ImageBackground~., data=dataset, method="knn", metric=metric, trControl=control)

# SVM
set.seed(7)
fit.svm <- train(ImageBackground~., data=dataset, method="svmRadial", metric=metric, trControl=control)

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
