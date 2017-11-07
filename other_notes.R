
# install.packages('e1071', dependencies=TRUE) # If the package has not already been installed
# install.packages('caret', dependencies=TRUE) # If the package has not already been installed
# install.packages('xgboost') # Make sure you are using a recent enough version of R for this to work
# if using linux and some of the below libraries are having trouble installing:
# sudo apt-get install g++
# sudo apt-get install libnlopt-dev
# install.packages()



# Reasonableness plots
plot(xgb_test_predictions,test_clean$SalePrice)
plot(xgb_test_predictions/max(test_clean$SalePrice),test_clean$SalePrice/max(test_clean$SalePrice))
plot((xgb_test_predictions - test_clean$SalePrice)/test_clean$SalePrice)


# Mean Square Error, R squared, compare averages
mse <- mean((xgb_test_predictions - test_clean$SalePrice)^2); mse_test_value
# 461198
r2 <- cor(xgb_test_predictions,test_clean$SalePrice)^2; r2
# 0.8843558
mean_diff <- mean(xgb_test_predictions - test_clean$SalePrice); mean_diff
# [1] 315.6391




# to save plots:
png(filename="plot1.png")
plot((xgb_test_predictions - test_clean$SalePrice)/test_clean$SalePrice)
dev.off()


