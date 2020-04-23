
# Load Libraries ----------------------------------------------------------

library(curatedMetagenomicData)
library(dplyr)
library(randomForest)
library(ROCR)
library(e1071)
library(glmnet)
library(ggplot2)

# Data Cleaning -----------------------------------------------------------

# QinN_2014 liver cirhosis
qin = curatedMetagenomicData("QinN_2014.metaphlan_bugs_list.stool", dryrun = FALSE)
qin_est = qin[[1]] 
experimentData(qin_est) # experiment-level metadata
head(pData(qin_est)) # subject-level metadata

# note HD are healthy individuals from discovery phase, HV from validation phase
# LD are cirrhosis individuals from discovey phase and LV from validation phase per subject ID
# https://www.nature.com/articles/nature13568

exp = exprs(qin_est) # relative abundance
t_exp = data.frame(t(exp))
t_exp$id = rownames(t_exp)

sub = pData(qin_est)
sub_condition = sub[c("study_condition", "subjectID")]
# sub_condition = select(sub, study_condition, subjectID, gender, age, BMI)
colnames(sub_condition)[2] = "id"
t_exp = left_join(t_exp, sub_condition, by = "id")
t_exp$study_condition = factor(t_exp$study_condition, levels = c("control", "cirrhosis"))
t_exp_c = t_exp[, -colSums(t_exp == 0)]
t_exp_c = t_exp_c[, -which(colnames(t_exp_c) %in% "id")]
# t_exp_c$gender = factor(t_exp_c$gender, levels = c("male", "female"))

# Data Split for nested CV ------------------------------------------------

testingData = t_exp_c # for nested cv to compute average performance and for rf

# k-fold validation outerloop
set.seed(123)
k = 10
fold_inds = sample(1:k, nrow(testingData), replace = TRUE)

# split  data into training & testing partitions
cv_data = lapply(1:k, 
                 function(index) list(est = testingData[fold_inds != index, , drop = FALSE],
                                      val = testingData[fold_inds == index, , drop = FALSE]))

# k-fold validation innerloop
trainingData_tuning = cv_data[[1]]$est
set.seed(123)
ki = 5
fold_inds = sample(1:ki, nrow(trainingData_tuning), replace = TRUE)

cv_data_inner = lapply(1:ki, 
                 function(index) list(est = trainingData_tuning[fold_inds != index, , drop = FALSE],
                                      val = trainingData_tuning[fold_inds == index, , drop = FALSE]))

est = cv_data_inner[[1]]$est # for rf
val = cv_data_inner[[1]]$val # for rf

# Random Forest -----------------------------------------------------------

accuracy_all = numeric()
auc_all = numeric()
sensitivity_all = numeric()
specificity_all = numeric()
precision_all = numeric()

for (i in 1:k) {
  train = cv_data[[i]]$est
  test = cv_data[[i]]$val
  rf_model = randomForest(study_condition ~ ., ntree = 500, 
                          mtry = floor(sqrt(ncol(train) - 1)),
                          importance = TRUE, data = train)
  
  predictions = predict(rf_model, test, type = "response")
  accuracy = mean(predictions == test$study_condition)
  
  rf_model_pred = predict(rf_model, test, type = "prob")
  auc = performance(prediction(rf_model_pred[, 2], test$study_condition, 
                               label.ordering = c("control", "cirrhosis")), "auc")
  auc = unlist(slot(auc, "y.values"))
  auc_all[i] = auc
  accuracy_all[i] = accuracy
  conf_matrix = table(Predicted = predictions, Actual = test$study_condition)
  TP = conf_matrix[2, 2]
  FN = conf_matrix[1, 2]
  sensitivity = TP/(TP+FN)
  TN = conf_matrix[1, 1]
  FP = conf_matrix[2, 1]
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  sensitivity_all[i] = sensitivity
  specificity_all[i] = specificity
  precision_all[i] = precision
}

mean(accuracy_all) # average overall accuracy
mean(auc_all) # average overall auc
mean(sensitivity_all) # average overall sensitivity, the recall (i.e., the number of correct positive samples divided by the total number of positive samples)
mean(specificity_all) # average overall specificity
mean(precision_all)

# Feature Importance + RF -------------------------------------------------

# feature importance selection
rf_model = randomForest(study_condition ~., ntree = 500, mtry = floor(sqrt(ncol(est) - 1)),
                        importance = TRUE, data = est)
importance = data.frame(rf_model$importance)
importance$species = rownames(importance)
importance = importance %>% arrange(., -MeanDecreaseGini)

# find the model with the best number of features
feature_list = c(5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 125, 150, 175, 200)
rf_list = vector(mode = "list", length = 15)
auc_list = vector(mode = "numeric", length = 15)
misclass_list = vector(mode = "numeric", length = 15)
for (i in 1:15) {
  n_feature = feature_list[i]
  features = importance$species[1:n_feature]
  trainingData_f = cbind(est[features], est[ncol(est)])
  rf_f = randomForest(study_condition ~ ., ntree = 500, mtry = floor(sqrt(n_feature)),
                       importance = TRUE, data = trainingData_f)
  rf_list[[i]] = rf_f
  predict = predict(rf_f, val, type = "class")
  misclass = mean(predict != val$study_condition)
  misclass_list[i] = misclass
  rf_model_pred = predict(rf_f, val, type = "prob")
  auc = performance(prediction(rf_model_pred[, 2], val$study_condition, 
                                label.ordering = c("control", "cirrhosis")), "auc")
  auc = unlist(slot(auc, "y.values"))
  auc_list[i] = auc
}

feature_select = which.max(auc_list) #find the model that maximize auc
which.min(misclass_list)


# 10-fold cv to calculate avg performance
# use the the feature number from the 5th feature importance list to build the final RF
features = importance$species[1:feature_select]

testingData_feature = cbind(testingData[features], testingData[ncol(testingData)])

set.seed(123)
k = 10
fold_inds = sample(1:k, nrow(testingData_feature), replace = TRUE)

## split  data into training & testing partitions
cv_data = lapply(1:k, 
                 function(index) list(est = testingData_feature[fold_inds != index, , drop = FALSE],
                                      val = testingData_feature[fold_inds == index, , drop = FALSE]))

accuracy_all = numeric()
auc_all = numeric()
sensitivity_all = numeric()
specificity_all = numeric()
precision_all = numeric()

for (i in 1:k) {
  train = cv_data[[i]]$est
  test = cv_data[[i]]$val
  rf_final = randomForest(study_condition ~ ., ntree = 500, 
                          mtry = floor(sqrt(feature_select)),
                            importance = TRUE, data = train)
    
  predictions = predict(rf_final, test, type = "response")
  accuracy = mean(predictions == test$study_condition)
    
  rf_model_pred = predict(rf_final, test, type = "prob")
  auc = performance(prediction(rf_model_pred[, 2], test$study_condition, 
                                 label.ordering = c("control", "cirrhosis")), "auc")
  auc = unlist(slot(auc, "y.values"))
  auc_all[i] = auc
  accuracy_all[i] = accuracy
  conf_matrix = table(Predicted = predictions, Actual = test$study_condition)
  TP = conf_matrix[2, 2]
  FN = conf_matrix[1, 2]
  sensitivity = TP/(TP+FN)
  TN = conf_matrix[1, 1]
  FP = conf_matrix[2, 1]
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  sensitivity_all[i] = sensitivity
  specificity_all[i] = specificity
  precision_all[i] = precision
}

mean(accuracy_all) # average overall accuracy
mean(auc_all) # average overall auc
mean(sensitivity_all)
mean(specificity_all)
mean(precision_all)

# SVM ---------------------------------------------------------------------

# define cost and gamma ranges
c = 2^seq(-5, 15, by = 2)
g = 2^seq(-15, 3, by = 2)

# tune model to find optimal cost, gamma values, 5-fold cross validation
set.seed(123)
tune.out = tune(svm, study_condition ~ ., 
                 data = trainingData_tuning, kernel = "radial",
                 ranges = list(cost = c,
                               gamma = g), 
                 tunecontrol = tune.control(cross = 5))

# show best model
svm.fit = tune.out$best.model

# 10-fold cv to calculate avg performance
set.seed(123)
k = 10
fold_inds = sample(1:k, nrow(testingData), replace = TRUE)

## split  data into training & testing partitions
cv_data = lapply(1:k, 
                 function(index) list(est = testingData[fold_inds != index, , drop = FALSE],
                                      val = testingData[fold_inds == index, , drop = FALSE]))

accuracy_all = numeric()
auc_all = numeric()
sensitivity_all = numeric()
specificity_all = numeric()
precision_all = numeric()

for (i in 1:k) {
  train = cv_data[[i]]$est
  test = cv_data[[i]]$val
  svm.fit.trn = svm(study_condition ~ ., data = train, 
                    kernel = "radial", gamma = svm.fit$gamma, cost = svm.fit$cost, 
                    probability = TRUE)
  
  svm_model_pred = predict(svm.fit.trn, test, probability = TRUE)
  accuracy = mean(svm_model_pred == test$study_condition)
  auc = performance(prediction(attributes(svm_model_pred)$probabilities[, 2], 
                               test$study_condition, 
                               label.ordering = c("control", "cirrhosis")), "auc")
  auc = unlist(slot(auc, "y.values"))
  auc_all[i] = auc
  accuracy_all[i] = accuracy
  conf_matrix = table(Predicted = svm_model_pred, Actual = test$study_condition)
  TP = conf_matrix[2, 2]
  FN = conf_matrix[1, 2]
  sensitivity = TP/(TP+FN)
  TN = conf_matrix[1, 1]
  FP = conf_matrix[2, 1]
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  sensitivity_all[i] = sensitivity
  specificity_all[i] = specificity
  precision_all[i] = precision
}

mean(accuracy_all)
mean(auc_all)
mean(sensitivity_all)
mean(specificity_all)
mean(precision_all)

# Lasso Regression --------------------------------------------------------

# define lambda range for lasso
lambda = 10^seq(-4, -0.5, by = 0.5)

# prepare data
est = cv_data_inner[[1]]$est # for tuning
val = cv_data_inner[[1]]$val # for tuning

x_train_tuning = model.matrix(study_condition ~ ., est)[,-1]
x_test_tuning = model.matrix(study_condition ~ ., val)[,-1]

y_train_tuning = est$study_condition
y_test_tuning = val$study_condition 

x_test = x
y_test = y

set.seed(123)
cv.lasso = cv.glmnet(x_train_tuning, y_train_tuning, alpha = 1, nfolds = 5, lambda = lambda, 
                      type.measure = "auc", 
                      family = "binomial") # Fit lasso model on training data

plot(cv.lasso) # Draw plot of training MSE as a function of lambda\
max(cv.lasso$cvm)
bestlambda_lasso = cv.lasso$lambda.min # Select lamda that minimizes training MSE

# 10-fold cv to calculate avg performance
set.seed(123)
k = 10
fold_inds = sample(1:k, nrow(testingData), replace = TRUE)

## split  data into training & testing partitions
cv_data = lapply(1:k, 
                 function(index) list(est = testingData[fold_inds != index, , drop = FALSE],
                                      val = testingData[fold_inds == index, , drop = FALSE]))

accuracy_all = numeric()
auc_all = numeric()
sensitivity_all = numeric()
specificity_all = numeric()
precision_all = numeric()

for (i in 1:k) {
  train = cv_data[[i]]$est
  test = cv_data[[i]]$val
  
  y_train = train$study_condition
  x_train = model.matrix(study_condition ~ ., train)[,-1]
  
  x_test = model.matrix(study_condition ~ ., test)[,-1]
  y_test = test$study_condition
  
  lasso = glmnet(x_train, y_train, alpha = 1, lambda = bestlambda_lasso, 
                 type.measure = "auc", 
                 family = "binomial")
  
  prediction = predict(lasso, x_test, type = "class")
  accuracy = mean(prediction == y_test)
  accuracy_all[i] = accuracy
  
  prob = predict(lasso, newx = x_test, type = "response")
  pred = prediction(prob, y_test, label.ordering = c("control", "cirrhosis"))
  auc = performance(pred,"auc") # shows calculated AUC for model
  auc = unlist(slot(auc, "y.values"))
  auc_all[i] = auc
  
  prediction = factor(prediction, levels = c("control","cirrhosis") )
  conf_matrix = table(Predicted = prediction, Actual = test$study_condition)
  TP = conf_matrix[2, 2]
  FN = conf_matrix[1, 2]
  sensitivity = TP/(TP+FN)
  TN = conf_matrix[1, 1]
  FP = conf_matrix[2, 1]
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  sensitivity_all[i] = sensitivity
  specificity_all[i] = specificity
  precision_all[i] = precision
}

mean(accuracy_all)
mean(auc_all)
mean(sensitivity_all)
mean(specificity_all)
mean(precision_all)

# ENet --------------------------------------------------------------------

#define alpha
L1_ratio = c(0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)

tuning_grid = data.frame(
  L1_ratio = L1_ratio,
  cvm_min    = NA,
  cvm_1se    = NA,
  lambda_min = NA,
  lambda_1se = NA
)

for(i in seq_along(tuning_grid$L1_ratio)) {
  
  # fit CV model for each alpha value
  fit = cv.glmnet(x_train_tuning, y_train_tuning, alpha = tuning_grid$L1_ratio[i], 
                   nfolds = 5, lambda = lambda,
                   type.measure = "auc", family = "binomial")
  
  # extract CVM and lambda values
  tuning_grid$cvm_min[i]    = fit$cvm[fit$lambda == fit$lambda.min]
  tuning_grid$cvm_1se[i]    = fit$cvm[fit$lambda == fit$lambda.1se]
  tuning_grid$lambda_min[i] = fit$lambda.min
  tuning_grid$lambda_1se[i] = fit$lambda.1se
}

tuning_grid = tuning_grid %>% mutate(se = cvm_1se - cvm_min)

ggplot(aes(L1_ratio, cvm_min), data = tuning_grid) +
  geom_line(size = 2) +
  geom_ribbon(aes(ymax = cvm_min + se, ymin = cvm_min - se), alpha = .25) +
  theme_classic() +
  ggtitle("CVM ± one standard error")

bestalpha = tuning_grid$L1_ratio[tuning_grid$cvm_min == min(tuning_grid$cvm_min)]
bestlambda_enet = tuning_grid$lambda_min[tuning_grid$cvm_min == min(tuning_grid$cvm_min)]

# 10-fold cv to calculate avg performance
set.seed(123)
k = 10
fold_inds = sample(1:k, nrow(testingData), replace = TRUE)

## split  data into training & testing partitions
cv_data = lapply(1:k, 
                 function(index) list(est = testingData[fold_inds != index, , drop = FALSE],
                                      val = testingData[fold_inds == index, , drop = FALSE]))

accuracy_all = numeric()
auc_all = numeric()
sensitivity_all = numeric()
specificity_all = numeric()
precision_all = numeric()

for (i in 1:k) {
  train = cv_data[[i]]$est
  test = cv_data[[i]]$val
  
  x_train = model.matrix(study_condition ~ ., train)[,-1]
  y_train = train$study_condition
  
  x_test = model.matrix(study_condition ~ ., test)[,-1]
  y_test = test$study_condition
  
  enet = glmnet(x_train, y_train, alpha = bestalpha, lambda = bestlambda_enet, 
                 type.measure = "auc", 
                 family = "binomial")
  
  prediction = predict(enet, x_test, type = "class")
  accuracy = mean(prediction == y_test)
  accuracy_all[i] = accuracy
  
  prob = predict(enet, newx = x_test, type = "response")
  pred = prediction(prob, y_test, label.ordering = c("control", "cirrhosis"))
  auc = performance(pred,"auc") # shows calculated AUC for model
  auc = unlist(slot(auc, "y.values"))
  auc_all[i] = auc
  
  prediction = factor(prediction, levels = c("control","cirrhosis") )
  conf_matrix = table(Predicted = prediction, Actual = y_test)
  TP = conf_matrix[2, 2]
  FN = conf_matrix[1, 2]
  sensitivity = TP/(TP+FN)
  TN = conf_matrix[1, 1]
  FP = conf_matrix[2, 1]
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  sensitivity_all[i] = sensitivity
  specificity_all[i] = specificity
  precision_all[i] = precision
}

mean(accuracy_all)
mean(auc_all)
mean(sensitivity_all)
mean(specificity_all)
mean(precision_all)

# Lasso + RF --------------------------------------------------------------

set.seed(123)
cv.lasso = cv.glmnet(x_train_tuning, y_train_tuning, alpha = 1, nfolds = 5, lambda = lambda, 
                     type.measure = "auc", 
                     family = "binomial") # Fit lasso model on training data

# extract coefficients from lasso regression
lasso_coef = predict(cv.lasso, type = "coefficients", s = bestlambda_lasso)[1:ncol(x_train_tuning),] # Display coefficients using lambda chosen by CV
length(lasso_coef[lasso_coef != 0]) # Display only non-zero coefficients
feature_coef = data.frame(lasso_coef[lasso_coef != 0])
features = rownames(feature_coef)[-1]

# RF using features from lasso
# 10-fold cv to calculate avg performance
testingData_feature = cbind(testingData[features], testingData[ncol(testingData)])
set.seed(123)
k = 10
fold_inds = sample(1:k, nrow(testingData_feature), replace = TRUE)

## split  data into training & testing partitions
cv_data = lapply(1:k, 
                 function(index) list(est = testingData_feature[fold_inds != index, , drop = FALSE],
                                      val = testingData_feature[fold_inds == index, , drop = FALSE]))

accuracy_all = numeric()
auc_all = numeric()
sensitivity_all = numeric()
specificity_all = numeric()
precision_all = numeric()

for (i in 1:k) {
  train = cv_data[[i]]$est
  test = cv_data[[i]]$val
  rf_final = randomForest(study_condition ~ ., ntree = 500, 
                          mtry = floor(sqrt(length(features))),
                          importance = TRUE, data = train)
  
  predictions = predict(rf_final, test, type = "response")
  accuracy = mean(predictions == test$study_condition)
  
  rf_model_pred = predict(rf_final, test, type = "prob")
  auc = performance(prediction(rf_model_pred[, 2], test$study_condition, 
                               label.ordering = c("control", "cirrhosis")), "auc")
  auc = unlist(slot(auc, "y.values"))
  auc_all[i] = auc
  accuracy_all[i] = accuracy
  conf_matrix = table(Predicted = predictions, Actual = test$study_condition)
  TP = conf_matrix[2, 2]
  FN = conf_matrix[1, 2]
  sensitivity = TP/(TP+FN)
  TN = conf_matrix[1, 1]
  FP = conf_matrix[2, 1]
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  sensitivity_all[i] = sensitivity
  specificity_all[i] = specificity
  precision_all[i] = precision
}

mean(accuracy_all) #average overall accuracy
mean(auc_all) #average overall auc
mean(sensitivity_all)
mean(specificity_all)
mean(precision_all)

# ENet + RF ---------------------------------------------------------------
set.seed(123)
cv.enet = cv.glmnet(x_train_tuning, y_train_tuning, alpha = bestalpha, lambda = lambda,
                    type.measure = "auc", 
                    family = "binomial")

# feature select
enet_coef = predict(cv.enet, type = "coefficients", s = bestlambda_enet)[1:ncol(x_train_tuning),] # Display coefficients using lambda chosen by CV
length(enet_coef[enet_coef != 0]) # Display only non-zero coefficients
feature_coef = data.frame(enet_coef[enet_coef != 0])
features = rownames(feature_coef)[-1]

# RF using features from lasso
# 10-fold cv to calculate avg performance
testingData_feature = cbind(testingData[features], testingData[ncol(testingData)])
set.seed(123)
k = 10
fold_inds = sample(1:k, nrow(testingData_feature), replace = TRUE)

# split  data into training & testing partitions
cv_data = lapply(1:k, 
                 function(index) list(est = testingData_feature[fold_inds != index, , drop = FALSE],
                                      val = testingData_feature[fold_inds == index, , drop = FALSE]))

accuracy_all = numeric()
auc_all = numeric()
sensitivity_all = numeric()
specificity_all = numeric()
precision_all = numeric()

for (i in 1:k) {
  train = cv_data[[i]]$est
  test = cv_data[[i]]$val
  rf_final = randomForest(study_condition ~ ., ntree = 500, 
                          mtry = floor(sqrt(length(features))),
                          importance = TRUE, data = train)
  
  predictions = predict(rf_final, test, type = "response")
  accuracy = mean(predictions == test$study_condition)
  
  rf_model_pred = predict(rf_final, test, type = "prob")
  auc = performance(prediction(rf_model_pred[, 2], test$study_condition, 
                               label.ordering = c("control", "cirrhosis")), "auc")
  auc = unlist(slot(auc, "y.values"))
  auc_all[i] = auc
  accuracy_all[i] = accuracy
  conf_matrix = table(Predicted = predictions, Actual = test$study_condition)
  TP = conf_matrix[2, 2]
  FN = conf_matrix[1, 2]
  sensitivity = TP/(TP+FN)
  TN = conf_matrix[1, 1]
  FP = conf_matrix[2, 1]
  specificity = TN/(TN+FP)
  precision = TP/(TP+FP)
  sensitivity_all[i] = sensitivity
  specificity_all[i] = specificity
  precision_all[i] = precision
}

mean(accuracy_all) #average overall accuracy
mean(auc_all) #average overall auc
mean(sensitivity_all)
mean(specificity_all)
mean(precision_all)
