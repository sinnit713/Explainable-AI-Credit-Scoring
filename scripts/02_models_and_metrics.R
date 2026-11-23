# Trains the three models and evaluates them on the test set.
# Input:  heloc_prepared.rds  (from 01_data_preparation.R)
# Output: model_results.rds, table_performance.csv

library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(xgboost)
library(pROC)

set.seed(123)

heloc <- readRDS("heloc_prepared.rds")
cat("Loaded:", nrow(heloc), "rows,", ncol(heloc), "columns\n")


# Build the modelling frame 
# Target is dropped because it was derived from RiskPerformance
# and would leak the outcome into the predictors.

model_data <- heloc %>% select(-Target)

# caret needs the outcome as a factor. Bad is the event of
# interest and is placed first so it is the positive class.
model_data$RiskPerformance <- factor(model_data$RiskPerformance,
                                     levels = c("Bad", "Good"))

cat("Predictors:", ncol(model_data) - 1, "\n")

# Train / test split 
# Stratified so that the Bad rate is preserved in both parts.

train_index <- createDataPartition(model_data$RiskPerformance,
                                   p = 0.70, list = FALSE)

train_data <- model_data[train_index, ]
test_data  <- model_data[-train_index, ]

cat("\nTraining rows:", nrow(train_data),
    " Bad rate:", round(mean(train_data$RiskPerformance == "Bad"), 4), "\n")
cat("Test rows:    ", nrow(test_data),
    " Bad rate:", round(mean(test_data$RiskPerformance == "Bad"), 4), "\n")

# Impute using training medians only 
# Medians are calculated on the training data and applied to both
# sets, so no information from the test set influences preparation.

numeric_vars <- names(train_data)[sapply(train_data, is.numeric)]

train_medians <- sapply(train_data[numeric_vars], median, na.rm = TRUE)

for (v in numeric_vars) {
  train_data[[v]][is.na(train_data[[v]])] <- train_medians[v]
  test_data[[v]][is.na(test_data[[v]])]   <- train_medians[v]
}

cat("Missing values after imputation - train:", sum(is.na(train_data)),
    " test:", sum(is.na(test_data)), "\n")

# Cross-validation settings 
# Five-fold CV on the training data is used for all tuning.

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

# Model 1: Logistic regression (lasso) 

set.seed(123)
grid_lr <- expand.grid(alpha = 1,
                       lambda = 10^seq(-4, 0, length = 20))

model_lr <- train(RiskPerformance ~ ., data = train_data,
                  method    = "glmnet",
                  family    = "binomial",
                  trControl = ctrl,
                  tuneGrid  = grid_lr,
                  metric    = "ROC")

cat("\nLogistic regression tuned. Best lambda:",
    round(model_lr$bestTune$lambda, 5),
    " CV ROC:", round(max(model_lr$results$ROC), 4), "\n")

# Model 2: Random forest
set.seed(123)
grid_rf <- expand.grid(mtry          = c(4, 6, 8),
                       splitrule     = "gini",
                       min.node.size = c(5, 20))

model_rf <- train(RiskPerformance ~ ., data = train_data,
                  method     = "ranger",
                  trControl  = ctrl,
                  tuneGrid   = grid_rf,
                  num.trees  = 500,
                  importance = "impurity",
                  metric     = "ROC")

cat("Random forest tuned. Best mtry:", model_rf$bestTune$mtry,
    " min.node.size:", model_rf$bestTune$min.node.size,
    " CV ROC:", round(max(model_rf$results$ROC), 4), "\n")

# Model 3: XGBoost 
# Trees need numeric input, so the two factor variables are
# converted to dummy columns. The design matrix is built on the
# combined data so that train and test share identical columns,
# then split back using the original row index. No statistics
# are learned from the test set at this stage.

all_data <- rbind(train_data, test_data)

design_matrix <- model.matrix(RiskPerformance ~ . - 1, data = all_data)

x_train <- design_matrix[1:nrow(train_data), ]
x_test  <- design_matrix[(nrow(train_data) + 1):nrow(all_data), ]

y_train <- ifelse(train_data$RiskPerformance == "Bad", 1, 0)
y_test  <- ifelse(test_data$RiskPerformance  == "Bad", 1, 0)

cat("\nDesign matrix columns:", ncol(x_train), "\n")

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test,  label = y_test)

# Tune by five-fold cross-validation on the training data.
# The number of rounds is chosen by early stopping rather than
# being fixed in advance.

set.seed(123)
folds <- createFolds(factor(y_train), k = 5)

xgb_grid <- expand.grid(max_depth = c(2, 4, 6),
                        eta       = c(0.05, 0.10))

xgb_tuning <- data.frame()

for (i in 1:nrow(xgb_grid)) {
  
  params <- list(
    objective        = "binary:logistic",
    eval_metric      = "auc",
    max_depth        = xgb_grid$max_depth[i],
    eta              = xgb_grid$eta[i],
    subsample        = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1
  )
  
  set.seed(123)
  cv <- xgb.cv(params                = params,
               data                  = dtrain,
               nrounds               = 1000,
               folds                 = folds,
               early_stopping_rounds = 30,
               verbose               = 0)
  
  best_round <- which.max(cv$evaluation_log$test_auc_mean)
  
  xgb_tuning <- rbind(xgb_tuning, data.frame(
    max_depth = xgb_grid$max_depth[i],
    eta       = xgb_grid$eta[i],
    nrounds   = best_round,
    CV_AUC    = round(cv$evaluation_log$test_auc_mean[best_round], 4)
  ))
  
  cat("depth", xgb_grid$max_depth[i],
      " eta", xgb_grid$eta[i],
      " rounds", best_round,
      " CV AUC", round(cv$evaluation_log$test_auc_mean[best_round], 4), "\n")
}

best <- xgb_tuning[which.max(xgb_tuning$CV_AUC), ]

cat("\nBest XGBoost setting: depth", best$max_depth,
    " eta", best$eta, " nrounds", best$nrounds,
    " CV AUC", best$CV_AUC, "\n")


# Fit the final model on the full training data.

final_params <- list(
  objective        = "binary:logistic",
  eval_metric      = "auc",
  max_depth        = best$max_depth,
  eta              = best$eta,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 1
)

set.seed(123)
model_xgb <- xgb.train(params  = final_params,
                       data    = dtrain,
                       nrounds = best$nrounds,
                       verbose = 0)

# Predicted probabilities of Bad 
# The test set is used once, now that tuning is complete.

prob_lr  <- predict(model_lr,  test_data, type = "prob")[, "Bad"]
prob_rf  <- predict(model_rf,  test_data, type = "prob")[, "Bad"]
prob_xgb <- predict(model_xgb, dtest)

# Evaluate on the test set 
get_metrics <- function(prob_bad, model_name) {
  
  pred <- factor(ifelse(prob_bad >= 0.50, "Bad", "Good"),
                 levels = c("Bad", "Good"))
  
  cm <- confusionMatrix(pred, test_data$RiskPerformance, positive = "Bad")
  
  roc_obj <- roc(response  = test_data$RiskPerformance,
                 predictor = prob_bad,
                 levels    = c("Good", "Bad"),
                 direction = "<",
                 quiet     = TRUE)
  
  data.frame(
    Model     = model_name,
    Accuracy  = round(as.numeric(cm$overall["Accuracy"]), 4),
    Precision = round(as.numeric(cm$byClass["Precision"]), 4),
    Recall    = round(as.numeric(cm$byClass["Recall"]), 4),
    F1        = round(as.numeric(cm$byClass["F1"]), 4),
    ROC_AUC   = round(as.numeric(auc(roc_obj)), 4)
  )
}

results <- rbind(
  get_metrics(prob_lr,  "Logistic Regression"),
  get_metrics(prob_rf,  "Random Forest"),
  get_metrics(prob_xgb, "XGBoost")
)

results$CV_AUC <- round(c(max(model_lr$results$ROC),
                          max(model_rf$results$ROC),
                          best$CV_AUC), 4)

cat("\n Test set performance (threshold 0.50, n =",
    nrow(test_data), ")\n")
print(results)

# Save
probs <- data.frame(
  LR     = prob_lr,
  RF     = prob_rf,
  XGB    = prob_xgb,
  Actual = test_data$RiskPerformance
)

saveRDS(list(model_lr = model_lr, model_rf = model_rf, model_xgb = model_xgb,
             train_data = train_data, test_data = test_data,
             x_train = x_train, x_test = x_test,
             y_train = y_train, y_test = y_test,
             train_medians = train_medians,
             xgb_tuning = xgb_tuning, probs = probs),
        "model_results.rds")

write.csv(results, "table_performance.csv", row.names = FALSE)
write.csv(xgb_tuning, "table_xgb_tuning.csv", row.names = FALSE)
cat("\nModels and results saved.\n")
