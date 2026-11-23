# Compares the explanations the three models give.
#
#   Part A  SHAP values for all three models
#   Part B  agreement between models
#   Part C  control for correlated features
#   Part D  stability of a model's own explanations
#
# Exact tree-based methods are used rather than kernel SHAP,
# which required roughly four hours for the random forest.
#
# Input:  model_results.rds
# Output: table_shap_importance.csv, table_shap_agreement.csv,
#         table_shap_agreement_grouped.csv,
#         table_shap_stability.csv, shap_values.rds

library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(xgboost)
library(treeshap)
library(pROC)

set.seed(123)

fit <- readRDS("model_results.rds")

N_EXPLAIN <- 300

# PART A: SHAP values
# All three models are refitted on the same numeric design
# matrix so that each is explained over an identical feature
# space, which is what makes the explanations comparable.

x_train <- as.data.frame(fit$x_train)
x_test  <- as.data.frame(fit$x_test)
y_train <- fit$y_train
y_test  <- fit$y_test

names(x_train) <- make.names(names(x_train))
names(x_test)  <- make.names(names(x_test))

cat("Design matrix:", ncol(x_train), "features\n")

best_lambda <- fit$model_lr$bestTune$lambda
best_mtry   <- fit$model_rf$bestTune$mtry
best_node   <- fit$model_rf$bestTune$min.node.size
best_xgb    <- fit$xgb_tuning[which.max(fit$xgb_tuning$CV_AUC), ]

set.seed(123)
m_lr <- glmnet(as.matrix(x_train), y_train, family = "binomial",
               alpha = 1, lambda = best_lambda)

set.seed(123)
m_rf <- ranger(x = x_train, y = factor(y_train),
               num.trees = 500, mtry = best_mtry,
               min.node.size = best_node, probability = TRUE)

set.seed(123)
m_xgb <- xgb.train(
  params = list(objective = "binary:logistic", eval_metric = "auc",
                max_depth = best_xgb$max_depth, eta = best_xgb$eta,
                subsample = 0.8, colsample_bytree = 0.8,
                min_child_weight = 1),
  data = xgb.DMatrix(data = as.matrix(x_train), label = y_train),
  nrounds = best_xgb$nrounds, verbose = 0)


# Check the refitted models reproduce the evaluated ones 

auc_of <- function(y, p) {
  as.numeric(auc(roc(response = y, predictor = p,
                     levels = c(0, 1), direction = "<", quiet = TRUE)))
}

check <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Refitted_AUC = round(c(
    auc_of(y_test, as.numeric(predict(m_lr, as.matrix(x_test),
                                      type = "response"))),
    auc_of(y_test, predict(m_rf, x_test)$predictions[, "1"]),
    auc_of(y_test, predict(m_xgb, xgb.DMatrix(data = as.matrix(x_test))))
  ), 4),
  Script02_AUC = c(0.7977, 0.7949, 0.7990)
)
cat("\n Refit check \n")
print(check, row.names = FALSE)


# Borrowers to explain 

set.seed(123)
explain_rows <- sample(1:nrow(x_test), N_EXPLAIN)
X_explain    <- x_test[explain_rows, ]

cat("\nExplaining", N_EXPLAIN, "borrowers\n")


# SHAP for XGBoost: exact, built into xgboost

contrib_xgb <- predict(m_xgb, xgb.DMatrix(data = as.matrix(X_explain)),
                       predcontrib = TRUE)
shap_xgb <- contrib_xgb[, colnames(contrib_xgb) != "BIAS"]
cat("XGBoost SHAP done\n")


# SHAP for random forest: exact TreeSHAP 

unified_rf <- ranger.unify(m_rf, x_train)
ts_rf      <- treeshap(unified_rf, X_explain, verbose = FALSE)
shap_rf    <- as.matrix(ts_rf$shaps)
cat("Random forest SHAP done\n")


# SHAP for logistic regression: closed form 
# For a linear model the Shapley value of a feature is exactly
# its coefficient times the deviation from the background mean.

beta <- as.numeric(coef(m_lr))[-1]
names(beta) <- rownames(coef(m_lr))[-1]
beta <- beta[names(X_explain)]
beta[is.na(beta)] <- 0

background_mean <- colMeans(x_train)[names(X_explain)]

shap_lr <- sweep(as.matrix(X_explain), 2, background_mean, "-")
shap_lr <- sweep(shap_lr, 2, beta, "*")
cat("Logistic regression SHAP done\n")


# Align columns across the three models 

common <- Reduce(intersect, list(colnames(shap_lr), colnames(shap_rf), colnames(shap_xgb)))

S <- list(LR  = shap_lr[,  common, drop = FALSE],
          RF  = shap_rf[,  common, drop = FALSE],
          XGB = shap_xgb[, common, drop = FALSE])

feature_names <- common
cat("SHAP matrices aligned on", length(feature_names), "features\n")


# PART B: agreement between models

global_importance <- function(m) colMeans(abs(m))

imp <- data.frame(
  Feature = feature_names,
  LR      = round(global_importance(S$LR), 5),
  RF      = round(global_importance(S$RF), 5),
  XGB     = round(global_importance(S$XGB), 5)
)
imp <- imp[order(-imp$XGB), ]

cat("\n Global SHAP importance, top 12 features \n")
print(head(imp, 12), row.names = FALSE)
write.csv(imp, "table_shap_importance.csv", row.names = FALSE)

top_n_features <- function(m, n = 10) {
  names(sort(global_importance(m), decreasing = TRUE))[1:n]
}

agreement <- function(a, b, label) {
  
  # Rank correlation across the union of each model's top ten.
  # Features with negligible attribution have arbitrary ranks,
  # so including all of them would overstate disagreement.
  top_u <- union(top_n_features(S[[a]], 10), top_n_features(S[[b]], 10))
  
  rho <- cor(global_importance(S[[a]])[top_u],
             global_importance(S[[b]])[top_u],
             method = "spearman")
  
  overlap3 <- length(intersect(top_n_features(S[[a]], 3),
                               top_n_features(S[[b]], 3)))
  
  # per borrower: the same single largest contributor?
  same_top <- mean(apply(abs(S[[a]]), 1, which.max) ==
                     apply(abs(S[[b]]), 1, which.max))
  
  data.frame(Comparison       = label,
             Rank_correlation = round(rho, 3),
             Top3_overlap     = paste0(overlap3, "/3"),
             Same_top_driver  = paste0(round(same_top * 100, 1), "%"))
}

agree <- rbind(
  agreement("LR",  "RF",  "LR vs RF"),
  agreement("LR",  "XGB", "LR vs XGB"),
  agreement("RF",  "XGB", "RF vs XGB")
)

cat("\n Agreement between models \n")
print(agree, row.names = FALSE)
write.csv(agree, "table_shap_agreement.csv", row.names = FALSE)


# PART C: control for correlated features
# SHAP divides credit arbitrarily between features carrying the
# same information. Attribution is summed within correlated
# groups and the comparison repeated, so that disagreement can
# be distinguished from an artefact of redundancy.

cmat <- cor(x_train[, feature_names], use = "pairwise.complete.obs")

group_of <- setNames(feature_names, feature_names)
for (i in seq_along(feature_names)) {
  for (j in seq_along(feature_names)) {
    if (i < j) {
      fi <- feature_names[i]; fj <- feature_names[j]
      if (!is.na(cmat[fi, fj]) && abs(cmat[fi, fj]) > 0.8) {
        group_of[fj] <- group_of[fi]
      }
    }
  }
}

cat("\nFeature groups:", length(unique(group_of)),
    "from", length(feature_names), "features\n")
merged <- group_of[group_of != names(group_of)]
if (length(merged) > 0) {
  cat("Merged into another feature:\n")
  print(data.frame(Feature = names(merged),
                   Grouped_with = as.character(merged)), row.names = FALSE)
}

collapse <- function(m) t(rowsum(t(m), group = group_of[colnames(m)]))
Sg <- lapply(S, collapse)

agreement_grouped <- function(a, b, label) {
  ga <- colMeans(abs(Sg[[a]])); gb <- colMeans(abs(Sg[[b]]))
  k <- min(10, length(ga))
  top_u <- union(names(sort(ga, decreasing = TRUE))[1:k],
                 names(sort(gb, decreasing = TRUE))[1:k])
  rho <- cor(ga[top_u], gb[top_u], method = "spearman")
  same_top <- mean(apply(abs(Sg[[a]]), 1, which.max) ==
                     apply(abs(Sg[[b]]), 1, which.max))
  data.frame(Comparison       = label,
             Rank_correlation = round(rho, 3),
             Same_top_driver  = paste0(round(same_top * 100, 1), "%"))
}

agree_grouped <- rbind(
  agreement_grouped("LR",  "RF",  "LR vs RF"),
  agreement_grouped("LR",  "XGB", "LR vs XGB"),
  agreement_grouped("RF",  "XGB", "RF vs XGB")
)

cat("\n Agreement after grouping correlated features \n")
print(agree_grouped, row.names = FALSE)
write.csv(agree_grouped, "table_shap_agreement_grouped.csv", row.names = FALSE)


# PART D: stability of a model's own explanations
# Each model is retrained on three further partitions and its
# explanations compared with itself on the same borrowers. This
# establishes how much disagreement retraining alone produces,
# providing a reference point for the between-model figures.

pool_x <- rbind(x_train, x_test)
pool_y <- c(y_train, y_test)

shap_by_seed <- list()

for (s in 101:103) {
  
  set.seed(s)
  idx <- createDataPartition(factor(pool_y), p = 0.70, list = FALSE)
  xtr <- pool_x[idx, ]
  ytr <- pool_y[idx]
  
  set.seed(s)
  rf_s <- ranger(x = xtr, y = factor(ytr), num.trees = 500,
                 mtry = best_mtry, min.node.size = best_node,
                 probability = TRUE)
  
  set.seed(s)
  xgb_s <- xgb.train(
    params = list(objective = "binary:logistic", eval_metric = "auc",
                  max_depth = best_xgb$max_depth, eta = best_xgb$eta,
                  subsample = 0.8, colsample_bytree = 0.8,
                  min_child_weight = 1),
    data = xgb.DMatrix(data = as.matrix(xtr), label = ytr),
    nrounds = best_xgb$nrounds, verbose = 0)
  
  set.seed(s)
  lr_s <- glmnet(as.matrix(xtr), ytr, family = "binomial",
                 alpha = 1, lambda = best_lambda)
  
  # random forest
  ts <- treeshap(ranger.unify(rf_s, xtr), X_explain, verbose = FALSE)
  shap_by_seed[[paste0("RF_", s)]] <-
    as.matrix(ts$shaps)[, common, drop = FALSE]
  
  # xgboost
  cb <- predict(xgb_s, xgb.DMatrix(data = as.matrix(X_explain)),
                predcontrib = TRUE)
  shap_by_seed[[paste0("XGB_", s)]] <-
    cb[, colnames(cb) != "BIAS"][, common, drop = FALSE]
  
  # logistic regression
  b <- as.numeric(coef(lr_s))[-1]
  names(b) <- rownames(coef(lr_s))[-1]
  b <- b[names(X_explain)]; b[is.na(b)] <- 0
  bm <- colMeans(xtr)[names(X_explain)]
  sl <- sweep(as.matrix(X_explain), 2, bm, "-")
  shap_by_seed[[paste0("LR_", s)]] <-
    sweep(sl, 2, b, "*")[, common, drop = FALSE]
  
  cat("stability seed", s, "done\n")
}

stability <- data.frame()
seeds <- c(101, 102, 103)

for (mod in c("LR", "RF", "XGB")) {
  vals <- c()
  for (i in 1:2) for (j in (i + 1):3) {
    A  <- shap_by_seed[[paste0(mod, "_", seeds[i])]]
    Bm <- shap_by_seed[[paste0(mod, "_", seeds[j])]]
    vals <- c(vals, mean(apply(abs(A), 1, which.max) ==
                           apply(abs(Bm), 1, which.max)))
  }
  stability <- rbind(stability, data.frame(
    Model = mod,
    Same_top_driver_across_retrains = paste0(round(mean(vals) * 100, 1), "%")))
}

cat("\n Stability of each model's own explanations \n")
print(stability, row.names = FALSE)
write.csv(stability, "table_shap_stability.csv", row.names = FALSE)


# Save 
saveRDS(list(S = S, Sg = Sg, X_explain = X_explain,
             explain_rows = explain_rows, group_of = group_of,
             shap_by_seed = shap_by_seed,
             x_train = x_train, x_test = x_test,
             y_train = y_train, y_test = y_test),
        "shap_values.rds")

cat("\nSHAP analysis complete.\n")


sv <- readRDS("shap_values.rds")
for (p in list(c("LR","RF"), c("LR","XGB"), c("RF","XGB"))) {
  k <- sum(apply(abs(sv$S[[p[1]]]),1,which.max) ==
             apply(abs(sv$S[[p[2]]]),1,which.max))
  ci <- binom.test(k, 300)$conf.int
  cat(p[1], "vs", p[2], ": ", round(k/300*100,1), "% [",
      round(ci[1]*100,1), ",", round(ci[2]*100,1), "]\n")
}

