# Tests whether the model comparison is stable.
#
#   Part A  bootstrap confidence intervals and equivalence
#   Part B  repeated train/test splits
#   Part C  sensitivity to data preparation
#
# Input:  model_results.rds, heloc_prepared.rds,
#         heloc_dataset_v1 (1).csv
# Output: table_auc_intervals.csv, table_equivalence.csv,
#         table_repeated_splits.csv, table_repeated_summary.csv,
#         table_sensitivity.csv, table_sensitivity_raw.csv

library(tidyverse)
library(caret)
library(glmnet)
library(ranger)
library(xgboost)
library(pROC)

set.seed(123)

fit <- readRDS("model_results.rds")

MARGIN <- 0.01   # equivalence margin, in AUC

auc_of <- function(y, p) {
  as.numeric(auc(roc(response = y, predictor = p,
                     levels = c("Good", "Bad"),
                     direction = "<", quiet = TRUE)))
}


# PART A: bootstrap confidence intervals on the test set
# The AUC difference between each pair of models is resampled
# 2,000 times. A 90 per cent interval is used because the
# equivalence test is two one-sided tests at 5 per cent each.

probs  <- fit$probs
actual <- probs$Actual
n_test <- nrow(probs)

set.seed(123)
B <- 2000
boot_auc <- data.frame(LR = numeric(B), RF = numeric(B), XGB = numeric(B))

for (b in 1:B) {
  i <- sample(1:n_test, n_test, replace = TRUE)
  boot_auc$LR[b]  <- auc_of(actual[i], probs$LR[i])
  boot_auc$RF[b]  <- auc_of(actual[i], probs$RF[i])
  boot_auc$XGB[b] <- auc_of(actual[i], probs$XGB[i])
}

cat("Bootstrap complete.\n")

auc_table <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  AUC   = round(c(auc_of(actual, probs$LR),
                  auc_of(actual, probs$RF),
                  auc_of(actual, probs$XGB)), 4),
  Lower = round(as.numeric(sapply(boot_auc, quantile, 0.05)), 4),
  Upper = round(as.numeric(sapply(boot_auc, quantile, 0.95)), 4)
)
cat(" Test AUC with 90% bootstrap interval \n")
print(auc_table, row.names = FALSE)

pairs <- list(c("RF", "LR"), c("XGB", "LR"), c("XGB", "RF"))
equiv <- data.frame()

for (p in pairs) {
  d  <- boot_auc[[p[1]]] - boot_auc[[p[2]]]
  lo <- as.numeric(quantile(d, 0.05))
  hi <- as.numeric(quantile(d, 0.95))
  
  equiv <- rbind(equiv, data.frame(
    Comparison = paste(p[1], "-", p[2]),
    Difference = round(mean(d), 4),
    Lower_90   = round(lo, 4),
    Upper_90   = round(hi, 4),
    Equivalent = ifelse(lo > -MARGIN & hi < MARGIN, "Yes", "No"),
    Differs    = ifelse(lo > 0 | hi < 0, "Yes", "No")
  ))
}

cat("\n Pairwise AUC differences, equivalence margin +/-",
    MARGIN, "---\n")
print(equiv, row.names = FALSE)

write.csv(auc_table, "table_auc_intervals.csv", row.names = FALSE)
write.csv(equiv, "table_equivalence.csv", row.names = FALSE)

# PART B: repeated train/test splits
# Hyperparameters are held at their tuned values so that the
# variation observed reflects the partition, not the tuning.

heloc <- readRDS("heloc_prepared.rds")
model_data <- heloc[, setdiff(names(heloc), "Target")]
model_data$RiskPerformance <- factor(model_data$RiskPerformance,
                                     levels = c("Bad", "Good"))

best_lambda <- fit$model_lr$bestTune$lambda
best_mtry   <- fit$model_rf$bestTune$mtry
best_node   <- fit$model_rf$bestTune$min.node.size
best_xgb    <- fit$xgb_tuning[which.max(fit$xgb_tuning$CV_AUC), ]

no_cv <- trainControl(method = "none", classProbs = TRUE)

# Fits all three models on one split and returns their test AUCs.
run_split <- function(data, seed) {
  
  set.seed(seed)
  idx <- createDataPartition(data$RiskPerformance, p = 0.70, list = FALSE)
  tr  <- data[idx, ]
  te  <- data[-idx, ]
  
  # impute with training medians
  num_vars <- names(tr)[sapply(tr, is.numeric)]
  meds <- sapply(tr[num_vars], median, na.rm = TRUE)
  for (v in num_vars) {
    tr[[v]][is.na(tr[[v]])] <- meds[v]
    te[[v]][is.na(te[[v]])] <- meds[v]
  }
  
  m_lr <- train(RiskPerformance ~ ., data = tr, method = "glmnet",
                family = "binomial", trControl = no_cv,
                tuneGrid = data.frame(alpha = 1, lambda = best_lambda))
  
  m_rf <- train(RiskPerformance ~ ., data = tr, method = "ranger",
                trControl = no_cv, num.trees = 500,
                tuneGrid = data.frame(mtry = best_mtry,
                                      splitrule = "gini",
                                      min.node.size = best_node))
  
  # xgboost needs a numeric matrix
  both <- rbind(tr, te)
  mm   <- model.matrix(RiskPerformance ~ . - 1, data = both)
  xtr  <- mm[1:nrow(tr), ]
  xte  <- mm[(nrow(tr) + 1):nrow(both), ]
  ytr  <- ifelse(tr$RiskPerformance == "Bad", 1, 0)
  
  set.seed(seed)
  m_xgb <- xgb.train(
    params = list(objective = "binary:logistic", eval_metric = "auc",
                  max_depth = best_xgb$max_depth, eta = best_xgb$eta,
                  subsample = 0.8, colsample_bytree = 0.8,
                  min_child_weight = 1),
    data = xgb.DMatrix(data = xtr, label = ytr),
    nrounds = best_xgb$nrounds, verbose = 0)
  
  data.frame(
    seed = seed,
    LR   = auc_of(te$RiskPerformance, predict(m_lr, te, type = "prob")[, "Bad"]),
    RF   = auc_of(te$RiskPerformance, predict(m_rf, te, type = "prob")[, "Bad"]),
    XGB  = auc_of(te$RiskPerformance, predict(m_xgb, xgb.DMatrix(data = xte)))
  )
}

cat("\nRunning 20 repeated splits. This will take some time.\n")

repeated <- data.frame()
for (s in 1:20) {
  repeated <- rbind(repeated, run_split(model_data, s))
  cat("  split", s, "done\n")
}

cat("\n AUC across 20 repeated splits \n")
summary_splits <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Mean  = round(as.numeric(sapply(repeated[c("LR","RF","XGB")], mean)), 4),
  SD    = round(as.numeric(sapply(repeated[c("LR","RF","XGB")], sd)), 4),
  Min   = round(as.numeric(sapply(repeated[c("LR","RF","XGB")], min)), 4),
  Max   = round(as.numeric(sapply(repeated[c("LR","RF","XGB")], max)), 4)
)
print(summary_splits, row.names = FALSE)

winner <- apply(repeated[c("LR","RF","XGB")], 1, function(r) names(which.max(r)))
cat("\nHighest AUC per split:\n")
print(table(winner))

cat("\nSD across splits within a model :",
    round(mean(sapply(repeated[c("LR","RF","XGB")], sd)), 4), "\n")
cat("SD across models within a split :",
    round(mean(apply(repeated[c("LR","RF","XGB")], 1, sd)), 4), "\n")

write.csv(repeated, "table_repeated_splits.csv", row.names = FALSE)
write.csv(summary_splits, "table_repeated_summary.csv", row.names = FALSE)


# PART C: sensitivity to data preparation
# Three specifications are compared. If the gap between models
# changes with preparation, then preparation, not model capacity,
# is driving part of the difference.

raw <- read.csv("heloc_dataset_v1 (1).csv")
preds_raw <- setdiff(names(raw), "RiskPerformance")

# The two delinquency variables are categorical: their numbers
# label states rather than measure them. A sentinel is therefore
# another category for these, not a missing value.
ordinal_vars <- c("MaxDelqEver", "MaxDelq2PublicRecLast12M")

make_variant <- function(variant) {
  
  d <- raw
  
  # keep or drop the no-bureau-record rows
  if (variant != "Keep no-record rows") {
    no_rec <- apply(d[preds_raw], 1, function(x) all(x == -9))
    d <- d[!no_rec, ]
  }
  d <- d[!duplicated(d), ]
  
  if (variant == "Sentinels left raw") {
    # naive treatment: codes kept as negative numbers, no
    # indicators, delinquency codes treated as numeric
    d$RiskPerformance <- factor(d$RiskPerformance, levels = c("Bad", "Good"))
    return(d)
  }
  
  # baseline treatment: indicators, then NA for numeric predictors
  for (v in preds_raw) {
    for (code in c(-7, -8, -9)) {
      if (sum(d[[v]] == code) > 0) {
        d[[paste0("Flag_", v, "_", abs(code))]] <- ifelse(d[[v]] == code, 1, 0)
      }
    }
  }
  
  numeric_preds <- setdiff(preds_raw, ordinal_vars)
  for (v in numeric_preds) d[[v]][d[[v]] %in% c(-7, -8, -9)] <- NA
  
  d$MaxDelqEver              <- as.factor(d$MaxDelqEver)
  d$MaxDelq2PublicRecLast12M <- as.factor(d$MaxDelq2PublicRecLast12M)
  d$RiskPerformance <- factor(d$RiskPerformance, levels = c("Bad", "Good"))
  d
}

variants <- c("Baseline", "Sentinels left raw", "Keep no-record rows")
sensitivity <- data.frame()

for (v in variants) {
  dv <- make_variant(v)
  cat("\n", v, ": ", nrow(dv), " rows, ", ncol(dv) - 1, " predictors\n", sep = "")
  
  for (s in 1:5) {
    r <- run_split(dv, s)
    r$Variant <- v
    sensitivity <- rbind(sensitivity, r)
  }
  cat("  five splits complete\n")
}

sens_summary <- sensitivity %>%
  group_by(Variant) %>%
  summarise(LR  = round(mean(LR), 4),
            RF  = round(mean(RF), 4),
            XGB = round(mean(XGB), 4),
            .groups = "drop") %>%
  mutate(RF_minus_LR  = round(RF - LR, 4),
         XGB_minus_LR = round(XGB - LR, 4))

cat("\n Mean AUC by preparation (five splits each)\n")
print(as.data.frame(sens_summary), row.names = FALSE)

write.csv(sensitivity, "table_sensitivity_raw.csv", row.names = FALSE)
write.csv(sens_summary, "table_sensitivity.csv", row.names = FALSE)

cat("\nRobustness analysis complete.\n")
sessionInfo()

sens <- read.csv("table_sensitivity_raw.csv")

sens %>%
  select(seed, Variant, LR, XGB) %>%
  mutate(gap = XGB - LR) %>%
  select(seed, Variant, gap) %>%
  pivot_wider(names_from = Variant, values_from = gap)


sens %>%
  mutate(gap = XGB - LR) %>%
  select(seed, Variant, gap) %>%
  pivot_wider(names_from = Variant, values_from = gap) %>%
  mutate(inflation = `Sentinels left raw` - Baseline) %>%
  as.data.frame() %>%
  print(row.names = FALSE)

sens %>%
  mutate(gap = XGB - LR) %>%
  select(seed, Variant, gap) %>%
  pivot_wider(names_from = Variant, values_from = gap) %>%
  mutate(inflation = `Sentinels left raw` - Baseline) %>%
  summarise(mean = round(mean(inflation), 4),
            sd   = round(sd(inflation), 4),
            min  = round(min(inflation), 4),
            max  = round(max(inflation), 4)) %>%
  as.data.frame()
