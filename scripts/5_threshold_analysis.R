# Translates model performance into lending decisions.
#
# At a matched approval rate, do the models admit different
# borrowers, and does the quality of the accepted book differ?
#
# Input:  model_results.rds
# Output: table_approval_rates.csv, table_decision_overlap.csv

library(tidyverse)

set.seed(123)

fit    <- readRDS("model_results.rds")
probs  <- fit$probs
actual <- ifelse(probs$Actual == "Bad", 1, 0)
n      <- nrow(probs)

cat("Test applicants:", n, "\n")
cat("Bad rate in the population:", round(mean(actual), 4), "\n")

# Bad rate among approved, at matched approval rates
# An applicant is approved when the predicted probability of Bad
# falls below a threshold. The threshold is set separately for
# each model so that all three approve the same proportion,
# which makes the comparison a like-for-like one.

approval_rates <- c(0.50, 0.60, 0.70, 0.80, 0.90)
models <- c("LR", "RF", "XGB")

results <- data.frame()

for (r in approval_rates) {
  for (m in models) {
    p  <- probs[[m]]
    th <- quantile(p, r)              # threshold giving approval rate r
    approved <- p <= th
    
    results <- rbind(results, data.frame(
      Approval_rate     = paste0(r * 100, "%"),
      Model             = m,
      Threshold         = round(as.numeric(th), 4),
      N_approved        = sum(approved),
      Bad_rate_approved = round(mean(actual[approved]), 4),
      Bad_per_1000      = round(mean(actual[approved]) * 1000, 1)
    ))
  }
}

wide <- results %>%
  select(Approval_rate, Model, Bad_rate_approved) %>%
  pivot_wider(names_from = Model, values_from = Bad_rate_approved) %>%
  mutate(Spread = round(pmax(LR, RF, XGB) - pmin(LR, RF, XGB), 4))

cat(" Bad rate among approved applicants \n")
print(as.data.frame(wide), row.names = FALSE)

cat("\n Bad loans accepted per 1,000 applicants \n")
per1000 <- results %>%
  select(Approval_rate, Model, Bad_per_1000) %>%
  pivot_wider(names_from = Model, values_from = Bad_per_1000) %>%
  mutate(Difference = round(pmax(LR, RF, XGB) - pmin(LR, RF, XGB), 1))
print(as.data.frame(per1000), row.names = FALSE)

write.csv(results, "table_approval_rates.csv", row.names = FALSE)


# Do the models approve the same people?
# Equivalent aggregate performance does not require that the
# same individuals are accepted. This measures how far the three
# models agree at the level of the individual applicant.

overlap <- data.frame()

for (r in approval_rates) {
  
  decisions <- sapply(models, function(m) probs[[m]] <= quantile(probs[[m]], r))
  
  for (p in list(c("LR","RF"), c("LR","XGB"), c("RF","XGB"))) {
    a <- decisions[, p[1]]
    b <- decisions[, p[2]]
    
    overlap <- rbind(overlap, data.frame(
      Approval_rate = paste0(r * 100, "%"),
      Comparison    = paste(p[1], "vs", p[2]),
      Same_decision = paste0(round(mean(a == b) * 100, 1), "%"),
      Disagree_n    = sum(a != b)
    ))
  }
}

cat("\n Agreement on individual decision \n")
print(as.data.frame(
  overlap %>%
    select(Approval_rate, Comparison, Same_decision) %>%
    pivot_wider(names_from = Comparison, values_from = Same_decision)
), row.names = FALSE)

cat("\n Applicants receiving a different decision \n")
print(as.data.frame(
  overlap %>%
    select(Approval_rate, Comparison, Disagree_n) %>%
    pivot_wider(names_from = Comparison, values_from = Disagree_n)
), row.names = FALSE)

write.csv(overlap, "table_decision_overlap.csv", row.names = FALSE)

# The 70 per cent operating point in detail

cat("\n Detail at a 70 per cent approval rate \n")
detail <- results %>% filter(Approval_rate == "70%")
print(as.data.frame(detail), row.names = FALSE)

cat("\nDifference between best and worst model:",
    round(max(detail$Bad_per_1000) - min(detail$Bad_per_1000), 1),
    "bad loans per 1,000 applicants\n")

cat("\nThreshold analysis complete.\n")

