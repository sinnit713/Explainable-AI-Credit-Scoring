# Prepares the FICO HELOC data for modelling.
# Output: heloc_prepared.rds  (used by 02_models_and_metrics.R)

library(tidyverse)

set.seed(123)

# Import Dataset 
heloc <- read.csv(file.choose("heloc_dataset_v1 (1).csv"))   # select heloc_dataset_v1_1.csv

cat("Rows imported:", nrow(heloc), "\n")
cat("Columns imported:", ncol(heloc), "\n")

predictors <- setdiff(names(heloc), "RiskPerformance")


# Exclusion 1: applicants with no bureau record 
# A -9 in every predictor means no bureau record exists for the
# applicant, so the row carries no information for modelling.

no_record <- apply(heloc[predictors], 1, function(x) all(x == -9))

cat("\nRecords with -9 in all 23 predictors:", sum(no_record), "\n")
cat("Bad rate within those records:",
    round(mean(heloc$RiskPerformance[no_record] == "Bad"), 3), "\n")

heloc <- heloc[!no_record, ]
cat("Rows remaining after exclusion:", nrow(heloc), "\n")


# Exclusion 2: duplicate records
# Most duplicates were the identical no-record rows removed above.
# Only genuine repeats should remain at this point.

cat("\nDuplicates remaining:", sum(duplicated(heloc)), "\n")

heloc <- heloc[!duplicated(heloc), ]
cat("Analysis sample:", nrow(heloc), "rows\n")
cat("Bad rate:", round(mean(heloc$RiskPerformance == "Bad") * 100, 2), "%\n")


# Evidence for the treatment of special codes 
# -7 and -8 are defined bureau conditions, not missing values.
# This table reports the Bad rate for each code so the choice of
# treatment can be justified rather than assumed.

special_evidence <- data.frame()

for (v in predictors) {
  for (code in c(-7, -8, -9)) {
    n_code <- sum(heloc[[v]] == code)
    if (n_code > 0) {
      special_evidence <- rbind(
        special_evidence,
        data.frame(
          Variable  = v,
          Code      = code,
          N         = n_code,
          Bad_rate  = round(mean(heloc$RiskPerformance[heloc[[v]] == code] == "Bad"), 3)
        )
      )
    }
  }
}

# Bad rate among ordinary (non-coded) values, for comparison
special_evidence$Bad_rate_valid <- sapply(
  special_evidence$Variable,
  function(v) round(mean(heloc$RiskPerformance[heloc[[v]] >= 0] == "Bad"), 3)
)

cat("\n--- Bad rate by special code (overall Bad rate",
    round(mean(heloc$RiskPerformance == "Bad"), 3), ") ---\n")
print(special_evidence)


# Binary indicators for each variable-and-code combination 
# The codes do not point in a consistent direction, so a single
# flag per variable would lose information. One flag is created
# for each combination that actually occurs in the data.

heloc <- heloc %>%
  mutate(
    Flag_MSinceMostRecentDelq_7          = ifelse(MSinceMostRecentDelq == -7, 1, 0),
    Flag_MSinceMostRecentDelq_8          = ifelse(MSinceMostRecentDelq == -8, 1, 0),
    Flag_MSinceMostRecentInq_7           = ifelse(MSinceMostRecentInqexcl7days == -7, 1, 0),
    Flag_MSinceMostRecentInq_8           = ifelse(MSinceMostRecentInqexcl7days == -8, 1, 0),
    Flag_NetFractionInstallBurden_8      = ifelse(NetFractionInstallBurden == -8, 1, 0),
    Flag_NumInstallTradesWBalance_8      = ifelse(NumInstallTradesWBalance == -8, 1, 0),
    Flag_NumBank2NatlHighUtil_8          = ifelse(NumBank2NatlTradesWHighUtilization == -8, 1, 0),
    Flag_MSinceOldestTradeOpen_8         = ifelse(MSinceOldestTradeOpen == -8, 1, 0),
    Flag_NetFractionRevolvingBurden_8    = ifelse(NetFractionRevolvingBurden == -8, 1, 0),
    Flag_NumRevolvingTradesWBalance_8    = ifelse(NumRevolvingTradesWBalance == -8, 1, 0),
    Flag_PercentTradesWBalance_8         = ifelse(PercentTradesWBalance == -8, 1, 0),
    Flag_ExternalRiskEstimate_9          = ifelse(ExternalRiskEstimate == -9, 1, 0)
  )

cat("\nIndicator columns created:", sum(grepl("^Flag_", names(heloc))), "\n")


# Set special codes to NA
# The information they carried is now held in the indicators.
# Imputation happens after the train/test split in script 02,
# so that only training data informs the imputed values.

heloc <- heloc %>%
  mutate(across(all_of(predictors), ~ ifelse(. %in% c(-7, -8, -9), NA, .)))

cat("Missing values now present:", sum(is.na(heloc)), "\n")


# Ordinal delinquency codes 
# These two variables label delinquency states; the numbers are
# category labels, not quantities, so they are treated as factors.

cat("\nBad rate by MaxDelqEver code:\n")
print(heloc %>% count(MaxDelqEver) %>%
        mutate(Bad_rate = round(sapply(MaxDelqEver, function(k)
          mean(heloc$RiskPerformance[heloc$MaxDelqEver == k] == "Bad")), 3)))

heloc <- heloc %>%
  mutate(
    MaxDelqEver              = as.factor(MaxDelqEver),
    MaxDelq2PublicRecLast12M = as.factor(MaxDelq2PublicRecLast12M)
  )


# Encode the target 
# Bad is the event of interest and is coded 1.

heloc <- heloc %>%
  mutate(Target = ifelse(RiskPerformance == "Bad", 1, 0))

cat("\nTarget coded. Event rate:", round(mean(heloc$Target), 4), "\n")


# Save 
saveRDS(heloc, "heloc_prepared.rds")
write.csv(special_evidence, "table_special_code_evidence.csv", row.names = FALSE)

cat("\nPrepared dataset saved.\n")
cat("Final dimensions:", nrow(heloc), "rows,", ncol(heloc), "columns\n")
