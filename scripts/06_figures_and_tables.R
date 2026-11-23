# Input:  model_results.rds, shap_values.rds,
#         table_*.csv from scripts 02, 03 and 05
#         table_equivalence.csv (from 03_robustness.R)
# Output: figure5_roc.png , figure8_local.png, 
#         figure6_equivalence.png
#         table_local_example.csv

library(tidyverse)
library(pROC)

set.seed(123)

fit <- readRDS("model_results.rds")
sv  <- readRDS("shap_values.rds")

# a consistent palette, as used in the proposal
cols <- c("Logistic Regression" = "#4E79A7",
          "Random Forest"       = "#E15759",
          "XGBoost"             = "#59A14F")

my_theme <- theme_minimal(base_size = 12) +
  theme(plot.title    = element_text(face = "bold", size = 13),
        axis.title    = element_text(face = "bold"),
        axis.text     = element_text(colour = "black"),
        panel.grid.minor = element_blank(),
        legend.position  = "bottom",
        legend.title     = element_blank())

# FIGURE : ROC curves

probs  <- fit$probs
actual <- probs$Actual

roc_df <- bind_rows(lapply(
  list(c("LR","Logistic Regression"),
       c("RF","Random Forest"),
       c("XGB","XGBoost")),
  function(m) {
    r <- roc(actual, probs[[m[1]]], levels = c("Good","Bad"),
             direction = "<", quiet = TRUE)
    data.frame(FPR = 1 - r$specificities,
               TPR = r$sensitivities,
               Model = m[2])
  }))

fig5 <- ggplot(roc_df, aes(FPR, TPR, colour = Model)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed",
              colour = "grey60") +
  geom_line(linewidth = 0.9) +
  scale_colour_manual(values = cols) +
  coord_equal() +
  labs(title = "Receiver operating characteristic curves",
       x = "False positive rate", y = "True positive rate") +
  my_theme

ggsave("figure5_roc.png", fig5, width = 6, height = 6, dpi = 300)
cat("Figure 5 saved\n")

# FIGURE 6: AUC across the twenty partitions

rep_df <- read.csv("table_repeated_splits.csv") %>%
  pivot_longer(c(LR, RF, XGB), names_to = "Model", values_to = "AUC") %>%
  mutate(Model = recode(Model, LR = "Logistic Regression",
                        RF = "Random Forest", XGB = "XGBoost"))

fig6 <- ggplot(rep_df, aes(x = factor(seed), y = AUC,
                           colour = Model, group = Model)) +
  geom_line(alpha = 0.5) +
  geom_point(size = 1.8) +
  scale_colour_manual(values = cols) +
  labs(title = "Test AUC across twenty stratified partitions",
       x = "Partition", y = "Test AUC") +
  my_theme

ggsave("figure6_splits.png", fig6, width = 9, height = 4.5, dpi = 300)
cat("Figure 6 saved\n")


# FIGURE : global SHAP importance, three models
# Attributions are rescaled within each model to a share of that
# model's total, because the random forest attributes on the
# probability scale and the other two on the log-odds scale.

imp <- read.csv("table_shap_importance.csv")

imp_long <- imp %>%
  pivot_longer(c(LR, RF, XGB), names_to = "Model", values_to = "Value") %>%
  group_by(Model) %>%
  mutate(Share = Value / sum(Value) * 100) %>%
  ungroup() %>%
  mutate(Model = recode(Model, LR = "Logistic Regression",
                        RF = "Random Forest", XGB = "XGBoost"))

top12 <- imp %>% arrange(-XGB) %>% slice(1:12) %>% pull(Feature)

fig7 <- imp_long %>%
  filter(Feature %in% top12) %>%
  mutate(Feature = factor(Feature, levels = rev(top12))) %>%
  ggplot(aes(Share, Feature, fill = Model)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = cols) +
  labs(title = "Global SHAP importance by model",
       subtitle = "Share of each model's total attribution",
       x = "Share of total attribution (%)", y = NULL) +
  my_theme

ggsave("figure7_shap_global.png", fig7, width = 9, height = 6, dpi = 300)
cat("Figure 7 saved\n")

# FIGURE 8: one applicant explained by all three models
# An applicant is chosen where the three models disagree on the
# largest contributing feature, since this is the case the
# analysis is concerned with.

S <- sv$S
top_lr  <- apply(abs(S$LR),  1, which.max)
top_rf  <- apply(abs(S$RF),  1, which.max)
top_xgb <- apply(abs(S$XGB), 1, which.max)

disagree <- which(top_lr != top_rf & top_rf != top_xgb & top_lr != top_xgb)
cat("Applicants where all three name a different top driver:",
    length(disagree), "\n")

i <- disagree[1]

local_df <- bind_rows(lapply(
  list(c("LR","Logistic Regression"),
       c("RF","Random Forest"),
       c("XGB","XGBoost")),
  function(m) {
    v <- S[[m[1]]][i, ]
    v <- v / sum(abs(v)) * 100     # share of that model's attribution
    data.frame(Feature = names(v), Share = as.numeric(v), Model = m[2])
  }))

show_feats <- local_df %>%
  group_by(Feature) %>% summarise(m = max(abs(Share))) %>%
  arrange(-m) %>% slice(1:8) %>% pull(Feature)

fig8 <- local_df %>%
  filter(Feature %in% show_feats) %>%
  mutate(Feature = factor(Feature, levels = rev(show_feats))) %>%
  ggplot(aes(Share, Feature, fill = Model)) +
  geom_col(position = "dodge") +
  geom_vline(xintercept = 0, colour = "grey40") +
  scale_fill_manual(values = cols) +
  labs(title = "One applicant, three explanations",
       subtitle = paste("Applicant", i,
                        "- contribution to the predicted probability of default"),
       x = "Share of the model's attribution (%)", y = NULL) +
  my_theme

ggsave("figure8_local.png", fig8, width = 9, height = 5, dpi = 300)
cat("Figure 8 saved\n")

# the numbers behind Figure 8, for quoting in the text
local_top <- local_df %>%
  group_by(Model) %>%
  slice_max(abs(Share), n = 1) %>%
  ungroup()
cat("\n Largest contributor for applicant", i, "\n")
print(as.data.frame(local_top), row.names = FALSE)
write.csv(local_df, "table_local_example.csv", row.names = FALSE)


# TABLE 2: sample construction

table2 <- data.frame(
  Step = c("Applications in the original dataset",
           "Records with code -9 in all 23 predictors",
           "Remaining duplicate record",
           "Analysis sample"),
  N = c(10459, -588, -1, 9870))
write.csv(table2, "table2_sample_construction.csv", row.names = FALSE)

#Figure_6
MARGIN <- 0.01

equiv <- read.csv("table_equivalence.csv")
print(equiv)

# Readable labels, ordered so the plot reads top to bottom in the
# same order as the text.
equiv <- equiv %>%
  mutate(Label = recode(as.character(Comparison),
                        "RF - LR"  = "Random Forest\nvs Logistic Regression",
                        "XGB - LR" = "XGBoost\nvs Logistic Regression",
                        "XGB - RF" = "XGBoost\nvs Random Forest"),
         Label = factor(Label, levels = rev(c(
           "Random Forest\nvs Logistic Regression",
           "XGBoost\nvs Logistic Regression",
           "XGBoost\nvs Random Forest"))))

fig6 <- ggplot(equiv, aes(x = Difference, y = Label)) +
  
  # shaded equivalence region
  annotate("rect", xmin = -MARGIN, xmax = MARGIN,
           ymin = -Inf, ymax = Inf,
           fill = "#4E79A7", alpha = 0.10) +
  
  # margin bounds and the line of no difference
  geom_vline(xintercept = c(-MARGIN, MARGIN),
             linetype = "dashed", colour = "#4E79A7", linewidth = 0.6) +
  geom_vline(xintercept = 0, colour = "grey40", linewidth = 0.4) +
  
  # the intervals
  geom_errorbarh(aes(xmin = Lower_90, xmax = Upper_90),
                 height = 0.14, linewidth = 0.8, colour = "#2F4B6E") +
  geom_point(size = 3, colour = "#2F4B6E") +
  
  scale_x_continuous(limits = c(-0.014, 0.014),
                     breaks = seq(-0.010, 0.010, 0.005)) +
  
  labs(x = "Difference in test AUC",
       y = NULL,
       caption = paste0("Shaded band shows the equivalence margin of ",
                        "\u00B1", MARGIN,
                        " AUC. Intervals are 90 per cent bootstrap ",
                        "confidence intervals.")) +
  
  theme_minimal(base_size = 12) +
  theme(axis.title    = element_text(face = "bold"),
        axis.text     = element_text(colour = "black"),
        panel.grid.minor   = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.caption  = element_text(hjust = 0, size = 9,
                                     colour = "grey30"),
        plot.margin   = margin(10, 14, 10, 10))

ggsave("figure6_equivalence.png", fig6,
       width = 7.5, height = 3.4, dpi = 300)

cat("\nFigure 6 saved.\n")
cat("Check: every interval should sit inside the shaded band.\n")
