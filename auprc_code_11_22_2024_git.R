# Use this R code to reproduce the simulated data used in the associated AUPRC manuscript. 
# Specifically, this code generates reproducible, simulated clinical data including cerebral edema binary outcomes for 
# virtual (simulated) children with diabetic ketoacidosis.

# Load necessary R packages
library(dplyr)
library(ggplot2)
library(caret)
library(pROC)
library(PRROC)
library(randomForest)
library(xgboost)
library(Matrix)
library(ROCR)
library(svglite)


# Set seed to ensure reproducibility
n_seed <- 163736 
set.seed(n_seed)


# Step 1: Data Synthesis

# Set number of simulated patients to equal 10,000
n <- 10000

# Assign patient age: range 1 to 18 years old, assume a uniform distribution
age <- sample(1:18, n, replace = TRUE)

# Assign Biologic Sex: Male or Female, assume 50/50 distribution
sex <- sample(c("Male", "Female"), n, replace = TRUE)

# Assign binary diabetes onset: New Onset vs. Known Type 1 DM, assume 60% New Onset
diabetes_onset <- sample(c("New Onset", "Known Diabetes"), n, replace = TRUE, prob = c(0.6, 0.4))

# Assign duration of pre-hospital symptoms in days: assume longer duration for new onset Type 1 DM
duration_symptoms <- ifelse(diabetes_onset == "New Onset",
                            rpois(n, lambda = 3) + 1,  # New onset patients to have longer symptoms
                            rpois(n, lambda = 1) + 1)  # Known patients may recognize symptoms sooner and thus likely will be shorter

# Assign blood gas pH level: 
blood_pH <- rnorm(n, mean = 7.1, sd = 0.1) #Assume acidotic mean pH
blood_pH <- ifelse(blood_pH < 6.8, 6.8, ifelse(blood_pH > 7.5, 7.5, blood_pH)) # truncate at limits of physiologically plausible values

# Assign PaCO2: normal range 35 to 45 mm Hg, lower values indicate hypocapnia (often present in DKA)
PaCO2 <- rnorm(n, mean = 30, sd = 5)
PaCO2 <- ifelse(PaCO2 < 10, 10, ifelse(PaCO2 > 50, 50, PaCO2)) # truncate at limits of that typically seen in DKA

# Assign Blood urea nitrogen (BUN) values: normal range is approximately 7 to 20 mg/dL, higher values indicate dehydration and hypovolemia
BUN <- rnorm(n, mean = 20, sd = 5)
BUN <- ifelse(BUN < 5, 5, ifelse(BUN > 50, 50, BUN)) # truncate at limits of values typically seen in DKA

# Assign binary bicarbonate treatment variable: given rarity of current use, assume treatment in 1-2% of patients
bicarbonate_treatment <- rbinom(n, 1, prob = 0.015)

# Set seed again to ensure reproducibility of CE risk score generation
set.seed(n_seed)

# Assign cerebral edema outcome: occurs in approximately 0.7% of patients per published reports, thus will target this outcome percentage.
# Create a risk score based on risk factors, will include non-linear terms such that data can't be perfectly modeled by logistic regression
risk_score <- (age < 5) * 1 +
  (diabetes_onset == "New Onset") * 1 + #New onset cases more likely to present with CE
  (blood_pH < 7.1) * 3 +                # Greater degree of acidosis more likely to co-occur with CE
  (PaCO2 < 18) * 2 +                    # Greater degree of hypocapnia more likely to co-occur with CE
  (BUN > 20) * 1 +                      # Greater degree of hypovolemia (evidenced by elevated BUN) more likely to co-occur with CE
  (bicarbonate_treatment + 1) ^ (1+ 18 / age) + # introduce non-linear interaction term between bicarb treatment and age (bicarb treatment and younger age both pose higher risk for CE)
  (blood_pH * PaCO2) * 2 +              # Additional interaction term to capture effect of hypercapnea, if pCO2 actually elevated could indicate severe cerebral dysfunction
  (BUN / age) * 2 +                     # Additional Interaction term: lower age and higher BUN (worse hypovolemia) to impart higher CE risk
  (1 / blood_pH) * 10 +                 # Additional non-linear term, increased acidosis to impart higher CE risk   
  (18/age * 2 * (BUN)) ^ 2 * 0.2        # non-linear interaction term, lower age and higher BUN (worse hypovolemia) to impart higher CE risk

# Normalize risk score
risk_score_scaled <- scale(risk_score)

# Assign probabilities using a modified logistic function
set.seed(n_seed)
prob_cerebral_edema <- plogis(-6 + risk_score_scaled^3)

# Adjust probabilities to achieve goal CE rate
perc_goal <- 0.007
expected_incidence <- mean(prob_cerebral_edema)
adjustment_factor <- (perc_goal / expected_incidence)
prob_cerebral_edema <- prob_cerebral_edema * adjustment_factor
prob_cerebral_edema <- pmin(prob_cerebral_edema, 1) # Ensure probabilities stay within bounds

# Assign cerebral edema outcome
set.seed(n_seed)
cerebral_edema <- rbinom(n, 1, prob = prob_cerebral_edema)

# Check incidence rate
incidence_rate <- mean(cerebral_edema)
print(paste("Incidence of cerebral edema:", round(incidence_rate * 100, 2), "%"))

# Combine into a data frame for analysis
data <- data.frame(
  age,
  sex = as.factor(sex),
  diabetes_onset = as.factor(diabetes_onset),
  duration_symptoms,
  blood_pH,
  PaCO2,
  BUN,
  bicarbonate_treatment = as.factor(bicarbonate_treatment),
  cerebral_edema = as.factor(cerebral_edema)
)

# Step 2: Split data into training and test sets for model development
set.seed(n_seed) #ensure seed set again to ensure reproducibility
trainIndex <- createDataPartition(data$cerebral_edema, p = 0.8, 
                                  list = FALSE, 
                                  times = 1)
dataTrain <- data[trainIndex, ] # random 80% to training set
dataTest  <- data[-trainIndex, ] # remaining 20% to test set


# Step 3: Train and Evaluate Models

## Logistic Regression Model

# Train the model to predict CE
set.seed(n_seed)
log_model <- glm(cerebral_edema ~ ., data = dataTrain, family = binomial)

# Predict probabilities using the hold-out test set data
log_pred_prob <- predict(log_model, newdata = dataTest, type = "response")

# Class predictions using 0.5 threshold in case accuracy metric desired for this probability threshold
log_pred_class <- ifelse(log_pred_prob > 0.5, "1", "0")
log_pred_class <- as.factor(log_pred_class)

# Calculate confusion matrix and accuracy
conf_matrix_log <- confusionMatrix(log_pred_class, dataTest$cerebral_edema, positive = "1")
accuracy_log <- conf_matrix_log$overall['Accuracy']

# Compute AUROC
roc_log <- roc(response = dataTest$cerebral_edema, predictor = log_pred_prob)
auroc_log <- auc(roc_log)

# Compute AUPRC
fg_log <- log_pred_prob[dataTest$cerebral_edema == "1"]  # Positive class scores
bg_log <- log_pred_prob[dataTest$cerebral_edema == "0"]  # Negative class scores
pr_log <- pr.curve(scores.class0 = fg_log, scores.class1 = bg_log, curve = TRUE) # create pr curve
auprc_log <- pr_log$auc.integral # calculate AUPRC


## Random Forest Model

# Train the model
set.seed(n_seed)
rf_model <- randomForest(cerebral_edema ~ ., data = dataTrain, ntree = 100)

# Predict probabilities for the test set
rf_pred_prob <- predict(rf_model, newdata = dataTest, type = "prob")[, 2]

# Class predictions
rf_pred_class <- predict(rf_model, newdata = dataTest, type = "response")

# Confusion matrix and accuracy using 0.5 probability threshold
conf_matrix_rf <- confusionMatrix(rf_pred_class, dataTest$cerebral_edema, positive = "1")
accuracy_rf <- conf_matrix_rf$overall['Accuracy']

# Compute AUROC
roc_rf <- roc(response = dataTest$cerebral_edema, predictor = rf_pred_prob)
auroc_rf <- auc(roc_rf)

# Compute AUPRC
fg_rf <- rf_pred_prob[dataTest$cerebral_edema == "1"]
bg_rf <- rf_pred_prob[dataTest$cerebral_edema == "0"]
pr_rf <- pr.curve(scores.class0 = fg_rf, scores.class1 = bg_rf, curve = TRUE)
auprc_rf <- pr_rf$auc.integral


## XGBoost Model

# Convert categorical variables to numeric to prepare data for structure needed for XGBoost
dataTrain_xgb <- dataTrain %>%
  mutate(across(where(is.factor), ~ as.numeric(as.factor(.))))
dataTest_xgb <- dataTest %>%
  mutate(across(where(is.factor), ~ as.numeric(as.factor(.))))

# Create matrices for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(dataTrain_xgb %>% select(-cerebral_edema)), 
                            label = as.numeric(dataTrain_xgb$cerebral_edema) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(dataTest_xgb %>% select(-cerebral_edema)), 
                           label = as.numeric(dataTest_xgb$cerebral_edema) - 1)

# Set parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eval_metric = "auc"
)

# Train the model
set.seed(n_seed)
xgb_model <- xgb.train(
  params = params,
  data = train_matrix,
  nrounds = 100,
  verbose = 0
)

# Predict probabilities on the test set
xgb_pred_prob <- predict(xgb_model, newdata = test_matrix)

# Class predictions if using default 0.5 probability threshold
xgb_pred_class <- ifelse(xgb_pred_prob > 0.5, "1", "0")
xgb_pred_class <- as.factor(xgb_pred_class)

# Confusion matrix and accuracy for 0.5 probability threshold
conf_matrix_xgb <- confusionMatrix(xgb_pred_class, dataTest$cerebral_edema, positive = "1")
accuracy_xgb <- conf_matrix_xgb$overall['Accuracy']

# Compute AUROC
roc_xgb <- roc(response = dataTest$cerebral_edema, predictor = xgb_pred_prob)
auroc_xgb <- auc(roc_xgb)

# Compute AUPRC
fg_xgb <- xgb_pred_prob[dataTest$cerebral_edema == "1"]
bg_xgb <- xgb_pred_prob[dataTest$cerebral_edema == "0"]
pr_xgb <- pr.curve(scores.class0 = fg_xgb, scores.class1 = bg_xgb, curve = TRUE)
auprc_xgb <- pr_xgb$auc.integral


# Step 6: Create Performance Table to summarize AUROC and AUPRC for each model

# Compile performance metrics into a data frame
performance_table <- data.frame(
  Model = c("Logistic Regression", "Random Forest", "XGBoost"),
  Accuracy = c(accuracy_log, accuracy_rf, accuracy_xgb),
  AUROC = c(auroc_log, auroc_rf, auroc_xgb),
  AUPRC = c(auprc_log, auprc_rf, auprc_xgb)
)

# Round the values
performance_table$Accuracy <- round(performance_table$Accuracy, 4)
performance_table$AUROC <- round(performance_table$AUROC, 4)
performance_table$AUPRC <- round(performance_table$AUPRC, 4)


# Display the performance table
print(performance_table)


# Step 5: Plot AUROC Curves

# Create sensitivity and specificity dataframes for ease of plotting
df_log <- data.frame(Specificity = roc_log$specificities, Sensitivity = roc_log$sensitivities, Model = "Logistic Regression")
df_rf <- data.frame(Specificity = roc_rf$specificities, Sensitivity = roc_rf$sensitivities, Model = "Random Forest")
df_xgb <- data.frame(Specificity = roc_xgb$specificities, Sensitivity = roc_xgb$sensitivities, Model = "XGBoost")

# Combine the data frames
df_combined <- rbind(df_log, df_rf, df_xgb)

# Create the ggplot
auroc_plot <- ggplot(df_combined, aes(x = 1 - Specificity, y = Sensitivity, color = Model, linetype = Model)) +
  geom_line(size = 1) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), labels = seq(1, 0, by = -0.1), name = "Specificity") +
  scale_y_continuous(breaks = seq(0, 1, by = 0.25), limits = c(0, 1), name = "Sensitivity") +
  labs(title = "AUROC Curves for Models") +
  scale_color_manual(
    values = c("blue", "red", "green"),
    labels = c(paste0("Logistic Regression: AUROC ", round(auroc_log, digits = 3)), # create labels with the AUROC value for each model
               paste0("Random Forest: AUROC ", round(auroc_rf, digits = 3)),
               paste0("XGBoost: AUROC ", round(auroc_xgb, digits = 3)))
  ) +
  scale_linetype_manual(
    values = c("solid", "dashed", "dotted"),
    labels = c(paste0("Logistic Regression: AUROC ", round(auroc_log, digits = 3)),
              paste0("Random Forest: AUROC ", round(auroc_rf, digits = 3)),
              paste0("XGBoost: AUROC ", round(auroc_xgb, digits = 3)))
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(size = 14, face = "bold"), # Larger, bold x-axis label
    axis.title.y = element_text(size = 14, face = "bold"), # Larger, bold y-axis label
    axis.text = element_text(size = 12), # Adjust size of axis tick labels if needed
    axis.ticks = element_line(color = "black"), # Add tick marks
    axis.ticks.length = unit(0.25, "cm"), # Adjust tick mark length
    legend.text = element_text(size = 14), # Make legend text larger
    legend.position = c(0.95, 0.25), # Position legend at the right side of plot within plot area
    legend.justification = c("right", "top"), # Anchor the legend to the top-right corner
    panel.grid = element_blank(), # Remove grid
    panel.background = element_rect(fill = "white", color = NA), # Set background to white
    legend.background = element_rect(color = "black", fill = NA), # Add a square border to the legend
    legend.key = element_rect(fill = "white", color = NA), # Set legend key background to white
    panel.border = element_rect(color = "black", fill = NA, size = 1), # Add black border around plot
    legend.title = element_blank()
  )

# Display the plot
auroc_plot

# Store image of the AUROC curve in svg forat
ggsave(filename = "~/auroc_fig_1.svg",  
       plot = auroc_plot, 
       device = "svg", 
       dpi = 600, 
       width = 10,  
       height = 8)


# Step 6: Plot AUPRC Curves

# Prepare data for ggplot (combine curves into a single data frame with labels)
df <- data.frame(
  Recall = c(pr_log$curve[, 1], pr_rf$curve[, 1], pr_xgb$curve[, 1]),
  Precision = c(pr_log$curve[, 2], pr_rf$curve[, 2], pr_xgb$curve[, 2]),
  Model = factor(rep(c("Logistic Regression", "Random Forest", "XGBoost"),
                     times = c(nrow(pr_log$curve), nrow(pr_rf$curve), nrow(pr_xgb$curve)))
  )
)

# Preprocess the data: select the maximum precision (PPV) for each recall (sensitivit) value within each model to aid in creating smooth plot
df_filtered <- df %>%
  group_by(Model, Recall) %>%
  summarize(Precision = max(Precision), .groups = "drop")

# Replace model type with text that includes the AUPRC value to aid in labeling of plot
df_filtered$Model <- as.character(df_filtered$Model)
df_filtered$Model[df_filtered$Model == "Logistic Regression"] <- paste0("Logistic Regression: AUPRC ", round(auprc_log, 3))
df_filtered$Model[df_filtered$Model == "Random Forest"] <- paste0("Random Forest: AUPRC ", round(auprc_rf, 3))
df_filtered$Model[df_filtered$Model == "XGBoost"] <- paste0("XGBoost: AUPRC ", round(auprc_xgb, 3))
df_filtered$Model <- as.factor(df_filtered$Model)

# Plot using ggplot2
auprc_plot <- ggplot(df_filtered, aes(x = Recall, y = Precision, color = Model, linetype = Model)) +
  geom_line(size = 1) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1), labels = seq(0, 1, by = 0.1), name = "Recall (Sensitivity)") +
  scale_y_continuous(breaks = seq(0, 1, by = 0.25), limits = c(0, 1), name = "Precision (PPV)") +
  labs(title = "AUPRC Curves for Models") +
  scale_color_manual(
    values = c("blue", "red", "green"),
    labels = c(paste0("Logistic Regression: AUPRC ", round(auprc_log, digits = 3)),
               paste0("Random Forest: AUPRC ", round(auprc_rf, digits = 3)),
               paste0("XGBoost: AUPRC ", round(auprc_xgb, digits = 3)))
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.title.x = element_text(size = 14, face = "bold"), # Larger, bold x-axis label
    axis.title.y = element_text(size = 14, face = "bold"), # Larger, bold y-axis label
    axis.text = element_text(size = 12), # Adjust size of axis tick labels if needed
    axis.ticks = element_line(color = "black"), # Add tick marks
    axis.ticks.length = unit(0.25, "cm"), # Adjust tick mark length
    legend.text = element_text(size = 14), # Make legend text larger
    legend.position = c(0.95, 0.45), # Position legend at the top right inside plot area
    legend.justification = c("right", "top"), # Anchor the legend to the top-right corner
    panel.grid = element_blank(), # Remove grid
    panel.background = element_rect(fill = "white", color = NA), # Set background to white
    legend.background = element_rect(color = "black", fill = NA), # Add a square border to the legend
    legend.key = element_rect(fill = "white", color = NA), # Set legend key background to white
    panel.border = element_rect(color = "black", fill = NA, size = 1), # Add black border around plot
    legend.title = element_blank()
  )

# Display the plot
auprc_plot 

# Store the model AUPRC curves in svg format
ggsave(filename = "~/auprc_fig_1.svg", 
       plot = auprc_plot, 
       device = "svg", 
       dpi = 300, 
       width = 10,  
       height = 8)  
