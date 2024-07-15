# Load necessary libraries
install.packages("tidyverse")
library(tidyverse)

# Load your dataset
# Assuming your dataset is named 'interior_design_data.csv'
data <- read.csv("interior_design_dataset.csv")
View(data)
# Select the relevant columns
selected_data <- data %>%
  select(client_satisfaction, client_age, designer_experience, num_designers, 
         furniture_cost, lighting_cost, decor_cost, renovation_cost, 
         project_completion_time, num_meetings)

# Handling missing values (if any)
# For simplicity, we'll remove rows with NA values
cleaned_data <- na.omit(selected_data)

# Standardizing the numerical variables (optional but recommended)
scaled_data <- cleaned_data %>%
  mutate(across(c(client_age, designer_experience, num_designers, furniture_cost, 
                  lighting_cost, decor_cost, renovation_cost, 
                  project_completion_time, num_meetings), scale))

# Splitting the data into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(scaled_data)), size = 0.8 * nrow(scaled_data))
train_data <- scaled_data[train_indices, ]
test_data <- scaled_data[-train_indices, ]

# Fit the multiple linear regression model
model <- lm(client_satisfaction ~ client_age + designer_experience + num_designers + 
              furniture_cost + lighting_cost + decor_cost + renovation_cost + 
              project_completion_time + num_meetings, data = train_data)

# Summarize the model
summary(model)

# Predictions on the test set
predictions <- predict(model, test_data)

# Model evaluation
actuals <- test_data$client_satisfaction
mse <- mean((predictions - actuals)^2)
rmse <- sqrt(mse)
mae <- mean(abs(predictions - actuals))

# Print the evaluation metrics
cat("Mean Squared Error (MSE): ", mse, "\n")
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")
cat("Mean Absolute Error (MAE): ", mae, "\n")

###############################################rigid##############################3
# Load necessary libraries
library(tidyverse)
install.packages("glmnet")
library(glmnet)

# Load your dataset
# Assuming your dataset is named 'interior_design_data.csv'
data <- read.csv("interior_design_dataset.csv")

# Select the relevant columns
selected_data <- data %>%
  select(client_satisfaction, client_age, designer_experience, num_designers, 
         furniture_cost, lighting_cost, decor_cost, renovation_cost, 
         project_completion_time, num_meetings)

# Handling missing values (if any)
# For simplicity, we'll remove rows with NA values
cleaned_data <- na.omit(selected_data)

# Splitting the data into features and target
x <- as.matrix(cleaned_data %>% select(-client_satisfaction))  # Features
y <- cleaned_data$client_satisfaction  # Target

# Standardizing the features (glmnet standardizes internally by default)
# Splitting the data into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(cleaned_data)), size = 0.8 * nrow(cleaned_data))
x_train <- x[train_indices, ]
y_train <- y[train_indices]
x_test <- x[-train_indices, ]
y_test <- y[-train_indices]

# Fit the Ridge Regression model
ridge_model <- glmnet(x_train, y_train, alpha = 0)

# Cross-validation to find the optimal lambda
cv_ridge <- cv.glmnet(x_train, y_train, alpha = 0)

# Best lambda value
best_lambda <- cv_ridge$lambda.min
cat("Best lambda: ", best_lambda, "\n")

# Predict on the test set using the best lambda
ridge_predictions <- predict(ridge_model, s = best_lambda, newx = x_test)

# Model evaluation
mse <- mean((ridge_predictions - y_test)^2)
rmse <- sqrt(mse)
mae <- mean(abs(ridge_predictions - y_test))

# Print the evaluation metrics
cat("Mean Squared Error (MSE): ", mse, "\n")
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")
cat("Mean Absolute Error (MAE): ", mae, "\n")
############################################lasso###########################
selected_data <- data %>%
  select(client_satisfaction, client_age, designer_experience, num_designers, 
         furniture_cost, lighting_cost, decor_cost, renovation_cost, 
         project_completion_time, num_meetings)

# Handling missing values (if any)
# For simplicity, we'll remove rows with NA values
cleaned_data <- na.omit(selected_data)

# Splitting the data into features and target
x <- as.matrix(cleaned_data %>% select(-client_satisfaction))  # Features
y <- cleaned_data$client_satisfaction  # Target

# Standardizing the features (glmnet standardizes internally by default)
# Splitting the data into training and testing sets
set.seed(123)  # For reproducibility
train_indices <- sample(seq_len(nrow(cleaned_data)), size = 0.8 * nrow(cleaned_data))
x_train <- x[train_indices, ]
y_train <- y[train_indices]
x_test <- x[-train_indices, ]
y_test <- y[-train_indices]

# Fit the Lasso Regression model
lasso_model <- glmnet(x_train, y_train, alpha = 1)

# Cross-validation to find the optimal lambda
cv_lasso <- cv.glmnet(x_train, y_train, alpha = 1)

# Best lambda value
best_lambda <- cv_lasso$lambda.min
cat("Best lambda: ", best_lambda, "\n")

# Predict on the test set using the best lambda
lasso_predictions <- predict(lasso_model, s = best_lambda, newx = x_test)

# Model evaluation
mse <- mean((lasso_predictions - y_test)^2)
rmse <- sqrt(mse)
mae <- mean(abs(lasso_predictions - y_test))

# Print the evaluation metrics
cat("Mean Squared Error (MSE): ", mse, "\n")
cat("Root Mean Squared Error (RMSE): ", rmse, "\n")
cat("Mean Absolute Error (MAE): ", mae, "\n")



