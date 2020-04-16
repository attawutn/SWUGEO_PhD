# Packages for spatial data processing & visualization
library(rgdal)
library(gdalUtils)
library(raster)
library(sf)
library(sp)
library(rasterVis)
library(mapview)
library(RColorBrewer)
library(ggplot2)
library(grDevices)

# Machine learning packages
library(caret)
library(xgboost)
library(randomForest)

# Packages for general data processing and parallel computation
library(data.table)
library(dplyr)
library(stringr)
library(doParallel)
library(snow)
library(parallel)

shp <- shapefile("Training_data/training_data1.shp")
ras <- stack("Process/Output/13month_Orb_Cal_Spk_TC_dB_Stack.tif")

dt <-  extract(ras, shp) %>% as.data.frame %>% mutate(id_cls = shp@data$Code_EN)

dt[is.na(dt)] <- 0
training <- dt[, colSums(is.na(dt)) == 0]
dt$id_cls <- as.factor(dt$id_cls)
sapply(dt,class)

#set the seed to make your partition reproducible
set.seed(123)
smp_size <- floor(0.70 * nrow(dt))
train_ind <- sample(seq_len(nrow(dt)), size = smp_size)
dt_train <- dt[train_ind, ]
dt_test <- dt[-train_ind, ]

x_train = dt_train %>% select(1:26)
x_test = dt_test %>% select(1:26)
y_train = dt_train %>% select(27)
y_test = dt_test %>% select(27)

# create cross-validation folds (splits the data into n random groups)
n_folds <- 28
set.seed(321)
folds <- createFolds(1:nrow(dt_train), k = n_folds)
# Set the seed at each resampling iteration. Useful when running CV in parallel.
seeds <- vector(mode = "list", length = n_folds + 1) # +1 for the final model
for(i in 1:n_folds) seeds[[i]] <- sample.int(1000, n_folds)
seeds[n_folds + 1] <- sample.int(1000, 1) # seed for the final model

ctrl <- trainControl(summaryFunction = multiClassSummary,
                     method = "cv",
                     number = n_folds,
                     search = "grid",
                     classProbs = TRUE, # not implemented for SVM; will just get a warning
                     savePredictions = 'final',
                     index = folds,
                     seeds = seeds)

# Register a doParallel cluster, using 3/4 (75%) of total CPU-s
cl <- makeCluster(3/4 * detectCores())
registerDoParallel(cl)

# Train the model using Random Forest
model_rf <- train(id_cls ~ .  ,data = dt_train, method = "rf",importance = TRUE, allowParallel = TRUE,trControl = ctrl)

model_rf
plot(model_rf)

cm_rf <- confusionMatrix(data = predict(model_rf, newdata = dt_test),
                         dt_test$id_cls)
cm_rf

# Train the model using SVM:Ridial Basis
model_svmRadial = train(id_cls ~ .  ,data = dt_train, method='svmRadial', tuneLength=15, trControl = ctrl)

model_svmRadial
plot(model_svmRadial)

cm_svm <- confusionMatrix(data = predict(model_svmRadial, newdata = dt_test),
                         dt_test$id_cls)
cm_svm

# Train the model usingXGBoost
grid_default <- expand.grid(
  nrounds = 100,
  max_depth = 6,
  eta = 0.1,
  obje
)

model_xgb <- train(x = x_train,y = y_train,trControl = ctrl,tuneGrid = grid_default,
  method = "xgbTree",verbose = TRUE
)

cm_XGBoost <- confusionMatrix(data = predict(xgb_base, newdata = dt_test),
                          dt_test$id_cls)
cm_XGBoost

#Train Neural Network
model_nn <- train(id_cls ~ .  ,data = dt_train, 
               method = "nnet", trControl = ctrl,
               linout = TRUE)
cm_nn <- confusionMatrix(data = predict(model_nn, newdata = dt_test),
                          dt_test$id_cls)
cm_nn

###Compare Model
models_compare <- resamples(list(RF=model_rf, SVM=model_svmRadial,NN=cm_nn))
summary(models_compare)
