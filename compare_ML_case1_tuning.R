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

shp <- shapefile("/Test/Training_data/training_data1.shp")
ras <- stack("/Test/Process/Output/13month_Orb_Cal_Spk_TC_dB_Stack.tif")

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

# create cross-validation folds (splits the data into n random groups)
set.seed(321)
n_folds <- 5
folds <- createFolds(1:nrow(dt_train), k = n_folds)
# Set the seed at each resampling iteration. Useful when running CV in parallel.
seeds <- vector(mode = "list", length = n_folds + 1) # +1 for the final model
for(i in 1:n_folds) seeds[[i]] <- sample.int(1000, n_folds)
seeds[n_folds + 1] <- sample.int(1000, 1) # seed for the final model

ctrl <- trainControl(summaryFunction = multiClassSummary,
                     method = "repeatedcv",
                     number = 10,
                     repeats = 5,
                     search = "grid",
                     classProbs = TRUE,
                     savePredictions = 'final',
                     allowParallel = TRUE,
                     verboseIter = TRUE)

nnet.grid = expand.grid(size = seq(from = 2, to = 10, by = 2), # number of neurons units in the hidden layer 
                        decay = seq(from = 0.1, to = 0.5, by = 0.1)) # regularization parameter to avoid over-fitting 

rf.grid <- expand.grid(mtry=1:20) # number of variables available for splitting at each tree node

svm.grid <- expand.grid(sigma=seq(from = 0.01, to = 0.10, by = 0.02), # controls for non-linearity in the hyperplane
                        C=seq(from = 2, to = 10, by = 2)) # controls the influence of each support vector


# Train the model using Random Forest
model_rf <- train(id_cls ~ .  ,data = dt_train, method = "rf",metric="Accuracy",importance = TRUE,trControl = ctrl,tuneGrid = rf.grid)
#model_rf
#plot(model_rf)

cm_rf <- confusionMatrix(data = predict(model_rf, newdata = dt_test),
                         dt_test$id_cls)

# Train the model using SVM:Ridial Basis
model_svmRadial = train(id_cls ~ .  ,data = dt_train, method='svmRadialSigma',metric="Accuracy", tuneGrid = svm.grid, trControl = ctrl)

#model_svmRadial
#plot(model_svmRadial)

cm_svm <- confusionMatrix(data = predict(model_svmRadial, newdata = dt_test),dt_test$id_cls)
#cm_svm


#Train Neural Network
model_nn <- train(id_cls ~ .  ,data = dt_train,method = "nnet", trControl = ctrl,metric="Accuracy",linout = TRUE,tuneGrid = nnet.grid)
cm_nn <- confusionMatrix(data = predict(model_nn, newdata = dt_test),dt_test$id_cls)
#cm_nn


# Train the model usingXGBoost
########################################################
xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,  
  allowParallel = TRUE,
  verboseIter = FALSE,
  returnData = FALSE
)

xgbGrid <- expand.grid(nrounds = c(100,200),
                       max_depth = c(10, 15, 20, 25),
                       colsample_bytree = seq(0.5, 0.9, length.out = 5),
                       eta = 0.1,
                       gamma=0,
                       min_child_weight = 1,
                       subsample = 1
)
set.seed(100) 
xgb_model = train(id_cls ~ .  ,data = dt_train,  
                  trControl = ctrl,
                  tuneGrid = xgbGrid,
                  method = "xgbTree"
)

cm_xgb <- confusionMatrix(data = predict(xgb_model, newdata = dt_test),
                          dt_test$id_cls)
cm_xgb


###Compare Model
models_compare <- resamples(list(RF=model_rf, SVM=model_svmRadial,NN=model_nn,xgb=xgb_model))
summary(models_compare)

##Predict
nnet_prediction = predict(ras, model=model_nn)
writeRaster(nnet_prediction,'test_nn.tif',options=c('TFW=YES'))

rf_prediction = predict(ras, model=model_rf)
writeRaster(rf_prediction,'test_rf.tif',options=c('TFW=YES'))

svm_prediction = predict(ras, model=model_svmRadial)
writeRaster(svm_prediction,'test_svm.tif',options=c('TFW=YES'))

xgb_prediction = predict(ras, model=xgb_model)
writeRaster(xgb_prediction,'test_xgb.tif',options=c('TFW=YES'))

