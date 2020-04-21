# Packages for spatial data processing & visualization
library(rgdal)
library(gdalUtils)
library(raster)
library(sf)
library(sp)
library(RStoolbox)
#library(getSpatialData)
library(rasterVis)
library(mapview)
library(RColorBrewer)
library(plotly)
library(grDevices)
library(rpart)
library(rpart.plot)

# Machine learning packages
library(caret)
library(randomForest)
library(ranger)
library(MLmetrics)
library(nnet)
library(NeuralNetTools)
library(LiblineaR)
library(xgboost)
library(e1071)

# Packages for general data processing and parallel computation
library(data.table)
library(dplyr)
library(stringr)
library(doParallel)
library(snow)
library(parallel)
###################################################################

shp <- shapefile("Training_data/training_data1.shp")
ras <- stack("Process/Output/13month_Orb_Cal_Spk_TC_dB_Stack.tif")
####################################################################

dt2 <- ras %>% 
  extract(y = shp) %>%  
  as.data.frame %>% 
  mutate(id_cls = shp@data$Code_EN)
########################################################################
#write.csv(dt2,"Training_data/Samples1.csv", row.names = FALSE)
vals <- extract(ras,shp)
train <- data.matrix(vals)
classes <- as.numeric(as.factor(shp@data$Code_EN)) - 1
#######################################################################
set.seed(100)  # For reproducibility

X_train = xgb.DMatrix(as.matrix(training %>% select(-PE)))
y_train = training$PE
X_test = xgb.DMatrix(as.matrix(testing %>% select(-PE)))
y_test = testing$PE



#70% of the sample size
smp_size <- floor(0.70 * nrow(dt2))

#set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dt2)), size = smp_size)

train <- dt2[train_ind, ]
test <- dt2[-train_ind, ]
train1 = train %>% select(1:26)
test1 = test %>% select(1:26)



#############################################################################
####################XGBOOST##################################################
xgb <- xgboost(data = train, 
               label = train$id_cls, 
               eta = 0.1,
               max_depth = 6, 
               nround=100, 
               objective = "multi:softmax",
               num_class = length(unique(classes)),
               nthread = 3)
result <- predict(xgb,ras[1:(nrow(ras)*ncol(ras))])
res <- raster(ras)
res1 <- setValues(res,result)
writeRaster(res1,'test4.tif',options=c('TFW=YES'))

###########################################################################
# Fitting SVM to the Training set 
dat = data.frame(x = train1, y = as.factor(train$id_cls))
svmfit = svm(y ~ ., data = dat, kernel = "linear", cost = 10, scale = FALSE)
print(svmfit)

# Predicting the Test set results 
y_pred = predict(classifier, newdata = test_set[-3]) 

#Create Test Data
#p <- stack("Process/Output/subset_0_of_13month_Orb_Cal_Spk_TC_dB_Stack.tif")
#test <- rasterToPoints(p)
#write.csv(test,"test_value.csv", row.names = TRUE)
