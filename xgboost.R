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
#vals <- extract(ras,shp)
#train <- data.matrix(vals)
#classes <- as.numeric(as.factor(shp@data$Code_EN)) - 1

#######################################################################
#70% of the sample size
smp_size <- floor(0.70 * nrow(dt2))

#set the seed to make your partition reproducible
set.seed(123)
train_ind <- sample(seq_len(nrow(dt2)), size = smp_size)

train <- dt2[train_ind, ]
test <- dt2[-train_ind, ]
#######################################################################
# create cross-validation folds (splits the data into n random groups)
n_folds <- 10
set.seed(321)
folds <- createFolds(1:nrow(train), k = n_folds)
# Set the seed at each resampling iteration. Useful when running CV in parallel.
seeds <- vector(mode = "list", length = n_folds + 1) # +1 for the final model
for(i in 1:n_folds) seeds[[i]] <- sample.int(1000, n_folds)
seeds[n_folds + 1] <- sample.int(1000, 1) # seed for the final model
###########################################################################
model.class <- rpart(as.factor(train$id_cls)~., data = train, method = 'class')
rpart.plot(model.class, box.palette = 0, main = "Classification Tree")

pr <- predict(ras, model.class, type ='class', progress = 'text') %>% 
  ratify()

levels(pr) <- levels(pr)[[1]] %>%
  mutate(legend = c("May","June","July","Sep","Other"))

##########################################
xgb <- xgboost(data = train, 
               label = classes, 
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

# Fitting SVM to the Training set 

library(e1071) 

classifier = svm(formula = classes, 
                 data = train, 
                 type = 'C-classification', 
                 kernel = 'linear') 

# Predicting the Test set results 
y_pred = predict(classifier, newdata = test_set[-3]) 

#Create Test Data
#p <- stack("Process/Output/subset_0_of_13month_Orb_Cal_Spk_TC_dB_Stack.tif")
#test <- rasterToPoints(p)
#write.csv(test,"test_value.csv", row.names = TRUE)
