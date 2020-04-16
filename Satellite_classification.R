# load required libraries (Note: if these packages are not installed, then install them first and then load)
library(rgdal) # a comprehansive repository for handling spatial data
library(raster) # for the manipulation of raster data
library(caret) # for the machine learning algorithms
library(sp) # for the manipulation of spatial objects
library(nnet) # Artificial Neural Network
library(randomForest) # Random Forest 
library(kernlab) # Support Vector Machines
library(dplyr)

shp <- shapefile("Training_data/training_data1.shp")
ras <- stack("Process/Output/13month_Orb_Cal_Spk_TC_dB_Stack.tif")

dt <- ras %>% 
  extract(y = shp) %>%  
  as.data.frame %>% 
  mutate(id_cls = shp@data$Code_EN)

smp_size <- floor(0.70 * nrow(dt2))

set.seed(123)
train_ind <- sample(seq_len(nrow(dt2)), size = smp_size)

train <- dt2[train_ind, ]
test <- dt2[-train_ind, ]

########################################################################
tc <- trainControl(method = "repeatedcv", # repeated cross-validation of the training data
                   number = 10, # number of folds
                   repeats = 5) # number of repeats

nnet.grid = expand.grid(size = seq(from = 2, to = 10, by = 2),
                        decay = seq(from = 0.1, to = 0.5, by = 0.1))

rf.grid <- expand.grid(mtry=1:20) 
svm.grid <- expand.grid(sigma=seq(from = 0.01, to = 0.10, by = 0.02), 
                        C=seq(from = 2, to = 10, by = 2)) 

## Begin training the models. This will take about 6 minutes for all three algorithms
# Run the neural network model
nnet_model <- caret::train(x = trn[,(1:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$id_cls))),method = "nnet", metric="Accuracy", trainControl = tc, tuneGrid = nnet.grid)

# Run the random forest model
rf_model <- caret::train(x = trn[,(1:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$id_cls))),method = "rf", metric="Accuracy", trainControl = tc, tuneGrid = rf.grid)

# Run the support vector machines model
svm_model <- caret::train(x = trn[,(1:ncol(trn)-1)], y = as.factor(as.integer(as.factor(trn$id_cls))),method = "svmRadialSigma", metric="Accuracy", trainControl = tc, tuneGrid = svm.grid)

## Apply the models to data
# Apply the neural network model to the Sentinel-2 data
nnet_prediction = raster::predict(s2data, model=nnet_model)

# Apply the random forest model to the Sentinel-2 data
rf_prediction = raster::predict(s2data, model=rf_model)

# Apply the support vector machines model to the Sentinel-2 data
svm_prediction = raster::predict(s2data, model=svm_model)

# Convert the evaluation data into a spatial object using the X and Y coordinates and extract predicted values
eva.sp = SpatialPointsDataFrame(coords = cbind(eva$x, eva$y), data = eva, 
                                proj4string = crs("+proj=utm +zone=47 +datum=WGS84 +units=m +no_defs +ellps=WGS84 +towgs84=0,0,0"))

## Superimpose evaluation points on the predicted classification and extract the values
# neural network
nnet_Eval = extract(nnet_prediction, eva.sp)
# random forest
rf_Eval = extract(rf_prediction, eva.sp)
# support vector machines
svm_Eval = extract((svm_prediction), eva.sp)

# Create an error matrix for each of the classifiers
nnet_errorM = confusionMatrix(as.factor(nnet_Eval),as.factor(eva$class))
rf_errorM = confusionMatrix(as.factor(rf_Eval),as.factor(eva$class))
svm_errorM = confusionMatrix(as.factor(svm_Eval),as.factor(eva$class))

# Plot the results next to one another along with the 2018 NMD dataset for comparison
nmd2018 = raster("MachineLearningTutorialPackage/NMD_S2Small.tif") # load NMD dataset (Nationella Marktäckedata, Swedish National Land Cover Dataset)
crs(nmd2018) <- crs(nnet_prediction) # Correct the coordinate reference system so it matches with the rest
rstack = stack(nmd2018, nnet_prediction, rf_prediction, svm_prediction) # combine the layers into one stack
names(rstack) = c("NMD 2018", "Single Layer Neural Network", "Random Forest", "Support Vector Machines") # name the stack
plot(rstack) # plot it! 

# Congratulations! You conducted your first machine learning classification.