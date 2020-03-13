library(raster)
library(xgboost)

shp <- shapefile("XGboost/Ri62_Suphanburi_Intersect_po.shp")
ras <- stack("XGboost/subset_0_of_S1B_IW_GRDH_1SDV_20200210T230834_20200210T230859_020213_02645C_DCDF_TNR_Orb_Cal_Spk_TC.tif")
vals <- extract(ras,shp)

train <- data.matrix(vals)
classes <- as.numeric(as.factor(shp@data$Code))

xgb <- xgboost(data = train,
               label = classes,
               nrounds = 100)

result <- predict(xgb,ras[1:(nrow(ras)*ncol(ras))])
res <- raster(ras)