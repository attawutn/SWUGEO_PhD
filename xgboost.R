library(raster)
library(xgboost)

shp <- shapefile("XGboost/Ri62_Suphanburi_Intersect_po.shp")
ras <- stack("XGboost/test_data.tif")
vals <- extract(ras,shp)

train <- data.matrix(vals)
classes <- as.numeric(as.factor(shp@data$Code)) - 1

xgb <- xgboost(data = train,
               label = classes,
               nrounds = 10000,
               nthread = 4,
               max_depth = 6,
               eta = 0.1
               )

result <- predict(xgb,ras[1:(nrow(ras)*ncol(ras))])
res <- raster(ras)
res1 <- setValues(res,result + 1)

writeRaster(res1,'test2.tif',options=c('TFW=YES'))
