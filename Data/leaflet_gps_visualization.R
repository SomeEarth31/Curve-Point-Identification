library(leaflet)
library(sf)
library(tidyverse)
library(htmlwidgets)


###Czech Data####
df <- read.delim("training-data.txt")

#sample_stretch<-df %>% filter(id==1)

# first, convert the data.frame to spdf
spatialdf <- st_as_sf(df, coords=c("x","y"), crs=st_crs(5514))

#convert the data frame
spatialdf <- st_transform(spatialdf, 4326) 

###create a colour palette for curve and tangent
pal <- colorFactor(
  palette = colorRampPalette(rainbow(2))(length(spatialdf$class)), 
  domain = spatialdf$class)

####create a leaflet
map <- leaflet(spatialdf) %>% addTiles() %>% 
  addCircleMarkers(radius = 2, label=paste0("segment: ",spatialdf$id), 
                 popup=paste0("segment: ",spatialdf$id),color = ~pal(class) )%>%
   addLegend("bottomright", pal = pal, values = ~class,
            title = "Curve (pre-defined classes)")

###view the leaflet
map

####export leaflet in R
saveWidget(map, file="czech_data_curves.html")


###Indian Data####
df2 <- read.csv("highway_geometry_data_every_10m_india_train.csv")

#sample_stretch<-df %>% filter(id==1)

# first, convert the data.frame to spdf
spatialdf2<-st_as_sf(df2, coords=c("X","Y"), crs=st_crs(24343))

#convert the data frame
spatialdf2 <- st_transform(spatialdf2, 4326) 

###create a colour palette for curve and tangent
pal <- colorFactor(
  palette = colorRampPalette(rainbow(2))(length(spatialdf2$Curve)), 
  domain = spatialdf2$Curve)

####create a leaflet
map2 <- leaflet(spatialdf2) %>% addTiles() %>% 
  addCircleMarkers(radius = 2, label=paste0("Radius (km): ",spatialdf2$Radius), 
                   popup=paste0("Radius (km): ",spatialdf2$Radius),color = ~pal(Curve) )%>%
  addLegend("bottomright", pal = pal, values = ~Curve,
            title = "Curve ( R threshold = 1.5 km)")

###view the leaflet
map2

####export leaflet in R
saveWidget(map2, file="indian_highway_curve.html")



###----------My Code--------#
#--for India data--#
test = as_tibble(df2)
ggplot(test, aes(x = X, y = Y, color = factor(Curve))) + geom_point() #Simple Plot
#Note: The India data set has one long road


#--for Czech data--#
test = as_tibble(df) %>% filter(id == 1)
ggplot(test, aes(x = x, y = y, color = factor(class))) + geom_point()
#Note: The Czech dataset has various roads

# first, convert the data.frame to spdf
spatialdf2 <- st_as_sf(test, coords=c("x","y"), crs=st_crs(5514))

#convert the data frame
spatialdf2 <- st_transform(spatialdf2, 4326) 

###create a colour palette for curve and tangent
pal2 <- colorFactor(
  palette = colorRampPalette(rainbow(2))(length(spatialdf2$class)), 
  domain = spatialdf2$class)

####create a leaflet
map <- leaflet(spatialdf2) %>% addTiles() %>% 
  addCircleMarkers(radius = 2, label=paste0("segment: ",spatialdf2$id), 
                   popup=paste0("segment: ",spatialdf2$id),color = ~pal2(class) )%>%
  addLegend("bottomright", pal = pal2, values = ~class,
            title = "Curve (pre-defined classes)")

###view the leaflet
map
