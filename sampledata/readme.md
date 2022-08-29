[Link to Geolife Data download](https://www.microsoft.com/en-us/download/details.aspx?id=52367&from=https%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2Fb16d359d-d164-469e-9fd4-daa38f2b2e13%2F)

In this work, we use four datasets: MDC, Geolife, Foursquare, Privamov.

In this sample data here, we use the Geolife to show example of how we convert the GPS data as POI data we use for the training.

Once imported into our architecture, each dataset was filtered and pre-processed individually to derive their respective train and test sets. 
We filter locations to a bounding box defining a city or region of interest and then transform continuous GPS coordinates by tessellating the space and encoding location as a discrete grid position to attain the location identifiers (i.e., POI).
In these spatial transformations, we convert the GPS coordinates to the discretizing locations via the Geohash algorithm with rectangular cells.
For instance, each bounding box defines the grid size of the interested region, and the grid granularity is 0.01 degrees, where each grid represents a 0.01 longitude x 0.01 latitude area. 
