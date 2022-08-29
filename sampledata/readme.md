
Once imported into our architecture, each dataset was filtered and pre-processed individually to derive their respective train and test sets. 
We filter locations to a bounding box defining a city or region of interest and then transform continuous GPS coordinates by tessellating the space and encoding location as a discrete grid position to attain the location identifiers (i.e., POI).
In these spatial transformations, we convert the GPS coordinates to the discretizing locations via the Geohash algorithm with rectangular cells.
For instance, each bounding box defines the grid size of the interested region, and the grid granularity is 0.01 degrees, where each grid represents a 0.01 longitude x 0.01 latitude area. 
