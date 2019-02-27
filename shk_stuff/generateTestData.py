from osgeo import gdal
import numpy as np

extentList = [(10139, 426),(9261, 428),(8871, 433),(8658, 428),(8684, 419),(8059, 421),(7630, 419),(7246, 419),(7018, 420),(6714, 420),(6387, 417)]

i = 0
for extent in extentList:
    arr = np.full(extent, i+49, dtype=np.int16)
    driver = gdal.GetDriverByName('Envi')
    destination = driver.Create(r"B:\temp\temp_clemens\aaa_crosstrack\results\flight line raster\flightline_" + str(i+49) + ".bsq", extent[1], extent[0], 1, gdal.GDT_Float32)
    destination.GetRasterBand(1).WriteArray(arr)
    destination = None
    i = i+1

for extent in extentList:
    print(extent[1], extent[0])