from sklearn import metrics
from sklearn import linear_model
from hubflow.core import *
from math import sqrt
from collections import Counter
import numpy as np
import csv
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from osgeo import gdal

# out dir
out_dir = r'B:\temp\temp_clemens\ComparisonBrightCorrMethods\\'

# read reference data
refdat = np.genfromtxt(r"B:/temp/temp_clemens/Validation/EnMAP-Pixel-Validation/ABCD_polygon_estimates_ordered_viewangle.csv", delimiter=',', dtype=str)

# specify metadata of reference data
modelname = 'savgol'
level = '2b'
classes = ['Unclassified', 'wve', 'nwve', 'bac']

# load prediction
raster = gdal.Open(r'B:\temp\temp_clemens\ComparisonBrightCorrMethods\classwisecroco\aggregations\mean.bsq')
num_bands = raster.RasterCount
arrList=[]
cols = raster.RasterXSize
rows = raster.RasterYSize
for band in range(num_bands):
    band += 1
    rb = raster.GetRasterBand(band)
    arr = rb.ReadAsArray(0,0, cols, rows)
    arrList.append(arr)

#       Validation          #
print('\tModel Validation')
        # settings
class_img = classes

num_class = len(classes)-1
validationNames = refdat[1:,2]
num_poly = len(validationNames)
npix_roi = 9

data_ref_all = np.transpose([validationNames])
temp1 = np.zeros((num_poly,num_class),dtype=float)
data_ref_all= np.append(data_ref_all,temp1,axis=1)

data_ref_all[:, 1] = refdat[1:, 9]       # l2_wVeg
data_ref_all[:, 2] = refdat[1:, 10]       # l2_nwVeg
data_ref_all[:, 3] = refdat[1:, 11]       # l2_back

data_ref_clipped = data_ref_all
validationNames_clipped = validationNames
already_deleted_ya_dingus = 0

array = np.array(arrList)

## Calculate means for each validation site
pred_mean = np.transpose([validationNames])
x = np.zeros((num_poly,num_class))
pred_mean= np.append(pred_mean,x,axis=1)

for band in range(0,len(array)):
    pred_mean[:,band+1]=np.mean(array[band],axis=1)

### Accuracy assessment ####
stats = np.zeros((5,num_class))

cur_class=0
for cur_class in range(num_class):
    val = ((data_ref_clipped[:, cur_class + 1]).astype(np.float)).reshape(-1,1)
    pred = (((pred_mean[:,cur_class+1]).astype(np.float))).reshape(-1,1)

    model = linear_model.LinearRegression()
    model.fit(val,pred)

    coef = model.coef_[0,0]
    intercept = model.intercept_[0]
    rsq = metrics.r2_score(val,pred)
    RMSE = sqrt(metrics.mean_squared_error(val,pred))
    MAE = metrics.mean_absolute_error(val,pred)

    stats[0, cur_class] = coef
    stats[1, cur_class] = intercept
    stats[2, cur_class] = rsq
    stats[3, cur_class] = RMSE
    stats[4, cur_class] = MAE

    ##plot##
    #by group
    groups = []
    for x in validationNames:
        groups.append(x[0])
    group_count = Counter(groups)
    site_colors = {'nadir':'green','offPOS':'purple','offNEG':'blue','exPOS':'orange','exNEG':'yellow'}

    plt.subplots()
    plt.plot([0, 100], [0, 100], color='grey', linestyle='dashed')

    counter = 0
    for x in group_count:
        plt.scatter(val[counter:counter+group_count[x]], pred[counter:counter+group_count[x]], color=site_colors[x],s=50)
        counter += group_count[x]

    plt.plot(val,coef*val+intercept,color='red')

    plt.title(season + " " + lib + " " + param +  ": " + class_img[cur_class+1])
    plt.xlabel('nadir: green offPOS: purple offNEG: blue exPOS: orange exNEG: yellow')
    plt.text(5,95, "y = "+str(round(coef,2))+'x + '+ str(round(intercept,2)))
    plt.text(5,90, "r-sq = " + (str(round(rsq,2))))
    plt.text(5,85, "RMSE = " + (str(round(RMSE,2))))
    plt.text(5,80, "MAE = " + (str(round(MAE,2))))

    plt.savefig(out_dir+r'\\'+level+'_'+modelname+'_'+class_img[cur_class+1]+'.png')

xlab = np.transpose([classes[1:]]).T
ylab = np.transpose([['','coef','int','rsq','RMSE','MAE']])
stats = np.append(xlab, stats,axis=0)
output = np.append(ylab,stats,axis=1)

with open(join(out_dir,'stats_'+level+'_sum_'+modelname+'.csv'),'w',newline='') as fh:
    writer = csv.writer(fh, delimiter=',')
    writer.writerows(output)

