import objbrowser

from os.path import join
from tempfile import gettempdir
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from hubflow.types import *

n_loops = 10
useRF =False

def synthMixRegressionWorkflow(x):
    wd = r'B:\temp\temp_clemens\BA_EnMap'
    speclibFilename = wd + r'\speclib\EnMAP\lib03_totalsub\EnMAP_speclib_area_b_totalsub_scaled.sli'
    enmapFilename = wd + r'\Raster\area_b_2_specsub_scaled'
    maskFilename = wd + r'\Raster\area_b_2_mask'
    img_out = wd + r'\Outputs\lib03'
    overwrite = True

	print('\n### SyntMixRegression Workflow ###' '\nOutput directory: {}'.format(wd))

    # create training data
    # - import ENVI speclib
	unsupervisedSample = UnsupervisedSample.fromENVISpectralLibrary(filename=speclibFilename)

    # - label spectra
	classDefinition = ClassDefinition(names=unsupervisedSample.metadata['level 4 class names'][1:], lookup=unsupervisedSample.metadata['level 4 class lookup'][3:])
	classificationSample = unsupervisedSample.classifyByName(names=unsupervisedSample.metadata['level 4 class spectra names'], classDefinition=classDefinition)

    # - generate synthetic mixtures
	probabilitySample = classificationSample.synthMix(mixingComplexities={2: 0.4, 3: 0.4, 4: 0.2}, classLikelihoods='proportional', n=1200)

    # train regressors
	if useRF:
		regressor = Regressor(sklEstimator=RandomForestRegressor())

	else:
		svr = SVR()
		param_grid = {'epsilon': [0.], 'kernel': ['rbf'], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000,10000], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]}
		tunedSVR = GridSearchCV(cv=3, estimator=svr, scoring='neg_mean_absolute_error', param_grid=param_grid)
		scaledAndTunedSVR = make_pipeline(StandardScaler(), tunedSVR)
		estimator = MultiOutputRegressor(scaledAndTunedSVR)
		regressor = Regressor(sklEstimator=estimator)


    # fit regressor
	image = Image(filename=enmapFilename)
	regressor.fit(sample=probabilitySample)
	prediction = regressor.predict(filename=img_out + str(x) + '.img', image=image, overwrite=overwrite, mask=Mask(maskFilename,0))
	return prediction



predictions = list()
for i in range(1,n_loops+1):
	prediction = synthMixRegressionWorkflow(i)
	predictions.append(prediction)


print(predictions)

from hubdc.model import Dataset, Open, CreateFromArray
import numpy as np
arrays = list()
for prediction in predictions:
	ds = Open(prediction.filename)
	array = ds.readAsArray()
	mask = array == -1
	np.clip(array, 0, 1, out=array)
	arrays.append(array)

mean = np.mean(arrays, axis=0)
mean[mask] = -1
CreateFromArray(pixelGrid=prediction.pixelGrid, array=mean,
				dstName=r'B:\temp\temp_clemens\BA_EnMap\Outputs\lib03\mean.bsq', format='ENVI', creationOptions=[])

