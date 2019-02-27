from hubflow.core import *
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

modelnames = ['classwisecroco','globalcroco']
#'uncorrected','classwisecroco','globalcroco','savgol','sumnorm','contrem',

levels = ['level2b', 'level3b'] #,'level2b'
#9.47

for level in levels:
    for modelname in modelnames:
        print(level, modelname)

        if modelname == "uncorrected" or "classwisecroco" or "globalcroco":
            library = ENVISpectralLibrary(filename='B:/temp/temp_clemens/speclibs/EnMAP/lib07_EnMAP_nadir/library_nadir_uncorrected.sli')
        else:
            library = ENVISpectralLibrary(filename='B:/temp/temp_clemens/speclibs/EnMAP/lib07_EnMAP_nadir/library_nadir_' + modelname + '.sli')


        raster = Raster(filename='B:/temp/temp_clemens/aaa_crosstrack/results/subsets/roisub/BA_' + modelname + '_roisub_specsub.bsq')
        #raster = Raster(filename='B:/temp/temp_clemens/aaa_crosstrack/results/subsets/BA_' + modelname + '_spatsub_specsub.bsq')

        classification = Classification.fromENVISpectralLibrary(filename='/vsimem/labels.bsq', library=library, attribute=level)
        classificationSample = ClassificationSample(raster=library.raster(), classification=classification)

        #outdir = 'B:/temp/temp_clemens/ComparisonBrightCorrMethods/02_nadir/spatsub/'+ level + '_' + modelname + '_spatsub_nadir'
        outdir = 'B:/temp/temp_clemens/ComparisonBrightCorrMethods/02_nadir/roisub/' + level + '_' + modelname + '_roisub_nadir'

        # build ensemble

        predictions = {target: list() for target in classificationSample.classification().classDefinition().labels()}
        runs = 10

        for run in range(runs):
            for target in classificationSample.classification().classDefinition().labels():
                stamp = '_run{}_target{}'.format(run + 1, target)
                print(stamp)

                fractionSample = classificationSample.synthMix2(
                    filenameFeatures=join(outdir, 'train', 'features{}.bsq'.format(stamp)),
                    filenameFractions=join(outdir, 'train', 'fractions_{}.bsq'.format(stamp)),
                    mixingComplexities={2: 0.7, 3: 0.3}, classLikelihoods='proportional', #proportional
                    n=1000, target=target, includeWithinclassMixtures=True, includeEndmember=True)

                svr = SVR()
                param_grid = {'epsilon': [0.1], 'kernel': ['rbf'],
                              'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  # , 0.01, 0.1, 1, 10, 100, 1000],
                              'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}  # , 0.01, 0.1, 1, 10, 100, 1000]}
                tunedSVR = GridSearchCV(cv=3, estimator=svr, scoring='neg_mean_absolute_error', param_grid=param_grid)
                scaledAndTunedSVR = make_pipeline(StandardScaler(), tunedSVR)
                regressor = Regressor(sklEstimator=scaledAndTunedSVR)
                regressor.fit(sample=fractionSample)
                prediction = regressor.predict(filename=join(outdir, 'predictions', 'prediction{}.bsq'.format(stamp)), raster=raster)

                predictions[target].append(prediction)

        # aggregate

        applier = Applier()
        for target in predictions:
            applier.setFlowRaster(name=str(target), raster=RasterStack(predictions[target]))

        for key in ['mean', 'std', 'median', 'iqr']:
            applier.setOutputRaster(name=key, filename=join(outdir, 'aggregations', '{}.bsq'.format(key)))

        class Aggregate(ApplierOperator):
            def ufunc(self, predictions):
                results = {key: list() for key in ['mean', 'std', 'median', 'iqr']}

                # reduce over runs and stack targets
                for target in predictions:
                    array = self.flowRasterArray(name=str(target), raster=RasterStack(predictions[target]))
                    np.clip(array, a_min=0, a_max=1, out=array) # clip to 0-1
                    results['mean'].append(np.mean(array, axis=0))
                    results['std'].append(np.std(array, axis=0))
                    p25, median, p75 = np.percentile(array, q=[25, 50, 75], axis=0)
                    results['median'].append(median)
                    results['iqr'].append(p75-p25)

                # normalize to 0-1
                for key in ['mean', 'median']:
                    total = np.sum(results[key], axis=0)
                    results[key] = [a / total for a in results[key]]

                for key in ['mean', 'std', 'median', 'iqr']:
                    self.outputRaster.raster(key=key).setArray(results[key])

        applier.apply(operatorType=Aggregate, predictions=predictions)

        # RGB

        for key in ['mean', 'median']:
            fraction = Fraction(filename=join(outdir, 'aggregations', '{}.bsq'.format(key)),
                                classDefinition=classificationSample.classification().classDefinition())
            fraction.asClassColorRGBRaster(filename=join(outdir, 'rgb', '{}_rgb.bsq'.format(key)))
