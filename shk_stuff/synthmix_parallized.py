from hubflow.core import *
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

modelnames = ['uncorrected','classwisecroco','globalcroco','contrem','savgol','sumnorm']

jobList = []
for modelname in modelnames:
    lib = ENVISpectralLibrary(filename='B:/temp/temp_clemens/speclibs/EnMAP/lib06_EnMAP_reduced/library_reduced_' + modelname + '.sli')
    ras = Raster(filename='B:/temp/temp_clemens/aaa_crosstrack/results/subsets/roisub/BA_' + modelname + '_roisub_specsub.bsq')
    out = 'B:/temp/temp_clemens/ComparisonBrightCorrMethods/' + modelname + '_roisub_reducedlib'
    subList = [lib, ras, out]

def workFunc(job):

    #library = ENVISpectralLibrary(filename='B:/temp/temp_clemens/speclibs/EnMAP/lib06_EnMAP_reduced/library_reduced_' + modelname + '.sli')
    library = job[0]

    #raster = Raster(filename='B:/temp/temp_clemens/aaa_crosstrack/results/subsets/roisub/BA_' + modelname + '_roisub_specsub.bsq')
    raster = job[1]

    classification = Classification.fromENVISpectralLibrary(filename= '/vsimem/labels.bsq', library=library, attribute='level2b')
    classificationSample = ClassificationSample(raster=library.raster(), classification=classification)

    #outdir = 'B:/temp/temp_clemens/ComparisonBrightCorrMethods/' + job[2] + '_roisub_reducedlib'
    outdir = job[2]

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
                mixingComplexities={2: 0.7, 3: 0.3}, classLikelihoods='equalized',
                n=1000, target=target, includeWithinclassMixtures=False, includeEndmember=False)

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

if __name__ == '__main__':
    Parallel(n_jobs=6)(delayed(workFunc)(i) for i in jobList)