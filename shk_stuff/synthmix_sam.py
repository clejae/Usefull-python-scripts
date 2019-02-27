from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn import linear_model
from hubflow.core import *
from math import sqrt
from collections import Counter
import numpy as np
import csv
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

# Parameterize #
regression_type = 'SVR'         # Regressor type: 'GPR', 'RF', 'KRR' or 'SVR'
alpha = 1e-10
n_loops = 1

wd = r'B:\temp\temp_clemens\ComparisonBrightCorrMethods\globalcroco'

lib = 'Clemens'
param = ''

seasonslist = ['su']
levellist = ['2b']
val_site_ignorelist = []


# synthmix params
mix_comp2 = .5     # must sum to 1
mix_comp3 = .4      # must sum to 1
mix_comp4 = .1       # must sum to 1
n_synthmix = 1000

classlists = {'1b': ['Unclassified', 'Veg', 'Back'],
        '2b': ['Unclassified', 'wve', 'nwve', 'bac'],
        '3b': ['Unclassified', 'Tree', 'Shrub', 'Grass', 'Back'],
        '3b-s': ['Unclassified', 'Tree', 'Shrub', 'Grass', 'Back', 'Shade'],
        '4b': ['Unclassified', 'Conifer', 'Broad', 'Shrub', 'Grass', 'Back'],

        '1c': ['Unclassified', 'Veg', 'Back','test'],
        '3c': ['Unclassified', 'Tree', 'Shrub', 'Grass', 'Back','test'],
        '4c': ['Unclassified', 'Conifer', 'Broad', 'Shrub','Back'],
        '4d': ['Unclassified', 'Conifer', 'Broad','Back'],

        '1': ['unclassified', 'Veg', 'NonVeg', 'Water'],
        '2': ['unclassified', 'WoodyVeg', 'NonWoodyVeg', 'Other'],
        '3': ['unclassified', 'Tree', 'Shrub', 'Grass', 'Ag','NonVeg','Water'],
        '4': ['unclassified', 'Conifer', 'Broadleaf', 'Shrub', 'PerennialAg', 'AnnualAg','Grass','IrrigatedGrass','Wetland','Imperv','Soil','Water'],
        '5' : ['Unclassified', 'Conifer', 'EvBroad','DecBroad', 'Shrub', 'PerennialAg', 'AnnualAg', 'Grass', 'IrrigatedGrass', 'Wetland', 'Imperv', 'Soil', 'Water']
              }

full_starttime = time.time()

# Model  workflow #
print('\n### SyntMixRegression Workflow ###')

def synthMixRegressionWorkflow(loop, classID):
    img_out = out_dir + r'\pred\\' + regression_type + classes[classID] + r'_'+str(loop)+'.img'
    conf_out = out_dir + r'\conf\\' + classes[classID] + r'_conf_' + str(loop) +'.img'
    print('   ' + classes[classID] + ' Loop '+str(loop))

    ## create training data ##
    library = ENVISpectralLibrary(filename=speclibFilename)

    classificationSample = ClassificationSample(raster=library.raster(),
                                                classification=Classification.fromENVISpectralLibrary(filename='/vsimem/labels.bsq',
                                                                                                      library=library,
                                                                                                      attribute= 'Level '+str(level)))

    ## Generate synthetic mixtures
    print('\tGenerating Synthetic Mixtures')
    stamp = '_run{}_'.format(loop) + classes[classID]

    ## old synthetic mixtures
    # fractionSample = classificationSample.synthMix(synthmix_out_dir+'features'+stamp+'.bsq',
    #                                                synthmix_out_dir+'fractions'+stamp+'.bsq',
    #                                                mixingComplexities={2: mix_comp2, 3: mix_comp3, 4: mix_comp4},
    #                                                classLikelihoods='proportional', n=n_synthmix)

    ## new synthmix
    fractionSample = classificationSample.synthMix2(
        filenameFeatures=join(synthmix_out_dir, 'train', 'features{}.bsq'.format(stamp)),
        filenameFractions=join(synthmix_out_dir, 'train', 'fractions_{}.bsq'.format(stamp)),
        mixingComplexities={2: mix_comp2, 3: mix_comp3, 4: mix_comp4}, classLikelihoods='proportional',
        n=n_synthmix, target=classID, includeWithinclassMixtures=True, includeEndmember=True)

    ## fit model
    if regression_type == 'RF':
        print('\tFitting Random Forest Regressor')
        regressor = Regressor(sklEstimator=RandomForestRegressor())

    elif regression_type == 'SVR':
        print('\tFitting SVR Regressor')
        svr = SVR()
        param_grid = {'epsilon': [0.1], 'kernel': ['rbf'],
                      'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],         #, 0.01, 0.1, 1, 10, 100, 1000],
                      'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}             #, 0.01, 0.1, 1, 10, 100, 1000]}
        tunedSVR = GridSearchCV(cv=3, estimator=svr, scoring='neg_mean_absolute_error', param_grid=param_grid)
        scaledAndTunedSVR = make_pipeline(StandardScaler(), tunedSVR)
        regressor = Regressor(sklEstimator=scaledAndTunedSVR)

    elif regression_type == 'GPR':
        print('\tFitting GPR Regressor')


        gpr = GaussianProcessRegressor(alpha=alpha, copy_X_train=True,          ### tunning alpha - run grid search (as with SVR) with alpha as only parameter
                                        kernel=RBF(length_scale=1),             ### and then insert that into here to allow us to extract the confidence maps
                                        n_restarts_optimizer=0,                 ### 'C' above would be alpha in the grid search
                                        normalize_y=False,
                                        optimizer='fmin_l_bfgs_b',
                                        random_state=None)
        scaledAndTunedGPR = make_pipeline(StandardScaler(), gpr)


        regressor = RegressorGPR(sklEstimator=scaledAndTunedGPR)

    elif regression_type == 'KRR':
        print("\tFitting Kernel Ridge Regressor")

        krr = KernelRidge()
        param_grid = {'kernel': ['rbf'],
                      'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                      'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
        tunedKRR = GridSearchCV(cv=3, estimator=krr, scoring='neg_mean_absolute_error', param_grid=param_grid)
        scaledAndTunedKRR = make_pipeline(StandardScaler(), tunedKRR)
        regressor = Regressor(sklEstimator=scaledAndTunedKRR)

    # predict fraction image (& uncertainty if GPR)
    image = Raster(filename=enmapFilename)
    #            to set no data values (default=0):image.dataset().setNoDataValue(-1) or  Raster.fromArray(array, noDataValues=[-1,-1,-1])

    # regressor.fit(sample=probabilitySample)
    regressor.fit(sample=fractionSample)

    print('\tApplying Model \n')
    prediction = regressor.predict(filename=img_out,raster=image, mask=None, overwrite=overwrite)

    #           for only writing to virtual temp file: filename = '/vsimem/im!=0age.bsq'
    #           can't remember what this was...: gdal.Unlink(filename)

    if regression_type == 'GPR':
        prediction_stdev = regressor.predictStd(filename=conf_out, image=image, overwrite=overwrite)
        return (prediction, prediction_stdev)
    else:
        return prediction



### added from hub workflow ###
class RegressorGPR(Regressor):

    def predictStd(self, filename, image, mask=None, **kwargs):
        applier = Applier(defaultGrid=image, **kwargs)
        applier.setFlowRaster('image', raster=image)
        applier.setFlowMask('mask', mask=mask)
        applier.setOutputRaster('prediction', filename=filename)
        applier.apply(operatorType=_RegressorGPRPredictStd, image=image, estimator=self, mask=mask)

        prediction = self.PREDICT_TYPE(filename=filename)
        assert isinstance(prediction, Raster)
        return prediction

class _RegressorGPRPredictStd(ApplierOperator):
    def ufunc(self, estimator, image, mask):
        self.features = self.flowRasterArray('image', raster=image)
        etype, dtype, noutputs = self.getInfos(estimator)
        noData = estimator.sample().regression().noDataValue()
        prediction = self.full(value=noData, bands=noutputs, dtype=dtype)

        valid = self.maskFromArray(array=self.features, noDataValueSource='image')
        valid *= self.flowMaskArray('mask', mask=mask)

        X = numpy.float64(self.features[:, valid[0]].T)

        def gprPredictStd(gprPipeline, X, return_std=True):
            Xt = X
            for name, transform in gprPipeline.steps[:-1]:
                if transform is not None:
                    Xt = transform.transform(Xt)
            return gprPipeline.steps[-1][-1].predict(Xt, return_std=return_std)

        y, y_std = gprPredictStd(gprPipeline=estimator.sklEstimator(), X=X, return_std=True)
        y = y_std

        prediction[:, valid[0]] = y.reshape(X.shape[0], -1).T

        self.outputRaster.raster(key='prediction').setArray(array=prediction)

        if isinstance(estimator.sample, FractionSample):
            self.setFlowMetadataFractionDefinition('prediction', classDefinition=estimator.sample().classDefinition())
        else:
            self.outputRaster.raster(key='prediction').setNoDataValue(value=noData)
        self.setFlowMetadataBandNames('prediction', bandNames=estimator.sample().fraction().outputNames())

    def getInfos(self, estimator):
        etype = estimator.sklEstimator()._estimator_type
        if etype in ['classifier', 'clusterer']:
            noutputs = 1
            dtype = numpy.uint8
        elif etype == 'regressor':
            X0 = numpy.float64(numpy.atleast_2d(self.features[:, 0, 0]))
            y0 = estimator.sklEstimator().predict(X=X0)
            noutputs = max(y0.shape)
            dtype = numpy.float32
        else:
            raise Exception('unexpected estimator type')
        return etype, dtype, noutputs


#  Execute model in loops  #
for cur_level in levellist:
    level = cur_level

    classes = classlists[level]
    n_classes = len(classes) - 1  # excluding 'unclassified' as first class

    for iteration in seasonslist:
        season = iteration
        print("### Level " + level + " " + season+ " ###")
        out_dir = wd + r'\02_predictions\\' + lib + r'_' + level + param + "_" + season + r'\\'
        stats_out_dir = wd + r'\04_stats\\'
        stack_out_dir = wd + r'\03_stacks\\'
        synthmix_out_dir = wd + r'\01_synthmix\\'

        #refdat = np.genfromtxt(r"N:\temp\temp_sam\GPR\data\reference\ABCD_validation3.csv", delimiter=',', dtype=str)
        #speclibFilename = wd + r"\00_data\01_speclib\00_" + lib + "_" + season + r"\00_" + lib + "_" + season + ".sli"
        #enmapFilename = r'N:\temp\temp_sam\GPR\clean_slate\00_data\02_image\ABCD_' + season + '.bsq' #wd + r'\00_data\02_image\ABCD_' + season + '.bsq'

        #speclibFilename = wd + r"\speclibs\\" + lib + "_" + season + ".sli"

        refdat = np.genfromtxt(r"N:\temp\temp_clemens\Validation\EnMAP-Pixel-Validation\ABCD_polygon_estimates_ordered.csv", delimiter=',', dtype=str)
        enmapFilename = r"N:\temp\temp_clemens\aaa_crosstrack\results\subsets\BA_classwisecroco_roisub_specsub.bsq"
        speclibFilename = r"N:\temp\temp_clemens\speclibs\EnMAP\lib05_EnMAP_preFinal\library_all_final_cleaned_v1_mean.sli"

        overwrite = True
        print('Output Directory: ' + stats_out_dir)

        for classloop in range(1,n_classes + 1):

            predictions = list()
            pred_error = list()
            for i in range(1,n_loops + 1):
                if regression_type == 'GPR':
                    result = synthMixRegressionWorkflow(i, classloop)
                    prediction = result[0]
                    predictions.append(prediction)
                    prediction_stdev = result[1]
                    pred_error.append(prediction_stdev)
                else:
                    prediction = synthMixRegressionWorkflow(i,classloop)
                    predictions.append(prediction)

            arrays_mean = list()
            arrays_std = list()

            for prediction in predictions:
                ds = Raster(prediction._filename)
                array_mean = ds.readAsArray()
                mask_mean = array_mean == -1
                np.clip(array_mean, 0, 1, out=array_mean)
                arrays_mean.append(array_mean)

                array_std = ds.readAsArray()
                mask_std = array_std == -1
                #np.clip(array_std, 0, 1, out=array_std)
                arrays_std.append(array_std)

            mean = np.mean(arrays_mean, axis=0)
            #np.clip(mean, 0, 1, out=mean)
            mean[mask_mean] = -1
            Raster.fromArray(array=mean, filename=out_dir + r'0_mean_' + classes[classloop] + '.bsq')

            SD = np.std(arrays_std, axis=0)
            #np.clip(SD, 0, 1, out=SD)
            SD[mask_std] = -1
            Raster.fromArray(array=SD, filename=out_dir + r'1_StDev_' + classes[classloop] + '.bsq')


            if regression_type == 'GPR':
                arrays_conf = list()
                for prediction in pred_error:
                    ds = Raster(prediction._filename)
                    array_conf = ds.readAsArray()
                    mask_conf = array_conf == -1
                    #np.clip(array_conf, 0, 1, out=array_conf)
                    arrays_conf.append(array_conf)

                conf = np.mean(arrays_conf, axis=0)
                #np.clip(mean3, 0, 1, out=mean3)
                conf[mask_conf] = -1
                Raster.fromArray(array=conf, filename=out_dir + r'2_Conf_' + classes[classloop] + '.bsq')



        #    stack    #
        stack =list()
        image = Raster(filename=enmapFilename)

        for name in classes:
            if name == 'Unclassified':
                filename = out_dir
            else:
                filename = out_dir + r'\0_mean_' + name + r'.bsq'
                img = Raster(filename)
                fileopen = gdal.Open(filename)
                array = fileopen.ReadAsArray()
                array = array * 100
                stack.append(array)

        stack2 = np.array(stack)

        Raster.fromArray(array=stack2, filename=stack_out_dir + r'0_'+season+'_pred_'+regression_type+level+'_'+lib+param+'.bsq')

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

        if level == '1b':
            data_ref_all[:, 1] = refdat[1:, 12]       # l1_veg
            data_ref_all[:, 2] = refdat[1:, 13]       # l1_back
        elif level == '2b':
            data_ref_all[:, 1] = refdat[1:, 9]       # l2_wVeg 14
            data_ref_all[:, 2] = refdat[1:, 10]       # l2_nwVeg 15
            data_ref_all[:, 3] = refdat[1:, 11]       # l2_back 16
        elif level == '3b':
            data_ref_all[:, 1] = refdat[1:, 17]       # l3_tree
            data_ref_all[:, 2] = refdat[1:, 18]       # l3_shrub
            data_ref_all[:, 3] = refdat[1:, 19]       # l3_grass
            data_ref_all[:, 4] = refdat[1:, 20]       # l3_back
        elif level == '4b':
            data_ref_all[:, 1] = refdat[1:, 21]       # l4_conifer
            data_ref_all[:, 2] = refdat[1:, 22]       # l4_broad
            data_ref_all[:, 3] = refdat[1:, 23]       # l4_shrub
            data_ref_all[:, 4] = refdat[1:, 24]       # l4_grass
            data_ref_all[:, 5] = refdat[1:, 25]       # l4_back

        data_ref_clipped = data_ref_all
        validationNames_clipped = validationNames
        already_deleted_ya_dingus = 0
        for site in val_site_ignorelist:
            rows_to_ignore = np.where(validationNames == site)
            data_ref_clipped = np.delete(data_ref_clipped, rows_to_ignore[0][0] - already_deleted_ya_dingus, 0)
            validationNames_clipped = np.delete(validationNames_clipped,rows_to_ignore[0][0] - already_deleted_ya_dingus, 0)
            already_deleted_ya_dingus += 1

        array = stack2

        ## create stack of residual images

        pixel_vals = np.genfromtxt(r"N:\temp\temp_sam\GPR\data\reference\ABCD_validation2.csv", delimiter=',', dtype=str)

        residuals = []




        ## Calculate means for each validation site
        pred_mean = np.transpose([validationNames])
        x = np.zeros((num_poly,num_class))
        pred_mean= np.append(pred_mean,x,axis=1)

        for band in range(0,len(array)):
            pred_mean[:,band+1]=np.mean(array[band],axis=1)

        ### ignore unwanted rows for validation (e.g. ag validation sites ###
        already_deleted_ya_dingus = 0
        pred_clip = pred_mean
        for site in val_site_ignorelist:
            rows_to_ignore = np.where(validationNames == site)
            pred_clip = np.delete(pred_clip, rows_to_ignore[0][0] - already_deleted_ya_dingus, 0)
            already_deleted_ya_dingus += 1

        ### Accuracy assessment ####


        stats = np.zeros((5,num_class))

        for cur_class in range(num_class):
            val = ((data_ref_clipped[:, cur_class + 1]).astype(np.float)).reshape(-1,1)
            pred = (((pred_clip[:,cur_class+1]).astype(np.float))).reshape(-1,1)


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
            for x in validationNames_clipped:
                groups.append(x[0])
            group_count = Counter(groups)
            site_colors = {'A':'green','B':'orange','C':'blue','D':'purple'}

            plt.subplots()
            plt.plot([0, 100], [0, 100], color='grey', linestyle='dashed')

            counter = 0
            for x in group_count:
                plt.scatter(val[counter:counter+group_count[x]], pred[counter:counter+group_count[x]], color=site_colors[x],s=50)
                counter += group_count[x]

            plt.plot(val,coef*val+intercept,color='red')

            plt.title(season + " " + lib + " " + param +  ": " + class_img[cur_class+1])
            plt.xlabel("A: Green    B: Orange    C: Blue    D: Purple")
            plt.text(5,95, "y = "+str(round(coef,2))+'x + '+ str(round(intercept,2)))
            plt.text(5,90, "r-sq = " + (str(round(rsq,2))))
            plt.text(5,85, "RMSE = " + (str(round(RMSE,2))))
            plt.text(5,80, "MAE = " + (str(round(MAE,2))))

            plt.savefig(stats_out_dir+r'\\'+regression_type+level+'_'+season+'_' +lib+param+'_'+class_img[cur_class+1]+'.png')

        xlab = np.transpose([classes[1:]]).T
        ylab = np.transpose([['','coef','int','rsq','RMSE','MAE']])
        stats = np.append(xlab, stats,axis=0)
        output = np.append(ylab,stats,axis=1)

        with open(join(stats_out_dir,'stats_'+regression_type+level+'_'+season+'_'+lib+param+'.csv'),'w',newline='') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerows(output)

        # export csv of validation vs predicted values

        blanks = np.empty((len(pred),1), dtype=str)

        valnames = []
        for n in classes[1:]:
            newname = 'val'+n
            valnames.append(newname)

        prednames = []
        for n in classes[1:]:
            newname = 'pred'+n
            prednames.append(newname)

        col_names = np.transpose([['site'] + valnames + ['','site'] + prednames]).T

        out_valVpred = np.append(data_ref_clipped,blanks,axis=1)
        out_valVpred = np.append(out_valVpred,pred_clip,axis=1)
        out_valVpred = np.append(col_names,out_valVpred,axis=0)

        with open(stats_out_dir+r'\valVpred_'+regression_type+level+'_'+season+"_"+lib+param+'.csv','w',newline='') as fh:
            writer = csv.writer(fh, delimiter=',')
            writer.writerows(out_valVpred)


full_endtime = time.time()
print("\nProcess Complete")
print("Total elapsed time: " + str(round((full_endtime - full_starttime)/60.,2)) + " minutes OR " + str(round(((full_endtime - full_starttime)/60.)/60.,2)) + " hours")
print("\tFigure directory: " + stats_out_dir)