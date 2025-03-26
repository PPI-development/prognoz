import statsmodels.api as sm
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import os
from fancyimpute import IterativeImputer
from copy import deepcopy
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from matplotlib.pyplot import rcParams
from methods import *
import seaborn
seaborn.set_style('darkgrid')
rcParams['figure.figsize'] = 18, 8
import os
import warnings
warnings.filterwarnings("ignore")

all_factors = [
 'TS',
 'T2M',
 'T2M_MAX',
 'T2M_MIN',
 'CLOUD_AMT',
 'T2MDEW',
 'T2MWET',
 'QV2M',
 'RH2M',
 'WS2M',
 'GWETTOP',
 'GWETROOT',
 'T2M_RANGE',
 'PRECTOTCORR',
 'ALLSKY_SRF_ALB',
 'ALLSKY_SFC_UVA',
 'ALLSKY_SFC_LW_DWN',
 'ALLSKY_SFC_SW_DWN',
 'ALLSKY_SFC_SW_DNI',
 'ALLSKY_SFC_PAR_TOT'
 ]

class AR_ModelLoader:

    def __init__(self, order, s_order, s_month, e_month, train_size = 0.8, scale = True, yearly = True, include_test = True):

        self.order, self.s_order, self.s_month, self.e_month = order, s_order, s_month, e_month
        self.train_size = train_size
        self.scale = scale
        self.yearly = yearly
        self.include_test = include_test

    def load_data(self, pest, area, population_type, target_var, dirpath = "data/", check_temp_first = False):

        self.pest, self.area, self.population_type, self.target_var = pest, area, population_type, target_var
        self.dirpath = dirpath

        filepath = self.dirpath + self.pest + "_" + self.area + "_train.xlsx"

        if not os.path.exists(filepath) and self.pest in ['breadbeetle', 'coloradobeetle']:
            filepath = self.dirpath + self.pest + "_" + self.area + ".xlsx"

        if self.include_test:
            filepath = filepath.replace("_train", "")

        if check_temp_first:
            if os.path.isfile(filepath.replace("data/", "data/temp/")):
                filepath = filepath.replace("data/", "data/temp/")

        if self.population_type == 'adult':
            sheet_name = 0
        elif self.population_type == 'larvae':
            sheet_name = 1

        df = pd.read_excel(filepath, sheet_name = sheet_name)
        self.df_orig = deepcopy(df)

        if 'Date' in df.columns:
            self.idx_col = 'Date'
            df.index = df.Date
        elif 'ds' in df.columns:
            self.idx_col = 'ds'
            df.index = df.ds

        self.columns = df.columns.tolist()

        df['y'] = df[self.target_var]
        df['y'] = df['y'].astype(float)

        df.drop(columns = self.columns, inplace = True)

        self.df = df

        return df

    def extract_features(self, factors, thresholds = [], expectation = np.nanmean, agg_strategy = 'effective_sum', only_last_year = False):

        self.factors = factors
        self.thresholds = thresholds

        jsonpath = self.dirpath + "agroclimatology/power_data_%s_daily.json" % self.area

        df = self.df

        with open(jsonpath) as f:
            data = json.load(f)

        X = []

        if not only_last_year:
            start_idx = 0
        else:
            start_idx = df.shape[0]-1

        for i in range(start_idx, df.shape[0]):

            if not only_last_year:
                try:
                    year = str(int(df.index[i].year)-1)
                except:
                    year = str(int(df.index[i])-1)
            else:
                try:
                    year = str(int(df.index[i].year))
                except:
                    year = str(int(df.index[i]))

            start_month = self.s_month
            end_month = self.e_month

            factor_vals = []

            for i, factor in enumerate(factors):
                data_obj = data['properties']['parameter'][factor]

                var_monthly = []

                for month in range(start_month, end_month+1):
                    month_val = get_monthly_avg(data_obj, year, month, expectation = expectation)
                    var_monthly.append(month_val)

                if agg_strategy == 'effective_sum':
                    if thresholds:
                        thresh = thresholds[i]
                    else:
                        thresh = 0

                    factor_vals.append(effective_sum(var_monthly, thresh = thresh))

                elif agg_strategy == 'mean':
                    factor_vals.append(np.mean(var_monthly))

            X.append(factor_vals)

        X = np.asarray(X)

        self.X = X

        return X

    def impute(self, impute_factors = None):

        if impute_factors is None:
            impute_factors = self.factors

        df = self.df

        if not df.isna().any()[0]:
            print("No NaN values found")
            return df
        else:
            print("Found NaN values, imputing...")

        df_orig = self.df_orig
        train_size = self.train_size
        X = self.X

        if impute_factors:
            for i, factor in enumerate(self.factors):
                df_orig[factor] = X[:,i]

        data_cols = df_orig.columns.tolist()
        data_cols = [x for x in data_cols if x != self.idx_col]

        df_orig.set_index(self.idx_col, inplace=True)

        scaler = StandardScaler()
        scaler.fit(df_orig.iloc[0:int(train_size*df.shape[0])])
        df_scaled = pd.DataFrame(scaler.transform(df_orig), columns=data_cols, index=df_orig.index)

        imputer = IterativeImputer(keep_empty_features = True)
        imputer.fit(df_scaled.iloc[0:int(train_size*df.shape[0])])
        df_imputed_scaled = pd.DataFrame(imputer.transform(df_scaled), columns=data_cols, index=df_scaled.index)

        df = pd.DataFrame(scaler.inverse_transform(df_imputed_scaled), columns=data_cols, index=df_imputed_scaled.index)

        df.ds = df.index
        df['y'] = df[self.target_var]
        df['y'] = df['y'].astype(float)

        df.drop(columns = data_cols, inplace = True)

        self.df = df

        return df

    def corrcoef(self, factors):

        savepath = "results/%s/" % self.pest
        os.makedirs(savepath, exist_ok=True)

        X = self.extract_features(factors, agg_strategy = 'mean')
        y = self.df.y

        correlations = []
        pvalues = []
        for i in range(X.shape[1]):
            spearman_corr = spearmanr(X[:, i], y, nan_policy = 'omit')[0]
            spearman_pval = spearmanr(X[:, i], y, nan_policy = 'omit')[1]
            correlations.append(spearman_corr)
            pvalues.append(spearman_pval)

        correlation_df = pd.DataFrame({'Feature': factors, "Spearman's r": correlations, "Spearman's p-value": pvalues})

        correlation_df.to_csv(savepath+'%s_correlation_table.csv' % self.pest, index=False)

        seaborn.set(font_scale=0.7)
        feature_names = factors+[self.target_var]

        correlation_matrix = spearmanr(X, y, nan_policy = 'omit')[0]

        """
        correlation_threshold = 0.5
        mask = np.abs(correlation_matrix) <= correlation_threshold
        """

        plt.figure(figsize=(12, 8))
        heatmap = seaborn.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
        plt.title('Spearman Correlation Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Target')
        plt.savefig(savepath+"%s_correlation_heatmap.png" % self.pest)

    def crossval(self, conf_level = 0.9, verbose = False, save_plots = True):

        df = self.df
        train_size = self.train_size
        X = self.X
        scale = self.scale
        order = self.order
        seasonal_order = self.s_order

        if self.factors:
            with_exog = True
        else:
            with_exog = False

        y = df.y.values

        scaler = StandardScaler()

        y_true = y[int(train_size*X.shape[0]):]
        y_pred = []
        low_pred = []
        high_pred = []

        test_len = X.shape[0]-int(train_size*X.shape[0])
        print("Test points: %i" % test_len)
        print("Total n: %i" % df.shape[0])

        for i in range(X.shape[0]-test_len, X.shape[0]):
            X_train, y_train = X[0:i], y[0:i]
            X_test, y_test = [X[i]], [y[i]]

            df_train, df_test = df.iloc[0:i], df.iloc[i]

            if scale:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            if with_exog:
                mod = sm.tsa.statespace.SARIMAX(df_train.y,
                                                order=order,
                                                seasonal_order=seasonal_order,
                                                enforce_stationarity=True,
                                                enforce_invertibility=False,
                                                exog = X_train)

            else:
                mod = sm.tsa.statespace.SARIMAX(df_train.y,
                                                order=order,
                                                seasonal_order=seasonal_order,
                                                enforce_stationarity=True,
                                                enforce_invertibility=False)

            results = mod.fit(maxiter = 200, disp = False)

            if with_exog:
                fcast = results.get_forecast(steps = df_test.shape[0], exog = X_test, signal_only = True, dynamic = True)
                fcast_vals = fcast.predicted_mean
                fcast_vals[fcast_vals < 0] = 0
                t_ci = fcast.conf_int(alpha=(1-conf_level))
                t_ci['lower y'][t_ci['lower y'] < 0] = 0
                forecast = fcast_vals

            else:
                fcast = results.get_forecast(steps = df_test.shape[0], signal_only = True, dynamic = True)
                fcast_vals = fcast.predicted_mean
                fcast_vals[fcast_vals < 0] = 0
                t_ci = fcast.conf_int(alpha=(1-conf_level))
                t_ci['lower y'][t_ci['lower y'] < 0] = 0
                forecast = fcast_vals

            y_pred.append(forecast.values[0])
            low_pred.append(t_ci['lower y'].values[0])
            high_pred.append(t_ci['upper y'].values[0])

        rmse = mean_squared_error(y_true, y_pred, squared = False)
        mape = mean_absolute_percentage_error(y_true, y_pred)*100
        hitrate = calc_hitrate(y_true, y_pred, backtest = True)
        normalized_bias = calc_normbias(y_true, y_pred)
        mpe = calc_mpe(y_true, y_pred)

        if verbose: print("Results for %s" % self.pest)
        if verbose: print("CV RMSE:", rmse)
        if verbose: print("CV MAPE:", mape)
        if verbose: print("CV Hit Rate:", hitrate)
        if verbose: print("CV Normalized Bias", normalized_bias)
        if verbose: print("CV Mean Percentage Error", mpe)
        if verbose: print()

        preds_tuple = (y_pred, low_pred, high_pred)
        df_train, df_test = df.iloc[0:int(train_size*df.shape[0])], df.iloc[int(train_size*df.shape[0]):]

        if save_plots:

            savepath = "results/%s/" % self.pest
            os.makedirs(savepath, exist_ok=True)

            predictions = pd.DataFrame(y_pred, columns = ['predicted_mean'])
            predictions.reset_index(drop = True, inplace = True)
            predictions.index = df_test.index
            predictions['Actual'] = y_true
            predictions.rename(columns={'predicted_mean': 'Pred'}, inplace = True)

            train_ind = list(df_train.index)
            train_vals = list(df_train.y)

            plt.close()
            plt.clf()
            plt.plot(train_ind, train_vals, color = 'blue', label = 'Train')
            plt.plot([df_train.index[-1], df_test.index[0]], [df_train.y.values[-1], predictions['Actual'].values[0]], \
                        color = 'green', linestyle = '-.')
            plt.plot([df_train.index[-1], df_test.index[0]], [df_train.y.values[-1], predictions['Pred'].values[0]], \
                        color = 'red', linestyle = '-.')
            plt.plot(df_test.index, predictions['Actual'], color = 'green', label = 'Actual', linestyle = '-.')
            plt.scatter(df_test.index, predictions['Pred'], color = 'red')
            plt.plot(df_test.index, predictions['Pred'], color = 'red', label = 'Pred', linestyle = '-.')
            plt.fill_between(df_test.index, low_pred, high_pred, color='r', alpha=0.1, label = "90% confidence interval")
            plt.legend()
            plt.savefig(savepath+"CV_results.png")

            try:
                self.corrcoef(all_factors)
            except KeyError:
                pass

        return df_train, df_test, preds_tuple

    def predict(self, n = 1):

        scale = self.scale
        order = self.order
        seasonal_order = self.s_order

        df = self.df
        X = self.X
        X_pred = self.extract_features(self.factors, self.thresholds, only_last_year = True)

        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_pred = scaler.transform(X_pred)

        mod = sm.tsa.statespace.SARIMAX(df.y,
                                        order=order,
                                        seasonal_order=seasonal_order,
                                        enforce_stationarity=True,
                                        enforce_invertibility=False,
                                        exog = X)

        results = mod.fit(maxiter = 200, disp = False)

        fcast = results.get_forecast(steps = n, exog = X_pred, signal_only = True, dynamic = True)

        return fcast

class R_ModelLoader:

    def __init__(self, train_size = 0.8, include_test = True, random_state = 42, scale = False, \
                 regressor = 'rf', n_estimators = 100, kernel = 'rbf', C = 0.75):

        self.train_size = train_size
        self.include_test = include_test
        self.scale = scale

        if regressor == 'rf':
            model = RandomForestRegressor(n_estimators = n_estimators, random_state = random_state)
        elif regressor == 'svr':
            model = SVR(kernel = kernel, C = C)

        self.model = model

    def load_data(self, pest, area, population_type, target_var, dirpath = "data/", check_temp_first = False):

        self.pest, self.area, self.population_type, self.target_var = pest, area, population_type, target_var
        self.dirpath = dirpath

        filepath = self.dirpath + "R_" + self.pest + "_" + self.area + "_train.xlsx"

        if self.include_test:
            filepath = filepath.replace("_train", "")

        if check_temp_first:
            if os.path.isfile(filepath.replace("data/", "data/temp/")):
                filepath = filepath.replace("data/", "data/temp/")

        if self.population_type == 'adult':
            sheet_name = 0
        elif self.population_type == 'larvae':
            sheet_name = 1

        df = pd.read_excel(filepath, sheet_name = sheet_name)
        self.df_orig = deepcopy(df)

        if 'Date' in df.columns:
            self.idx_col = 'Date'
            df.index = df.Date
        elif 'ds' in df.columns:
            self.idx_col = 'ds'
            df.index = df.ds

        self.columns = df.columns.tolist()

        df['y'] = df[self.target_var]
        df['y'] = df['y'].astype(float)

        df.drop(columns = self.columns, inplace = True)

        self.df = df

        return df

    def extract_features(self, factors, thresholds = [], expectation = np.nanmean, agg_strategy = 'effective_sum', only_last_day = False):

        self.factors = factors
        self.thresholds = thresholds

        jsonpath = self.dirpath + "agroclimatology/power_data_%s_daily.json" % self.area

        df = self.df

        with open(jsonpath) as f:
            data = json.load(f)

        X = []

        if not only_last_day:
            start_idx = 1
        else:
            start_idx = df.shape[0]-1

        for i in range(start_idx, df.shape[0]):

            if not only_last_day:
                year, month, day = str(int(df.index[i].year)), str(int(df.index[i].month)), str(int(df.index[i-1].day))
            else:
                year, month, day = str(int(df.index[i].year)), str(int(df.index[i].month)), str(int(df.index[i].day))

            factor_vals = []

            for i, factor in enumerate(factors):
                data_obj = data['properties']['parameter'][factor]
                var_val = get_daily_val(data_obj, year, month, day)

                if agg_strategy == 'effective_sum':
                    if thresholds:
                        thresh = thresholds[i]
                    else:
                        thresh = 0

                    factor_vals.append(effective_sum(var_val, thresh = thresh))

                elif agg_strategy == 'mean':
                    factor_vals.append(np.mean(var_val))

            X.append(factor_vals)

        X = np.asarray(X)

        self.X = X

        return X

    def corrcoef(self, factors):

        savepath = "results/%s/" % self.pest
        os.makedirs(savepath, exist_ok=True)

        X = self.extract_features(factors, agg_strategy = 'mean')
        y = self.df.y.iloc[1:]

        correlations = []
        pvalues = []
        for i in range(X.shape[1]):
            spearman_corr = spearmanr(X[:, i], y, nan_policy = 'omit')[0]
            spearman_pval = spearmanr(X[:, i], y, nan_policy = 'omit')[1]
            correlations.append(spearman_corr)
            pvalues.append(spearman_pval)

        correlation_df = pd.DataFrame({'Feature': factors, "Spearman's r": correlations, "Spearman's p-value": pvalues})

        correlation_df.to_csv(savepath+'%s_correlation_table.csv' % self.pest, index=False)

        seaborn.set(font_scale=0.7)
        feature_names = factors+[self.target_var]

        correlation_matrix = spearmanr(X, y, nan_policy = 'omit')[0]

        """
        correlation_threshold = 0.5
        mask = np.abs(correlation_matrix) <= correlation_threshold
        """

        plt.figure(figsize=(12, 8))
        heatmap = seaborn.heatmap(correlation_matrix, annot=True, fmt=".1f", cmap='coolwarm', xticklabels=feature_names, yticklabels=feature_names)
        plt.title('Spearman Correlation Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Target')
        plt.savefig(savepath+"%s_correlation_heatmap.png" % self.pest)

    def crossval(self, verbose = False, save_plots = True):

        model = self.model
        scale = self.scale

        df = self.df.iloc[1:]
        train_size = self.train_size
        X = self.X

        y = df.y.values

        y_true = y[int(train_size*X.shape[0]):]
        y_pred = []

        test_len = X.shape[0]-int(train_size*X.shape[0])
        print("Test points: %i" % test_len)
        print("Total n: %i" % df.shape[0])

        scaler = StandardScaler()

        for i in range(X.shape[0]-test_len, X.shape[0]):
            X_train, y_train = X[0:i], y[0:i]
            X_test, y_test = [X[i]], [y[i]]

            if scale:
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_train)

            model.fit(X_train, y_train)
            y_pred.append(model.predict(X_test))

        rmse = mean_squared_error(y_true, y_pred, squared = False)
        mape = mean_absolute_percentage_error(y_true, y_pred)*100
        hitrate = calc_hitrate(y_true, y_pred, backtest = True)
        normalized_bias = calc_normbias(y_true, y_pred)
        mpe = calc_mpe(y_true, y_pred)

        if verbose: print("Results for %s" % self.pest)
        if verbose: print("CV RMSE:", rmse)
        if verbose: print("CV MAPE:", mape)
        if verbose: print("CV Hit Rate:", hitrate)
        if verbose: print("CV Normalized Bias", normalized_bias)
        if verbose: print("CV Mean Percentage Error", mpe)
        if verbose: print()

        df_train, df_test = df.iloc[0:int(train_size*df.shape[0])], df.iloc[int(train_size*df.shape[0]):]

        if save_plots:

            savepath = "results/%s/" % self.pest
            os.makedirs(savepath, exist_ok=True)

            predictions = pd.DataFrame(y_pred, columns = ['predicted_mean'])
            predictions.reset_index(drop = True, inplace = True)
            predictions.index = df_test.index
            predictions['Actual'] = y_true
            predictions.rename(columns={'predicted_mean': 'Pred'}, inplace = True)

            train_ind = list(df_train.index)
            train_vals = list(df_train.y)

            plt.close()
            plt.clf()
            plt.plot(train_ind, train_vals, color = 'blue', label = 'Train')
            plt.plot([df_train.index[-1], df_test.index[0]], [df_train.y.values[-1], predictions['Actual'].values[0]], \
                        color = 'green', linestyle = '-.')
            plt.plot([df_train.index[-1], df_test.index[0]], [df_train.y.values[-1], predictions['Pred'].values[0]], \
                        color = 'red', linestyle = '-.')
            plt.plot(df_test.index, predictions['Actual'], color = 'green', label = 'Actual', linestyle = '-.')
            plt.scatter(df_test.index, predictions['Pred'], color = 'red')
            plt.plot(df_test.index, predictions['Pred'], color = 'red', label = 'Pred', linestyle = '-.')
            plt.legend()
            plt.savefig(savepath+"CV_results.png")

            try:
                self.corrcoef(all_factors)
            except KeyError:
                pass

        return df_train, df_test, (y_pred,)

    def predict(self):

        model = self.model
        scale = self.scale

        df = self.df.iloc[1:]
        X = self.X
        X_pred = self.extract_features(self.factors, self.thresholds, only_last_day = True)

        y = df.y.values

        if scale:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            X_pred = scaler.transform(X_pred)

        model.fit(X, y)

        fcast = model.predict(X_pred)

        return fcast