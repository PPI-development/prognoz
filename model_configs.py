from model_loaders import AR_ModelLoader
from collections import OrderedDict
import numpy as np

config = OrderedDict()

params_short = [
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

param_descriptions = OrderedDict([
    ('TS', 'Температура почвы'),
    ('T2M', 'Температура на высоте 2 метров'),
    ('T2M_MAX', 'Макс. температура на высоте 2 метров'),
    ('T2M_MIN', 'Мин. температура на высоте 2 метров'),
    ('CLOUD_AMT', 'Облачность'),
    ('T2MDEW', 'Точка росы на высоте 2 метров'),
    ('T2MWET', 'Температура влажного термометра на высоте 2 метров'),
    ('QV2M', 'Удельная влажность на высоте 2 метров'),
    ('RH2M', 'Относительная влажность на высоте 2 метров'),
    ('WS2M', 'Скорость ветра на высоте 2 метров'),
    ('GWETTOP', 'Влажность почвы - поверхностного слоя'),
    ('GWETROOT', 'Влажность почвы - корневой зоны'),
    ('T2M_RANGE', 'Разница температур на высоте 2 метров'),
    ('PRECTOTCORR', 'Осадки'),
    ('ALLSKY_SRF_ALB', 'Общий показатель отражательной способности поверхности Земли в любую погоду'),
    ('ALLSKY_SFC_UVA', 'Уфа излучение: всей поверхности неба'),
    ('ALLSKY_SFC_LW_DWN', 'Длинноволновое нисходящее излучение всей поверхности неба'),
    ('ALLSKY_SFC_SW_DWN', 'Коротковолновое нисходящее излучение'),
    ('ALLSKY_SFC_SW_DNI', 'Прямое солнечное излучение, падающее на горизонтальную плоскость'),
    ('ALLSKY_SFC_PAR_TOT', 'Общий ПАР всей поверхности неба')
])

##

id = 'breadbeetle_ZKO'

config[id] = {'order': (0,1,1), 'seasonal_order': (1,0,1,2), 's_month': 1, 'e_month': 3, 'train_size': 0.8, \
              'pest': 'breadbeetle', 'area': 'ZKO', 'population_type': 'adult', 'target_var': 'Weighted sum (ед/м^2)', \
              'features': ['QV2M', 'T2MDEW', 'T2MWET'], 'thresholds': [2.5, 2.5, 2.5],
              'imputer_features': None, 'coords': [51.14, 51.22], 'json': "power_data_ZKO_daily.json"}

##

id = 'breadbeetle-larvae_ZKO'

config[id] = {'order': (1,1,1), 'seasonal_order': (0,0,0,2), 's_month': 1, 'e_month': 2, 'train_size': 0.8, \
              'pest': 'breadbeetle', 'area': 'ZKO', 'population_type': 'larvae', 'target_var': 'Weighted sum (ед/м^2)', \
              'features': ['QV2M', 'T2MDEW', 'T2MWET'], 'thresholds': [2.5, 2.5, 2.5],
              'imputer_features': None, 'coords': [51.14, 51.22], 'json': "power_data_ZKO_daily.json"}

##

id = 'coloradobeetle_SKO'

config[id] = {'order': (1,1,1), 'seasonal_order': (0,1,1,2), 's_month': 1, 'e_month': 9, 'train_size': 0.8, \
              'pest': 'coloradobeetle', 'area': 'SKO', 'population_type': 'adult', 'target_var': 'Perc_cont', \
              'features': ['RH2M', 'T2MDEW', 'T2MWET'], 'thresholds': [3, 3, 3],
              'imputer_features': None, 'coords': [54.16, 69.94], 'json': "power_data_SKO_daily.json"}

##

id = 'coloradobeetle_ZKO'

config[id] = {'order': (1,1,1), 'seasonal_order': (1,0,2,2), 's_month': 1, 'e_month': 4, 'train_size': 0.8, \
              'pest': 'coloradobeetle', 'area': 'ZKO', 'population_type': 'adult', 'target_var': 'Perc_cont', \
              'features': ['QV2M', 'T2MDEW', 'T2MWET'], 'thresholds': [2, 2, 2],
              'imputer_features': None, 'coords': [51.14, 51.22], 'json': "power_data_ZKO_daily.json"}

##

id = 'sovka_TO'

config[id] = {'order': (1,1,1), 'seasonal_order': (1,1,1,2), 's_month': 1, 'e_month': 3, 'train_size': 0.8, \
              'pest': 'sovka', 'area': 'TO', 'population_type': 'adult', 'target_var': 'Perc_cont', \
              'features': ['WS2M', 'ALLSKY_SFC_SW_DNI', 'ALLSKY_SFC_LW_DWN', 'PRECTOTCORR'], 'thresholds': [3, 3, 3, 0.21],
              'imputer_features': ['WS2M', 'PRECTOTCORR_SUM', 'ALLSKY_SFC_LW_DWN', 'QV2M'], \
              'coords': [43.41, 68.25], 'json': "power_data_TO_daily.json"}

##

id = 'tmnt_ZKO'

config[id] = {'order': (0,1,1), 'seasonal_order': (0,1,1,2), 's_month': 1, 'e_month': 4, 'train_size': 0.75, \
              'pest': 'tmnt', 'area': 'ZKO', 'population_type': 'adult', 'target_var': 'Cont_perc', \
              'features': ['ALLSKY_SFC_LW_DWN', 'GWETROOT', 'ALLSKY_SFC_SW_DNI'], 'thresholds': [288, 0.59, 5.6],
              'imputer_features': None, 'coords': [51.14, 51.22], "json": "power_data_ZKO_daily.json"}

##

id = 'septoriosis_AKM'

config[id] = {'order': (0,1,0), 'seasonal_order': (1,0,1,2), 's_month': 5, 'e_month': 9, 'train_size': 0.8, \
              'pest': 'septoriosis', 'area': 'AKM', 'population_type': 'adult', 'target_var': 'Weighted Sum', \
              'features': ['ALLSKY_SFC_SW_DNI', 'GWETTOP'], 'thresholds': [7.8, 0.31],
              'imputer_features': None, 'coords': [45.02, 75.66], 'json': "power_data_AKM_daily.json"}

##

id = 'oak_ALM1'

config[id] = {'order': (1,0,1), 'seasonal_order': (1,0,1,2), 's_month': 1, 'e_month': 10, 'train_size': 0.8, \
              'pest': 'oak', 'area': 'ALM1', 'population_type': 'adult', 'target_var': 'average', \
              'features': ['TS', 'RH2M'], 'thresholds': [29, 47],
              'imputer_features': ['QV2M', 'T2M'], 'coords': [43.454, 77.235], 'json': "power_data_ALM1_daily.json"}

##

id = 'repens_ALM'

config[id] = {'order': (1,1,1), 'seasonal_order': (0,0,1,2), 's_month': 1, 'e_month': 5, 'train_size': 0.8, \
              'pest': 'repens', 'area': 'ALM', 'population_type': 'adult', 'target_var': 'Perc_Cont', \
              'features': ['PRECTOTCORR', 'ALLSKY_SFC_LW_DWN', 'ALLSKY_SFC_SW_DWN'], 'thresholds': [0.69, 296, 4.2],
              'imputer_features': ['T2MWET', 'WS2M'], 'coords': [43.24, 76.51], 'json': "power_data_ALM_daily.json"}

##

id = 'golden_VKO'

config[id] = {'order': (1,0,0), 'seasonal_order': (1,0,0,2), 's_month': 1, 'e_month': 10, 'train_size': 0.8, \
              'pest': 'golden', 'area': 'VKO', 'population_type': 'adult', 'target_var': 'Total area', \
              'features': ['ALLSKY_SFC_PAR_TOT', 'T2M_RANGE'], 'thresholds': [86, 13.5],
              'imputer_features': ['T2M', 'ALLSKY_SFC_SW_DNI'], 'coords': [50.35, 80.4], 'json': "power_data_VKO_daily.json"}

##

id = 'spider_ALM'

config[id] = {'order': (1,1,0), 'seasonal_order': (1,1,1,2), 's_month': 1, 'e_month': 7, 'train_size': 0.8, \
              'pest': 'spider', 'area': 'ALM', 'population_type': 'adult', 'target_var': 'Total', \
              'features': ['QV2M', 'ALLSKY_SFC_UVA'], 'thresholds': [5.8, 15],
              'imputer_features': None, 'coords': [43.24, 76.51], 'json': "power_data_ALM_daily.json"}

##

id = 'hessianfly_KOS'

config[id] = {'order': (0,0,1), 'seasonal_order': (1,0,1,2), 's_month': 2, 'e_month': 8, 'train_size': 0.8, \
              'pest': 'hessianfly', 'area': 'KOS', 'population_type': 'adult', 'target_var': 'Total', \
              'features': ['GWETTOP'], 'thresholds': [],
              'imputer_features': None, 'coords': [54.04, 65.33], 'json': "power_data_KOS_daily.json"}

##

id = 'fireblight_ENB'

config[id] = {'order': (1,1,1), 'seasonal_order': (0,1,1,2), 's_month': 2, 'e_month': 7, 'train_size': 0.8, \
              'pest': 'fireblight', 'area': 'ENB', 'population_type': 'adult', 'target_var': 'Perc_Cont', \
              'features': ['ALLSKY_SFC_SW_DNI', 'TS'], 'thresholds': [2, 10.5],
              'imputer_features': None, 'coords': [43.46, 77.69], 'json': "power_data_ENB_daily.json"}

##

id = 'R_cabbagemoth_ALM'

config[id] = {'regressor': 'rf', 'train_size': 0.8, 'n_estimators': 100, \
              'pest': 'cabbagemoth', 'area': 'ALM', 'population_type': 'adult', 'target_var': 'Perc_Cont', \
              'features': ['RH2M', 'T2M_RANGE'], 'thresholds': [75, 10.5],
              'imputer_features': None, 'coords': [43.24, 76.51], 'json': "power_data_ALM_daily.json"}

##

id = 'R_satomato_ALM'

config[id] = {'regressor': 'rf', 'train_size': 0.9, 'n_estimators': 100, \
              'pest': 'satomato', 'area': 'ALM', 'population_type': 'adult', 'target_var': 'Adult', \
              'features': ['ALLSKY_SFC_SW_DNI', 'T2M_MIN'], 'thresholds': [0.1, 6.8],
              'imputer_features': None, 'coords': [43.24, 76.51], 'json': "power_data_ALM_daily.json"}

##

id = 'R_cornborer_ENB'

config[id] = {'regressor': 'rf', 'train_size': 0.8, 'n_estimators': 100, \
              'pest': 'cornborer', 'area': 'ENB', 'population_type': 'adult', 'target_var': 'Num of catepillars for 100 plants', \
              'features': ['CLOUD_AMT', 'ALLSKY_SFC_SW_DWN'], 'thresholds': [37, 2],
              'imputer_features': None, 'coords': [43.46, 77.69], 'json': "power_data_ENB_daily.json"}

##

id = 'R_darkbeetle_KER'

config[id] = {'regressor': 'rf', 'train_size': 0.8, 'n_estimators': 100, \
              'pest': 'darkbeetle', 'area': 'KER', 'population_type': 'adult', 'target_var': 'Population', \
              'features': ['T2M', 'ALLSKY_SFC_SW_DWN'], 'thresholds': [],
              'imputer_features': None, 'coords': [44.25, 78.18], 'json': "power_data_KER_daily.json"}


##

id = 'R_clickbeetle_KER'

config[id] = {'regressor': 'svr', 'train_size': 0.8, 'kernel': 'rbf', 'C': 1, \
              'pest': 'clickbeetle', 'area': 'KER', 'population_type': 'adult', 'target_var': 'Population', \
              'features': ['T2M_MAX', 'T2M'], 'thresholds': [28, 13],
              'imputer_features': None, 'coords': [44.25, 78.18], 'json': "power_data_KER_daily.json"}

