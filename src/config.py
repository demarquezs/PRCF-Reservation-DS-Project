from scipy.stats import randint, uniform

columns_to_drop = {'columns_to_drop':['row_id', 'center_name', 'center_address_1', 'center_city', 'center_state', 'center_country',
                                    'center_zip_code', 'center_geolocation', 'center_phone_number_1', 'center_fax', 'transaction_site',
                                    'permit_number', 'permit_status', 'event_type', 'event_end_year', 'event_end_month',
                                    'customer_zip_code', 'customer_city', 'customer_state', 'center_address_2', 'center_phone_number_2',
                                    'facility_name', 'site_name', 'event_start_month', 'permit_month','day_of_week']}

client_params = {
    'domain': "citydata.mesaaz.gov",
    'dataset_identifier': "resf-c4x9",
    'limit': 495000
}


path_params = {
    'raw_data_path':"data/raw/PRCF_reservation.csv",
    'processed_data_path':"data/processed/df_filtered_to_model.csv",
}



model_params = {
    'LinearRegression': {},

    'Ridge': {
        'alpha': uniform(0.1, 10),
        'solver': ['auto', 'svd', 'lsqr', 'sparse_cg', 'sag'],
        'max_iter': randint(100, 1000),
        'tol': uniform(1e-6, 1e-2),
    },

    'Lasso': {
        'alpha': uniform(0.1, 10),
        'tol': uniform(1e-6, 1e-2),
        'selection': ['cyclic', 'random']
    },

    'ElasticNet': {
        'alpha': uniform(0.1, 10),
        'l1_ratio': uniform(0, 1),
        'tol': uniform(1e-6, 1e-2),
        'selection': ['cyclic', 'random']
    },
    'RandomForestRegressor': {
        'n_estimators': randint(50, 500),
        'max_depth': randint(10, 100),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2'],
    },

    'XGBRegressor': {
        'n_estimators': randint(50, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 1),
        'subsample': uniform(0.1, 0.9,),
        'colsample_bytree': uniform(0.1,0.8),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1),
    }

}


training_params = {
    'n_splits': 5,
    'n_iter': 4,
    'random_state': 42,
    'n_jobs': 8
}


data_params = {
    'date_column': 'event_start_time',
    'target_column': 'attendance'
}


