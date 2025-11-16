from sodapy import Socrata
import pandas as pd
import os
from src.config import client_params, path_params, columns_to_drop, data_params
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


def load_data(custom_params=None):

        params = custom_params or client_params

        #extract the data using Socrata’s client library:
        client = Socrata(params['domain'], None)
        results = client.get(params['dataset_identifier'], limit=params['limit'])
        df = pd.DataFrame.from_records(results)

        # create directories if they do not exist and save raw data
        project_dir_l = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        raw_dir = os.path.dirname(os.path.join(project_dir_l, path_params['raw_data_path']))
        os.makedirs(raw_dir, exist_ok=True)

        #save raw data to csv
        csv_path = os.path.join(project_dir_l, path_params['raw_data_path'])
        df.to_csv(csv_path, index=False)

        return df


def filter_and_process_data(df):

    # remove the unnecessarycolumns from the dataset and NaN values:
    df_filtered = df.drop(columns=df[columns_to_drop['columns_to_drop']])
    df_filtered.dropna(axis=0, how='any', subset=None, inplace=True)
    df_filtered.reset_index(drop=True, inplace=True)


    #extract the permit hour from column 'permit_date', and create a new column
    permit_hour=[]

    for i in range(0,len(df_filtered['permit_date'])):
        permit_hour.append(df_filtered['permit_date'][i][11:13])
        
    permit_hour_col = pd.DataFrame(permit_hour, columns=['permit_hour'])

    df_filtered['permit_hour'] = permit_hour_col

    #extract the month and day from the 'event_start_date' column and create new columns
    mounth_numeric=[]
    mounth_range = slice(5, 7)

                                
    for i in range(0,len(df_filtered['event_start_date'])):
        mounth_numeric.append(df_filtered['event_start_date'][i][mounth_range])
        
    mounth_numeric_col = pd.DataFrame(mounth_numeric, columns=['event_mounth'])

    df_filtered['event_start_mounth_numeric'] = mounth_numeric_col

    df_filtered.reset_index(drop=True, inplace=True)
    day_numeric=[]

    for i in range(0,len(df_filtered['event_start_date'])):
        day_numeric.append(df_filtered['event_start_date'][i][8:10])

    day_numeric_col = pd.DataFrame(day_numeric, columns=['event_day'])

    df_filtered['event_start_day_numeric'] = day_numeric_col


    #extract the day from the 'event_end_date' column and create a new column
    df_filtered.reset_index(drop=True, inplace=True)
    day_end_numeric=[]
    day_end_range = slice(8, 10)

    for i in range(0,len(df_filtered['event_end_date'])):
        day_end_numeric.append(df_filtered['event_end_date'][i][day_end_range])

    day_end_numeric_col = pd.DataFrame(day_end_numeric, columns=['event_end_day'])
    df_filtered['event_end_day_numeric'] = day_end_numeric_col

    #format the 'event_start_time' column to datetime format
    df_filtered['event_start_time'] = pd.to_datetime(df_filtered['event_start_time'], format='%H:%M:%S')

    #change the datatype of some columns
    df_filtered['attendance'] = df_filtered['attendance'].astype(int)
    df_filtered['hours_reserved'] = df_filtered[['hours_reserved']].astype(float)
    df_filtered['center_longitude'] = df_filtered[['center_longitude']].astype(float)
    df_filtered['center_latitude'] = df_filtered[['center_latitude']].astype(float)
    df_filtered['permit_year'] = df_filtered[['permit_year']].astype(int)
    df_filtered['permit_hour'] = df_filtered[['permit_hour']].astype(int)
    df_filtered['event_start_mounth_numeric'] = df_filtered[['event_start_mounth_numeric']].astype(int)
    df_filtered['event_start_day_numeric'] = df_filtered[['event_start_day_numeric']].astype(int)
    df_filtered['event_end_day_numeric'] = df_filtered[['event_end_day_numeric']].astype(int)

    #drop the obsolete columns that will not be used
    df_filtered=df_filtered.drop(columns=['permit_date','event_start_date','event_end_date','event_end_time'])

    #reduce outliers in 'hours_reserved' column
    df_2 = df_filtered[df_filtered['hours_reserved']<=20]
    df_2.shape

    #extrac records smaller than 100
    df_3 = df_2[df_2['attendance']<=50]
    df_3.shape

    return df_3


def transform_and_split_data(df_3):

    #transform categorical columns to numerical using LabelEncoder
    le= LabelEncoder()

    #functions to encode the columns
    def label_encoder_function(dataframe, column):
        column_encoded = le.fit_transform(dataframe[column])
        return column_encoded

    def add_nwe_column(dataframe, column):
        dataframe.loc[:,f'{column}_numeric']= label_encoder_function(dataframe, column)
        return dataframe

    # encode and add new columns
    add_nwe_column(df_3, 'facility_type')
    add_nwe_column(df_3, 'schedule_type')
    add_nwe_column(df_3, 'residency_flag')
    add_nwe_column(df_3, 'customer_gender')

    #drop the original categorical columns
    df_3 = df_3.drop(columns=['facility_type', 'schedule_type', 'residency_flag', 'customer_gender', 'event_end_day_numeric'])
    df_3["event_start_year"] = pd.to_numeric(df_3["event_start_year"], errors="coerce")
    df_3= df_3[(df_3["event_start_year"] > 2014) & (df_3["event_start_year"] < 2024)]

    #save the filtered data in a CSV file
    project_dir_s = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    data_dir_s = os.path.join(project_dir_s, path_params['processed_data_path'])
    os.makedirs(data_dir_s, exist_ok=True)

    output_path = os.path.join(data_dir_s, "df_to_ml.csv")
    df_3.to_csv(output_path, index=False)

    df_3 = df_3.sort_values(by=data_params['date_column'], ascending=True).reset_index(drop=True)

    # Split into training and validation sets
    train_size = int(len(df_3) * 0.7)
    train_df, val_df = df_3.iloc[:train_size], df_3.iloc[train_size:]

    X_train = train_df.drop(columns=[data_params['target_column'], data_params['date_column']])
    y_train = train_df[data_params['target_column']]
    X_val = val_df.drop(columns=[data_params['target_column'], data_params['date_column']])
    y_val = val_df[data_params['target_column']]

    # 3️Scale based on training data only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)


    return X_train_scaled, X_val, y_train, y_val