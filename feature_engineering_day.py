import pandas as pd
import requests




def data_preprocessing(filename, start_date, end_date, is_scotish):
    electric_df = pd.read_csv(filename)

    # change datetime datatype
    electric_df["SETTLEMENT_DATE"] = pd.to_datetime(electric_df["SETTLEMENT_DATE"])
    
    # define time zone -> start = midnight of London timezone
    start = electric_df["SETTLEMENT_DATE"].dt.tz_localize("Europe/London")
    electric_df["datetime"] = (start + pd.to_timedelta((electric_df["SETTLEMENT_PERIOD"] - 1) * 30, unit="min")).dt.tz_convert("UTC")
    
    if is_scotish:
        daily = electric_df.groupby(['datetime']).agg(
            demand_sum = ('ENGLAND_WALES_DEMAND','mean'),
            wind_generation = ('EMBEDDED_WIND_GENERATION','sum'),
            wind_capacity = ('EMBEDDED_WIND_CAPACITY','sum'),
            NON_BM_STOR = ('NON_BM_STOR','sum'),
            pump_generation = ('PUMP_STORAGE_PUMPING','sum'),
            scotish_flow = ('SCOTTISH_TRANSFER','sum'),
            IFA_FLOW = ('IFA_FLOW','sum'),
            IFA2_FLOW = ('IFA2_FLOW','sum'),
            BRITNED_FLOW = ('BRITNED_FLOW','sum'),
            MOYLE_FLOW = ('MOYLE_FLOW','sum'),
            EAST_WEST_FLOW = ('EAST_WEST_FLOW','sum'),
            NEMO_FLOW = ('NEMO_FLOW','sum'),
            NSL_FLOW = ('NSL_FLOW','sum'),
            ELECLINK_FLOW = ('ELECLINK_FLOW','sum'),
            VIKING_FLOW = ('VIKING_FLOW','sum'),
            GREENLINK_FLOW = ('GREENLINK_FLOW','sum'),
            solar_generation = ('EMBEDDED_SOLAR_GENERATION','sum'),
            solar_capacity = ('EMBEDDED_SOLAR_CAPACITY','sum')).round(2)
        
        cols = ['scotish_flow',"IFA_FLOW", "IFA2_FLOW", "BRITNED_FLOW", "MOYLE_FLOW",'EAST_WEST_FLOW','NEMO_FLOW','NSL_FLOW','ELECLINK_FLOW','VIKING_FLOW','GREENLINK_FLOW','NON_BM_STOR']
        daily["total_flow"] = daily[cols].sum(axis=1)
        daily["total_ytd_flow"] = daily["total_flow"].shift(1)
        daily['final_demand'] = daily['demand_sum']/2 + daily['total_flow']
        daily['final_ytd_demand'] = daily['final_demand'].shift(1)
    else:
        daily = electric_df.groupby(['datetime']).agg(
            demand_sum = ('ENGLAND_WALES_DEMAND','mean'),
            wind_generation = ('EMBEDDED_WIND_GENERATION','sum'),
            wind_capacity = ('EMBEDDED_WIND_CAPACITY','sum'),
            NON_BM_STOR = ('NON_BM_STOR','sum'),
            pump_generation = ('PUMP_STORAGE_PUMPING','sum'),
            IFA_FLOW = ('IFA_FLOW','sum'),
            IFA2_FLOW = ('IFA2_FLOW','sum'),
            BRITNED_FLOW = ('BRITNED_FLOW','sum'),
            MOYLE_FLOW = ('MOYLE_FLOW','sum'),
            EAST_WEST_FLOW = ('EAST_WEST_FLOW','sum'),
            NEMO_FLOW = ('NEMO_FLOW','sum'),
            NSL_FLOW = ('NSL_FLOW','sum'),
            ELECLINK_FLOW = ('ELECLINK_FLOW','sum'),
            VIKING_FLOW = ('VIKING_FLOW','sum'),
            GREENLINK_FLOW = ('GREENLINK_FLOW','sum'),
            solar_generation = ('EMBEDDED_SOLAR_GENERATION','sum'),
            solar_capacity = ('EMBEDDED_SOLAR_CAPACITY','sum')).round(2)
        
        cols = ["IFA_FLOW", "IFA2_FLOW", "BRITNED_FLOW", "MOYLE_FLOW",'EAST_WEST_FLOW','NEMO_FLOW','NSL_FLOW','ELECLINK_FLOW','VIKING_FLOW','GREENLINK_FLOW','NON_BM_STOR']
        
        daily["total_flow"] = daily[cols].sum(axis=1)
        daily["total_ytd_flow"] = daily["total_flow"].shift(1)
        daily['final_demand'] = daily['demand_sum']/2 + daily['total_flow']
        daily['final_ytd_demand'] = daily['final_demand'].shift(1)
    
    
    daily['same_yesterday_demand'] = daily['demand_sum'].shift(1)/2
    daily['same_last_week_demand'] = daily['demand_sum'].shift(7)
    daily['weekly_avg'] = daily['demand_sum'].shift(1).rolling(7).mean().round(2)
    
    days = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    daily['day_of_week'] = daily.index.dayofweek
    daily['is_weekend'] = (daily['day_of_week']>=5).astype(int)
    daily['day_of_week'] = pd.Categorical(daily['day_of_week'], categories=range(7))

    dummies = pd.get_dummies(daily['day_of_week'], prefix='dow', dtype=int)
    dummies.columns = [f'dow_{d}' for d in days]

    daily = pd.concat([daily, dummies], axis=1)
    daily = daily.drop(columns=cols)
    
    daily = daily.drop(columns='demand_sum')
    daily['renew_generation'] = daily['solar_generation'] + daily['wind_generation']
    daily['renew_ytd_generation'] = daily['solar_generation'].shift(1) + daily['wind_generation'].shift(1)
    
    
    print('Download weather information....')
    # import weather data
    url = "https://archive-api.open-meteo.com/v1/era5"
    params = {
        "latitude": 53.4,
        "longitude": -2.24,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_mean,wind_speed_10m_mean",
        "timezone": "UTC",
    }
    
    r = requests.get(url, params=params)
    r.raise_for_status()
    weather_daily = pd.DataFrame(r.json()["daily"])
    print('Download weather information successfully')
    weather_daily["datetime"] = pd.to_datetime(weather_daily["time"])
    
    # add timezone
    weather_daily["datetime"] = weather_daily["datetime"].dt.tz_localize("UTC")

    
    
    data = daily.merge(weather_daily, on='datetime', how='inner')
    data = data.drop(columns='time')
    
    print('Add weather information successfully')
    
    # add stress column
    data['stress'] = (data['final_demand'] > data['final_demand'].quantile(0.85)) & (data['renew_generation'] < data['renew_generation'].quantile(0.3))
    # save data
    data.to_csv(filename.replace(".csv", "_cleaned_day.csv"))
    
    print('data is ready to be used <3')

data_preprocessing('demanddata_2019.csv','2019-01-01','2019-12-31',False)
data_preprocessing('demanddata_2020.csv','2020-01-01','2020-12-31',False)
data_preprocessing('demanddata_2021.csv','2021-01-01','2021-12-31',False)
data_preprocessing('demanddata_2022.csv','2022-01-01','2022-12-31',False)
data_preprocessing('demanddata_2023.csv','2023-01-01','2023-12-31',True)
data_preprocessing('demanddata_2024.csv','2024-01-01','2024-12-31',True)
data_preprocessing('demanddata_2025.csv','2025-01-01','2025-12-31',True)















