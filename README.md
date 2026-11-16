# energy_stress_day_prediction
Use energy and weather data to predict a stress day in term of enery consumption

Feature engineering, 
- total_flow : a total of import and export energy
- demand_sum : a total of demand over great britain
- day_of_week : a number 0-6 representing each day of week. 0 = Sunday, 1 = Monday and so on.
- dow_xx : encoding column for each day. A value of 0,1 indicates whether it belongs to that column. For example, 1 in dow_mon means that row is data of Monday
- is_weekend : indicates whether that row is data of weekend
- 

Models
- linear regression, class weight and random forest


Evaluation metric
- Classification metric, average precision score and 95% CI using bootstrap with 1,000 times of resample.


Summary
The result shows that 
