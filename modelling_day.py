import pandas as pd
import numpy as np
# import data
files = [f"demanddata_{y}_cleaned_day.csv" for y in range(2019, 2024)]

training_data = pd.concat(
    [pd.read_csv(f, index_col="datetime", parse_dates=True) for f in files]
).sort_index()

validation_data = pd.read_csv('demanddata_2024_cleaned_day.csv')
testing_data = pd.read_csv('demanddata_2025_cleaned_day.csv')

training_data = training_data.dropna()
validation_data = validation_data.dropna()
testing_data = testing_data.dropna()

print(f'training size : {len(training_data)}')
print(f'testing size : {len(testing_data)}')

feature = ['dow_mon','dow_tue','dow_wed','dow_thu','dow_fri','dow_sat','dow_sun','is_weekend','total_ytd_flow','renew_ytd_generation','final_ytd_demand','same_last_week_demand']
target = 'stress'

X_train, y_train = training_data[feature], training_data[target]
X_val, y_val = validation_data[feature], validation_data[target]
X_test, y_test = testing_data[feature], testing_data[target]
print(y_train.shape)
print('data has been split to train and test set')

# training with class weight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model_weight = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight='balanced', max_iter=1000))

model_weight.fit(X_train, y_train)

# training with randomforest
from sklearn.ensemble import RandomForestClassifier

model_forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, class_weight='balanced', random_state=42)

model_forest.fit(X_train, y_train)

# validate
pred_val_weight = model_weight.predict(X_val)
prob_val_weight = model_weight.predict_proba(X_val)[:,1]
print(f'weight prediction : {pred_val_weight}')
unique_weight, count_weight = np.unique_counts(pred_val_weight)
print(dict(zip(unique_weight, count_weight)))

pred_val_forest = model_forest.predict(X_val)
prob_val_forest = model_forest.predict_proba(X_val)[:,1]
print(f'\nforest prediction : {pred_val_forest}')
unique_forest, count_forest = np.unique_counts(pred_val_forest)
print(dict(zip(unique_forest, count_forest)))


# make prediction
pred_weight = model_weight.predict(X_test)
prob_weight = model_weight.predict_proba(X_test)[:,1]
print(f'weight prediction : {pred_weight}')
unique_weight, count_weight = np.unique_counts(pred_weight)
print(dict(zip(unique_weight, count_weight)))

pred_forest = model_forest.predict(X_test)
prob_forest = model_forest.predict_proba(X_test)[:,1]
print(f'\nforest prediction : {pred_forest}')
unique_forest, count_forest = np.unique_counts(pred_forest)
print(dict(zip(unique_forest, count_forest)))


# see coefficient
logistic = model_weight.named_steps['logisticregression']
coef = pd.DataFrame({
    'Feature' : feature,
    'Coefficient' : logistic.coef_[0]})

coef['abs_coef'] = coef['Coefficient'].abs()
coef = coef.sort_values(by='abs_coef', ascending=False)
print(coef)

# see result
from sklearn.metrics import classification_report, average_precision_score
print('\n ########### VALIDATION ###########')
print(f' Class weight : \n{classification_report(y_val, pred_val_weight)}')
print(f' Class weight prob : {average_precision_score(y_val, prob_val_weight)}')

print(f'\nrandom forest : \n{classification_report(y_val, pred_val_forest)}')
print(f'random forest prob : {average_precision_score(y_val, prob_val_forest)}')

# baseline model

baseline = LogisticRegression(max_iter=2000).fit(X_train, y_train)
baseline_pred = baseline.predict(X_test)
baseline_prob = baseline.predict_proba(X_test)[:,1]
print('Baseline result : \n', classification_report(y_test, baseline_pred))
print(f'baseline prob : {average_precision_score(y_test, baseline_prob)}')



print('\n ########### FINAL RESULT ###########')
print(f' Class weight : \n{classification_report(y_test, pred_weight)}')
print(f' Class weight prob : {average_precision_score(y_test, prob_weight)}')

print(f'\nrandom forest : \n{classification_report(y_test, pred_forest)}')
print(f'random forest prob : {average_precision_score(y_test, prob_forest)}')

rng = np.random.default_rng(42)

y = y_test.to_numpy()
p = prob_weight
n = len(y)
block = 7
n_blocks = int(np.ceil(n / block))

scores = []
for _ in range(1000):
    starts = rng.integers(0, n - block + 1, n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    if y[idx].sum() == 0:
        continue
    scores.append(average_precision_score(y[idx], p[idx]))

low, high = np.percentile(scores, [2.5, 97.5])
print(f'AP 95% CI of class weight: {low:.3f} to {high:.3f}')

rng = np.random.default_rng(42)

y = y_test.to_numpy()
p = prob_forest
n = len(y)
block = 7                              # 7 days for daily data (use 24*7 for hourly)
n_blocks = int(np.ceil(n / block))

scores = []
for _ in range(1000):
    starts = rng.integers(0, n - block + 1, n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    if y[idx].sum() == 0:
        continue
    scores.append(average_precision_score(y[idx], p[idx]))

low, high = np.percentile(scores, [2.5, 97.5])
print(f'AP 95% CI of random forest: {low:.3f} to {high:.3f}')

y = y_test.to_numpy()
p = baseline_prob
n = len(y)
block = 7                              # 7 days for daily data (use 24*7 for hourly)
n_blocks = int(np.ceil(n / block))

scores = []
for _ in range(1000):
    starts = rng.integers(0, n - block + 1, n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    if y[idx].sum() == 0:
        continue
    scores.append(average_precision_score(y[idx], p[idx]))

low, high = np.percentile(scores, [2.5, 97.5])
print(f'AP 95% CI of baseline result : {low:.3f} to {high:.3f}')





