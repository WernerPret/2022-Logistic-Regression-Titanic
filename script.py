import codecademylib3_seaborn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the passenger data
passengers = pd.read_csv("passengers.csv")
# print(passengers)

# 2. Convert sex column to numerical
sex_keys = (passengers["Sex"].unique())
sex_vals = list(range(0,len(sex_keys)))

sex_dict = {}
for i in range(len(sex_keys)):
  sex_dict[sex_keys[i]] = sex_vals[i]
# print(sex_dict)
passengers["Sex"] = passengers["Sex"].apply(lambda x: sex_dict[x])

# 3. Fill the nan values in the age column
passengers.fillna(passengers.mean().round(0), inplace=True)

# 4, 5. First & Second Class column
passengers["FirstClass"] = passengers.Pclass.apply(lambda x: 1 if x==1 else 0)
passengers["SecondClass"] = passengers.Pclass.apply(lambda x: 1 if x==2 else 0)

# 6. Define Features
features = passengers[["Sex", "Age", "FirstClass", "SecondClass"]]
survival = passengers["Survived"]

# 7. Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival, train_size = 0.8, test_size = 0.2)

# 8. Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

# 9. Create and train the model
logger = LogisticRegression()
logger.fit(x_train, y_train)

# Score the model on the train data
data_score_train = logger.score(x_train, y_train)
# print(data_score_train)

# Score the model on the test data
data_score_test = logger.score(x_test, y_test)
# print(data_score_test)

# 12. Analyze the coefficients
feature_coefficients = logger.coef_
assess_features = list(zip(features, feature_coefficients[0]))
# print(assess_features)
# print(feature_coefficients)

# 13. Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
Werner = np.array([0, 25, 0, 0])

# 14. Combine passenger arrays
sample_passengers = [Jack, Rose, Werner]

# 15. Scale the sample passenger features
sample_scaler = StandardScaler()
sample_passengers = sample_scaler.fit_transform(sample_passengers)

# Make survival predictions!
titanic_prediction = logger.predict(sample_passengers)
titanic_probabilities = logger.predict_proba(sample_passengers)
print(titanic_probabilities)
