import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler,LabelEncoder

df = pd.read_csv(r"E:\ML\Project.py\processed_lung_cancer.csv")
scaler = StandardScaler()
df[['age', 'pack_years']] = scaler.fit_transform(df[['age', 'pack_years']])

le_gender = LabelEncoder()
le_radon = LabelEncoder()
le_asbestos = LabelEncoder()
le_secondhand = LabelEncoder()
le_copd = LabelEncoder()
le_alcohol = LabelEncoder()
le_family = LabelEncoder()
le_cancer = LabelEncoder()

# Apply to columns
df['gender'] = le_gender.fit_transform(df['gender'])
df['radon_exposure'] = le_radon.fit_transform(df['radon_exposure'])
df['asbestos_exposure'] = le_asbestos.fit_transform(df['asbestos_exposure'])
df['secondhand_smoke_exposure'] = le_secondhand.fit_transform(df['secondhand_smoke_exposure'])
df['copd_diagnosis'] = le_copd.fit_transform(df['copd_diagnosis'])
df['alcohol_consumption'] = le_alcohol.fit_transform(df['alcohol_consumption'])
df['family_history'] = le_family.fit_transform(df['family_history'])
df['lung_cancer'] = le_cancer.fit_transform(df['lung_cancer'])


X = df.drop('lung_cancer',axis=1)
y = df['lung_cancer']


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=4,)



model = KNeighborsClassifier(n_neighbors=182, weights='distance')
model.fit(X_train,y_train)
answer = model.predict(X_test)
print(answer)
print(accuracy_score(y_test,answer))
print(confusion_matrix(y_test,answer))
print(classification_report(y_test,answer))



age = float(input("Age: "))
gender_input = input("Gender (Male/Female): ").strip().lower()


smoker = input("Do you smoke? (Yes/No): ").strip().lower()
if smoker == "yes":
    cigs_per_day = float(input("How many cigarettes per day?: "))
    years_smoked = float(input("How many years have you been smoking?: "))
    pack_years = (cigs_per_day / 20) * years_smoked
else:
    pack_years = 0.0

# Radon Exposure
radon_input = input("Radon exposure (Low/Medium/High): ").strip().lower()


# Asbestos Exposure
asbestos_input = input("Exposed to asbestos? (Yes/No): ").strip().lower()


# Secondhand Smoke Exposure
secondhand_input = input("Secondhand smoke around you? (Yes/No): ").strip().lower()


# COPD Diagnosis
copd_input = input("Diagnosed with COPD? (Yes/No): ").strip().lower()


# Alcohol Consumption
alcohol_input = input("Alcohol Consumption (Not Drinking/Moderate/Heavy): ")


# Family History
family_input = input("Family history of lung cancer? (Yes/No): ").strip().lower()


# Final prediction
# Convert user input into dataframe with same columns

gender = le_gender.transform([gender_input.capitalize()])[0]
radon_exposure = le_radon.transform([radon_input.capitalize()])[0]
asbestos_exposure = le_asbestos.transform([asbestos_input.capitalize()])[0]
secondhand_smoke_exposure = le_secondhand.transform([secondhand_input.capitalize()])[0]
copd_diagnosis = le_copd.transform([copd_input.capitalize()])[0]
alcohol_consumption = le_alcohol.transform([alcohol_input])[0]
family_history = le_family.transform([family_input.capitalize()])[0]

user_df = pd.DataFrame([[age, gender, pack_years, radon_exposure, asbestos_exposure,
                         secondhand_smoke_exposure, copd_diagnosis, alcohol_consumption,
                         family_history]], columns=X.columns)
user_df[['age','pack_years']] = scaler.transform(user_df[['age','pack_years']])
prediction = model.predict(user_df)
predicted_label = le_cancer.inverse_transform([prediction[0]])[0]

if predicted_label.lower() == "yes":
    print("\nPrediction: Lung Cancer Risk")
else:
    print("\nPrediction: No Significant Risk")



