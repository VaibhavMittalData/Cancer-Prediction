import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =====================
# Load Data
# =====================
df = pd.read_csv(r"E:\ML\Project.py\processed_lung_cancer.csv")

scaler = StandardScaler()
df[['age', 'pack_years']] = scaler.fit_transform(df[['age', 'pack_years']])

encoders = {}
cols = [
    'gender','radon_exposure','asbestos_exposure',
    'secondhand_smoke_exposure','copd_diagnosis',
    'alcohol_consumption','family_history','lung_cancer'
]

for col in cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df.drop('lung_cancer', axis=1)
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.8, random_state=42
)

# =====================
# Models (FAST)
# =====================
nb = GaussianNB()
knn = KNeighborsClassifier(n_neighbors=5)
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
sv = SVC(kernel='rbf', probability=True, random_state=42)

voting_model = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('knn', knn),
        ('dt', dt),
        ('sv', sv)
        
    ],
    voting='hard'   # ✅ FAST (no probabilities)
)

voting_model.fit(X_train, y_train)

pred = voting_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

# =====================
# User Input
# =====================
# age = float(input("Age: "))
# gender = encoders['gender'].transform([input("Gender (Male/Female): ").capitalize()])[0]

# smoker = input("Do you smoke? (Yes/No): ").lower()
# if smoker == "yes":
#     cigs = float(input("Cigarettes per day: "))
#     years = float(input("Years smoked: "))
#     pack_years = (cigs / 20) * years
# else:
#     pack_years = 0.0

# radon = encoders['radon_exposure'].transform([input("Radon (Low/Medium/High): ").capitalize()])[0]
# asbestos = encoders['asbestos_exposure'].transform([input("Asbestos (Yes/No): ").capitalize()])[0]
# secondhand = encoders['secondhand_smoke_exposure'].transform([input("Secondhand]()
