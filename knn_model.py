import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = df.dropna().sample(n=2000, random_state=42)  # Reducing dataset size


    reverse_mappings = {}
    # Encode categorical variables and save reverse mappings
    feas = ["Job Title", "Key Skills", "Role Category", "Location", "Functional Area", "Industry", "Role"]
    for col in feas:
        df[col], mapping = pd.factorize(df[col])
        reverse_mappings[col] = dict(enumerate(mapping))

    return df, reverse_mappings

def train_knn(df, features, target):
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    return knn_model

def predict_knn(model, user_features):
    return model.predict([user_features])[0]



