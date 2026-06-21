import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Users/lenipreni/Desktop/House_Price_Prediction/House_Price_Prediction/dataset/archive (1).zip")


df = df.drop(['id', 'date'], axis=1)


for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


X = df.drop('price', axis=1)
y = df['price']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

print("Model training completed")
y_pred = model.predict(X_test)

print(y_pred[:5])
from sklearn.metrics import r2_score

print("R2 Score:", r2_score(y_test, y_pred))
from sklearn.metrics import mean_absolute_error

print("MAE:", mean_absolute_error(y_test, y_pred))
print("Model Performance:")
print("R2 Score closer to 1 = better model")
print("MAE lower = better prediction")
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())
sample = X_test.iloc[[0]]   # keep as DataFrame
prediction = model.predict(sample)

print("Predicted Price:", prediction[0])
print("Predicted Price:", prediction[0])
import pickle

pickle.dump(model, open("house_price_model.pkl", "wb"))
print("Model saved successfully")
plt.scatter(y_test, y_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()
