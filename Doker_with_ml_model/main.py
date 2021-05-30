import joblib
model=joblib.load('salaray_model.pk1')

a=float(input("Enter your experience"))

ans=model.predict([[a]])
print(type(ans))

