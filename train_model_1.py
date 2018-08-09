#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import joblib as jl
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


x_data = jl.load('x_dataset.jl').astype(float)
y_data = jl.load('y_dataset.jl').astype(float)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.20, random_state=42)

model = RandomForestClassifier(n_estimators=500)
model.fit(x_train, y_train)

print(model.score(x_test,y_test))


joblib.dump(model, 'ai2.0_model_1.jl') 






