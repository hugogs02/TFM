# TFM
Trabajo Final de Máster: Aplicación de algoritmos de Machine Learning para predicción de rendimiento de acciones. Máster en Big Data, Data Science &amp; Inteligencia Artificial de la UCM.

Este código está compuesto por un archivo que realiza un EDA de los datos y los limpia y formatea. A continuación hay dos archivos, code_H1.py y code_H4.py, que entrenan los modelos de ML sobre una submuestra, aplican bagging y reentrenan los modelos sobre el dataset completo, obteniendo las predicciones correspondientes. Además, hay un archivo, codigo_ML.py, que intenta replicar el comportamiento de los anteriores pero en un script unificado, obteniendo los modelos y predicciones para ambos horizontes en la misma ejecución.

Para la ejecución correcta del código, son necesarios los siguientes paquetes en las siguientes versiones:

numpy==1.26.4
pandas==2.3.2
matplotlib==3.10.6
seaborn==0.13.2
scikit-learn==1.7.1
xgboost==3.0.5
lightgbm==4.6.0
tensorflow==2.17.1
keras==3.11.3
statsmodels==0.14.5
scipy==1.16.1
joblib==1.5.2
tqdm==4.67.1
