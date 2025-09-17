import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.base import clone

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from itertools import product

import warnings
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)

def create_supervised_with_dates(df, W=25, H=1):
    """
    Genera un conjunto de datos supervisado a partir de una serie temporal,
    utilizando ventanas deslizantes y manteniendo las fechas correspondientes.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con índices de fechas y columnas de series temporales.
    W : int, optional (default=25)
        Longitud de la ventana de entrada (número de observaciones pasadas).
    H : int, optional (default=1)
        Horizonte de predicción (número de pasos a predecir).
    
    Returns
    -------
    X : numpy.ndarray
        Matriz de entradas con las ventanas de tamaño W.
    y : numpy.ndarray
        Matriz de salidas con los valores futuros a predecir.
    dates : numpy.ndarray
        Fechas correspondientes a cada salida en y.
        """
    X, y, dates = [], [], []
    for col in df.columns:
        data = df[col].values
        idx = df.index
        for i in range(len(data) - W - H + 1):
            X.append(data[i:i+W])
            y.append(data[i+W:i+W+H])
            dates.append(idx[i+W:i+W+H])
    return np.array(X), np.array(y), np.array(dates)


def compute_metrics(y_true, y_pred):
    """
    Calcula varias métricas de evaluación para modelos de predicción.
    
    Parameters
    ----------
    y_true : array-like
        Valores reales de la serie.
    y_pred : array-like
        Valores predichos por el modelo.
    
    Returns
    -------
    dict
        Diccionario con las métricas calculadas:
        - "MSE" : Error cuadrático medio.
        - "RMSE": Raíz del error cuadrático medio.
        - "MAE" : Error absoluto medio.
        - "R2"  : Coeficiente de determinación.
        - "MAPE": Error porcentual absoluto medio.
        - "SMAPE": Error porcentual absoluto medio simétrico.
        - "EVS" : Varianza explicada.
    """
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-6))) * 100
    smape = 100*np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-6))
    evs = explained_variance_score(y_true, y_pred)
    return {"MSE":mse,"RMSE":rmse,"MAE":mae,"R2":r2,"MAPE":mape,"SMAPE":smape,"EVS":evs}

# MODEL BUILDERS
def build_lstm(input_shape, units=16, dropout=0.3, H=1, **kwargs):
    """
    Construye un modelo LSTM para predicción de series temporales.
    
    Parameters
    ----------
    input_shape : tuple
        Forma de la entrada (timesteps, features).
    units : int, optional (default=16)
        Número de unidades en la capa LSTM.
    dropout : float, optional (default=0.3)
        Tasa de dropout para regularización.
    H : int, optional (default=1)
        Horizonte de predicción (número de pasos a predecir).
    **kwargs : dict
        Argumentos adicionales para compatibilidad.
    
    Returns
    -------
    keras.Sequential
        Modelo LSTM compilado con optimizador Adam y pérdida MSE.
    """
    model = Sequential([LSTM(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru(input_shape, units=16, dropout=0.3, H=1, **kwargs):
    """
    Construye un modelo GRU para predicción de series temporales.
    
    Parameters
    ----------
    input_shape : tuple
        Forma de la entrada (timesteps, features).
    units : int, optional (default=16)
        Número de unidades en la capa GRU.
    dropout : float, optional (default=0.3)
        Tasa de dropout para regularización.
    H : int, optional (default=1)
        Horizonte de predicción (número de pasos a predecir).
    **kwargs : dict
        Argumentos adicionales para compatibilidad.
    
    Returns
    -------
    keras.Sequential
        Modelo GRU compilado con optimizador Adam y pérdida MSE.
    """
    model = Sequential([GRU(units, input_shape=input_shape),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_conv1d(input_shape, units=16, dropout=0.3, kernel_size=3, H=1, **kwargs):
    """
    Construye un modelo Conv1D para predicción de series temporales.
    
    Parameters
    ----------
    input_shape : tuple
        Forma de la entrada (timesteps, features).
    units : int, optional (default=16)
        Número de filtros en la capa Conv1D.
    dropout : float, optional (default=0.3)
        Tasa de dropout para regularización.
    kernel_size : int, optional (default=3)
        Tamaño de la ventana de convolución.
    H : int, optional (default=1)
        Horizonte de predicción (número de pasos a predecir).
    **kwargs : dict
        Argumentos adicionales para compatibilidad.
    
    Returns
    -------
    keras.Sequential
        Modelo Conv1D compilado con optimizador Adam y pérdida MSE.
    """
    model = Sequential([Conv1D(filters=units, kernel_size=kernel_size, activation='relu', input_shape=input_shape),
                        Flatten(),
                        Dropout(dropout),
                        Dense(H)])
    model.compile(optimizer='adam', loss='mse')
    return model

dl_models = {"LSTM": build_lstm, "GRU": build_gru, "Conv1D": build_conv1d}

# CROSS VALIDATION
def tscv_ml(model, X, y, n_splits=5):
    """
    Realiza validación cruzada en series temporales para modelos de machine learning.
    
    Parameters
    ----------
    model : estimator
        Modelo base a entrenar (compatible con scikit-learn).
    X : numpy.ndarray
        Conjunto de entradas, de forma (n_muestras, ventana, features).
    y : numpy.ndarray
        Conjunto de salidas, de forma (n_muestras, horizonte).
    n_splits : int, optional (default=5)
        Número de divisiones para TimeSeriesSplit.
    
    Returns
    -------
    dict
        Diccionario con el promedio de métricas calculadas en cada split:
        - "MSE"   : Error cuadrático medio.
        - "RMSE"  : Raíz del error cuadrático medio.
        - "MAE"   : Error absoluto medio.
        - "R2"    : Coeficiente de determinación.
        - "MAPE"  : Error porcentual absoluto medio.
        - "SMAPE" : Error porcentual absoluto medio simétrico.
        - "EVS"   : Varianza explicada.
    """
    X_flat = X.reshape((X.shape[0], -1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for train_idx, test_idx in tscv.split(X_flat):
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_fold = MultiOutputRegressor(clone(model)) if y.shape[1] > 1 else clone(model)
        model_fold.fit(X_train_scaled, y_train if y_train.shape[1] > 1 else np.ravel(y_train))
        y_pred = model_fold.predict(X_test_scaled)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}


def tscv_dl(model_builder, X, y, n_splits=3, units=16, dropout=0.3, epochs=10, batch_size=32, kernel_size=3):
    """
    Realiza validación cruzada en series temporales para modelos de deep learning.

    Parameters
    ----------
    model_builder : callable
        Función constructora del modelo (ej. build_lstm, build_gru, build_conv1d).
    X : numpy.ndarray
        Conjunto de entradas de forma (n_muestras, ventana).
    y : numpy.ndarray
        Conjunto de salidas de forma (n_muestras, horizonte).
    n_splits : int, optional (default=3)
        Número de divisiones para TimeSeriesSplit.
    units : int, optional (default=16)
        Número de unidades/neurona en la capa principal del modelo.
    dropout : float, optional (default=0.3)
        Tasa de dropout para regularización.
    epochs : int, optional (default=10)
        Número máximo de épocas de entrenamiento.
    batch_size : int, optional (default=32)
        Tamaño del lote para el entrenamiento.
    kernel_size : int, optional (default=3)
        Tamaño de kernel (solo usado en modelos Conv1D).

    Returns
    -------
    dict
        Diccionario con el promedio de métricas calculadas en cada split:
        - "MSE"   : Error cuadrático medio.
        - "RMSE"  : Raíz del error cuadrático medio.
        - "MAE"   : Error absoluto medio.
        - "R2"    : Coeficiente de determinación.
        - "MAPE"  : Error porcentual absoluto medio.
        - "SMAPE" : Error porcentual absoluto medio simétrico.
        - "EVS"   : Varianza explicada.
    """
    X_seq = X.reshape((X.shape[0], X.shape[1], 1))
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    for train_idx, test_idx in tscv.split(X_seq):
        X_train, X_test = X_seq[train_idx], X_seq[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_2d = X_train.reshape((X_train.shape[0], -1))
        X_test_2d = X_test.reshape((X_test.shape[0], -1))
        X_train_scaled = scaler.fit_transform(X_train_2d).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test_2d).reshape(X_test.shape)
        model = model_builder(input_shape=(X_train.shape[1],1),
                              units=units, dropout=dropout, H=y_train.shape[1], kernel_size=kernel_size)
        es = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1)
        model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
                  epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es, lr])
        y_pred = model.predict(X_test_scaled, verbose=0)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

# RUN EXPERIMENT
def run_experiment(path_csv, W=25, H=1, sub_sample_ratio=0.4):
    """
    Ejecuta un experimento de predicción con múltiples modelos (ML y DL),
    evaluando su rendimiento mediante validación cruzada en series temporales.

    Parameters
    ----------
    path_csv : str
        Ruta al archivo CSV con precios históricos (índice temporal y columnas por empresa).
    W : int, optional (default=25)
        Tamaño de la ventana temporal utilizada como entrada.
    H : int, optional (default=1)
        Horizonte de predicción (número de pasos futuros a estimar).
    sub_sample_ratio : float, optional (default=0.4)
        Proporción de columnas/empresas seleccionadas aleatoriamente para el experimento.

    Returns
    -------
    results : dict
        Métricas de rendimiento (R2) obtenidas para cada modelo.
    best_models : dict
        Mejor configuración encontrada para cada modelo (hiperparámetros o arquitectura).
    df_sub : pandas.DataFrame
        Subconjunto de datos utilizado en el experimento.
    """
    np.random.seed(SEED)
    df_weekly = pd.read_csv(path_csv, index_col=0, parse_dates=True)
    df_weekly.index = pd.to_datetime(df_weekly.index, utc=True)

    sampled_cols = np.random.choice(df_weekly.columns, int(len(df_weekly.columns)*sub_sample_ratio), replace=False)
    df_sub = df_weekly[sampled_cols]

    X_sub, y_sub, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X_sub, y_sub = np.nan_to_num(X_sub), np.nan_to_num(y_sub)

    xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
    lgb_params = {"n_estimators":[100,200], "max_depth":[5,10], "learning_rate":[0.01,0.05]}
    dl_params = {'units':[16,32], 'dropout':[0.2,0.3]}
    c1d_params = {'units':[16,32], 'dropout':[0.2,0.3], 'kernel_size':[3,5]}

    ml_models = {"XGB": XGBRegressor, "LGBM": LGBMRegressor}
    results, best_models = {}, {}

    # ML
    for name, ModelClass in ml_models.items():
        best_score, best_model = -np.inf, None
        param_grid = xgb_params if name=="XGB" else lgb_params
        for params in ParameterGrid(param_grid):
            print(f"\nEntrenando {name} con {params}")
            model = ModelClass(random_state=SEED, verbosity=0, **params) if name=="XGB" else ModelClass(random_state=SEED, verbose=-1, **params)
            metrics = tscv_ml(model, X_sub, y_sub, n_splits=3)
            print(f"R2={metrics['R2']:.4f}")
            if metrics['R2'] > best_score:
                best_score, best_model = metrics['R2'], model
        results[name] = best_score
        best_models[name] = best_model

    # DL
    for name, builder in dl_models.items():
        best_score, best_config = -np.inf, None
        combos = product(c1d_params['units'], c1d_params['dropout'], c1d_params['kernel_size']) if name=="Conv1D" else product(dl_params['units'], dl_params['dropout'])
        for combo in combos:
            config = {'units':combo[0], 'dropout':combo[1]} if name!="Conv1D" else {'units':combo[0],'dropout':combo[1],'kernel_size':combo[2]}
            print(f"\nEntrenando {name} con {config}")
            metrics = tscv_dl(builder, X_sub, y_sub, n_splits=3, **config)
            print(f"R2={metrics['R2']:.4f}")
            if metrics['R2'] > best_score:
                best_score, best_config = metrics['R2'], config
        results[name] = best_score
        best_models[name] = best_config

    return results, best_models, df_sub

# BAGGING AND PREDICTIONS
def run_bagging_and_forecast(df_sub, W=25, H=1, best_models=None, top_models=None, n_bags=5):
    """
    Ejecuta un esquema de Bagging con los mejores modelos y genera predicciones futuras.

    Parameters
    ----------
    df_sub : pandas.DataFrame
        Subconjunto de datos con series temporales (una columna por empresa).
    W : int, optional (default=25)
        Tamaño de la ventana temporal utilizada como entrada.
    H : int, optional (default=1)
        Horizonte de predicción (número de pasos futuros a estimar).
    best_models : dict, optional
        Modelos o configuraciones óptimas obtenidas en la fase de búsqueda de hiperparámetros.
    top_models : list of str, optional
        Lista de modelos a incluir en el ensamble. Por defecto ["XGB", "Conv1D"].
    n_bags : int, optional (default=5)
        Número de repeticiones en el ensamble Bagging por cada modelo.

    Returns
    -------
    y_pred_last : numpy.ndarray
        Predicciones futuras para cada empresa en df_sub.
        - Forma (n_empresas, H) si H > 1.
        - Forma (n_empresas,) si H = 1.
    metrics : dict
        Métricas de evaluación promedio en el conjunto de test:
        - "MSE"   : Error cuadrático medio.
        - "RMSE"  : Raíz del error cuadrático medio.
        - "MAE"   : Error absoluto medio.
        - "R2"    : Coeficiente de determinación.
        - "MAPE"  : Error porcentual absoluto medio.
        - "SMAPE" : Error porcentual absoluto medio simétrico.
        - "EVS"   : Varianza explicada.
    """
    if top_models is None:
        top_models = ["XGB", "Conv1D"]

    X, y, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X, y = np.nan_to_num(X), np.nan_to_num(y)
    X_flat = X.reshape((X.shape[0], -1))

    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X_flat))[-1]
    X_train, X_test = X_flat[train_idx], X_flat[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Bagging
    bagging_preds = np.zeros_like(y_test, dtype=float)
    for name in top_models:
        preds_bag = np.zeros_like(y_test, dtype=float)
        for _ in range(n_bags):
            if name in ["XGB", "LGBM"]:
                base = clone(best_models[name])
                if H > 1:
                    model = MultiOutputRegressor(base)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                else:
                    base.fit(X_train, np.ravel(y_train))
                    preds = base.predict(X_test).reshape(-1,1)
            else:
                X_train_seq = X_train.reshape((X_train.shape[0], W, 1))
                X_test_seq = X_test.reshape((X_test.shape[0], W, 1))
                config = best_models[name]
                model = dl_models[name](input_shape=(W,1), H=H, **config)
                es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
                model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
                          epochs=20, batch_size=32, verbose=0, callbacks=[es, lr])
                preds = model.predict(X_test_seq, verbose=0)
            preds_bag += preds
        preds_bag /= n_bags
        bagging_preds += preds_bag
    bagging_preds /= len(top_models)

    metrics = compute_metrics(y_test, bagging_preds)
    print(f"Métricas Bagging (H={H}):", {k: round(v,4) for k,v in metrics.items()})

    # Reentreno con todo
    X_full, y_full, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X_full, y_full = np.nan_to_num(X_full), np.nan_to_num(y_full)
    X_full_flat = X_full.reshape((X_full.shape[0], -1))
    X_full_scaled = scaler.fit_transform(X_full_flat)

    trained_models = {}
    for name in top_models:
        if name in ["XGB", "LGBM"]:
            base = clone(best_models[name])
            if H > 1:
                model = MultiOutputRegressor(base)
                model.fit(X_full_scaled, y_full)
            else:
                base.fit(X_full_scaled, np.ravel(y_full))
                model = base
        else:
            X_full_seq = X_full_scaled.reshape((X_full_scaled.shape[0], W, 1))
            config = best_models[name]
            model = dl_models[name](input_shape=(W,1), H=H, **config)
            es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
            model.fit(X_full_seq, y_full, epochs=20, batch_size=32, verbose=0, callbacks=[es, lr])
        trained_models[name] = model

    # Predicción futura
    n_companies = df_sub.shape[1]
    y_pred_last = np.zeros((n_companies, H)) if H > 1 else np.zeros(n_companies)

    for i, company in enumerate(df_sub.columns):
        series = df_sub[company].values
        last_window = series[-W:].reshape(1, -1)
        last_window_scaled = scaler.transform(last_window)

        preds_models = []
        for name, model in trained_models.items():
            if name in ["XGB", "LGBM"]:
                preds_models.append(model.predict(last_window_scaled))
            else:
                last_seq = last_window_scaled.reshape((1, W, 1))
                preds_models.append(model.predict(last_seq, verbose=0))
        preds_models = [np.ravel(p) for p in preds_models]
        preds_models = np.vstack(preds_models)
        y_pred_last[i] = preds_models.mean(axis=0)

    return y_pred_last, metrics

# RECURSIVE FORECASTING
def recursive_forecast_H1(df_sub, trained_model_H1, scaler, W=25, steps=4):
    """
    Genera predicciones recursivas a partir de un modelo entrenado para H=1.

    Parameters
    ----------
    df_sub : pandas.DataFrame
        Subconjunto de datos con series temporales (una columna por empresa).
    trained_model_H1 : object
        Modelo entrenado para horizonte H=1 (puede ser de ML o DL).
    scaler : sklearn.preprocessing.StandardScaler
        Objeto de escalado utilizado en el entrenamiento.
    W : int, optional (default=25)
        Tamaño de la ventana temporal utilizada como entrada.
    steps : int, optional (default=4)
        Número de pasos futuros a predecir de forma recursiva.

    Returns
    -------
    y_pred_recursive : numpy.ndarray
        Predicciones recursivas para cada empresa, de forma (n_empresas, steps).
    """
    n_companies = df_sub.shape[1]
    y_pred_recursive = np.zeros((n_companies, steps))

    for i, company in enumerate(df_sub.columns):
        series = df_sub[company].values
        window = series[-W:].reshape(1, -1)
        for s in range(steps):
            window_scaled = scaler.transform(window)
            if hasattr(trained_model_H1, "predict"):  # ML
                pred = trained_model_H1.predict(window_scaled)
            else:  # DL
                pred = trained_model_H1.predict(window_scaled.reshape((1, W, 1)), verbose=0)
            pred = float(np.ravel(pred))
            y_pred_recursive[i, s] = pred
            window = np.roll(window, -1)
            window[0, -1] = pred
    return y_pred_recursive


# PLOT FORECASTS
def plot_forecasts_full(df_sub, y_pred_last_H1, y_pred_last_H4, H1=1, H4=4, title="Predicciones (Full)"):
    """
    Genera gráficos de predicciones históricas y futuras (H=1 y H=4)
    para las 6 empresas con mayor crecimiento absoluto previsto.

    Parameters
    ----------
    df_sub : pandas.DataFrame
        Subconjunto de datos con series históricas (una columna por empresa).
    y_pred_last_H1 : numpy.ndarray
        Predicciones de horizonte H=1 (un paso adelante).
    y_pred_last_H4 : numpy.ndarray
        Predicciones de horizonte H=4 (múltiples pasos).
    H1 : int, optional (default=1)
        Horizonte usado en las predicciones H=1.
    H4 : int, optional (default=4)
        Horizonte usado en las predicciones H=4.
    title : str, optional
        Título de la figura.

    Returns
    -------
    None
        Muestra los gráficos en pantalla.
    """
    last_values = df_sub.iloc[-1].values
    diff_abs = y_pred_last_H4[:,-1] - last_values
    top6_idx = np.argsort(diff_abs)[-6:][::-1]

    fig, axes = plt.subplots(3,2, figsize=(14,10))
    axes = axes.flatten()
    for j, idx in enumerate(top6_idx):
        company = df_sub.columns[idx]
        series = df_sub[company]
        last_date = series.index[-1]
        axes[j].plot(series.index, series.values, color="black", label="Histórico")
        next_date = last_date + pd.Timedelta(weeks=1)
        pred_val_H1 = float(y_pred_last_H1[idx])
        axes[j].plot([last_date, next_date],
                     [series.values[-1], pred_val_H1],
                     "bo--", label=f"Pred H={H1}")
        future_dates = [last_date + pd.Timedelta(weeks=k) for k in range(1,H4+1)]
        full_dates = [last_date] + future_dates
        full_preds = [series.values[-1]] + list(y_pred_last_H4[idx])
        axes[j].plot(full_dates, full_preds, "ro-", label=f"Pred H={H4}")
        axes[j].set_title(company)
        axes[j].legend()
        axes[j].grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_forecasts_zoom(df_sub, y_pred_last_H1, y_pred_last_H4, H1=1, H4=4, title="Predicciones (Zoom)"):
    """
    Genera gráficos comparativos entre predicciones recursivas H=1 y directas H=4
    para las 6 empresas con mayor crecimiento absoluto previsto.

    Parameters
    ----------
    df_sub : pandas.DataFrame
        Subconjunto de datos con series históricas (una columna por empresa).
    y_pred_last_H1 : numpy.ndarray
        Predicciones recursivas basadas en horizonte H=1.
    y_pred_last_H4 : numpy.ndarray
        Predicciones directas basadas en horizonte H=4.
    H1 : int, optional (default=1)
        Horizonte usado en las predicciones H=1 recursivas.
    H4 : int, optional (default=4)
        Horizonte usado en las predicciones H=4 directas.
    title : str, optional
        Título de la figura.

    Returns
    -------
    None
        Muestra los gráficos en pantalla.
    """
    last_values = df_sub.iloc[-1].values
    diff_abs = y_pred_last_H4[:,-1] - last_values
    top6_idx = np.argsort(diff_abs)[-6:][::-1]

    fig, axes = plt.subplots(3,2, figsize=(14,10))
    axes = axes.flatten()
    for j, idx in enumerate(top6_idx):
        company = df_sub.columns[idx]
        series = df_sub[company]
        last_date = series.index[-1]

        # H=1 recursivo
        future_dates = [last_date + pd.Timedelta(weeks=k) for k in range(1,H4+1)]
        full_dates = [last_date] + future_dates
        full_preds_H1 = [series.values[-1]] + list(y_pred_last_H1[idx])
        axes[j].plot(full_dates, full_preds_H1, "bo--", label=f"Pred H=1 recursivo")

        # H=4 directo
        full_preds_H4 = [series.values[-1]] + list(y_pred_last_H4[idx])
        axes[j].plot(full_dates, full_preds_H4, "ro-", label=f"Pred H=4 directo")

        axes[j].set_title(company)
        axes[j].legend()
        axes[j].grid(True)
        axes[j].tick_params(axis="x", rotation=45)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# MAIN
if __name__=="__main__":
    path_csv = "stock_weekly_clean.csv"

    # ---------- H=1 ----------
    print("\n=== Experimento H=1 ===")
    res1, models1, df_sub1 = run_experiment(path_csv, H=1)
    y_pred_last_H1, metrics_H1 = run_bagging_and_forecast(df_sub1, W=25, H=1, best_models=models1)

    # Guardamos scaler y modelo H=1 para el forecast recursivo
    X_full, y_full, _ = create_supervised_with_dates(df_sub1, W=25, H=1)
    X_full = np.nan_to_num(X_full)
    X_full_flat = X_full.reshape((X_full.shape[0], -1))
    scaler_H1 = StandardScaler().fit(X_full_flat)

    best_model_name = max(res1, key=res1.get)
    if best_model_name in ["XGB", "LGBM"]:
        trained_model_H1 = clone(models1[best_model_name])
        trained_model_H1.fit(scaler_H1.transform(X_full_flat), np.ravel(y_full))
    else:
        X_full_seq = scaler_H1.transform(X_full_flat).reshape((X_full.shape[0], 25, 1))
        config = models1[best_model_name]
        trained_model_H1 = dl_models[best_model_name](input_shape=(25,1), H=1, **config)
        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        trained_model_H1.fit(X_full_seq, y_full, epochs=20, batch_size=32, verbose=0, callbacks=[es, lr])

    y_pred_last_H1_recursive = recursive_forecast_H1(df_sub1, trained_model_H1, scaler_H1, W=25, steps=4)

    # ---------- H=4 ----------
    print("\n=== Experimento H=4 ===")
    res4, models4, df_sub4 = run_experiment(path_csv, H=4)
    y_pred_last_H4, metrics_H4 = run_bagging_and_forecast(df_sub4, W=25, H=4, best_models=models4)

    # ---------- Comparaciones ----------
    print("\nComparacion R2 medio por modelo:")
    for k in set(res1.keys()) | set(res4.keys()):
        print(f"{k}: H=1: {res1.get(k,'-')}, H=4: {res4.get(k,'-')}")

    print("\n=== Gráfico comparando H=1 recursivo vs H=4 directo ===")
    plot_forecasts_zoom(df_sub4, y_pred_last_H1_recursive, y_pred_last_H4,
                        H1=1, H4=4, title="Zoom: H=1 recursivo vs H=4 directo")
