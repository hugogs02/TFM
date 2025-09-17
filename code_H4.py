import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Conv1D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from itertools import product
from sklearn.base import clone
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

SEED = 42

def create_supervised_with_dates(df, W=25, H=4):
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
        for i in range(len(data)-W-H+1):
            X.append(data[i:i+W])
            y.append(data[i+W:i+W+H])
            dates.append(idx[i+W:i+W+H])
    return np.array(X), np.array(y), np.array(dates)

# METRICS
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
        - "MSE"   : Error cuadrático medio.
        - "RMSE"  : Raíz del error cuadrático medio.
        - "MAE"   : Error absoluto medio.
        - "R2"    : Coeficiente de determinación.
        - "MAPE"  : Error porcentual absoluto medio.
        - "EVS"   : Varianza explicada.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/(y_true+1e-6))) * 100
    da = np.mean((y_true>0) == (y_pred>0))
    smape = 100*np.mean(2*np.abs(y_pred-y_true)/(np.abs(y_true)+np.abs(y_pred)+1e-6))
    evs = explained_variance_score(y_true, y_pred)
    return {"MSE":mse,"RMSE":rmse,"MAE":mae,"R2":r2,"MAPE":mape,"EVS":evs}

# DL MODELS
def build_lstm(input_shape, units=16, dropout=0.3, H=4, **kwargs):
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
    H : int, optional (default=4)
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

def build_gru(input_shape, units=16, dropout=0.3, H=4, **kwargs):
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
    H : int, optional (default=4)
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

def build_conv1d(input_shape, units=16, dropout=0.3, kernel_size=3, H=4, **kwargs):
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
    H : int, optional (default=4)
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

# ML con TimeSeriesSplit
def tscv_ml(model, X, y, n_splits):
    """
    Realiza validación cruzada para modelos de machine learning en series temporales.

    Parameters
    ----------
    model : estimator
        Modelo base compatible con scikit-learn.
    X : numpy.ndarray
        Conjunto de entradas, de forma (n_muestras, ventana, features).
    y : numpy.ndarray
        Conjunto de salidas, de forma (n_muestras, horizonte).
    n_splits : int
        Número de divisiones para TimeSeriesSplit.

    Returns
    -------
    dict
        Diccionario con el promedio de métricas en todos los splits:
        - "MSE"   : Error cuadrático medio.
        - "RMSE"  : Raíz del error cuadrático medio.
        - "MAE"   : Error absoluto medio.
        - "R2"    : Coeficiente de determinación.
        - "MAPE"  : Error porcentual absoluto medio.
        - "EVS"   : Varianza explicada.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    X_flat = X.reshape((X.shape[0], -1))
    for train_idx, test_idx in tscv.split(X_flat):
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_fold = MultiOutputRegressor(clone(model)) if y.shape[1]>1 else clone(model)
        model_fold.fit(X_train_scaled, y_train)
        y_pred = model_fold.predict(X_test_scaled)
        metrics_list.append(compute_metrics(y_test, y_pred))
    return {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}

# DL con TimeSeriesSplit
def tscv_dl(model_builder, X, y, n_splits, units, dropout, epochs, batch_size, kernel_size=3):
    """
    Realiza validación cruzada para modelos de deep learning en series temporales.
    
    Parameters
    ----------
    model_builder : callable
        Función constructora del modelo (ej. build_lstm, build_gru, build_conv1d).
    X : numpy.ndarray
        Conjunto de entradas, de forma (n_muestras, ventana, features).
    y : numpy.ndarray
        Conjunto de salidas, de forma (n_muestras, horizonte).
    n_splits : int
        Número de divisiones para TimeSeriesSplit.
    units : int
        Número de unidades/filtros en la capa principal.
    dropout : float
        Tasa de dropout para regularización.
    epochs : int
        Número de épocas de entrenamiento.
    batch_size : int
        Tamaño del lote en el entrenamiento.
    kernel_size : int, optional (default=3)
        Tamaño de kernel en la capa Conv1D (si aplica).
    
    Returns
    -------
    dict
        Diccionario con el promedio de métricas en todos los splits:
        - "MSE"   : Error cuadrático medio.
        - "RMSE"  : Raíz del error cuadrático medio.
        - "MAE"   : Error absoluto medio.
        - "R2"    : Coeficiente de determinación.
        - "MAPE"  : Error porcentual absoluto medio.
        - "EVS"   : Varianza explicada.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics_list = []
    X_seq = X.reshape((X.shape[0], X.shape[1],1))
    
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

# GRAFICOS
def plot_learning_curves(histories, title="Curvas de entrenamiento/validación"):
    """
    Grafica las curvas de pérdida (loss) en entrenamiento y validación
    a partir de objetos History de Keras.

    Parameters
    ----------
    histories : keras.callbacks.History or dict
        Puede ser:
        - Un único objeto History (para un solo modelo).
        - Un diccionario {nombre: history} para graficar varios modelos juntos.
    title : str, optional
        Título del gráfico.

    Returns
    -------
    None
        Muestra el gráfico en pantalla.
    """
    plt.figure(figsize=(8,6))

    if isinstance(histories, dict):
        # Varios modelos
        for name, hist in histories.items():
            plt.plot(hist.history['loss'], label=f'{name} - Entrenamiento')
            if 'val_loss' in hist.history:
                plt.plot(hist.history['val_loss'], label=f'{name} - Validación')
    else:
        # Un solo modelo
        hist = histories
        plt.plot(hist.history['loss'], label='Entrenamiento')
        if 'val_loss' in hist.history:
            plt.plot(hist.history['val_loss'], label='Validación')

    plt.xlabel("Epocas")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__=="__main__":
    path_csv = "stock_weekly_clean.csv"
    W,H = 25,4
    sub_sample_ratio = 0.4
    np.random.seed(SEED)
    
    df_weekly = pd.read_csv(path_csv, index_col=0, parse_dates=True)
    df_weekly.index = pd.to_datetime(df_weekly.index, utc=True)
    sampled_cols = np.random.choice(df_weekly.columns, int(len(df_weekly.columns)*sub_sample_ratio), replace=False)
    df_sub = df_weekly[sampled_cols]
    
    X_sub, y_sub, _ = create_supervised_with_dates(df_sub, W=W, H=H)
    X_sub = np.nan_to_num(X_sub)
    y_sub = np.nan_to_num(y_sub)
    
    # HYPERPARAMETERS
    xgb_params = {"n_estimators":[100,200], "max_depth":[3,5], "learning_rate":[0.01,0.05]}
    lgb_params = {"n_estimators":[100,200], "max_depth":[5,10], "learning_rate":[0.01,0.05]}
    dl_params = {'units':[16,32,64], 'dropout':[0.2,0.3]}
    c1d_params = {'units':[16,32,64], 'dropout':[0.2,0.3], 'kernel_size':[3,5]}
    
    best_models = {}
    tuning_results = {}
    results_sub = {}

    # ML MODELS
    ml_models = {"XGB": XGBRegressor, "LGBM": LGBMRegressor}
    ml_param_grids = {"XGB": xgb_params, "LGBM": lgb_params}
    
    for name, ModelClass in ml_models.items():
        best_score, best_model = -np.inf, None
        for params in ParameterGrid(ml_param_grids[name]):
            print(f"{name} con {params}")
            if name=="XGB":
                model = ModelClass(random_state=SEED, verbosity=0, **params)
            else:
                model = ModelClass(random_state=SEED, verbose=-1, **params)
            metrics = tscv_ml(model, X_sub, y_sub, n_splits=5)
            tuning_results.setdefault(name, []).append({'params': params, 'metrics': metrics})
            if metrics['R2'] > best_score:
                best_score = metrics['R2']
                best_model = model
        results_sub[name] = tscv_ml(best_model, X_sub, y_sub, n_splits=5)
        best_models[name] = best_model

    # DL MODELS
    dl_models = {"LSTM": build_lstm, "GRU": build_gru, "Conv1D": build_conv1d}
    dl_param_grids = {"LSTM": dl_params, "GRU": dl_params, "Conv1D": c1d_params}
    
    for name, builder in dl_models.items():
        best_score, best_config, best_metrics = -np.inf, None, None
        param_grid = dl_param_grids[name]
        combos = product(*(param_grid.values())) if name!="Conv1D" else product(param_grid['units'], param_grid['dropout'], param_grid['kernel_size'])
        for combo in combos:
            print(f"{name} con {combo}")
            if name=="Conv1D":
                units, dropout, kernel_size = combo
                metrics = tscv_dl(builder, X_sub, y_sub, n_splits=3, units=units, dropout=dropout,
                                  kernel_size=kernel_size, epochs=10, batch_size=32)
                config = {'units': units, 'dropout': dropout, 'kernel_size': kernel_size}
            else:
                units, dropout = combo
                metrics = tscv_dl(builder, X_sub, y_sub, n_splits=3, units=units, dropout=dropout,
                                  epochs=10, batch_size=32, kernel_size=3)
                config = {'units': units, 'dropout': dropout}
    
            tuning_results.setdefault(name, []).append({'params': config, 'metrics': metrics})
    
            if metrics['R2'] > best_score:
                best_score = metrics['R2']
                best_config = config
                best_metrics = metrics
    
        # Guardamos todas las métricas del mejor modelo
        results_sub[name] = {**best_metrics, 'config': best_config}
        best_models[name] = best_config
    
    # SELECT BEST MODEL
    best_r2 = -np.inf
    best_model_name = None
    for name, m in best_models.items():
        if name in ["XGB", "LGBM"]:
            r2 = tscv_ml(m, X_sub, y_sub, n_splits=5)['R2']
        else:
            r2 = tscv_dl(dl_models[name], X_sub, y_sub, n_splits=3, **m, epochs=10, batch_size=32)['R2']
        if r2 > best_r2:
            best_r2 = r2
            best_model_name = name
    
    print("\n=== Métricas de los mejores modelos (submuestra) ===")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(pd.DataFrame(results_sub).T)
    
    top_models = ["LGBM", "Conv1D"]
    
    # BAGGING GLOBAL
    print("=== Bagging simple por empresa ===")
    
    n_bags = 5
    
    # Preparamos datos completos supervisados
    X_full, y_full, dates_full = create_supervised_with_dates(df_sub, W=W, H=H)
    X_full = np.nan_to_num(X_full)
    y_full = np.nan_to_num(y_full)
    
    # Escalamos
    scaler_full = StandardScaler()
    X_full_scaled = scaler_full.fit_transform(X_full)
    
    # Ultimo fold train/test
    tscv = TimeSeriesSplit(n_splits=5)
    train_idx, test_idx = list(tscv.split(X_full_scaled))[-1]
    X_train, X_test = X_full_scaled[train_idx], X_full_scaled[test_idx]
    y_train, y_test = y_full[train_idx], y_full[test_idx]
    
    # Bagging sobre todos los modelos
    bagging_preds = np.zeros_like(y_test, dtype=float)
    histories = {}
    
    for name in top_models:
        preds_bag = np.zeros_like(y_test, dtype=float)
        for _ in range(n_bags):
            if name in ["XGB", "LGBM"]:
                model = MultiOutputRegressor(clone(best_models[name])) if y_train.shape[1] > 1 else clone(best_models[name])
                model.fit(X_train, y_train)
                preds_bag += model.predict(X_test)
            else:  # Conv1D
                config = best_models[name]
                X_train_seq = X_train.reshape((X_train.shape[0], W, 1))
                X_test_seq = X_test.reshape((X_test.shape[0], W, 1))
                model = dl_models[name](input_shape=(W, 1), H=y_train.shape[1], **config)
                es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
                history = model.fit(X_train_seq, y_train, validation_data=(X_test_seq, y_test),
                          epochs=20, batch_size=32, verbose=0, callbacks=[es, lr])
                histories[name] = history
                preds_bag += model.predict(X_test_seq, verbose=0)
        preds_bag /= n_bags
        bagging_preds += preds_bag
    
    bagging_preds /= len(top_models)
    plot_learning_curves(histories, title="Curvas conjuntas de entrenamiento/validación (DL)")
    bagging_metrics = compute_metrics(y_test, bagging_preds)
    print("Métricas Bagging (H=4):", {k: round(v,4) for k,v in metrics.items()})

    # PREDICCIÓN FUTURA FINAL
    print("\n=== Predicción futura (última ventana de cada empresa) ===")
    
    n_companies = df_sub.shape[1]
    y_pred_last = np.zeros((n_companies, H))   # Asegurar matriz 2D
    
    # Entrenamos cada modelo una vez con todos los datos
    trained_models = {}
    for name in top_models:
        if name in ["XGB", "LGBM"]:
            model = MultiOutputRegressor(clone(best_models[name])) if H > 1 else clone(best_models[name])
            model.fit(X_full_scaled, y_full)
        else:
            config = best_models[name]
            X_full_seq = X_full_scaled.reshape((X_full_scaled.shape[0], W, 1))
            model = dl_models[name](input_shape=(W,1), H=H, **config)
            es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
            model.fit(X_full_seq, y_full, epochs=20, batch_size=32, verbose=0, callbacks=[es, lr])
        trained_models[name] = model
    
    # Predicción empresa por empresa
    for i, company in enumerate(df_sub.columns):
        series = df_sub[company].values
        last_window = series[-W:].reshape(1, -1)
        last_window_scaled = scaler_full.transform(last_window)
    
        preds_models = []
        for name, model in trained_models.items():
            if name in ["XGB","LGBM"]:
                pred = model.predict(last_window_scaled)
            else:  # DL
                last_seq = last_window_scaled.reshape((1,W,1))
                pred = model.predict(last_seq, verbose=0)
            preds_models.append(pred.reshape(-1))   # Aseguramos (H,)
        
        # Bagging simple: promedio entre modelos
        y_pred_last[i] = np.mean(preds_models, axis=0)
    
      
    # VISUALIZACIÓN PARA H=4
    print("\n=== Predicciones futuras (4 semanas) ===")
    
    # Seleccionamos las 6 empresas con mayor crecimiento previsto en el último paso (semana 4)
    last_values = df_sub.iloc[-1].values
    diff_abs = y_pred_last[:, -1] - last_values
    diff_pct = (diff_abs / last_values) * 100
    top6_idx = np.argsort(diff_pct)[-6:][::-1]
    
    for i in top6_idx:
        company = df_sub.columns[i]
        current = last_values[i]
        preds = y_pred_last[i]
        print(f"\nEmpresa: {company}")
        print(f"  Ultimo valor: {current:.2f}")
        for h, pred in enumerate(preds, start=1):
            abs_diff = pred - current
            pct_diff = (abs_diff / current) * 100
            print(f"   +{h} semana(s): Pred={pred:.2f} | Dif={abs_diff:.2f} | Dif%={pct_diff:.2f}%")
    
    # GRAFICO HISTÓRICO + PREDICCIONES
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for j, idx in enumerate(top6_idx):
        company = df_sub.columns[idx]
        series = df_sub[company]
    
        # Histórico
        axes[j].plot(series.index, series.values, label="Histórico", color="black")
    
        # Fechas futuras
        future_dates = [series.index[-1] + pd.Timedelta(weeks=k) for k in range(1, H+1)]
    
        # Unimos último valor real + predicciones
        full_dates = [series.index[-1]] + future_dates
        full_preds = [series.values[-1]] + list(y_pred_last[idx])
    
        # Graficamos predicciones unidas
        axes[j].plot(full_dates, full_preds, "ro-", label="Predicción (4 semanas)")
    
        axes[j].set_title(company)
        axes[j].legend()
        axes[j].grid(True)
    
    plt.tight_layout()
    plt.show()

