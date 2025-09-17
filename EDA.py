import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox

max_nan_ratio = 0.4
path_csv = "stock_details_5_years.csv"
export_path = "stock_weekly_clean.csv"

# ---------------- MAIN ----------------
if __name__ == "__main__":
    path_csv = "stock_details_5_years.csv"

    df = pd.read_csv(path_csv)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    
    # EDA preliminar antes del pivotado
    print("\n=== EDA RAW (datos crudos) ===")
    print(f"Empresas totales: {df['Company'].nunique()}")
    print(f"Rango de fechas: {df['Date'].min().date()} a {df['Date'].max().date()}")

    # Conteo por empresa
    counts = df.groupby("Company")['Date'].count().sort_values(ascending=False)
    print("\nTop 10 empresas con más registros:")
    print(counts.head(10))
    print("\nBottom 10 empresas con menos registros:")
    print(counts.tail(10))
    
    # Resample semanal
    df_pivot = df.pivot(index='Date', columns='Company', values='Close').resample('W').last()

    print("\n=== EDA Pivot (serie temporal por empresa) ===")
    print(f"Dimensiones: {df_pivot.shape}")
    print(f"Rango de fechas: {df_pivot.index.min().date()} a {df_pivot.index.max().date()}")

    # Missing por empresa
    na_ratio = df_pivot.isna().mean().sort_values(ascending=False)
    na_ratio_row = na_ratio.to_frame().T
    print("\n--- Porcentaje de missing por empresa (Top 10) ---")
    print(na_ratio.head(10))

    # Missing por fecha
    missing_by_date = df_pivot.isna().sum(axis=1)
    plt.figure(figsize=(12, 4))
    missing_by_date.plot()
    plt.title("Número de valores faltantes por fecha (pre-limpieza)")
    plt.xlabel("Fecha")
    plt.ylabel("Num. de missing")
    plt.show()

    # Empresas eliminadas por exceso de NaN
    dropped = na_ratio[na_ratio > max_nan_ratio]
    if not dropped.empty:
        print(f"\nEmpresas eliminadas (> {max_nan_ratio*100:.0f}% NaN): {list(dropped.index)}")
        df_pivot = df_pivot.drop(columns=dropped.index)

    # Missing por fecha (tras eliminar empresas malas)
    missing_by_date = df_pivot.isna().sum(axis=1)
    plt.figure(figsize=(12, 4))
    missing_by_date.plot()
    plt.title("Número de valores faltantes por fecha (tras limpieza)")
    plt.xlabel("Fecha")
    plt.ylabel("Num. de missing")
    plt.show()

    # Imputación: interpolación temporal + ffill + bfill
    df_pivot = df_pivot.interpolate(method='time').bfill()
    print(f"Valores missing totales: {df_pivot.isna().sum().sum()}")
    
    # Estadísticas
    print("\n--- Estadísticas descriptivas (primeras 10 empresas) ---")
    descr = df_pivot.describe().T
    print(descr.head(10))
    print("\n--- Estadísticas descriptivas (todas las empresas) ---")
    print(descr)

    # Chequeo de estacionalidad
    print("\n--- Chequeo de estacionalidad ---")
    seasonal_strength = {}
    for company in df_pivot.columns:
        try:
            series = df_pivot[company].dropna()
            decomposition = seasonal_decompose(series, period=52)
            strength = decomposition.seasonal.var() / (series.var() + 1e-6)
            seasonal_strength[company] = strength
        except:
            seasonal_strength[company] = np.nan

    seasonal_strength_series = pd.Series(seasonal_strength).sort_values(ascending=False)

    # Top 20 en fuerza estacional
    plt.figure(figsize=(12, 5))
    sns.barplot(x=seasonal_strength_series.index[:20], y=seasonal_strength_series.values[:20], color="steelblue")
    plt.xticks(rotation=90)
    plt.ylabel("Fuerza estacionalidad")
    plt.title("Top 20 empresas por fuerza estacional (periodo=52 semanas)")
    plt.show()

    # Descomposición de 3 ejemplos: mayor, menor y mediana fuerza estacional
    examples = [
        seasonal_strength_series.dropna().idxmax(),
        seasonal_strength_series.dropna().idxmin(),
        seasonal_strength_series.dropna().index[len(seasonal_strength_series)//2]
    ]
    print(f"\nEjemplos de descomposición estacional (estacionalidad máxima, mínima y mediana): {examples}")
    for ex in examples:
        series = df_pivot[ex].dropna()
        decomposition = seasonal_decompose(series, period=52)
        decomposition.plot()
        plt.show()
        
        residual = decomposition.resid
        ljung_box_result = acorr_ljungbox(residual.dropna(), lags=[40], return_df=True)
        print(f"\nResultado de la prueba de Ljung-Box para la serie {ex}: {ljung_box_result['lb_pvalue'].iloc[0]}")

    # Gráfico comparativo de las 3 series
    plt.figure(figsize=(12, 5))
    for ex in examples:
        plt.plot(df_pivot.index, df_pivot[ex], label=ex)
    plt.legend()
    plt.title("Comparación de series: mayor, menor y mediana estacionalidad")
    plt.grid(True)
    plt.show()
    
    # Gráfico de todas las empresas
    plt.figure(figsize=(12, 5))
    for company in df_pivot.columns:
        plt.plot(df_pivot.index, df_pivot[company], label=company)
    #plt.legend()
    plt.title("Gráfico de todas las empresas (tras limpieza)")
    plt.grid(True)
    plt.show()

    # Exportar dataset limpio
    df_pivot.to_csv(export_path)
    print(f"\nDataset limpio exportado a: {export_path}")