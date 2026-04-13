import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

RESULTADOS = Path('resultados')
RESULTADOS.mkdir(exist_ok=True)


def generar_datos(n=120, seed=42):
    rng = np.random.default_rng(seed)
    dist = rng.uniform(2.0, 15.0, n).round(1)
    hora = rng.integers(6, 9, n)
    pico = ((hora >= 7) & (rng.random(n) < 0.75)).astype(int)
    obras = (rng.random(n) < 0.25).astype(int)
    paradas = rng.integers(1, 6, n)

    tiempo = np.clip(3*dist + 5*pico + 7*obras + 1.5*paradas + rng.normal(0, 3, n), 12, 60).round(1)
    ocupacion = np.clip(0.04*dist + 0.25*pico + 0.10*obras + 0.30 + rng.normal(0, 0.08, n), 0.2, 1.0).round(3)

    return pd.DataFrame({
        'distancia_km': dist, 'hora_salida': hora, 'es_hora_pico': pico,
        'hay_obras': obras, 'num_paradas': paradas,
        'tiempo_viaje': tiempo, 'ocupacion': ocupacion,
    })


def ejecutar():
    # Datos
    print("Generando datos...")
    df = generar_datos()
    df.to_csv('datos.csv', index=False)
    print(f"  {len(df)} registros")

    # Preparar
    print("Preparando datos...")
    FEATURES = ['distancia_km', 'hora_salida', 'es_hora_pico', 'hay_obras', 'num_paradas']
    X = df[FEATURES]
    y_t = df['tiempo_viaje']
    y_o = df['ocupacion']

    X_train, X_test, yt_train, yt_test, yo_train, yo_test = train_test_split(
        X, y_t, y_o, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Modelos
    print("Entrenando modelos...")
    m_t = LinearRegression().fit(X_train, yt_train)
    m_o = LinearRegression().fit(X_train, yo_train)

    yt_pred = m_t.predict(X_test)
    yo_pred = m_o.predict(X_test)

    # Metricas
    r2_t = r2_score(yt_test, yt_pred)
    mae_t = mean_absolute_error(yt_test, yt_pred)
    rmse_t = np.sqrt(mean_squared_error(yt_test, yt_pred))

    r2_o = r2_score(yo_test, yo_pred)
    mae_o = mean_absolute_error(yo_test, yo_pred)
    rmse_o = np.sqrt(mean_squared_error(yo_test, yo_pred))

    print(f"\nTiempo de Viaje: R2={r2_t:.3f}, MAE={mae_t:.2f}min, RMSE={rmse_t:.2f}min")
    print(f"Ocupacion:      R2={r2_o:.3f}, MAE={mae_o:.4f}, RMSE={rmse_o:.4f}")

    # Graficas
    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    ax[0, 0].scatter(yt_test, yt_pred, alpha=0.6)
    ax[0, 0].plot([yt_test.min(), yt_test.max()], [yt_test.min(), yt_test.max()], 'k--')
    ax[0, 0].set_title(f'Tiempo Real vs Predicho (R2={r2_t:.3f})')

    residuos_t = yt_test - yt_pred
    ax[0, 1].scatter(yt_pred, residuos_t, alpha=0.6, color='coral')
    ax[0, 1].axhline(0, color='black', linestyle='--')
    ax[0, 1].set_title('Residuos - Tiempo')

    ax[1, 0].scatter(yo_test, yo_pred, alpha=0.6, color='green')
    ax[1, 0].plot([yo_test.min(), yo_test.max()], [yo_test.min(), yo_test.max()], 'k--')
    ax[1, 0].set_title(f'Ocupacion Real vs Predicha (R2={r2_o:.3f})')

    residuos_o = yo_test - yo_pred
    ax[1, 1].scatter(yo_pred, residuos_o, alpha=0.6, color='coral')
    ax[1, 1].axhline(0, color='black', linestyle='--')
    ax[1, 1].set_title('Residuos - Ocupacion')

    plt.tight_layout()
    plt.savefig(RESULTADOS / 'resultados.png', dpi=110)
    print(f"\nGraficas: resultados/resultados.png")

    # Metricas archivo
    metricas = f"""=== MODELO: Tiempo de Viaje ===
R2:   {r2_t:.4f}
MAE:  {mae_t:.4f} min
RMSE: {rmse_t:.4f} min

=== MODELO: Ocupacion ===
R2:   {r2_o:.4f}
MAE:  {mae_o:.4f}
RMSE: {rmse_o:.4f}
"""
    (RESULTADOS / 'metricas.txt').write_text(metricas)
    print("Metricas: resultados/metricas.txt")


if __name__ == '__main__':
    ejecutar()
