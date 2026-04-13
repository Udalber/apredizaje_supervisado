"""
Tests minimos para Tarea 5
Ejecucion: python test.py
"""

from main import generar_datos
import numpy as np
from sklearn.preprocessing import StandardScaler


def test_datos():
    df = generar_datos(n=50)
    assert len(df) == 50
    assert df.isnull().sum().sum() == 0
    assert df['distancia_km'].between(2.0, 15.0).all()
    assert df['tiempo_viaje'].between(12, 60).all()
    assert df['ocupacion'].between(0.2, 1.0).all()


def test_scaler():
    df = generar_datos(n=50)
    X = df[['distancia_km', 'hora_salida', 'es_hora_pico', 'hay_obras', 'num_paradas']]
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    assert abs(X_norm.mean()) < 0.1
    assert abs(X_norm.std() - 1.0) < 0.1


if __name__ == '__main__':
    test_datos()
    test_scaler()
    print("OK - 2/2 tests pasados")
