# Tarea 5 — Regresión Lineal (Versión Simplificada)

**Estudiante:** Ángel Udalber Rodríguez Moya (100216147)

## Qué hace

1. Genera dataset de 120 viajes simulados (SITP/TransMilenio Bogotá)
2. Entrena 2 modelos de regresión lineal:
   - Modelo A: predice tiempo_viaje (minutos)
   - Modelo B: predice ocupacion (proporción 0-1)
3. Calcula R², MAE, RMSE
4. Genera gráficas: Real vs Predicho + Residuos

## Cómo ejecutar

```bash
cd Inteliegencia_artificial/scripts/tarea5
python main.py       
python test.py 
```

## Archivos generados

- `datos.csv` — dataset de 120 viajes
- `resultados/resultados.png` — gráficas (4 subplots)
- `resultados/metricas.txt` — R², MAE, RMSE

## Resultados esperados

```
Tiempo de Viaje: R2=0.927, MAE=2.46min, RMSE=3.21min
Ocupacion:      R2=0.818, MAE=0.0561, RMSE=0.0691
```
