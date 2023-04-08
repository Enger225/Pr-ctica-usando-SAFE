[![PyPI version](https://badge.fury.io/py/safe-transformer.svg)](https://badge.fury.io/py/safe-transformer)
# PRÁCTICA USANDO SAFE-TRANSFORMER
## REQUERIMIENTOS:

Instalar la siguiente librería:
```
pip install safe-transformer
```
Nota: Esta instalación ya se encuentra en el Notebook SAFE.ipynb

## EVALUACIÓN CON EL MODELO DE LOGÍSTICA VAINILLA:
```python
# Crear un modelo de regresión logística vainilla y ajustarlo a los datos de entrenamiento
logreg_H1 = LogisticRegression()
logreg_H1 = logreg_H1.fit(X_train1, y_train1)
logreg_H2 = LogisticRegression()
logreg_H2 = logreg_H2.fit(X_train2, y_train2)

# Evaluar el modelo en los datos de prueba
logreg_y_pred1 = logreg_H1.predict(X_test1)
logreg_y_pred2 = logreg_H2.predict(X_test2)
```

```bash
Coeficiente de determinación del modelo regresión lineal en H1: 0.97128935529898019485
Coeficiente de determinación del modelo regresión lineal en H2: 0.98027851283511102665

MSE (Error cuadrático medio) del modelo regresión lineal en H1: 0.00574138791812281619
MSE (Error cuadrático medio) del modelo regresión lineal en H2: 0.00479011723181646271
```
https://github.com/Enger225/Pr-ctica-usando-SAFE/blob/main/Graficas/logreg_H1_mse.png
