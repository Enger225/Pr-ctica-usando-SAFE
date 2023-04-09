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
logreg_H1 = LogisticRegression(solver='lbfgs')
logreg_H1 = logreg_H1.fit(X_train1, y_train1)
logreg_H2 = LogisticRegression(solver='lbfgs')
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
### MSE:
![*Fit method algorithm*](/Graficas/vainilla_mse_h1.png)
![*Fit method algorithm*](/Graficas/vainilla_mse_h2.png)

### COEFICIENTE:
![*Fit method algorithm*](/Graficas/vainilla_r2_h1.png)
![*Fit method algorithm*](/Graficas/vainilla_r2_h2.png)

## EVALUACIÓN CON EL MODELO LINEAL:
```python
# Crear un modelo de regresión lineal y ajustarlo a los datos de entrenamiento
linreg_H1 = LinearRegression()
linreg_H1 = linreg_H1.fit(X_train1_lin, y_train1_lin)
linreg_H2 = LinearRegression()
linreg_H2 = linreg_H2.fit(X_train2_lin, y_train2_lin)

# Evaluar el modelo en los datos de prueba
linreg_y_pred1 = linreg_H1.predict(X_test1_lin)
linreg_y_pred2 = linreg_H2.predict(X_test2_lin)
```

```bash
Coeficiente de determinación del modelo regresión lineal en H1: 0.99999999999543620621
Coeficiente de determinación del modelo regresión lineal en H2: 1.00000000000000000000

MSE (Error cuadrático medio) del modelo regresión lineal en H1: 0.00000000000091264252
MSE (Error cuadrático medio) del modelo regresión lineal en H2: 0.00000000000000000000
```
### MSE:
![*Fit method algorithm*](/Graficas/lineal_mse_h1.png)
![*Fit method algorithm*](/Graficas/lineal_mse_h2.png)

### COEFICIENTE:
![*Fit method algorithm*](/Graficas/lineal_r2_h1.png)
![*Fit method algorithm*](/Graficas/lineal_r2_h2.png)

## IMPLEMENTACIÓN DEL MÉTODO SAFE:
```python
# Entrenar un modelo complejo (bosque aleatorio supervisor flexible)
surrogate_model_H1 = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=2)
surrogate_model_H1 = surrogate_model_H1.fit(X_train1_SAFE, y_train1_SAFE)
surrogate_model_H2 = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=2)
surrogate_model_H2 = surrogate_model_H2.fit(X_train2_SAFE, y_train2_SAFE)
```
```python
# Utilizar SAFE para encontrar transformaciones de variables
safe_transformer_H1 = SafeTransformer(surrogate_model_H1)
X_transformed_H1 = safe_transformer_H1.fit_transform(X_train1_SAFE, y_train1_SAFE)
safe_transformer_H2 = SafeTransformer(surrogate_model_H2)
X_transformed_H2 = safe_transformer_H2.fit_transform(X_train2_SAFE, y_train2_SAFE)
```
## EVALUACIÓN CON EL MODELO LOGÍSTICO:
```python
# Entrenar un modelo de regresión logística con las variables transformadas
logistic_model_H1 = LogisticRegression(solver="lbfgs")
LM_H1 = logistic_model_H1.fit(X_transformed_H1, y_train1_SAFE)
logistic_model_H2 = LogisticRegression(solver="lbfgs")
LM_H2 = logistic_model_H2.fit(X_transformed_H2, y_train2_SAFE)
```
```python
# Evaluar el modelo en el conjunto de prueba
X_test_transformed_H1 = safe_transformer_H1.transform(X_test1_SAFE)
y_pred_H1 = LM_H1.predict(X_test_transformed_H1)
X_test_transformed_H2 = safe_transformer_H2.transform(X_test2_SAFE)
y_pred_H2 = LM_H2.predict(X_test_transformed_H2)
```
```bash
Coeficiente de determinación del modelo Logístico en H1: -0.38185581234908605452
Coeficiente de determinación del modelo Logístico en H2: -0.44797234710632571897

MSE (Error cuadrático medio) del modelo Logístico en H1: 0.27633549675486768216
MSE (Error cuadrático medio) del modelo Logístico en H2: 0.35169544938862978833
```
### MSE:
![*Fit method algorithm*](/Graficas/logistico_mse_h1.png)
![*Fit method algorithm*](/Graficas/logistico_mse_h2.png)

### COEFICIENTE:
![*Fit method algorithm*](/Graficas/logistico_r2_h1.png)
![*Fit method algorithm*](/Graficas/logistico_r2_h2.png)
