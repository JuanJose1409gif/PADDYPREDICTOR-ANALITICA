# 🌾 PaddyAI — Predictor de Rendimiento de Arroz

Proyecto Final · Analítica de Datos · Entrega: 14-Mayo-2026

## Descripción

Aplicación web que usa **Regresión Logística** y **Red Neuronal Artificial** para predecir si una parcela de arroz (Paddy) tendrá rendimiento **alto o bajo**, basado en 44 variables agrícolas, climatológicas y de manejo de cultivo.

- **Dataset:** `paddydataset.csv` — 2,789 registros
- **Target:** Rendimiento alto = ≥ mediana (24,636 Kg)
- **Accuracy Reg. Logística:** 91.22%
- **Accuracy Red Neuronal:** 90.68%

## Estructura del proyecto

```
paddy-predictor/
├── index.html              ← Página principal
├── individual.html         ← Predicción individual
├── lotes.html              ← Predicción por lotes (CSV)
├── style.css               ← Estilos
├── paddydataset.csv        ← Dataset original
└── models/
    ├── model_data.js       ← Modelos entrenados (pesos y encoders)
    └── model.js            ← Motor de inferencia JS
```

## Cómo abrir en VS Code

1. Abre la carpeta `paddy-predictor/` en VS Code
2. Instala la extensión **Live Server** (ritwickdey.liveserver)
3. Clic derecho en `index.html` → **Open with Live Server**
4. Listo! La app corre en `http://127.0.0.1:5500`

## Funcionalidades

### 1. Predicción Individual
- Formulario con todos los campos del dataset
- Selector de modelo (Regresión Logística o Red Neuronal)
- Muestra: predicción (ALTO/BAJO) + probabilidad con barra visual

### 2. Predicción por Lotes
- Subir CSV con múltiples registros
- Botón de dataset demo (100 registros generados)
- Selector: modelo individual o **comparar ambos**
- Muestra: tabla de predicciones, **matriz de confusión**, métricas de desempeño

## Deploy en GitHub Pages

```bash
# 1. Crea un repositorio en GitHub
# 2. Sube todos los archivos de paddy-predictor/
git init
git add .
git commit -m "Paddy AI - Proyecto Final"
git remote add origin https://github.com/TU_USUARIO/paddy-predictor.git
git push -u origin main

# 3. En GitHub: Settings → Pages → Source: main branch / root
# 4. URL: https://TU_USUARIO.github.io/paddy-predictor/
```

**Nota:** No se necesita servidor backend. Los modelos están embebidos en JavaScript
y toda la inferencia corre en el navegador del cliente.

## Modelos

Los modelos fueron entrenados con `scikit-learn` y los pesos exportados a JavaScript:

- **Regresión Logística:** `LogisticRegression(max_iter=1000)`
- **Red Neuronal:** `MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500)`
- **Preprocesamiento:** `StandardScaler` + `LabelEncoder` para variables categóricas
- **Split:** 80% train / 20% test, estratificado

## Script de entrenamiento

Ver `train_models.py` para re-entrenar los modelos con nuevos datos.
