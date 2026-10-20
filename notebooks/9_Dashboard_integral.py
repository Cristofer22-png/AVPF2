import os
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Definición de Rutas Relativas
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
data_dir = os.path.join(project_dir, 'data')

# Detección automática del archivo CSV
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
if not csv_files:
    raise FileNotFoundError("No se encontró ningún archivo .csv dentro de la carpeta data/")

data_path = os.path.join(data_dir, csv_files[0])
output_html = os.path.join(project_dir, 'dashboard', '9_Dashboard.html')

# 2. Carga y Limpieza de Datos
df = pd.read_csv(data_path)
df.columns = [col.strip().lower() for col in df.columns]

col_map = {}
for col in df.columns:
    if 'cement' in col: col_map[col] = 'cemento'
    elif 'slag' in col: col_map[col] = 'escoria'
    elif 'fly' in col or 'ash' in col: col_map[col] = 'ceniza_volante'
    elif 'water' in col: col_map[col] = 'agua'
    elif 'super' in col or 'plastic' in col: col_map[col] = 'superplastificante'
    elif 'coarse' in col: col_map[col] = 'agregado_grueso'
    elif 'fine' in col: col_map[col] = 'agregado_fino'
    elif 'age' in col: col_map[col] = 'edad_dias'
    elif 'strength' in col or 'compressive' in col: col_map[col] = 'resistencia_mpa'

if col_map:
    df.rename(columns=col_map, inplace=True)

df_clean = df.dropna().drop_duplicates().copy()

# 3. KPIs
kpi_promedio = round(df_clean['resistencia_mpa'].mean(), 2)
kpi_maximo = round(df_clean['resistencia_mpa'].max(), 2)
kpi_edad_prom = round(df_clean['edad_dias'].mean(), 1)
kpi_registros = len(df_clean)

# 4. Modelo Predictivo
X = df_clean[['cemento', 'agua', 'edad_dias']]
y = df_clean['resistencia_mpa']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2_val = round(r2_score(y_test, y_pred), 3)

cemento_prom = df_clean['cemento'].mean()
agua_prom = df_clean['agua'].mean()

pred_input = pd.DataFrame([[cemento_prom, agua_prom, 120]], columns=['cemento', 'agua', 'edad_dias'])
pred_120_dias = round(model.predict(pred_input)[0], 2)

# 5. Gráficos Interactivos
fig_hist = px.histogram(df_clean, x='resistencia_mpa', nbins=30,
                         title="Distribución de la Resistencia a la Compresión (f'c)",
                         labels={'resistencia_mpa': 'Resistencia (MPa)', 'count': 'Frecuencia'},
                         color_discrete_sequence=['#1f77b4'])

corr = df_clean.corr(numeric_only=True).round(2)
fig_corr = px.imshow(corr, text_auto=True, title='Matriz de Correlación de Pearson',
                     color_continuous_scale='Blues')

fig_scatter = px.scatter(df_clean, x='cemento', y='resistencia_mpa', color='edad_dias',
                         title='Relación Cemento vs. Resistencia según Edad de Curado',
                         labels={'cemento': 'Cemento (kg/m³)', 'resistencia_mpa': 'Resistencia (MPa)', 'edad_dias': 'Edad (Días)'})

time_series = df_clean.groupby('edad_dias')['resistencia_mpa'].mean().reset_index()
fig_time = px.line(time_series, x='edad_dias', y='resistencia_mpa', markers=True,
                   title='Evolución de Resistencia Promedio según Edad de Curado',
                   labels={'edad_dias': 'Edad (Días)', 'resistencia_mpa': 'Resistencia Promedio (MPa)'})

html_hist = fig_hist.to_html(full_html=False, include_plotlyjs='cdn')
html_corr = fig_corr.to_html(full_html=False, include_plotlyjs=False)
html_scatter = fig_scatter.to_html(full_html=False, include_plotlyjs=False)
html_time = fig_time.to_html(full_html=False, include_plotlyjs=False)

# 6. HTML del Dashboard
html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Dashboard Interactivo - Resistencia del Concreto</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f6f9; color: #333; }}
        h1 {{ text-align: center; color: #003366; }}
        .subtitle {{ text-align: center; color: #666; margin-bottom: 30px; }}
        .kpi-container {{ display: flex; justify-content: space-around; margin-bottom: 30px; flex-wrap: wrap; }}
        .kpi-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); width: 20%; text-align: center; min-width: 180px; margin: 5px; }}
        .kpi-value {{ font-size: 24px; font-weight: bold; color: #003366; margin-top: 10px; }}
        .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
        .graph-card {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .section-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); margin-bottom: 20px; }}
    </style>
</head>
<body>

    <h1>Dashboard Interactivo - Análisis de Resistencia del Concreto</h1>
    <p class="subtitle">Evaluación de dosificaciones y modelo predictivo aplicado a Ingeniería Civil</p>

    <div class="kpi-container">
        <div class="kpi-card">
            <div>Resistencia Promedio</div>
            <div class="kpi-value">{kpi_promedio} MPa</div>
        </div>
        <div class="kpi-card">
            <div>Resistencia Máxima</div>
            <div class="kpi-value">{kpi_maximo} MPa</div>
        </div>
        <div class="kpi-card">
            <div>Edad Promedio</div>
            <div class="kpi-value">{kpi_edad_prom} Días</div>
        </div>
        <div class="kpi-card">
            <div>Registros Válidos</div>
            <div class="kpi-value">{kpi_registros} Muestras</div>
        </div>
    </div>

    <div class="grid-container">
        <div class="graph-card">{html_hist}</div>
        <div class="graph-card">{html_corr}</div>
        <div class="graph-card">{html_scatter}</div>
        <div class="graph-card">{html_time}</div>
    </div>

    <div class="section-card">
        <h2>Modelo Predictivo (Regresión Lineal)</h2>
        <p><b>Modelo Utilizado:</b> Regresión Lineal Múltiple (Scikit-Learn)</p>
        <p><b>Coeficiente de Determinación (R²):</b> {r2_val}</p>
        <p><b>Predicción Estimada a 120 días:</b> {pred_120_dias} MPa (utilizando contenido promedio de cemento y agua)</p>
    </div>

    <div class="section-card">
        <h2>Conclusiones Técnicas</h2>
        <ul>
            <li><b>Efecto del Cemento:</b> Existe una correlación positiva directa entre el contenido de cemento en la mezcla y la resistencia a la compresión (f'c), confirmando que mayores dosis incrementan la capacidad portante del elemento estructural.</li>
            <li><b>Curva de Madurez del Concreto:</b> La resistencia promedio presenta un crecimiento exponencial acelerado durante los primeros 28 días de curado, alcanzando una meseta de estabilización paulatina hacia los 90 y 120 días.</li>
            <li><b>Precisión del Modelo:</b> El modelo de Regresión Lineal obtiene un coeficiente de determinación R² de {r2_val}, permitiendo estimaciones de laboratorio preliminares para verificar el cumplimiento de especificaciones del proyecto.</li>
        </ul>
    </div>

</body>
</html>
"""

os.makedirs(os.path.dirname(output_html), exist_ok=True)
with open(output_html, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"Dashboard generado exitosamente en: {output_html}")