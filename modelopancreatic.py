import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import plotly.express as px
import plotly.graph_objects as go

# Configuração inicial
st.set_page_config(page_title="Dashboard de Previsão de Câncer Pancreático, Feito por Matheus Prado", layout="wide")

# Título e subtítulo
st.title("📊 Dashboard de Previsão de Câncer Pancreático 🩺 - Matheus Prado")
st.subheader("🔍 Explore os dados do projeto")

# Carregar os dados
@st.cache_data
def load_data():
    df = pd.read_csv('/home/prado/.cache/kagglehub/datasets/johnjdavisiv/urinary-biomarkers-for-pancreatic-cancer/versions/1/Debernardi et al 2020 data.csv')
    return df

df = load_data()

# Pré-processamento dos dados
df['diagnosis'] = df['diagnosis'] == 3  # Codificar diagnóstico como True (classe 3) ou False
df['sex'] = df['sex'].map({'M': 1, 'F': 0})  # Codificar sexo como 1 (masculino) ou 0 (feminino)
df = df[['creatinine', 'plasma_CA19_9', 'age', 'sex', 'LYVE1', 'REG1B', 'TFF1', 'diagnosis']].copy()

# Mostrar os dados brutos
if st.checkbox("Mostrar dados brutos"):
    st.dataframe(df)

# EDA (Análise Exploratória de Dados)
st.header("📊 Análise Exploratória de Dados (EDA)")

# Gráficos de Pares
st.subheader("Gráficos de Pares")
st.text("Perceba no gráfico abaixo que no LYVE1-LYVE1, notamos um aumento nos valores dos pacientes com câncer (true) comparados aos que não possuem. O plasma CA19, possui uma frequência alta para um determinado valor entre aqueles que não possuem câncer (Falso).")

fig = px.scatter_matrix(
    df,
    dimensions=['REG1B', 'plasma_CA19_9', 'creatinine', 'LYVE1', 'TFF1', 'age'],
    color='diagnosis',
    title="Gráficos de Pares",
    labels={'diagnosis': 'Diagnóstico'},
    color_discrete_map={True: 'red', False: 'blue'}
)
fig.update_layout(height=800, width=1000)
st.plotly_chart(fig)

# Matriz de Correlação
st.subheader("Matriz de Correlação")
st.text("O LYVE1 se destaca na correlação com o diagnóstico, novamente demonstrando-se um ótimo parâmetro para o diagnóstico.")

corr = df.dropna().corr()
fig_corr = px.imshow(
    corr,
    text_auto=True,
    title="Matriz de Correlação",
    color_continuous_scale='RdBu'
)
st.plotly_chart(fig_corr)

# Treinamento do Modelo
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_model():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb', XGBClassifier(eval_metric='logloss', random_state=42))
    ])
    param_grid = {
        'xgb__n_estimators': [50, 100, 200],
        'xgb__learning_rate': [0.01, 0.1, 0.2]
    }
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

best_model = train_model()

# Avaliação do Modelo
st.header("🎯 Avaliação do Modelo")

y_pred = best_model.predict(X_test)

st.subheader("Métricas de Desempenho")
st.write(f"**Acurácia no conjunto de teste:** {accuracy_score(y_test, y_pred):.2f}")
st.text("Relatório de Classificação:")
st.text(classification_report(y_test, y_pred))

# Matriz de Confusão
st.subheader("Matriz de Confusão")
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(
    cm,
    text_auto=True,
    title="Matriz de Confusão",
    labels=dict(x="Previsão", y="Real"),
    x=['Não Classe 3', 'Classe 3'],
    y=['Não Classe 3', 'Classe 3'],
    color_continuous_scale='Blues'
)
st.plotly_chart(fig_cm)

# Importância das Features
st.subheader("Importância das Features")
st.text("Percebemos que, de fato, o LYVE1 era um importante marcador para realizar as previsões do modelo. No entanto, após o aprendizado do modelo, há uma grande importância do Plasma CA19.")

importances = best_model.named_steps['xgb'].feature_importances_
indices = np.argsort(importances)[::-1]

fig_importance = go.Figure()
fig_importance.add_trace(go.Bar(
    x=importances[indices],
    y=X.columns[indices],
    orientation='h',
    marker_color='red'
))
fig_importance.update_layout(
    title="Importância das Features",
    xaxis_title="Importância",
    yaxis_title="Feature"
)
st.plotly_chart(fig_importance)

# Interface para Previsão Individual
st.header("🔮 Faça uma Previsão Individual")
st.subheader("Insira os valores abaixo para prever a probabilidade de câncer pancreático:")

creatinine = st.number_input("Creatinine", min_value=0.0, max_value=1000.0, value=1.0)
plasma_CA19_9 = st.number_input("Plasma CA19-9", min_value=0.0, max_value=10000.0, value=100.0)
age = st.number_input("Idade", min_value=0, max_value=120, value=50)
sex = st.selectbox("Sexo", ["Masculino", "Feminino"])
LYVE1 = st.number_input("LYVE1", min_value=0.0, max_value=1000.0, value=1.0)
REG1B = st.number_input("REG1B", min_value=0.0, max_value=1000.0, value=1.0)
TFF1 = st.number_input("TFF1", min_value=0.0, max_value=1000.0, value=1.0)

# Mapear sexo para números
sex_mapped = 1 if sex == "Masculino" else 0

# Criar um DataFrame com os dados inseridos
input_data = pd.DataFrame({
    'creatinine': [creatinine],
    'plasma_CA19_9': [plasma_CA19_9],
    'age': [age],
    'sex': [sex_mapped],
    'LYVE1': [LYVE1],
    'REG1B': [REG1B],
    'TFF1': [TFF1]
})

# Fazer a previsão
if st.button("Prever"):
    prediction = best_model.predict(input_data)
    result = "Alta probabilidade de ser câncer" if prediction[0] else "Baixa probabilidade de ser câncer"
    st.success(f"Resultado da Previsão: **{result}**")