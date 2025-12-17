
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Clasificación Calidad Agua SVM", page_icon="💧", layout="wide")
st.title("💧 Clasificación de Calidad de Agua con SVM")
st.markdown("---")

# ============================================================================
# FUNCIONES
# ============================================================================
@st.cache_data
def generar_datos(n_samples=1000):
    np.random.seed(42)
    ph = np.random.normal(7.2, 0.8, n_samples)
    turbidez = np.abs(np.random.gamma(2, 2.5, n_samples))
    conductividad = np.random.normal(500, 150, n_samples)
    oxigeno = np.random.normal(7.5, 1.8, n_samples)
    temperatura = np.random.normal(20, 4, n_samples)
    tds = conductividad * 0.65 + np.random.normal(0, 20, n_samples)

    calidad = []
    for i in range(n_samples):
        score = sum([
            2 if 6.5 <= ph[i] <= 8.5 else 0,
            2 if turbidez[i] < 5 else 0,
            2 if oxigeno[i] > 6 else 0,
            1 if 200 <= conductividad[i] <= 800 else 0,
            1 if tds[i] < 500 else 0
        ])
        calidad.append(0 if score >= 6 else (1 if score >= 3 else 2))

    return pd.DataFrame({
        'pH': ph, 'Turbidez': turbidez, 'Conductividad': conductividad,
        'Oxigeno': oxigeno, 'Temperatura': temperatura, 'TDS': tds, 'Calidad': calidad
    })

def entrenar_modelo(df, kernel, C, gamma, usar_smote):
    X = df.drop('Calidad', axis=1)
    y = df['Calidad']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    if usar_smote:
        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    modelo = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, random_state=42)
    modelo.fit(X_train_scaled, y_train)

    y_pred = modelo.predict(X_test_scaled)
    y_pred_proba = modelo.predict_proba(X_test_scaled)
    cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5)

    return {
        'modelo': modelo, 'scaler': scaler, 'X_test': X_test_scaled,
        'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba,
        'cv_scores': cv_scores
    }

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("⚙️ Configuración")
n_samples = st.sidebar.slider("Muestras", 100, 2000, 1000, 100)
kernel = st.sidebar.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
C = st.sidebar.slider("Parámetro C", 0.1, 10.0, 1.0, 0.1)
gamma = st.sidebar.select_slider("Gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1.0], value='scale')
usar_smote = st.sidebar.checkbox("Usar SMOTE", True)

entrenar = st.sidebar.button("🚀 Entrenar Modelo", type="primary")

# ============================================================================
# INICIALIZACIÓN
# ============================================================================
if 'modelo_entrenado' not in st.session_state:
    st.session_state.modelo_entrenado = False
if 'df' not in st.session_state or len(st.session_state.df) != n_samples:
    st.session_state.df = generar_datos(n_samples)

df = st.session_state.df

# ============================================================================
# TABS
# ============================================================================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Datos", "🤖 Modelo", "📈 Resultados", "🔍 Predicción"])

# TAB 1: DATOS
with tab1:
    st.header("📊 Exploración de Datos")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total", len(df))
    col2.metric("Buena", len(df[df['Calidad']==0]), f"{len(df[df['Calidad']==0])/len(df)*100:.1f}%")
    col3.metric("Regular", len(df[df['Calidad']==1]), f"{len(df[df['Calidad']==1])/len(df)*100:.1f}%")
    col4.metric("Mala", len(df[df['Calidad']==2]), f"{len(df[df['Calidad']==2])/len(df)*100:.1f}%")

    st.dataframe(df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig = px.bar(df['Calidad'].value_counts().sort_index(),
                    title='Distribución de Calidad',
                    color_discrete_sequence=['green', 'orange', 'red'])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(df, x='Turbidez', y='Oxigeno', color='Calidad',
                        title='Turbidez vs Oxígeno',
                        color_discrete_map={0: 'green', 1: 'orange', 2: 'red'})
        st.plotly_chart(fig, use_container_width=True)

    fig = px.imshow(df.drop('Calidad', axis=1).corr(), text_auto='.2f',
                   title='Matriz de Correlación')
    st.plotly_chart(fig, use_container_width=True)

# TAB 2: MODELO
with tab2:
    st.header("🤖 Entrenamiento del Modelo")

    if entrenar:
        with st.spinner('Entrenando...'):
            st.session_state.resultados = entrenar_modelo(df, kernel, C, gamma, usar_smote)
            st.session_state.modelo_entrenado = True
            st.success("✅ Modelo entrenado")

    if st.session_state.modelo_entrenado:
        res = st.session_state.resultados

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{accuracy_score(res['y_test'], res['y_pred']):.2%}")
        col2.metric("CV Score", f"{res['cv_scores'].mean():.2%}")
        col3.metric("CV Std", f"{res['cv_scores'].std():.4f}")

        st.subheader("Parámetros")
        st.info(f"Kernel: {kernel} | C: {C} | Gamma: {gamma} | SMOTE: {usar_smote}")
    else:
        st.info("👈 Configura y entrena el modelo")

# TAB 3: RESULTADOS
with tab3:
    st.header("📈 Resultados")

    if st.session_state.modelo_entrenado:
        res = st.session_state.resultados
        class_names = ['Buena', 'Regular', 'Mala']

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(res['y_test'], res['y_pred'])
            fig = px.imshow(cm, x=class_names, y=class_names, text_auto=True,
                           title='Matriz de Confusión', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Visualización PCA")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(res['X_test'])
            df_pca = pd.DataFrame({'PC1': X_pca[:, 0], 'PC2': X_pca[:, 1], 'Calidad': res['y_test']})
            fig = px.scatter(df_pca, x='PC1', y='PC2', color='Calidad',
                           color_discrete_map={0: 'green', 1: 'orange', 2: 'red'})
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Reporte de Clasificación")
        report = classification_report(res['y_test'], res['y_pred'],
                                      target_names=class_names, output_dict=True)
        st.dataframe(pd.DataFrame(report).T, use_container_width=True)
    else:
        st.warning("⚠️ Entrena el modelo primero")

# TAB 4: PREDICCIÓN
with tab4:
    st.header("🔍 Predicción")

    if st.session_state.modelo_entrenado:
        res = st.session_state.resultados

        st.subheader("Ingresa parámetros del agua")

        col1, col2, col3 = st.columns(3)
        ph = col1.number_input("pH", 0.0, 14.0, 7.0, 0.1)
        turbidez = col1.number_input("Turbidez (NTU)", 0.0, 50.0, 5.0, 0.5)
        conductividad = col2.number_input("Conductividad (µS/cm)", 0.0, 2000.0, 500.0, 10.0)
        oxigeno = col2.number_input("Oxígeno (mg/L)", 0.0, 20.0, 8.0, 0.1)
        temperatura = col3.number_input("Temperatura (°C)", 0.0, 40.0, 20.0, 0.5)
        tds = col3.number_input("TDS (mg/L)", 0.0, 2000.0, 300.0, 10.0)

        if st.button("🎯 Predecir", type="primary"):
            datos = pd.DataFrame([[ph, turbidez, conductividad, oxigeno, temperatura, tds]],
                                columns=['pH', 'Turbidez', 'Conductividad', 'Oxigeno', 'Temperatura', 'TDS'])

            datos_scaled = res['scaler'].transform(datos)
            pred = res['modelo'].predict(datos_scaled)[0]
            probs = res['modelo'].predict_proba(datos_scaled)[0]

            class_names = ['Buena', 'Regular', 'Mala']
            colors = ['green', 'orange', 'red']

            st.markdown("---")

            if pred == 0:
                st.success(f"## 🟢 Calidad: {class_names[pred]}")
                st.info("✅ Agua apta para consumo")
            elif pred == 1:
                st.warning(f"## 🟠 Calidad: {class_names[pred]}")
                st.info("⚠️ Requiere tratamiento")
            else:
                st.error(f"## 🔴 Calidad: {class_names[pred]}")
                st.info("❌ No apta para consumo")

            col1, col2, col3 = st.columns(3)
            col1.metric("Buena", f"{probs[0]:.1%}")
            col2.metric("Regular", f"{probs[1]:.1%}")
            col3.metric("Mala", f"{probs[2]:.1%}")

            fig = go.Figure(data=[go.Bar(x=class_names, y=probs,
                                        marker_color=colors,
                                        text=[f"{p:.1%}" for p in probs],
                                        textposition='auto')])
            fig.update_layout(title="Probabilidades", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Entrena el modelo primero")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("<div style='text-align: center'>💧 Clasificación de Calidad de Agua con SVM | Streamlit</div>",
           unsafe_allow_html=True)
