
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="ColpoVision + IA", layout="wide")
st.title("🔬 ColpoVision – Análisis IA + Informe Clínico")

st.markdown("### 1. Cargar imagen colposcópica")
imagen = st.file_uploader("Seleccioná una imagen", type=["jpg", "jpeg", "png"])

if imagen:
    st.image(imagen, caption="Imagen cargada", use_column_width=True)

    st.markdown("### 2. Diagnóstico asistido por IA")
    # Simulación de IA hasta que se integre el modelo real
    resultado_ia = "🟡 Sospechosa de lesión de bajo grado (NIC 1)"
    confianza = "86%"

    st.success(f"**Resultado IA:** {resultado_ia}")
    st.info(f"**Confianza del modelo:** {confianza}")

    st.markdown("### 3. Informe clínico estructurado")
    nombre = st.text_input("Nombre del paciente:")
    edad = st.text_input("Edad:")
    fecha = st.date_input("Fecha del estudio:", value=datetime.today())

    motivo = st.text_area("Motivo de consulta:")
    tecnica = st.text_area("Técnica y métodos utilizados:")
    hallazgos = st.text_area("Hallazgos colposcópicos:")
    impresion = st.text_area("Impresión diagnóstica:")
    recomendaciones = st.text_area("Recomendaciones:")

    if st.button("🔍 Ver informe completo"):
        st.markdown("---")
        st.markdown(f"""
        ### 🧾 Informe Colposcópico Final

        **Paciente:** {nombre or '[No proporcionado]'}  
        **Edad:** {edad or '[No especificada]'}  
        **Fecha del estudio:** {fecha.strftime('%d/%m/%Y')}  
        **Motivo de consulta:** {motivo or '[No especificado]'}  

        ---
        **📸 Análisis automático por IA:**  
        Resultado: {resultado_ia}  
        Confianza: {confianza}  

        ---
        **Técnica y métodos utilizados:**  
        {tecnica}

        **Hallazgos:**  
        {hallazgos}

        **Impresión diagnóstica:**  
        {impresion}

        **Recomendaciones:**  
        {recomendaciones}
        """)
