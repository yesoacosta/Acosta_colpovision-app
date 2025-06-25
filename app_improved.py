# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from datetime import datetime
import numpy as np
from PIL import Image
import io
import sqlite3
import os
import logging
import re
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
# import tensorflow as tf
# from tensorflow.keras.applications import VGG16
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.preprocessing import image as tf_image

# Set up logging to track errors and events
logging.basicConfig(filename='colpovision.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Configure Streamlit page
st.set_page_config(page_title="ColpoVision - Análisis de Colposcopía", page_icon="🔬", layout="wide")

# Custom CSS for a clean interface
st.markdown("""
<style>
    .header { background: #2a5298; color: white; padding: 1rem; border-radius: 10px; text-align: center; }
    .card { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# Initialize database
def init_db():
    # Create SQLite database and tables if they don't exist
    conn = sqlite3.connect('colpovision.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS patients (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 nombre TEXT, apellido TEXT, identificacion TEXT UNIQUE,
                 fecha_nacimiento TEXT, edad INTEGER, email TEXT, telefono TEXT,
                 created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS analyses (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 patient_id INTEGER, image_name TEXT, analysis_date TEXT,
                 predictions TEXT, confidence REAL, acetic_findings TEXT,
                 lugol_findings TEXT, transformation_zone TEXT, diagnosis TEXT,
                 technique TEXT, FOREIGN KEY(patient_id) REFERENCES patients(id))''')
    conn.commit()
    conn.close()

init_db()

# CNN Model for Colposcopy Analysis
class ColpoCNN:
    def __init__(self):
        # Load or initialize the CNN model (VGG16-based)
        self.model = self.load_model()
        self.labels = ['Normal', 'CIN I', 'CIN II', 'CIN III', 'Carcinoma']

    def load_model(self):
        # Simulate loading a pre-trained model fine-tuned on Servi/UCI dataset
        # In practice, you'd train VGG16 on the dataset and save the model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        model = Sequential([
            base_model,
            Flatten(),
            Dense(256, activation='relu'),
            Dense(5, activation='softmax')  # 5 classes for colposcopy
        ])
        # Compile model (simulated; in reality, load weights from trained model)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def preprocess_image(self, img):
        # Resize and normalize image for CNN input
        img = img.resize((224, 224))
        img_array = tf_image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    def analyze_image(self, img):
        # Analyze image using CNN
        try:
            img_array = self.preprocess_image(img)
            predictions = self.model.predict(img_array)[0]
            confidence = np.max(predictions)
            pred_dict = {label: float(prob) for label, prob in zip(self.labels, predictions)}
            # Simulate acetic acid and Lugol findings (in practice, derived from image features)
            acetic_findings = "Blanqueamiento acetoblanco leve en cuadrante superior."
            lugol_findings = "Captación parcial de Lugol en zona central."
            # Transformation zone characteristics
            tz_chars = "Zona de transformación tipo 1, márgenes definidos."
            # Diagnostic impression
            max_label = self.labels[np.argmax(predictions)]
            diagnosis = f"Impresión diagnóstica: {max_label} con confianza {confidence:.2f}."
            return {
                'predictions': pred_dict,
                'confidence': confidence,
                'acetic_findings': acetic_findings,
                'lugol_findings': lugol_findings,
                'transformation_zone': tz_chars,
                'diagnosis': diagnosis,
                'technique': 'CNN (VGG16)'
            }
        except Exception as e:
            logging.error(f"Error en análisis de imagen: {e}")
            return None

# Patient Management
class PatientManager:
    @staticmethod
    def add_patient(data):
        # Add patient to database
        conn = sqlite3.connect('colpovision.db')
        c = conn.cursor()
        try:
            c.execute('''INSERT INTO patients (nombre, apellido, identificacion, fecha_nacimiento, 
                         edad, email, telefono, created_at)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (data['nombre'], data['apellido'], data['identificacion'],
                       str(data['fecha_nacimiento']), data['edad'], data['email'],
                       data['telefono'], datetime.now().strftime('%Y-%m-%d %H:%M')))
            conn.commit()
            patient_id = c.lastrowid
            conn.close()
            return patient_id
        except sqlite3.IntegrityError:
            conn.close()
            return None

    @staticmethod
    def get_patients():
        # Retrieve all patients
        conn = sqlite3.connect('colpovision.db')
        c = conn.cursor()
        c.execute('SELECT * FROM patients')
        patients = c.fetchall()
        conn.close()
        return patients

    @staticmethod
    def get_patient(patient_id):
        # Get patient by ID
        conn = sqlite3.connect('colpovision.db')
        c = conn.cursor()
        c.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
        patient = c.fetchone()
        conn.close()
        return patient

# Report Generator
class ReportGenerator:
    @staticmethod
    def create_pdf_report(patient, analysis_results):
        # Generate detailed PDF report
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', fontSize=18, alignment=1, textColor=colors.darkblue)
        story = []

        # Title
        story.append(Paragraph("Reporte de Colposcopía - ColpoVision", title_style))
        story.append(Spacer(1, 20))

        # Patient Info
        patient_info = [
            ['Datos del Paciente', ''],
            ['Nombre', f"{patient[1]} {patient[2]}"],
            ['Identificación', patient[3]],
            ['Fecha de Nacimiento', patient[4]],
            ['Edad', str(patient[5])],
            ['Email', patient[6] or 'N/A'],
            ['Teléfono', patient[7] or 'N/A'],
            ['Fecha del Análisis', datetime.now().strftime('%d/%m/%Y %H:%M')]
        ]
        table = Table(patient_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Analysis Results
        story.append(Paragraph("Resultados del Análisis", styles['Heading2']))
        results_data = [['Diagnóstico', 'Probabilidad (%)']]
        for diag, prob in analysis_results['predictions'].items():
            results_data.append([diag, f"{prob*100:.1f}"])
        table = Table(results_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(table)
        story.append(Spacer(1, 20))

        # Detailed Findings
        story.append(Paragraph("Hallazgos Clínicos", styles['Heading2']))
        findings = f"""
        <b>Técnica Utilizada:</b> {analysis_results['technique']}<br/>
        <b>Hallazgos con Ácido Acético:</b> {analysis_results['acetic_findings']}<br/>
        <b>Hallazgos con Lugol:</b> {analysis_results['lugol_findings']}<br/>
        <b>Características de la Zona de Transformación:</b> {analysis_results['transformation_zone']}<br/>
        <b>Impresión Diagnóstica:</b> {analysis_results['diagnosis']}<br/>
        """
        story.append(Paragraph(findings, styles['Normal']))
        story.append(Spacer(1, 20))

        # Footer
        story.append(Paragraph(
            "<i>Generado por ColpoVision. Consulte a un médico para interpretación.</i>",
            styles['Italic']
        ))

        doc.build(story)
        buffer.seek(0)
        return buffer

# Data Validator
class DataValidator:
    @staticmethod
    def validate_email(email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))

    @staticmethod
    def validate_patient(data):
        errors = []
        if not data['nombre'] or len(data['nombre'].strip()) < 2:
            errors.append("Nombre inválido.")
        if not data['apellido'] or len(data['apellido'].strip()) < 2:
            errors.append("Apellido inválido.")
        if not data['identificacion'] or len(data['identificacion']) < 5:
            errors.append("Identificación inválida.")
        if data['email'] and not DataValidator.validate_email(data['email']):
            errors.append("Email inválido.")
        if data['edad'] < 0 or data['edad'] > 120:
            errors.append("Edad inválida.")
        return errors

# Main UI
def main():
    st.markdown("<div class='header'><h1>ColpoVision</h1><p>Análisis de Colposcopía con IA</p></div>", unsafe_allow_html=True)
    menu = st.sidebar.selectbox("Menú", ["Dashboard", "Pacientes", "Análisis", "Reportes"])
    
    if menu == "Dashboard":
        st.header("Dashboard")
        patients = PatientManager.get_patients()
        st.metric("Pacientes Registrados", len(patients))
    
    elif menu == "Pacientes":
        st.header("Gestión de Pacientes")
        with st.form("new_patient"):
            col1, col2 = st.columns(2)
            with col1:
                nombre = st.text_input("Nombre")
                identificacion = st.text_input("Identificación")
                edad = st.number_input("Edad", 0, 120, 30)
            with col2:
                apellido = st.text_input("Apellido")
                email = st.text_input("Email")
                telefono = st.text_input("Teléfono")
            fecha_nacimiento = st.date_input("Fecha de Nacimiento")
            if st.form_submit_button("Guardar"):
                patient_data = {
                    'nombre': nombre,
                    'apellido': apellido,
                    'identificacion': identificacion,
                    'fecha_nacimiento': fecha_nacimiento,
                    'edad': edad,
                    'email': email,
                    'telefono': telefono
                }
                errors = DataValidator.validate_patient(patient_data)
                if not errors:
                    patient_id = PatientManager.add_patient(patient_data)
                    if patient_id:
                        st.success("Paciente guardado.")
                    else:
                        st.error("Identificación duplicada.")
                else:
                    for error in errors:
                        st.error(error)
        
        # List patients
        patients = PatientManager.get_patients()
        if patients:
            for p in patients:
                st.markdown(f"<div class='card'>ID: {p[0]} | {p[1]} {p[2]} | Edad: {p[5]}</div>", unsafe_allow_html=True)

    elif menu == "Análisis":
        st.header("Análisis de Imágenes")
        patients = PatientManager.get_patients()
        if patients:
            patient_options = {f"{p[1]} {p[2]} - {p[3]}": p[0] for p in patients}
            selected = st.selectbox("Paciente", list(patient_options.keys()))
            patient_id = patient_options[selected]
            patient = PatientManager.get_patient(patient_id)
            
            uploaded_file = st.file_uploader("Cargar Imagen", type=['png', 'jpg', 'jpeg'])
            if uploaded_file:
                image = Image.open(uploaded_file)
                st.image(image, caption="Imagen Cargada")
                if st.button("Analizar"):
                    with st.spinner("Analizando..."):
                        cnn = ColpoCNN()
                        results = cnn.analyze_image(image)
                        if results:
                            # Save analysis to database
                            conn = sqlite3.connect('colpovision.db')
                            c = conn.cursor()
                            c.execute('''INSERT INTO analyses (patient_id, image_name, analysis_date, predictions,
                                         confidence, acetic_findings, lugol_findings, transformation_zone,
                                         diagnosis, technique)
                                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                                      (patient_id, uploaded_file.name, datetime.now().strftime('%Y-%m-%d %H:%M'),
                                       str(results['predictions']), results['confidence'],
                                       results['acetic_findings'], results['lugol_findings'],
                                       results['transformation_zone'], results['diagnosis'], results['technique']))
                            conn.commit()
                            conn.close()
                            
                            # Display results
                            st.subheader("Resultados")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Diagnóstico:** {results['diagnosis']}")
                                st.write(f"**Confianza:** {results['confidence']*100:.1f}%")
                            with col2:
                                st.write(f"**Ácido Acético:** {results['acetic_findings']}")
                                st.write(f"**Lugol:** {results['lugol_findings']}")
                            st.write(f"**Zona de Transformación:** {results['transformation_zone']}")
                            
                            # Generate report
                            pdf_buffer = ReportGenerator.create_pdf_report(patient, results)
                            st.download_button(
                                label="Descargar Reporte",
                                data=pdf_buffer,
                                file_name=f"Reporte_{patient[2]}.pdf",
                                mime="application/pdf"
                            )
                        else:
                            st.error("Error en el análisis.")
    
    elif menu == "Reportes":
        st.header("Reportes")
        conn = sqlite3.connect('colpovision.db')
        c = conn.cursor()
        c.execute('SELECT a.id, a.analysis_date, p.nombre, p.apellido FROM analyses a JOIN patients p ON a.patient_id = p.id')
        analyses = c.fetchall()
        conn.close()
        if analyses:
            for a in analyses:
                st.markdown(f"<div class='card'>Análisis #{a[0]} | {a[2]} {a[3]} | {a[1]}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
