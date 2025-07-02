import hashlib
import re
from datetime import timedelta, datetime
import pickle
import os
import logging
import sqlite3
import tensorflow as tf
model = tf.keras.models.load_model('colpo_model.h5')
import numpy as np
from PIL import ImageEnhance
import streamlit as st

# 1. Validación de datos
class DataValidator:
    @staticmethod
    def validate_email(email):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    @staticmethod
    def validate_identification(identification):
        return identification.isalnum() and len(identification) >= 5
    
    @staticmethod
    def validate_patient_data(data):
        errors = []
        if not data.get('nombre') or len(data['nombre'].strip()) < 2:
            errors.append("Nombre debe tener al menos 2 caracteres")
        if not data.get('apellido') or len(data['apellido'].strip()) < 2:
            errors.append("Apellido debe tener al menos 2 caracteres")
        if not DataValidator.validate_identification(data.get('identificacion', '')):
            errors.append("Identificación debe ser alfanumérica y tener al menos 5 caracteres")
        if data.get('email') and not DataValidator.validate_email(data['email']):
            errors.append("Formato de email inválido")
        if data.get('edad', 0) < 0 or data.get('edad', 0) > 120:
            errors.append("Edad debe estar entre 0 y 120 años")
        return errors

# 2. Persistencia con SQLite
class SQLDataPersistence:
    DB_FILE = 'colpovision.db'

    @staticmethod
    def init_db():
        with sqlite3.connect(SQLDataPersistence.DB_FILE) as conn:
            c = conn.cursor()
            c.execute('''CREATE TABLE IF NOT EXISTS patients
                        (id INTEGER PRIMARY KEY, 
                         identificacion TEXT UNIQUE,
                         nombre TEXT,
                         apellido TEXT,
                         email TEXT,
                         edad INTEGER,
                         timestamp DATETIME)''')
            c.execute('''CREATE TABLE IF NOT EXISTS analysis_results
                        (id INTEGER PRIMARY KEY,
                         patient_id INTEGER,
                         result_type TEXT,
                         confidence REAL,
                         timestamp DATETIME,
                         FOREIGN KEY(patient_id) REFERENCES patients(id))''')
            conn.commit()

    @staticmethod
    def save_patient(data):
        try:
            with sqlite3.connect(SQLDataPersistence.DB_FILE) as conn:
                c = conn.cursor()
                c.execute('''INSERT OR REPLACE INTO patients 
                            (identificacion, nombre, apellido, email, edad, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?)''',
                         (data['identificacion'], data['nombre'], data['apellido'],
                          data.get('email'), data.get('edad'), datetime.now()))
                conn.commit()
                return c.lastrowid
        except Exception as e:
            Logger.log_error(f"Error saving patient: {e}", "SQLDataPersistence.save_patient")
            return None

    @staticmethod
    def save_analysis_result(patient_id, result_type, confidence):
        try:
            with sqlite3.connect(SQLDataPersistence.DB_FILE) as conn:
                c = conn.cursor()
                c.execute('''INSERT INTO analysis_results 
                            (patient_id, result_type, confidence, timestamp)
                            VALUES (?, ?, ?, ?)''',
                         (patient_id, result_type, confidence, datetime.now()))
                conn.commit()
                return True
        except Exception as e:
            Logger.log_error(f"Error saving analysis: {e}", "SQLDataPersistence.save_analysis_result")
            return False

    @staticmethod
    def load_patients():
        try:
            with sqlite3.connect(SQLDataPersistence.DB_FILE) as conn:
                c = conn.cursor()
                c.execute('SELECT * FROM patients')
                return c.fetchall()
        except Exception as e:
            Logger.log_error(f"Error loading patients: {e}", "SQLDataPersistence.load_patients")
            return []

# 3. Seguridad
class SecurityManager:
    @staticmethod
    def hash_password(password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password, hashed):
        return SecurityManager.hash_password(password) == hashed
    
    @staticmethod
    def sanitize_filename(filename):
        safe_chars = re.sub(r'[^\w\s-]', '', filename)
        return re.sub(r'[-\s]+', '-', safe_chars)

# 4. Logging
class Logger:
    @staticmethod
    def setup_logging():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('colpovision.log'),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def log_analysis(patient_id, result_type, confidence):
        logger = logging.getLogger(__name__)
        logger.info(f"Análisis realizado - Paciente: {patient_id}, Tipo: {result_type}, Confianza: {confidence}")
    
    @staticmethod
    def log_error(error_msg, context=""):
        logger = logging.getLogger(__name__)
        logger.error(f"Error: {error_msg} - Contexto: {context}")

# 5. Análisis de imágenes con TensorFlow
class EnhancedImageAnalyzer:
    def __init__(self, model_path='colpo_model'):
        self.model = self.load_model(model_path)

    def load_model(self, model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            Logger.log_error("Modelo TensorFlow cargado exitosamente", "EnhancedImageAnalyzer.load_model")
            return model
        except Exception as e:
            Logger.log_error(f"Error cargando modelo: {e}", "EnhancedImageAnalyzer.load_model")
            return None

    @staticmethod
    def preprocess_image(image):
        import cv2
        img_array = np.array(image)
        enhancer = ImageEnhance.Contrast(image)
        enhanced_img = enhancer.enhance(1.2)
        # Redimensionar a tamaño esperado por el modelo
        processed_img = cv2.resize(img_array, (224, 224))
        processed_img = processed_img / 255.0  # Normalización
        return np.expand_dims(processed_img, axis=0)

    def analyze_image(self, image, patient_id):
        if not self.model:
            return None, "Modelo no cargado"
        
        quality_ok, quality_msg = self.validate_image_quality(image)
        if not quality_ok:
            return None, quality_msg

        try:
            processed_img = self.preprocess_image(image)
            prediction = self.model.predict(processed_img)
            result_type = 'Normal' if prediction[0][0] > 0.5 else 'Anormal'
            confidence = float(max(prediction[0]))
            
            Logger.log_analysis(patient_id, result_type, confidence)
            SQLDataPersistence.save_analysis_result(patient_id, result_type, confidence)
            return result_type, confidence
        except Exception as e:
            Logger.log_error(f"Error en análisis de imagen: {e}", "EnhancedImageAnalyzer.analyze_image")
            return None, str(e)

    @staticmethod
    def validate_image_quality(image):
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        if height < 224 or width < 224:
            return False, "Imagen muy pequeña (mínimo 224x224)"
        mean_intensity = np.mean(img_array)
        if mean_intensity < 10:
            return False, "Imagen muy oscura"
        if mean_intensity > 245:
            return False, "Imagen muy clara"
        return True, "Calidad aceptable"

# 6. Configuración
class Config:
    DEFAULT_CONFIG = {
        'ui': {
            'theme': 'light',
            'primary_color': '#1e3c72',
            'secondary_color': '#2a5298'
        },
        'model': {
            'confidence_threshold': 0.75,
            'batch_size': 8,
            'max_image_size': 512
        },
        'email': {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'use_tls': True
        }
    }
    
    @staticmethod
    def load_config():
        if 'app_config' not in st.session_state:
            st.session_state.app_config = Config.DEFAULT_CONFIG.copy()
        return st.session_state.app_config
    
    @staticmethod
    def save_config(config):
        st.session_state.app_config = config
    
    @staticmethod
    def get_config_value(path, default=None):
        config = Config.load_config()
        keys = path.split('.')
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        return config

# Función principal
def enhanced_main():
    Logger.setup_logging()
    SQLDataPersistence.init_db()
    
    if 'data_loaded' not in st.session_state:
        st.session_state.patients_db = SQLDataPersistence.load_patients()
        st.session_state.data_loaded = True
    
    config = Config.load_config()
    
    # Interfaz de Streamlit
    st.title("ColpoVision - Sistema de Análisis")
    
    with st.form("patient_form"):
        nombre = st.text_input("Nombre")
        apellido = st.text_input("Apellido")
        identificacion = st.text_input("Identificación")
        email = st.text_input("Email")
        edad = st.number_input("Edad", min_value=0, max_value=120)
        image = st.file_uploader("Subir imagen", type=['png', 'jpg', 'jpeg'])
        submitted = st.form_submit_button("Analizar")
        
        if submitted:
            data = {
                'nombre': nombre,
                'apellido': apellido,
                'identificacion': identificacion,
                'email': email,
                'edad': edad
            }
            
            errors = DataValidator.validate_patient_data(data)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                patient_id = SQLDataPersistence.save_patient(data)
                if patient_id and image:
                    analyzer = EnhancedImageAnalyzer()
                    result, msg = analyzer.analyze_image(image, patient_id)
                    if result:
                        st.success(f"Resultado: {result} (Confianza: {msg:.2%})")
                    else:
                        st.error(msg)

if __name__ == "__main__":
    enhanced_main()
