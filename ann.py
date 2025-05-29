import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer 


st.title('Clasificación de Sentimiento de Reseñas de Producto (ANN)')
st.write('Esta aplicación te permite cargar un dataset de reseñas, preprocesar el texto, entrenar una ANN y evaluar su rendimiento.')

st.header('1. Carga de Datos')
st.write("Sube tu archivo CSV (ej. `IMDB Dataset.csv`).")
uploaded_file = st.file_uploader("Sube tu archivo CSV", type="csv")

df = None
if uploaded_file is not None:
    try:
        
        df = pd.read_csv(
            uploaded_file, 
            
            sep=',',            
            quotechar='"',      
            doublequote=True,   
            engine='python'    
        )
        
        st.success("Archivo cargado exitosamente.")
        st.write("Primeras 5 filas de tu dataset:")
        st.dataframe(df.head())
        st.write(f"Dimensiones del dataset: {df.shape[0]} filas, {df.shape[1]} columnas.")

        st.header('2. Preprocesamiento Específico del Dataset')
       
        if 'sentiment' in df.columns:
            if df['sentiment'].dtype == 'object': 
                st.write("Mapeando la columna 'sentiment' de texto a números (1=positivo, 0=negativo).")
                df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
                if df['sentiment'].isnull().any():
                    st.warning("Se encontraron valores nulos después del mapeo de sentimiento. Posibles valores no reconocidos en la columna 'sentiment'.")
                    df.dropna(subset=['sentiment'], inplace=True) 
            
            y = df['sentiment']
            X_text = df['review'] 
            
            df.dropna(subset=['review'], inplace=True) 
            
            st.write("\nDistribución de la clase objetivo (sentimiento mapeado):")
            st.write(y.value_counts())


            st.markdown("""
            <div style="background-color:#fff3cd; color:#856404; border:1px solid #ffeeba; padding:15px; border-radius:5px;">
                <h4>⚠️ Vectorización de Texto ⚠️</h4>
                <p>Convirtiendo el texto de las reseñas a características numéricas usando TF-IDF.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader('Vectorizando el Texto (TF-IDF)')
            st.write("Convirtiendo las reseñas a características numéricas usando TF-IDF...")
            
            vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2)) # Añadido ngram_range para más contexto
            X = vectorizer.fit_transform(X_text)
            
            st.success(f"Texto vectorizado. Ahora tienes {X.shape[1]} características numéricas.")
            st.write(f"Dimensiones de las características (X) después de TF-IDF: {X.shape}")

            st.subheader('3. Preprocesamiento Final y Entrenamiento')
            if st.button('Entrenar Modelo ANN'):

                st.write("Realizando división de datos y escalado de características...")
                X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, random_state=42, stratify=y)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                st.success("Datos preprocesados y escalados correctamente.")

                st.write("Construyendo el modelo de Red Neuronal Artificial...")
                model = Sequential()
                model.add(Dense(units=128, activation='relu', input_shape=(X_train_scaled.shape[1],))) 
                model.add(Dense(units=64, activation='relu'))
                model.add(Dense(units=1, activation='sigmoid'))
                
                with st.expander("Ver resumen del modelo ANN"):
                    model_summary = []
                    model.summary(print_fn=lambda x: model_summary.append(x))
                    st.text("\n".join(model_summary))

                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                st.success("Modelo compilado.")

                st.write("Entrenando el modelo... esto puede tardar un tiempo.")
                progress_bar = st.progress(0)
                status_text = st.empty()

                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / 10 
                        progress_bar.progress(progress)
                        status_text.text(f"Época {epoch + 1}/10 - Pérdida: {logs['loss']:.4f}, Precisión: {logs['accuracy']:.4f}")

                history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, 
                                    validation_split=0.1, verbose=0, callbacks=[StreamlitCallback()])
                
                progress_bar.progress(1.0)
                status_text.text("Entrenamiento completado!")

                st.header('4. Evaluación del Modelo')
                
                loss, accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
                st.metric("Precisión (Accuracy) en el conjunto de prueba", f"{accuracy:.4f}")

                y_pred_proba = model.predict(X_test_scaled).flatten()
                y_pred_class = (y_pred_proba > 0.5).astype(int)

                st.subheader('Matriz de Confusión')
                cm = confusion_matrix(y_test, y_pred_class)
                fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                            xticklabels=['Predicción Negativa (0)', 'Predicción Positiva (1)'],
                            yticklabels=['Real Negativa (0)', 'Real Positiva (1)'])
                ax_cm.set_xlabel('Predicción')
                ax_cm.set_ylabel('Valor Real')
                ax_cm.set_title('Matriz de Confusión')
                st.pyplot(fig_cm)

                st.subheader('Curva ROC y AUC')
                fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
                ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
                ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Clasificador aleatorio')
                ax_roc.set_xlim([0.0, 1.0])
                ax_roc.set_ylim([0.0, 1.05])
                ax_roc.set_xlabel('Tasa de Falsos Positivos (FPR)')
                ax_roc.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
                ax_roc.set_title('Curva Característica Operativa del Receptor (ROC)')
                ax_roc.legend(loc="lower right")
                ax_roc.grid(True)
                st.pyplot(fig_roc)

                st.subheader('Historial de Entrenamiento')
                fig_history, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

                ax1.plot(history.history['accuracy'], label='Precisión de Entrenamiento')
                ax1.plot(history.history['val_accuracy'], label='Precisión de Validación')
                ax1.set_title('Precisión del Modelo durante el Entrenamiento')
                ax1.set_xlabel('Época')
                ax1.set_ylabel('Precisión')
                ax1.legend()
                ax1.grid(True)

                ax2.plot(history.history['loss'], label='Pérdida de Entrenamiento')
                ax2.plot(history.history['val_loss'], label='Pérdida de Validación')
                ax2.set_title('Pérdida del Modelo durante el Entrenamiento')
                ax2.set_xlabel('Época')
                ax2.set_ylabel('Pérdida')
                ax2.legend()
                ax2.grid(True)

                plt.tight_layout()
                st.pyplot(fig_history)
            
            else:
                st.error("No se encontró la columna 'sentiment' en el dataset. Asegúrate de que el archivo es el correcto.")

    except Exception as e:
        st.error(f"Ocurrió un error al cargar o procesar el archivo CSV: {e}")
        st.warning("""
        Asegúrate de que:
        1. Tu archivo es el `IMDB Dataset.csv` de Kaggle.
        2. La codificación (`utf-8` o `latin1` si hay problemas) y los parámetros de `pd.read_csv` sean correctos.
        """)

else:
    st.info("Por favor, sube tu archivo `IMDB Dataset.csv` para comenzar.")