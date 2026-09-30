import streamlit as st
from groq import Groq
import tiktoken
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# CONFIGURACIÓN DE LA APLICACIÓN
# ============================================================
st.set_page_config(page_title="Laboratorio GenAI", page_icon="🤖", layout="wide")
st.title("🤖 Laboratorio de Generative AI")

# ============================================================
# MENÚ Y MANEJO DE API KEY
# ============================================================
st.sidebar.title("Configuración")

if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("✅ API Key de Groq configurada.")
else:
    api_key = st.sidebar.text_input("🔑 Ingresa tu API Key de Groq:", type="password")
    if not api_key:
        st.warning("⚠ Debes ingresar tu API Key de Groq para usar la generación de texto.")

st.sidebar.markdown("---")
st.sidebar.title("Menú")
opcion = st.sidebar.radio(
    "Selecciona una sección:",
    ["Generación de texto", "Tokenización", "Embeddings", "Similitud del coseno"]
)

# ============================================================
# FUNCIONES AUXILIARES (Caché)
# ============================================================
@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

COLORES = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E8BAFF", "#FFB3F7", "#DCD3FF"]

# ============================================================
# 1. GENERACIÓN DE TEXTO
# ============================================================
if opcion == "Generación de texto":
    st.header("1️⃣ Generación de texto")
    
    if api_key:
        client = Groq(api_key=api_key)
        modelo = st.selectbox("Selecciona el modelo de Groq:", ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"])
        prompt = st.text_area("Prompt:", placeholder="Ejemplo: Explica qué es la inteligencia artificial.")
        
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.1)
        with col2:
            max_tokens = st.slider("Máximo de tokens de salida", 50, 2048, 500, 50)

        if st.button("🚀 Generar respuesta"):
            if prompt.strip():
                with st.spinner("Generando..."):
                    try:
                        response = client.chat.completions.create(
                            model=modelo,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=temperature,
                            max_tokens=max_tokens
                        )
                        st.subheader("Respuesta:")
                        st.write(response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Error con la API: {e}")
            else:
                st.warning("Escribe un prompt.")

# ============================================================
# 2. TOKENIZACIÓN
# ============================================================
elif opcion == "Tokenización":
    st.header("2️⃣ Tokenización")
    
    modelo_token = st.selectbox("Selecciona el modelo de tokenización:", ["cl100k_base (GPT-4)", "p50k_base (GPT-3)"])
    encoding_name = modelo_token.split(" ")[0]
    texto = st.text_area("Escribe un texto:", "La inteligencia artificial está cambiando el mundo.")

    if st.button("🔤 Tokenizar texto"):
        if texto.strip():
            tokenizer = tiktoken.get_encoding(encoding_name)
            tokens_ids = tokenizer.encode(texto)
            
            st.subheader("Resultados:")
            st.write(f"**Cantidad de tokens:** {len(tokens_ids)}")
            
            html_content = "<div style='line-height: 2; font-size: 18px;'>"
            for i, token_id in enumerate(tokens_ids):
                token_str = tokenizer.decode([token_id]).replace('<', '&lt;').replace('>', '&gt;')
                color = COLORES[i % len(COLORES)]
                html_content += f"<span style='background-color: {color}; padding: 2px 6px; border-radius: 4px; margin-right: 4px; color: black;'>{token_str} <sub>[{token_id}]</sub></span>"
            html_content += "</div>"
            st.markdown(html_content, unsafe_allow_html=True)
        else:
            st.warning("Escribe algún texto.")

# ============================================================
# 3. EMBEDDINGS
# ============================================================
elif opcion == "Embeddings":
    st.header("3️⃣ Embeddings")
    
    modelo_emb = st.selectbox("Selecciona el modelo:", ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"])
    frase = st.text_input("Escribe una frase:", "Me gusta programar en Python.")

    if st.button("📊 Generar embedding"):
        if frase.strip():
            with st.spinner("Cargando modelo..."):
                model = load_embedding_model(modelo_emb)
                vector = model.encode(frase)
                st.success("¡Éxito!")
                st.write(f"**Dimensiones:** {len(vector)}")
                st.write("**Primeros valores:**", vector[:50])
        else:
            st.warning("Escribe una frase.")

# ============================================================
# 4. SIMILITUD DEL COSENO
# ============================================================
elif opcion == "Similitud del coseno":
    st.header("4️⃣ Similitud del coseno")
    
    modelo_emb = st.selectbox("Selecciona el modelo:", ["all-MiniLM-L6-v2", "paraphrase-multilingual-MiniLM-L12-v2"])
    frase1 = st.text_input("Frase 1:", "Me encanta jugar fútbol los fines de semana.")
    frase2 = st.text_input("Frase 2:", "Disfruto practicar balompié los sábados.")

    if st.button("📐 Calcular similitud"):
        if frase1.strip() and frase2.strip():
            with st.spinner("Calculando..."):
                model = load_embedding_model(modelo_emb)
                emb1 = model.encode([frase1])
                emb2 = model.encode([frase2])
                similitud = cosine_similarity(emb1, emb2)[0][0]
                
                st.subheader("Resultado")
                st.metric(label="Similitud", value=f"{similitud:.4f}")
        else:
            st.warning("Escribe ambas frases.")
