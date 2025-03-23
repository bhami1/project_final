# app.py
import streamlit as st
import sqlite3
import hashlib
import requests
import json
import pandas as pd
from datetime import datetime, timedelta
import base64
from io import BytesIO
from PIL import Image
import os
import random
import groq  # For using Groq API'


# Set page configuration
st.set_page_config(
    page_title="AInstein - Space Knowledge Explorer",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Database setup
def setup_database():
    conn = sqlite3.connect('ainstein_users.db', check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            email TEXT,
            date_joined TEXT,
            preferences TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            timestamp TEXT,
            user_message TEXT,
            bot_message TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            query TEXT,
            timestamp TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    return conn, cursor

# Password hashing with salt
def hash_password(password):
    salt = "AInstein2025"  # In production, use a secure random salt per user
    return hashlib.sha256((password + salt).encode()).hexdigest()

# Space Data Functions
def get_space_picture_of_day():
    """Get a space picture from free sources instead of NASA API"""
    # Local database of space images and descriptions
    space_images = [
        {
            "title": "The Andromeda Galaxy",
            "explanation": "The Andromeda Galaxy (M31) is the closest spiral galaxy to our Milky Way, located about 2.5 million light-years away. It's visible to the naked eye from dark sky locations and contains approximately one trillion stars.",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Andromeda_Galaxy_560mm_FL.jpg/2560px-Andromeda_Galaxy_560mm_FL.jpg"
        },
        {
            "title": "The Pillars of Creation",
            "explanation": "The Pillars of Creation are elephant trunks of interstellar gas and dust in the Eagle Nebula, about 6,500-7,000 light years from Earth. These columns of cosmic dust and hydrogen gas are birthplaces of new stars.",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg/1280px-Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg"
        },
        {
            "title": "The Crab Nebula",
            "explanation": "The Crab Nebula (M1) is a supernova remnant in the constellation of Taurus. The nebula was created by a supernova explosion observed by Chinese astronomers in 1054 AD, and is now about 11 light-years across.",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Crab_Nebula.jpg/2560px-Crab_Nebula.jpg"
        },
        {
            "title": "Saturn and Its Rings",
            "explanation": "Saturn is the sixth planet from the Sun and the second-largest in the Solar System, known for its prominent ring system. The rings are made up of billions of particles of ice and rock, ranging in size from micrometers to meters.",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Saturn_during_Equinox.jpg/2560px-Saturn_during_Equinox.jpg"
        },
        {
            "title": "The Orion Nebula", 
            "explanation": "The Orion Nebula (M42) is a diffuse nebula situated in the Milky Way, south of Orion's Belt. It's one of the brightest nebulae and is visible to the naked eye. The nebula is an active star-forming region, with many young stars hidden within.",
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Orion_Nebula_-_Hubble_2006_mosaic_18000.jpg/2560px-Orion_Nebula_-_Hubble_2006_mosaic_18000.jpg"
        }
    ]
    return random.choice(space_images)

# Planetary data function
def get_planetary_data():
    """Get comprehensive data about planets in our solar system"""
    planets_data = {
        "Mercury": {
            "radius": 2439.7,  # km
            "moons": [],
            "distance_from_sun": 57.9,  # million km
            "orbital_period": 88,  # days
            "rotation_period": 58.6,  # days
            "gravity": 3.7,  # m/s²
            "temperature": {"min": -173, "max": 427},  # °C
            "atmosphere": "Minimal - sodium, potassium, oxygen, hydrogen, helium"
        },
        "Venus": {
            "radius": 6051.8,  # km
            "moons": [],
            "distance_from_sun": 108.2,  # million km
            "orbital_period": 225,  # days
            "rotation_period": 243,  # days (retrograde)
            "gravity": 8.87,  # m/s²
            "temperature": {"min": 462, "max": 462},  # °C (uniform)
            "atmosphere": "96.5% carbon dioxide, 3.5% nitrogen, sulfur dioxide"
        },
        "Earth": {
            "radius": 6371.0,  # km
            "moons": ["Moon (Luna)"],
            "distance_from_sun": 149.6,  # million km
            "orbital_period": 365.25,  # days
            "rotation_period": 1,  # day
            "gravity": 9.8,  # m/s²
            "temperature": {"min": -88, "max": 58},  # °C
            "atmosphere": "78% nitrogen, 21% oxygen, 1% argon, carbon dioxide, water vapor"
        },
        "Mars": {
            "radius": 3389.5,  # km
            "moons": ["Phobos", "Deimos"],
            "distance_from_sun": 227.9,  # million km
            "orbital_period": 687,  # days
            "rotation_period": 1.03,  # days
            "gravity": 3.71,  # m/s²
            "temperature": {"min": -140, "max": 20},  # °C
            "atmosphere": "95% carbon dioxide, 2.7% nitrogen, 1.6% argon"
        },
        "Jupiter": {
            "radius": 69911,  # km
            "moons": ["Io", "Europa", "Ganymede", "Callisto", "Amalthea", "Himalia", "Elara", "Pasiphae", "Sinope", "Lysithea", "Carme", "Ananke", "Leda", "Thebe", "Adrastea", "Metis", "Callirrhoe", "Themisto", "Megaclite", "Taygete", "Chaldene", "Harpalyke", "Kalyke", "Iocaste", "Erinome", "Isonoe", "Praxidike", "Autonoe", "Thyone", "Hermippe", "Aitne", "Eurydome", "Euanthe", "Euporie", "Orthosie", "Sponde", "Kale", "Pasithee", "Hegemone", "Mneme", "Aoede", "Thelxinoe", "Arche", "Kallichore", "Helike", "Carpo", "Eukelade", "Cyllene", "Kore", "Herse"],
            "distance_from_sun": 778.5,  # million km
            "orbital_period": 4333,  # days
            "rotation_period": 0.41,  # days
            "gravity": 24.79,  # m/s²
            "temperature": {"min": -145, "max": -145},  # °C (avg. cloud temp)
            "atmosphere": "90% hydrogen, 10% helium, methane, ammonia, water vapor"
        },
        "Saturn": {
            "radius": 58232,  # km
            "moons": ["Titan", "Rhea", "Iapetus", "Dione", "Tethys", "Enceladus", "Mimas", "Hyperion", "Phoebe", "Janus", "Epimetheus", "Prometheus", "Pandora", "Helene", "Atlas", "Pan", "Ymir", "Paaliaq", "Tarvos", "Ijiraq", "Suttungr", "Kiviuq", "Mundilfari", "Albiorix", "Skathi", "Erriapus", "Siarnaq", "Thrymr", "Narvi", "Methone", "Pallene", "Polydeuces", "Daphnis", "Aegir", "Bebhionn", "Bergelmir"],
            "distance_from_sun": 1434.0,  # million km
            "orbital_period": 10759,  # days
            "rotation_period": 0.44,  # days
            "gravity": 10.44,  # m/s²
            "temperature": {"min": -178, "max": -178},  # °C (avg. cloud temp)
            "atmosphere": "96% hydrogen, 3% helium, methane, ammonia, water vapor"
        },
        "Uranus": {
            "radius": 25362,  # km
            "moons": ["Titania", "Oberon", "Miranda", "Ariel", "Umbriel", "Puck", "Sycorax", "Portia", "Juliet", "Belinda", "Cressida", "Desdemona", "Rosalind", "Bianca", "Ophelia", "Cordelia", "Caliban", "Stephano", "Trinculo", "Setebos", "Francisco", "Ferdinand", "Prospero", "Margaret", "Perdita", "Mab", "Cupid"],
            "distance_from_sun": 2871.0,  # million km
            "orbital_period": 30687,  # days
            "rotation_period": 0.72,  # days (retrograde)
            "gravity": 8.87,  # m/s²
            "temperature": {"min": -224, "max": -224},  # °C (avg. cloud temp)
            "atmosphere": "83% hydrogen, 15% helium, 2% methane"
        },
        "Neptune": {
            "radius": 24622,  # km
            "moons": ["Triton", "Proteus", "Nereid", "Larissa", "Galatea", "Despina", "Thalassa", "Naiad", "Halimede", "Neso", "Sao", "Laomedeia", "Psamathe", "S/2004 N 1"],
            "distance_from_sun": 4495.0,  # million km
            "orbital_period": 60190,  # days
            "rotation_period": 0.67,  # days
            "gravity": 11.15,  # m/s²
            "temperature": {"min": -218, "max": -218},  # °C (avg. cloud temp)
            "atmosphere": "80% hydrogen, 19% helium, 1.5% methane"
        }
    }
    return planets_data

# Satellite launch data function
def get_satellite_data():
    """Get comprehensive data about satellites"""
    satellites = [
        {"name": "Sputnik 1", "launch_date": "1957-10-04", "country": "USSR", "purpose": "First artificial satellite", "status": "Decayed"},
        {"name": "Explorer 1", "launch_date": "1958-01-31", "country": "USA", "purpose": "First US satellite", "status": "Decayed"},
        {"name": "Vanguard 1", "launch_date": "1958-03-17", "country": "USA", "purpose": "Weather research", "status": "In orbit"},
        {"name": "TIROS-1", "launch_date": "1960-04-01", "country": "USA", "purpose": "First weather satellite", "status": "Decayed"},
        {"name": "Telstar 1", "launch_date": "1962-07-10", "country": "USA", "purpose": "Communications", "status": "In orbit"},
        {"name": "Syncom 3", "launch_date": "1964-08-19", "country": "USA", "purpose": "First geostationary satellite", "status": "In orbit"},
        {"name": "Intelsat I", "launch_date": "1965-04-06", "country": "USA", "purpose": "Commercial communications", "status": "In orbit"},
        {"name": "ATS-6", "launch_date": "1974-05-30", "country": "USA", "purpose": "Communications", "status": "In orbit"},
        {"name": "GPS I-1", "launch_date": "1978-02-22", "country": "USA", "purpose": "Navigation", "status": "Decommissioned"},
        {"name": "GOES-1", "launch_date": "1975-10-16", "country": "USA", "purpose": "Weather monitoring", "status": "Decommissioned"},
        {"name": "Hubble Space Telescope", "launch_date": "1990-04-24", "country": "USA", "purpose": "Space telescope", "status": "Operational"},
        {"name": "International Space Station", "launch_date": "1998-11-20", "country": "International", "purpose": "Space station", "status": "Operational"},
        {"name": "Terra", "launch_date": "1999-12-18", "country": "USA", "purpose": "Earth observation", "status": "Operational"},
        {"name": "Envisat", "launch_date": "2002-03-01", "country": "ESA", "purpose": "Earth observation", "status": "Non-operational"},
        {"name": "WMAP", "launch_date": "2001-06-30", "country": "USA", "purpose": "Cosmic microwave background", "status": "Mission complete"},
        {"name": "Kepler", "launch_date": "2009-03-07", "country": "USA", "purpose": "Exoplanet search", "status": "Mission complete"},
        {"name": "Gaia", "launch_date": "2013-12-19", "country": "ESA", "purpose": "Astrometry", "status": "Operational"},
        {"name": "TESS", "launch_date": "2018-04-18", "country": "USA", "purpose": "Exoplanet search", "status": "Operational"},
        {"name": "James Webb Space Telescope", "launch_date": "2021-12-25", "country": "International", "purpose": "Space telescope", "status": "Operational"},
        {"name": "Chandrayaan-3", "launch_date": "2023-07-14", "country": "India", "purpose": "Lunar exploration", "status": "Operational"}
    ]
    return satellites

# Groq API Integration
def get_groq_response(query, context=None, model="llama3-70b-8192"):
    """Get response from Groq API"""
    try:
        # Make sure the API key is set
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return "Error: GROQ_API_KEY environment variable is not set. Please set it before using the chat."
        
        # Initialize the Groq client
        client = groq.Client(api_key=api_key)
        
        # Prepare the prompt with space context
        system_message = """You are AInstein, a helpful AI assistant specializing in space knowledge. 
        Focus on planetary data including radii, moons, satellites, and other astronomical facts. 
        Provide information about planets, their properties, moons, and human-made satellites.
        Keep responses concise, informative, and engaging."""
        
        if context:
            prompt = f"Context information: {context}\n\nUser question: {query}"
        else:
            prompt = query
        
        # Call the Groq API
        chat_completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        
        # Extract the response
        if hasattr(chat_completion, 'choices') and len(chat_completion.choices) > 0:
            if hasattr(chat_completion.choices[0], 'message') and hasattr(chat_completion.choices[0].message, 'content'):
                response = chat_completion.choices[0].message.content
                return response
        return "I couldn't generate a response. Please try again."
    
    except Exception as e:
        return f"Error connecting to Groq API: {str(e)}"

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_file):
    import PyPDF2
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to create finance vector embeddings
def create_finance_vector_embeddings(doc_file):
    import pandas as pd
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    
    # Initialize embeddings
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    try:
        if doc_file.type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
            st.session_state.finance_df = pd.read_excel(doc_file)
            st.session_state.finance_docs = [Document(page_content=str(row)) for row in st.session_state.finance_df.values.tolist()]
        elif doc_file.type == "application/pdf":
            pdf_text = extract_text_from_pdf(doc_file)
            st.session_state.finance_docs = [Document(page_content=pdf_text)]
        else:
            st.error("Unsupported file format.")
            return
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(st.session_state.finance_docs)
    
    # Create vector store
    st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)

# Process user input function
# Modify this function
def process_user_input():
    if st.session_state.get('user_input') and not st.session_state.get('processing', False):
        user_message = st.session_state["user_input"]
        
        # Set processing flag to prevent multiple responses
        st.session_state.processing = True
        
        # Add user message to chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        try:
            # Get response from AI based on current settings
            ai_response = get_groq_response(
                user_message, 
                model=st.session_state.get('groq_model', "llama3-70b-8192")
            )
            
            # Add AI response to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
            
            # Save chat to database
            if st.session_state.get('user_id'):
                conn, cursor = setup_database()
                try:
                    cursor.execute(
                        "INSERT INTO chat_history (user_id, timestamp, user_message, bot_message) VALUES (?, ?, ?, ?)",
                        (st.session_state.user_id, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), user_message, ai_response)
                    )
                    
                    # Save the query for future reference
                    cursor.execute(
                        "INSERT INTO saved_queries (user_id, query, timestamp) VALUES (?, ?, ?)",
                        (st.session_state.user_id, user_message, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    )
                    
                    conn.commit()
                except Exception as e:
                    st.error(f"Database error: {str(e)}")
                finally:
                    conn.close()
        except Exception as e:
            st.error(f"Error processing message: {str(e)}")
            # Add error message to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}"})
        
        # Reset processing flag
        st.session_state.processing = False
        
        # Rerun to update the UI
        st.rerun()

# Function to save user preferences
def save_user_preferences(user_id, preferences):
    conn, cursor = setup_database()
    try:
        preferences_json = json.dumps(preferences)
        cursor.execute("UPDATE users SET preferences = ? WHERE id = ?", (preferences_json, user_id))
        conn.commit()
    except Exception as e:
        print(f"Error saving preferences: {str(e)}")
    finally:
        conn.close()


# Function to load user preferences
def load_user_preferences(user_id):
    conn, cursor = setup_database()
    try:
        cursor.execute("SELECT preferences FROM users WHERE id = ?", (user_id,))
        result = cursor.fetchone()
        if result and result[0]:
            return json.loads(result[0])
        return {}
    except Exception as e:
        print(f"Error loading preferences: {str(e)}")
        return {}
    finally:
        conn.close()

# Main app function
def main():
    # Import required libraries
    import streamlit as st
    import os
    import pandas as pd
    import PyPDF2
    from langchain_groq import ChatGroq
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain.schema import Document
    from dotenv import load_dotenv
    import time
    import json
    import sqlite3
    from datetime import datetime
    
    # Load environment variables
    load_dotenv()
    
    groq_api_key = os.getenv('GROQ_API_KEY')
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    
    # Setup database connection
    conn, cursor = setup_database()
    
    # Initialize session state variables
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'llm_type' not in st.session_state:
        st.session_state.llm_type = 'groq'  # Default to Groq
    if 'groq_model' not in st.session_state:
        st.session_state.groq_model = 'llama3-70b-8192'  # Default model
    if 'processing' not in st.session_state:
        st.session_state.processing = False  # Flag to prevent multiple responses
    
    # App title
    st.title("🔭 AInstein - Space Knowledge Explorer")
    
    # Login/Registration System
    if not st.session_state.logged_in:
        auth_tab1, auth_tab2 = st.tabs(["Login", "Register"])
        
        with auth_tab1:
            login_username = st.text_input("Username", key="login_username")
            login_password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login"):
                if login_username and login_password:
                    hashed_password = hash_password(login_password)
                    cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", 
                                  (login_username, hashed_password))
                    user = cursor.fetchone()
                    
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.username = login_username
                        st.session_state.user_id = user[0]
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.warning("Please enter both username and password")
        
        with auth_tab2:
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            reg_email = st.text_input("Email (optional)", key="reg_email")
            
            if st.button("Register"):
                if reg_username and reg_password:
                    if reg_password == reg_confirm:
                        hashed_password = hash_password(reg_password)
                        try:
                            cursor.execute("INSERT INTO users (username, password, email, date_joined) VALUES (?, ?, ?, ?)",
                                          (reg_username, hashed_password, reg_email, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                            conn.commit()
                            st.success("Registration successful! Please log in.")
                        except sqlite3.IntegrityError:
                            st.error("Username already exists")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.warning("Please enter both username and password")
    
    # Main Interface (after login)
    else:
        st.sidebar.title(f"Welcome, {st.session_state.username}!")
        
        # LLM Selection
        st.sidebar.header("AI Settings")
        st.sidebar.selectbox(
            "Select LLM Backend",
            options=["groq"],
            index=0,
            key="llm_type"
        )
        
        # Groq model selection
        groq_models = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"]
        st.sidebar.selectbox(
            "Select Groq Model",
            options=groq_models,
            index=groq_models.index(st.session_state.groq_model),
            key="groq_model_select",
            on_change=lambda: setattr(st.session_state, 'groq_model', st.session_state.groq_model_select)
        )
        
        # Check API Key
        if st.sidebar.checkbox("Check API Key"):
            api_key = os.environ.get("GROQ_API_KEY")
            if api_key:
                st.sidebar.success("GROQ API Key is set ✓")
            else:
                st.sidebar.error("GROQ API Key is not set! Please set it in your environment variables.")
                st.sidebar.info("You can add this to your script or .env file: os.environ['GROQ_API_KEY'] = 'your-key-here'")
        
        # Main content area with tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Space Chat", "Astronomy Picture", "Planetary Data", "Satellites", "Finance Advisor"])
        
        # Tab 1: Space Chat
        with tab1:
            st.header("Chat with AInstein about Space 🚀")
            st.write("Ask questions about planets, moons, satellites, and other space data!")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                if st.session_state.chat_history:
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            st.write(f"👤 **You:** {message['content']}")
                        else:
                            st.write(f"🤖 **AInstein:** {message['content']}")
                else:
                    st.info("Start chatting about space! Ask about planets, their properties, moons, or satellites.")
            
            # Chat input
            user_input = st.text_input(
                "Ask a question about space...",
                key="user_input"
            )
            
            # Submit button for chat
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("Send"):
                    if user_input:
                        process_user_input()
                    else:
                        st.warning("Please enter a question first")
            
            with col2:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Tab 2: Astronomy Picture of the Day
        with tab2:
            st.header("Astronomy Picture of the Day")
            
            # Get picture data
            picture_data = get_space_picture_of_day()
            
            # Display picture and info
            st.subheader(picture_data["title"])
            st.image(picture_data["url"], use_container_width=True)
            st.write(picture_data["explanation"])
            
            # Option to explain picture with AI
            if st.button("Ask AI about this image") and not st.session_state.processing:
                st.session_state.processing = True
                    
                    # Generate prompt about the current image
                prompt = f"Please provide more detailed scientific information about {picture_data['title']}. Include interesting facts that might not be well known."
                    
                try:
                        # Get AI explanation
                        ai_explanation = get_groq_response(prompt, model=st.session_state.get('groq_model', "llama3-70b-8192"))
                        st.markdown("### AI Explanation")
                        st.write(ai_explanation)
                except Exception as e:
                        st.error(f"Error getting AI explanation: {str(e)}")
                finally:
                        st.session_state.processing = False
        
        # Tab 3: Planetary Data
        with tab3:
            st.header("Planetary Data Explorer")
            
            # Get planetary data
            planets_data = get_planetary_data()
            
            # Display planet selection
            selected_planet = st.selectbox(
                "Select a planet to view detailed information:",
                options=list(planets_data.keys())
            )
            
            # Display selected planet data
            if selected_planet:
                planet_info = planets_data[selected_planet]
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader(f"{selected_planet} - Basic Information")
                    st.write(f"**Radius:** {planet_info['radius']} km")
                    st.write(f"**Distance from Sun:** {planet_info['distance_from_sun']} million km")
                    st.write(f"**Orbital Period:** {planet_info['orbital_period']} days")
                    st.write(f"**Rotation Period:** {planet_info['rotation_period']} days")
                    st.write(f"**Gravity:** {planet_info['gravity']} m/s²")
                    
                with col2:
                    st.subheader("Environmental Conditions")
                    st.write(f"**Temperature Range:** {planet_info['temperature']['min']}°C to {planet_info['temperature']['max']}°C")
                    st.write(f"**Atmosphere:** {planet_info['atmosphere']}")
                    
                    # Display moons
                    st.subheader("Moons")
                    if planet_info["moons"]:
                        st.write(f"{selected_planet} has {len(planet_info['moons'])} moons")
                        if len(planet_info['moons']) <= 10:
                            for moon in planet_info["moons"]:
                                st.write(f"- {moon}")
                        else:
                            st.write(f"Major moons: {', '.join(planet_info['moons'][:10])}...")
                            if st.button(f"Show all {len(planet_info['moons'])} moons"):
                                st.write(", ".join(planet_info["moons"]))
                    else:
                        st.write(f"{selected_planet} has no moons")
                
                # Visualization option
                if st.button("Visualize Planet Data"):
                    import matplotlib.pyplot as plt
                    import numpy as np
                    
                    # Create figure with two subplots
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # Plot 1: Compare planet size with Earth
                    planets_to_compare = ["Earth", selected_planet]
                    radii = [planets_data["Earth"]["radius"], planet_info["radius"]]
                    
                    # Create bar chart for size comparison
                    ax1.bar(planets_to_compare, radii, color=['blue', 'red'])
                    ax1.set_ylabel('Radius (km)')
                    ax1.set_title('Size Comparison with Earth')
                    
                    # Plot 2: Temperature range
                    labels = ['Min', 'Max']
                    temps = [planet_info['temperature']['min'], planet_info['temperature']['max']]
                    
                    # Create bar chart for temperature
                    ax2.bar(labels, temps, color=['lightblue', 'orange'])
                    ax2.set_ylabel('Temperature (°C)')
                    ax2.set_title(f'{selected_planet} Temperature Range')
                    
                    # Adjust layout and display
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Ask AI about the planet
                if st.button("Ask AI about this planet"):
                    prompt = f"Tell me some fascinating facts about {selected_planet} that most people don't know."
                    try:
                        ai_response = get_groq_response(prompt, model=st.session_state.get('groq_model', "llama3-70b-8192"))
                        st.markdown("### AI Insights")
                        st.write(ai_response)
                    except Exception as e:
                        st.error(f"Error getting AI response: {str(e)}")
        
        # Tab 4: Satellites
        with tab4:
            st.header("Satellite Database")
            
            # Get satellite data
            satellite_data = get_satellite_data()
            
            # Convert to DataFrame for easier filtering and display
            satellites_df = pd.DataFrame(satellite_data)
            
            # Filters
            st.subheader("Filter Satellites")
            
            # Filter by country
            countries = ["All"] + sorted(satellites_df["country"].unique().tolist())
            selected_country = st.selectbox("Select Country", countries)
            
            # Filter by status
            statuses = ["All"] + sorted(satellites_df["status"].unique().tolist())
            selected_status = st.selectbox("Select Status", statuses)
            
            # Filter by launch date
            min_date = pd.to_datetime(satellites_df["launch_date"]).min().date()
            max_date = pd.to_datetime(satellites_df["launch_date"]).max().date()
            
            selected_date_range = st.slider(
                "Launch Date Range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date)
            )
            
            # Apply filters
            filtered_df = satellites_df.copy()
            
            # Apply country filter
            if selected_country != "All":
                filtered_df = filtered_df[filtered_df["country"] == selected_country]
            
            # Apply status filter
            if selected_status != "All":
                filtered_df = filtered_df[filtered_df["status"] == selected_status]
            
            # Apply date filter
            filtered_df["launch_date"] = pd.to_datetime(filtered_df["launch_date"])
            filtered_df = filtered_df[
                (filtered_df["launch_date"].dt.date >= selected_date_range[0]) &
                (filtered_df["launch_date"].dt.date <= selected_date_range[1])
            ]
            
            # Display filtered satellites
            st.subheader(f"Satellite List ({len(filtered_df)} satellites)")
            st.dataframe(filtered_df)
            
            # Visualize satellites
            if st.button("Visualize Satellite Data"):
                import matplotlib.pyplot as plt
                import numpy as np
                
                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot 1: Satellites by country
                country_counts = satellites_df["country"].value_counts().sort_values(ascending=False).head(5)
                ax1.bar(country_counts.index, country_counts.values, color='skyblue')
                ax1.set_xlabel('Country')
                ax1.set_ylabel('Number of Satellites')
                ax1.set_title('Top 5 Countries by Satellite Count')
                plt.xticks(rotation=45)
                
                # Plot 2: Satellites by status
                status_counts = satellites_df["status"].value_counts()
                ax2.pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%')
                ax2.set_title('Satellites by Current Status')
                
                # Adjust layout and display
                plt.tight_layout()
                st.pyplot(fig)
        
        # Tab 5: Finance Advisor
        with tab5:
            st.header("Finance Advisor")
            
            # File uploader
            st.subheader("Upload Financial Documents")
            uploaded_file = st.file_uploader("Upload Excel, CSV, or PDF files", type=["xlsx", "csv", "pdf"])
            
            if uploaded_file is not None:
                st.success(f"Successfully uploaded {uploaded_file.name}")
                
                # Create vector embeddings
                if st.button("Process Document"):
                    with st.spinner("Processing document..."):
                        try:
                            create_finance_vector_embeddings(uploaded_file)
                            st.success("Document processed successfully!")
                        except Exception as e:
                            st.error(f"Error processing document: {str(e)}")
            
                # Finance query
                st.subheader("Ask about your financial data")
                finance_query = st.text_input("Enter your financial question")
                
                if finance_query and st.button("Submit Query"):
                    if 'vectors' in st.session_state:
                        try:
                            # Perform similarity search
                            relevant_docs = st.session_state.vectors.similarity_search(finance_query)
                            
                            # Prepare context for the AI
                            context = "\n".join([doc.page_content for doc in relevant_docs])
                            
                            # Get AI response with context
                            finance_response = get_groq_response(
                                finance_query,
                                context=context,
                                model=st.session_state.get('groq_model', "llama3-70b-8192")
                            )
                            
                            st.subheader("AI Response")
                            st.write(finance_response)
                            
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                    else:
                        st.warning("Please process a document first")
        
        # User account and preferences
        st.sidebar.header("Account Settings")
        
        # Load user preferences
        user_preferences = load_user_preferences(st.session_state.user_id)
        
        
        # Settings form
        with st.sidebar.form("preferences_form"):
            st.subheader("Preferences")
            
            # Theme preference
            theme = st.selectbox(
                "Theme",
                options=["Light", "Dark"],
                index=0 if user_preferences.get("theme", "Light") == "Light" else 1
            )
            
            # Notification preferences
            email_notifications = st.checkbox(
                "Email Notifications",
                value=user_preferences.get("email_notifications", False)
            )
            
            # Save button
            if st.form_submit_button("Save Preferences"):
                # Update preferences
                user_preferences = {
                    "theme": theme,
                    "email_notifications": email_notifications
                }
                
                # Save to database
                save_user_preferences(st.session_state.user_id, user_preferences)
                st.sidebar.success("Preferences saved!")
        
        # Logout button
        if st.sidebar.button("Logout"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Close database connection when app is done
    conn.close()

# Run the app
if __name__ == "__main__":
    main()