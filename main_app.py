import streamlit as st
import sqlite3
import bcrypt

#Database Initialization and Helper Functions
def init_db():
    """Initializes the database and creates the 'users' table if it doesn't exist."""
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    # Use IF NOT EXISTS to prevent errors on subsequent runs
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    """)
    conn.commit()
    conn.close()


def check_username(username):
    """Checks if a username already exists in the database to prevent duplicates."""
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    result = cur.fetchone()
    conn.close()
    return result is not None


def add_username(username, password):
    """Hashes a new user's password using bcrypt and stores the credentials."""
    if check_username(username):
        st.warning("This username already exists.")
    else:
        conn = sqlite3.connect('users.db')
        cur = conn.cursor()
        # Hash the password with a salt for secure storage
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed))
        conn.commit()
        conn.close()
        st.success("Sign-up successful!")


def verify_user(username, password):
    """Verifies user credentials by comparing the input password with the stored hash."""
    conn = sqlite3.connect('users.db')
    cur = conn.cursor()
    cur.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = cur.fetchone()
    conn.close()
    if result:
        stored_hash = result[0]
        # Use bcrypt's checkpw to securely compare the password and hash
        return bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8'))
    return False

#Application Setup
# Ensure the database and table are ready before the app runs
init_db()

# Initialize session state to manage user login status
if "control" not in st.session_state:
    st.session_state.control = False

#Main Application Logic
# Conditionally display the login screen or the main app based on login state
if not st.session_state.control:
    # --- Login/Signup UI ---
    st.markdown("<h1 style='color:#FF9B85;'>Welcome</h1>", unsafe_allow_html=True)
    st.header(":green[_Dataset explorer_]")
    st.markdown(":rainbow[Please log in or sign up]")

    # Get user input for login or signup
    username = st.text_input("Username", key="kullanıcı_adı")
    st.session_state.username = username
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    # Handle login logic
    with col1:
        if st.button("Log in", key="login_button"):
            if not username or not password:
                st.warning("Please fill in all fields")
            elif verify_user(username, password):
                st.success("Login successful!")
                st.session_state.control = True # Update login state
                st.rerun() # Rerun the script to show the main app
            else:
                st.error("Incorrect username or password.")

    # Handle signup logic
    with col2:
        if st.button("Sign up", key="signup_button"):
            if not username or not password:
                st.warning("Please fill in all fields to register.")
            else:
                add_username(username, password)
else:
    #Main Application Interface
    st.success("Login successful")
    st.header(f"Welcome, {st.session_state.username}")

    # Define the pages for multi-page navigation
    home = st.Page("home.py", title="Home")
    ml = st.Page("ml_methods.py", title="ML Methods (Explore the Sub-Pages")
    dl = st.Page("dl_methods.py", title="DL Methods (Design your own Neural Network)")

    # Configure and run the multi-page navigation
    pg = st.navigation(
        {"Home": [home],
         "ML": [ml],
         "DL": [dl]},
    )
    pg.run()

    # Add a logout button to the sidebar
    with st.sidebar:
        # Use on_click to reset the session state and log the user out
        st.button("Log out", on_click=lambda: st.session_state.update(control=False))