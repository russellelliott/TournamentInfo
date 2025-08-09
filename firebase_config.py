import os
from dotenv import load_dotenv

def create_firebase_config():
    """Creates a Python object representing the Firebase configuration."""
    load_dotenv()

    try:
        config = {
            "type": os.getenv("TYPE"),
            "project_id": os.getenv("PROJECT_ID"),
            "private_key_id": os.getenv("PRIVATE_KEY_ID"),
            "private_key": os.getenv("PRIVATE_KEY"),
            "client_email": os.getenv("CLIENT_EMAIL"),
            "client_id": os.getenv("CLIENT_ID"),
            "auth_uri": os.getenv("AUTH_URI"),
            "token_uri": os.getenv("TOKEN_URI"),
            "auth_provider_x509_cert_url": os.getenv("AUTH_PROVIDER_X509_CERT_URL"),
            "client_x509_cert_url": os.getenv("CLIENT_X509_CERT_URL"),
            "universe_domain": os.getenv("UNIVERSE_DOMAIN")
        }
        return config
    except Exception as e:
        print(f"Error loading Firebase configuration from .env file: {e}")
        return None

# Make it directly importable
firebase_config = create_firebase_config()
