import os
import sys
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
print(f"DEBUG: os.environ.get('FRAMEWORK') after load_dotenv: {os.environ.get('FRAMEWORK')}")

# Add the current directory to the path to ensure imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Determine which framework to use based on environment variable
# Default to FastAPI if not specified
FRAMEWORK = os.getenv("FRAMEWORK", "fastapi").lower()
logger.info(f"Starting application with framework: {FRAMEWORK}")
print(f"DEBUG: FRAMEWORK environment variable is set to: {FRAMEWORK}")

port = int(os.getenv("PORT", 8000))
host = os.getenv("HOST", "0.0.0.0")

# Import the appropriate application based on the framework
if FRAMEWORK == "nicegui":
    try:
        # Import the enhanced NiceGUI implementation
        from app.frontend.nicegui_app import ui, app as nicegui_app
        logger.info("NiceGUI framework initialized successfully")
        application = nicegui_app
        
        # Configure NiceGUI app settings
        nicegui_app.title = "Project Base - Python-Native Web Application"
        nicegui_app.favicon = "🚀"
        # Ensure the server starts
        if __name__ in {"__main__", "__mp_main__"}:
            ui.run(host=host, port=port, title="Project Base")
    except ImportError as e:
        logger.error(f"Failed to initialize NiceGUI: {e}")
        print("NiceGUI not installed. Please install with: pip install nicegui")
        exit(1)
elif FRAMEWORK == "reactpy":
    try:
        # Import the main FastAPI app
        from app import app as fastapi_app
        
        # Import ReactPy components
        from app.frontend.reactpy_app import App
        
        # Import the correct configure function from reactpy.backend.fastapi
        from reactpy.backend.fastapi import configure
        
        # Configure ReactPy with the FastAPI app
        # This will set up all necessary routes and WebSocket handlers
        configure(fastapi_app, App)
        
        # Remove the default root endpoint to avoid conflicts with ReactPy
        # Find and remove the root endpoint handler if it exists
        for route in list(fastapi_app.routes):
            if route.path == "/" and route.methods == {"GET"}:
                fastapi_app.routes.remove(route)
                logger.info("Removed default root endpoint to avoid conflicts with ReactPy")
                break
        
        logger.info("ReactPy framework initialized and configured with FastAPI app successfully")
        application = fastapi_app
    except ImportError as e:
        logger.error(f"Failed to initialize ReactPy: {e}")
        print("ReactPy or its FastAPI backend not installed. Please install with: pip install reactpy reactpy-fastapi")
        exit(1)
elif FRAMEWORK == "generated":
    try:
        from app.generated.preview_app import application as generated_application
        logger.info("Generated framework initialized successfully")
        application = generated_application
    except ImportError as e:
        logger.error(f"Failed to initialize generated application from app.generated.preview_app: {e}")
        print("Generated application not found or contains errors. Ensure app/generated/preview_app.py exists and defines an 'application' instance.")
        exit(1)
else:
    # Default to FastAPI
    from app import app
    logger.info("FastAPI framework initialized successfully")
    application = app

# This is used by ASGI servers like Uvicorn
app = application

if __name__ in {"__main__", "__mp_main__"}:
    import uvicorn
    logger.info(f"Starting server on {host}:{port}")

    # Run the application with uvicorn
    if FRAMEWORK != "nicegui":
        # Start FastAPI with uvicorn
        uvicorn.run("main:app", host=host, port=port)