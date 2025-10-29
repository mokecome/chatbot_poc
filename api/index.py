from app import app as application

# Vercel expects an object named `app` in Python functions that expose a WSGI
# application. We alias the existing Flask instance so the existing routes (for
# example `/api/chat`) continue to work without duplicating logic.
app = application
