"""
WSGI entry point for production deployment.

Gunicorn / most PaaS platforms look for a module-level `app` (or
`application`) callable. This also makes sure the database tables exist
before the first request, without relying on `if __name__ == "__main__"`
(which doesn't run under gunicorn).

Run locally with:
    gunicorn wsgi:app --bind 0.0.0.0:8000 --workers 2 --timeout 120
"""

from app import app, create_db

create_db()

if __name__ == "__main__":
    app.run()
