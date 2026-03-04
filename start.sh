#!/bin/bash
gunicorn bot:flask_app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
