"""Create a throwaway demo login for LOCAL testing only.
    AURELIUS_DEV=1 AURELIUS_DATA=synthetic python seed_demo.py
    AURELIUS_DEV=1 AURELIUS_DATA=synthetic python server.py serve

Then sign in at http://127.0.0.1:8000 with:
    email:    demo@aurelius.test
    password: aurelius-demo-2026

This password is public (it is in this file). It refuses to run unless
AURELIUS_DEV=1, and you must never create it on a server anyone else can reach.
"""
import os, sys

if os.environ.get("AURELIUS_DEV") != "1":
    sys.exit("Refusing: set AURELIUS_DEV=1. A published password must never exist on a real deployment.")

import server

EMAIL, PASSWORD = "demo@aurelius.test", "aurelius-demo-2026"
server.init_db()
try:
    server.create_user(EMAIL, PASSWORD, name="Demo", role="admin")
    print(f"Created {EMAIL} / {PASSWORD}")
except Exception as e:
    print(f"Not created ({e}). It may already exist; use 'python server.py set-password {EMAIL}' to reset it.")
