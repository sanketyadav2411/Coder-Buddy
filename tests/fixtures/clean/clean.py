import hashlib
import json
import os
import subprocess

API_KEY = os.environ.get("API_KEY", "")


def get_user(conn, uid):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = ?", (uid,))
    return cur.fetchall()


def run(args):
    return subprocess.run(args, check=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(blob):
    return json.loads(blob)
