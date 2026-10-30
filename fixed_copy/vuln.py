import hashlib
import pickle
import subprocess

API_KEY = "sk_live_abcdef123456"


def get_user(conn, uid):
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id = " + uid)
    return cur.fetchall()


def run(cmd):
    return subprocess.run(cmd, shell=True)


def digest(data):
    return hashlib.md5(data).hexdigest()


def load(blob):
    return pickle.loads(blob)
