import os #λειτουργίες του λειτουργικού συστήματος
import base64 #κωδικοποίηση/αποκωδικοποίηση δεδομένων
import hashlib #κρυπτογραφικά hash functions
import psycopg2 #Driver για PostgreSQL database
import bcrypt #ασφαλές hashing κωδικών

#localhost

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))         
DB_NAME = os.getenv("DB_NAME", "crypto_app")
DB_USER = os.getenv("DB_USER", "crypto_user")
DB_PASS = os.getenv("DB_PASS", "strongpassword")

# κεντρικό μηχανισμό σύνδεσης με βάση δεδομένων PostgreSQL.

def get_conn():
    return psycopg2.connect(
        host=DB_HOST,                       
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

#φίλτρο ασφαλείας

def _prehash(password: str) -> bytes:
    if password is None:
        password = ""
    return hashlib.sha256(password.encode("utf-8")).digest()  # 32 bytes

#δημιουργία νέου χρήστη με ασφαλή αποθήκευση κωδικού στη βάση δεδομένων.

def create_user(email: str, password: str) -> bool:
    email = (email or "").strip().lower()

    pre = _prehash(password)
    pre_b64 = base64.b64encode(pre)  # bytes (~44)

    # bcrypt needs bytes; gensalt default cost is fine
    pw_hash = bcrypt.hashpw(pre_b64, bcrypt.gensalt()).decode("utf-8")

    q = """
    INSERT INTO app_user (email, password_hash)
    VALUES (%s, %s)
    ON CONFLICT (email) DO NOTHING
    RETURNING id;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (email, pw_hash))
        return cur.fetchone() is not None

#έλεγχο ταυτότητας χρήστη

def verify_user(email: str, password: str) -> bool:
    email = (email or "").strip().lower()

    q = "SELECT password_hash FROM app_user WHERE email=%s;"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (email,))
        row = cur.fetchone()
        if not row:
            return False

        stored = row[0].encode("utf-8")
        pre = _prehash(password)
        pre_b64 = base64.b64encode(pre)  # bytes (~44)

        return bcrypt.checkpw(pre_b64, stored)


#βρίσκει και επιστρέφει το μοναδικό αναγνωριστικό (id) ενός χρήστη με βάση το email του.

def get_user_id(email: str):
    email = (email or "").strip().lower()
    q = "SELECT id FROM app_user WHERE email=%s;"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (email,))
        row = cur.fetchone()
        return row[0] if row else None


#επιστρέφει τη λίστα με τα αγαπημένα (favorites) ενός χρήστη, με βάση το email του.

def list_favorites(email: str) -> list[str]:
    uid = get_user_id(email)
    if not uid:
        return []
    q = "SELECT pair FROM user_favorite WHERE user_id=%s ORDER BY created_at DESC;"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (uid,))
        return [r[0].upper() for r in cur.fetchall()]


#προσθέτει ένα νέο favorite

def add_favorite(email: str, pair: str):
    uid = get_user_id(email)
    if not uid:
        return
    pair = (pair or "").strip().upper()
    q = """
    INSERT INTO user_favorite (user_id, pair)
    VALUES (%s, %s)
    ON CONFLICT DO NOTHING;
    """
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (uid, pair))

#αφαιρεί ένα favorite

def remove_favorite(email: str, pair: str):
    uid = get_user_id(email)
    if not uid:
        return
    pair = (pair or "").strip().upper()
    q = "DELETE FROM user_favorite WHERE user_id=%s AND pair=%s;"
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(q, (uid, pair))
