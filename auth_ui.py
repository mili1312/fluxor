import re
import hashlib

from dash import dcc, html, Input, Output, State, no_update
from dash.exceptions import PreventUpdate

from db import create_user, verify_user, get_user_id


# -------------------------
# Validation helpers
# -------------------------

def hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def is_valid_email(email: str) -> bool:
    if not email:
        return False
    email = email.strip()
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email))


# -------------------------
# UI helpers
# -------------------------

def glass_card(children, width=440):
    """Reusable frosted glass card used in auth pages."""
    return html.Div(
        style={
            "width": f"{width}px",
            "maxWidth": "92vw",
            "background": "linear-gradient(180deg, rgba(14,24,48,0.86), rgba(8,12,24,0.86))",
            "border": "1px solid rgba(255,255,255,0.10)",
            "borderRadius": "18px",
            "padding": "18px",
            "boxSizing": "border-box",  # ✅ extra safety
            "boxShadow": "0 20px 60px rgba(0,0,0,0.40)",
            "backdropFilter": "blur(10px)",
            "animation": "popIn 420ms ease-out",
        },
        children=children,
    )


def input_box(_id, placeholder, _type="text"):
    """Reusable input style (fixes overflow with boxSizing)."""
    return dcc.Input(
        id=_id,
        type=_type,
        placeholder=placeholder,
        style={
            "width": "100%",
            "padding": "12px 12px",
            "borderRadius": "12px",
            "border": "1px solid rgba(255,255,255,0.14)",
            "backgroundColor": "rgba(7,11,20,0.90)",
            "color": "#E6E6E6",
            "outline": "none",
            "fontSize": "13px",
            "boxSizing": "border-box",  # ✅ FIX για να μην “βγαίνει έξω”
        },
    )


def primary_button(text, _id):
    return html.Button(
        text,
        id=_id,
        n_clicks=0,
        style={
            "width": "100%",
            "padding": "12px 12px",
            "borderRadius": "12px",
            "border": "1px solid rgba(39,76,255,0.35)",
            "background": "linear-gradient(180deg, rgba(39,76,255,0.85), rgba(39,76,255,0.55))",
            "color": "#E6E6E6",
            "cursor": "pointer",
            "fontWeight": 900,
            "fontSize": "13px",
            "boxSizing": "border-box",
        },
    )


def link_button(text, _id):
    return html.Button(
        text,
        id=_id,
        n_clicks=0,
        style={
            "border": "none",
            "background": "transparent",
            "color": "rgba(230,230,230,0.85)",
            "cursor": "pointer",
            "padding": 0,
            "fontSize": "12px",
            "textDecoration": "underline",
        },
    )


def alert_box(text, kind="warn"):
    bg = "rgba(255,185,70,0.14)"
    bd = "1px solid rgba(255,185,70,0.25)"
    if kind == "ok":
        bg = "rgba(35,205,130,0.14)"
        bd = "1px solid rgba(35,205,130,0.25)"
    if kind == "err":
        bg = "rgba(255,70,70,0.12)"
        bd = "1px solid rgba(255,70,70,0.25)"
    return html.Div(
        text,
        style={
            "marginTop": "10px",
            "padding": "10px 12px",
            "borderRadius": "12px",
            "background": bg,
            "border": bd,
            "fontSize": "12px",
            "opacity": 0.95,
            "whiteSpace": "pre-line",
            "boxSizing": "border-box",
        },
    )


def logo_block():
    return html.Div(
        style={"display": "flex", "justifyContent": "center", "marginBottom": "10px"},
        children=[
            html.Div(
                style={
                    "width": "132px",
                    "height": "132px",
                    "borderRadius": "999px",
                    "background": "radial-gradient(circle at 30% 20%, rgba(62,180,255,0.22), rgba(7,11,20,0.0))",
                    "border": "1px solid rgba(255,255,255,0.10)",
                    "boxShadow": "0 18px 55px rgba(0,0,0,0.45), 0 0 60px rgba(62,180,255,0.12)",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "backdropFilter": "blur(6px)",
                    "animation": "logoIn 520ms ease-out",
                    "boxSizing": "border-box",
                },
                children=[
                    html.Img(
                        src="/assets/fluxor.png",
                        style={
                            "width": "110px",
                            "height": "auto",
                            "display": "block",
                            "filter": "drop-shadow(0 10px 24px rgba(0,0,0,0.45))",
                            "opacity": 0.95,
                        },
                    )
                ],
            )
        ],
    )


# -------------------------
# Animated bubbles background (AUTH)
# -------------------------

def bubbles_background():
    """
    Animated bubbles layer behind the auth card.
    Uses fixed specs (no random) so it looks stable on reload.
    """
    specs = [
        # (size_px, left_percent, duration_sec, delay_sec, opacity)
        (90,  8,  28,  0,  0.22),
        (140, 18, 36,  6,  0.18),
        (110, 28, 30,  2,  0.20),
        (200, 42, 44,  9,  0.16),
        (120, 55, 33,  4,  0.20),
        (160, 66, 40,  12, 0.17),
        (100, 76, 29,  1,  0.22),
        (220, 88, 48,  14, 0.14),
        (130, 94, 34,  7,  0.18),
        (80,  36, 24,  3,  0.22),
        (150, 6,  38,  10, 0.17),
        (170, 60, 46,  16, 0.14),
    ]

    bubbles = []
    for size, left, dur, delay, op in specs:
        bubbles.append(
            html.Div(
                className="bubble",
                style={
                    "width": f"{size}px",
                    "height": f"{size}px",
                    "left": f"{left}%",
                    "animationDuration": f"{dur}s",
                    "animationDelay": f"{delay}s",
                    "opacity": op,
                },
            )
        )

    return html.Div(bubbles, className="bubbles-layer")



def background_slogan():
    return html.Div(
        "WHERE DATA MEETS DECISION",
        style={
            "position": "absolute",
            "top": "10%",                 # ⬆️ ΠΑΝΩ ΨΗΛΑ
            "left": "50%",
            "transform": "translateX(-50%)",
            "fontSize": "clamp(28px, 7vw, 90px)",
            "fontWeight": 900,
            "letterSpacing": "4px",
            "textTransform": "uppercase",
            "color": "rgba(255,255,255,0.06)",
            "whiteSpace": "nowrap",
            "zIndex": 1,
            "pointerEvents": "none",
            "userSelect": "none",
            "textAlign": "center",
        },
    )





# -------------------------
# Cards
# -------------------------

def login_card():
    return glass_card(
        [
            logo_block(),
            
            html.Div("Sign in to continue", style={"opacity": 0.75, "fontSize": "12px", "marginTop": "4px"}),

            html.Div(style={"height": "12px"}),

            html.Label("Email", style={"fontSize": "12px", "opacity": 0.85}),
            html.Div(style={"height": "6px"}),
            input_box("login_email", "email@example.com", "email"),

            html.Div(style={"height": "10px"}),

            html.Label("Password", style={"fontSize": "12px", "opacity": 0.85}),
            html.Div(style={"height": "6px"}),
            input_box("login_password", "••••••••", "password"),

            html.Div(style={"height": "12px"}),

            primary_button("Login", "btn_login"),

            html.Div(id="auth_msg", children=""),

            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginTop": "12px", "opacity": 0.9},
                children=[
                    html.Div("No account?", style={"fontSize": "12px", "opacity": 0.8}),
                    link_button("Create one →", "go_register"),
                ],
            ),
        ],
        width=440,
    )


def register_card():
    return glass_card(
        [
            logo_block(),
            html.Div("Create account", style={"fontSize": "18px", "fontWeight": 900}),
            html.Div("Register with email", style={"opacity": 0.75, "fontSize": "12px", "marginTop": "4px"}),

            html.Div(style={"height": "12px"}),

            html.Label("Email", style={"fontSize": "12px", "opacity": 0.85}),
            html.Div(style={"height": "6px"}),
            input_box("reg_email", "email@example.com", "email"),

            html.Div(style={"height": "10px"}),

            html.Label("Password", style={"fontSize": "12px", "opacity": 0.85}),
            html.Div(style={"height": "6px"}),
            input_box("reg_password", "••••••••", "password"),

            html.Div(style={"height": "10px"}),

            html.Label("Confirm password", style={"fontSize": "12px", "opacity": 0.85}),
            html.Div(style={"height": "6px"}),
            input_box("reg_password2", "••••••••", "password"),

            html.Div(style={"height": "12px"}),

            primary_button("Create account", "btn_register"),

            html.Div(id="reg_msg", children=""),

            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "marginTop": "12px", "opacity": 0.9},
                children=[
                    link_button("← Back to login", "go_login"),
                    html.Div("", style={"fontSize": "12px"}),
                ],
            ),
        ],
        width=440,
    )


def auth_shell(children):
    """Full-screen centered auth page shell + animated bubbles background."""
    return html.Div(
        style={
            "height": "100vh",
            "width": "100vw",
            "position": "relative",  # ✅ important for absolute background layer
            "background": "radial-gradient(1100px 650px at 20% 10%, rgba(39,76,255,0.18), rgba(7,11,20,0)) , #070B14",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "padding": "18px",
            "overflow": "hidden",
        },
        children=[
            bubbles_background(),
            background_slogan(),  # ✅ background layer
            html.Div(
                children,
                style={
                    "position": "relative",
                    "zIndex": 2,  # ✅ keep the card above bubbles
                    "width": "100%",
                    "display": "flex",
                    "justifyContent": "center",
                },
            ),
        ],
    )


# -------------------------
# Callbacks
# -------------------------

def register_auth_callbacks(app):
    """Register auth callbacks on the given Dash app."""

    @app.callback(
        Output("view", "data", allow_duplicate=True),
        Input("go_register", "n_clicks"),
        State("view", "data"),
        prevent_initial_call=True,
    )
    def nav_to_register(n, view):
        if not n:
            raise PreventUpdate
        return {"page": "register"}

    @app.callback(
        Output("view", "data", allow_duplicate=True),
        Input("go_login", "n_clicks"),
        State("view", "data"),
        prevent_initial_call=True,
    )
    def nav_to_login(n, view):
        if not n:
            raise PreventUpdate
        return {"page": "login"}

    @app.callback(
        Output("reg_msg", "children"),
        Output("view", "data", allow_duplicate=True),
        Input("btn_register", "n_clicks"),
        State("reg_email", "value"),
        State("reg_password", "value"),
        State("reg_password2", "value"),
        prevent_initial_call=True,
    )
    def do_register(_, email, pw1, pw2):
        email = (email or "").strip().lower()
        pw1 = pw1 or ""
        pw2 = pw2 or ""

        if not is_valid_email(email):
            return alert_box("⚠️ Βάλε σωστό email.", "warn"), no_update
        if len(pw1) < 6:
            return alert_box("⚠️ Password τουλάχιστον 6 χαρακτήρες.", "warn"), no_update
        if pw1 != pw2:
            return alert_box("⚠️ Τα passwords δεν ταιριάζουν.", "warn"), no_update

        ok = create_user(email, pw1)
        if not ok:
            return alert_box("⚠️ Υπάρχει ήδη λογαριασμός με αυτό το email.", "warn"), no_update

        return alert_box("✅ Έγινε εγγραφή! Κάνε login.", "ok"), {"page": "login"}

    @app.callback(
        Output("auth_msg", "children"),
        Output("session", "data"),
        Input("btn_login", "n_clicks"),
        State("login_email", "value"),
        State("login_password", "value"),
        prevent_initial_call=True,
    )
    def do_login(_, email, pw):
        email = (email or "").strip().lower()
        pw = pw or ""

        if not is_valid_email(email) or not pw:
            return alert_box("⚠️ Συμπλήρωσε email & password.", "warn"), no_update

        if not verify_user(email, pw):
            return alert_box("❌ Λάθος email ή password.", "err"), no_update

        user_id = get_user_id(email)
        if not user_id:
            return alert_box("❌ Δεν βρέθηκε χρήστης.", "err"), no_update

        return "", {"logged_in": True, "email": email, "user_id": user_id}

    @app.callback(
        Output("session", "data", allow_duplicate=True),
        Output("view", "data", allow_duplicate=True),
        Input("btn_logout", "n_clicks"),
        prevent_initial_call=True,
    )
    def do_logout(n):
        if not n:
            raise PreventUpdate
        return {"logged_in": False, "email": None, "user_id": None}, {"page": "login"}
