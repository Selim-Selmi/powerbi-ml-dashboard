from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin
from werkzeug.security import check_password_hash, generate_password_hash
from pathlib import Path
import json

auth_bp = Blueprint("auth", __name__)

# -------- Users store (can be JSON file, DB, etc.) --------
DATA_PATH = Path(__file__).resolve().parent / "users.json"

# Create sample file if not exists
if not DATA_PATH.exists():
    users = {
        "admin@example.com": generate_password_hash("admin123"),
        "amine@example.com": generate_password_hash("amine123"),
    }
    json.dump(users, open(DATA_PATH, "w"))

# Load users
USERS = json.load(open(DATA_PATH))

# ✅ Add these lines here (AFTER USERS is defined)
print("🔍 Loading users from:", DATA_PATH)
print("Users loaded:", list(USERS.keys()))


# -------- Flask-Login User model --------
class User(UserMixin):
    def __init__(self, id):
        self.id = id


# -------- Routes --------
@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if email in USERS and check_password_hash(USERS[email], password):
            user = User(email)
            login_user(user)
            flash("Welcome back!", "success")
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials.", "danger")

    return render_template("login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "info")
    return redirect(url_for("auth.login"))
