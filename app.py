
import os
import secrets
import sqlite3
from datetime import datetime

import pandas as pd
from flask import (
    Flask, render_template, request, redirect,
    url_for, session, flash
)
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY")

if not app.secret_key:
    raise RuntimeError("SECRET_KEY environment variable is required")

DB = os.environ.get("DATABASE_PATH", "bank.db")


def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()

    conn.executescript("""
    CREATE TABLE IF NOT EXISTS admin_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password_hash TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL UNIQUE,
        phone TEXT,
        city TEXT,
        income REAL DEFAULT 0,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS accounts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER NOT NULL,
        account_no TEXT NOT NULL UNIQUE,
        account_type TEXT NOT NULL,
        balance REAL DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(customer_id) REFERENCES customers(id)
    );

    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        account_id INTEGER NOT NULL,
        txn_type TEXT NOT NULL,
        amount REAL NOT NULL,
        description TEXT,
        is_fraud INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(account_id) REFERENCES accounts(id)
    );
    """)

    # Create or update the admin account only when a password is supplied.
    admin_password = os.environ.get("ADMIN_PASSWORD")

    if admin_password:
        hashed_pw = generate_password_hash(admin_password)

        conn.execute("""
            INSERT INTO admin_users(username, password_hash)
            VALUES(?, ?)
            ON CONFLICT(username)
            DO UPDATE SET password_hash = excluded.password_hash
        """, ("admin", hashed_pw))

    conn.commit()
    conn.close()


def login_required():
    return "user" in session


@app.route("/")
def home():
    if not login_required():
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        admin = conn.execute(
            "SELECT * FROM admin_users WHERE username = ?",
            (username,)
        ).fetchone()
        conn.close()

        if admin and check_password_hash(admin["password_hash"], password):
            session["user"] = username
            return redirect(url_for("dashboard"))

        flash("Invalid username or password")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
def dashboard():
    if not login_required():
        return redirect(url_for("login"))

    conn = get_db()

    customers = conn.execute(
        "SELECT COUNT(*) c FROM customers"
    ).fetchone()["c"]

    accounts = conn.execute(
        "SELECT COUNT(*) c FROM accounts"
    ).fetchone()["c"]

    balance = conn.execute(
        "SELECT COALESCE(SUM(balance),0) b FROM accounts"
    ).fetchone()["b"]

    deposits = conn.execute(
        "SELECT COALESCE(SUM(amount),0) a FROM transactions WHERE txn_type='DEPOSIT'"
    ).fetchone()["a"]

    withdrawals = conn.execute(
        "SELECT COALESCE(SUM(amount),0) a FROM transactions WHERE txn_type='WITHDRAW'"
    ).fetchone()["a"]

    recent = conn.execute("""
        SELECT t.*, a.account_no, c.name
        FROM transactions t
        JOIN accounts a ON a.id=t.account_id
        JOIN customers c ON c.id=a.customer_id
        ORDER BY t.id DESC LIMIT 10
    """).fetchall()

    rows = conn.execute("""
        SELECT c.id, c.name, c.income, COUNT(t.id) txn_count,
               COALESCE(SUM(a.balance),0) balance
        FROM customers c
        LEFT JOIN accounts a ON a.customer_id=c.id
        LEFT JOIN transactions t ON t.account_id=a.id
        GROUP BY c.id
    """).fetchall()

    conn.close()

    data = pd.DataFrame([dict(r) for r in rows])
    segments = []

    if len(data) >= 3:
        X = data[["income", "txn_count", "balance"]].fillna(0)
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        data["segment"] = model.fit_predict(X)
        segments = data[
            ["name", "income", "txn_count", "balance", "segment"]
        ].to_dict("records")
    else:
        segments = data.to_dict("records")

    return render_template(
        "dashboard.html",
        customers=customers,
        accounts=accounts,
        balance=balance,
        deposits=deposits,
        withdrawals=withdrawals,
        recent=recent,
        segments=segments
    )


@app.route("/customers")
def customers():
    if not login_required():
        return redirect(url_for("login"))

    conn = get_db()
    rows = conn.execute("""
        SELECT c.*, COUNT(a.id) account_count,
               COALESCE(SUM(a.balance),0) total_balance
        FROM customers c
        LEFT JOIN accounts a ON a.customer_id=c.id
        GROUP BY c.id ORDER BY c.id DESC
    """).fetchall()
    conn.close()

    return render_template("customers.html", customers=rows)


@app.route("/customers/add", methods=["GET", "POST"])
def add_customer():
    if not login_required():
        return redirect(url_for("login"))

    if request.method == "POST":
        conn = get_db()
        try:
            conn.execute("""
                INSERT INTO customers(name,email,phone,city,income,created_at)
                VALUES(?,?,?,?,?,?)
            """, (
                request.form["name"],
                request.form["email"],
                request.form["phone"],
                request.form["city"],
                float(request.form.get("income") or 0),
                datetime.now().isoformat(timespec="seconds")
            ))

            conn.commit()
            flash("Customer added successfully")
            return redirect(url_for("customers"))

        except sqlite3.IntegrityError:
            flash("Email already exists")
        except ValueError:
            flash("Please enter a valid income amount")
        finally:
            conn.close()

    return render_template("add_customer.html")


@app.route("/customers/<int:customer_id>")
def customer_detail(customer_id):
    if not login_required():
        return redirect(url_for("login"))

    conn = get_db()

    customer = conn.execute(
        "SELECT * FROM customers WHERE id=?",
        (customer_id,)
    ).fetchone()

    accounts = conn.execute(
        "SELECT * FROM accounts WHERE customer_id=?",
        (customer_id,)
    ).fetchall()

    conn.close()

    if not customer:
        return "Customer not found", 404

    return render_template(
        "customer_detail.html",
        customer=customer,
        accounts=accounts
    )


@app.route("/accounts/add/<int:customer_id>", methods=["POST"])
def add_account(customer_id):
    if not login_required():
        return redirect(url_for("login"))

    account_no = "SB" + str(secrets.randbelow(90000000) + 10000000)
    account_type = request.form["account_type"]

    conn = get_db()

    try:
        conn.execute("""
            INSERT INTO accounts(customer_id,account_no,account_type,balance,created_at)
            VALUES(?,?,?,?,?)
        """, (
            customer_id,
            account_no,
            account_type,
            0,
            datetime.now().isoformat(timespec="seconds")
        ))

        conn.commit()
        flash("Account created: " + account_no)

    except sqlite3.IntegrityError:
        flash("Could not create account")

    finally:
        conn.close()

    return redirect(url_for("customer_detail", customer_id=customer_id))


def update_balance(account_id, amount, txn_type, description):
    if amount <= 0:
        return False, "Amount must be positive"

    if txn_type not in ("DEPOSIT", "WITHDRAW"):
        return False, "Invalid transaction type"

    conn = get_db()

    try:
        account = conn.execute(
            "SELECT * FROM accounts WHERE id=?",
            (account_id,)
        ).fetchone()

        if not account:
            return False, "Account not found"

        if txn_type == "WITHDRAW" and account["balance"] < amount:
            return False, "Insufficient balance"

        if txn_type == "DEPOSIT":
            new_balance = account["balance"] + amount
        else:
            new_balance = account["balance"] - amount

        all_txns = pd.read_sql_query(
            "SELECT amount FROM transactions",
            conn
        )

        is_fraud = 0

        if len(all_txns) >= 5:
            clf = IsolationForest(
                contamination=0.05,
                random_state=42
            )
            clf.fit(all_txns[["amount"]])

            current_pred = clf.predict([[amount]])

            if current_pred[0] == -1:
                is_fraud = 1

        conn.execute(
            "UPDATE accounts SET balance=? WHERE id=?",
            (new_balance, account_id)
        )

        conn.execute("""
            INSERT INTO transactions(
                account_id,txn_type,amount,description,is_fraud,created_at
            )
            VALUES(?,?,?,?,?,?)
        """, (
            account_id,
            txn_type,
            amount,
            description,
            is_fraud,
            datetime.now().isoformat(timespec="seconds")
        ))

        conn.commit()

        msg = "Transaction successful"
        if is_fraud:
            msg += " (Flagged as potential anomaly!)"

        return True, msg

    except Exception:
        conn.rollback()
        raise

    finally:
        conn.close()


@app.route("/transaction/<int:account_id>", methods=["POST"])
def transaction(account_id):
    if not login_required():
        return redirect(url_for("login"))

    try:
        amount = float(request.form["amount"])

        if amount <= 0:
            flash("Amount must be positive")
            return redirect(request.referrer or url_for("dashboard"))

        txn_type = request.form["txn_type"]
        description = request.form.get("description", "")

        ok, msg = update_balance(
            account_id,
            amount,
            txn_type,
            description
        )

        flash(msg)

    except ValueError:
        flash("Invalid amount")

    return redirect(request.referrer or url_for("dashboard"))


@app.route("/transactions")
def transactions():
    if not login_required():
        return redirect(url_for("login"))

    conn = get_db()

    rows = conn.execute("""
        SELECT t.*, a.account_no, c.name
        FROM transactions t
        JOIN accounts a ON a.id=t.account_id
        JOIN customers c ON c.id=a.customer_id
        ORDER BY t.id DESC
    """).fetchall()

    conn.close()

    return render_template("transactions.html", transactions=rows)


@app.route("/analytics")
def analytics():
    if not login_required():
        return redirect(url_for("login"))

    conn = get_db()

    rows = conn.execute("""
        SELECT c.id, c.name, c.income, COUNT(t.id) txn_count,
               COALESCE(SUM(CASE WHEN t.txn_type='DEPOSIT' THEN t.amount ELSE 0 END),0) deposits,
               COALESCE(SUM(CASE WHEN t.txn_type='WITHDRAW' THEN t.amount ELSE 0 END),0) withdrawals,
               COALESCE(SUM(a.balance),0) balance
        FROM customers c
        LEFT JOIN accounts a ON a.customer_id=c.id
        LEFT JOIN transactions t ON t.account_id=a.id
        GROUP BY c.id
    """).fetchall()

    conn.close()

    data = pd.DataFrame([dict(r) for r in rows])

    chart_labels = data["name"].tolist() if not data.empty else []
    chart_balances = data["balance"].tolist() if not data.empty else []
    chart_incomes = data["income"].tolist() if not data.empty else []

    return render_template(
        "analytics.html",
        chart_labels=chart_labels,
        chart_balances=chart_balances,
        chart_incomes=chart_incomes
    )

init_db()

if __name__ == "__main__":
    app.run(debug=False)