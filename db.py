"""
Supabase wrapper for Aletheia.

Responsibilities:
- Provide a cached Supabase client (one per Streamlit session).
- Wrap auth (sign up / sign in / sign out) so app.py stays clean.
- Wrap audits CRUD (save, list, get) so the rest of the app does not need to
  know about RLS or the underlying SQL.

Row-Level Security in Supabase scopes every query to the currently
authenticated user, so we never need to filter by user_id manually here —
the policies in audits_schema.sql do that on the database side.
"""

from __future__ import annotations

import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_client() -> Client:
    """Return a cached Supabase client for this Streamlit session."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise RuntimeError(
            "SUPABASE_URL / SUPABASE_PUBLISHABLE_KEY are missing from .env. "
            "See .env.example for the template."
        )
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def _client() -> Client:
    return get_client()


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def sign_up(email: str, password: str, display_name: str) -> dict:
    """
    Register a new user. Returns the Supabase user object on success.
    Raises on failure.

    Display name is stored in the user's auth metadata (no separate
    profiles table needed).
    """
    res = _client().auth.sign_up(
        {
            "email": email,
            "password": password,
            "options": {"data": {"display_name": display_name}},
        }
    )
    if res.user is None:
        raise RuntimeError("Sign up failed — no user returned.")
    return res.user.model_dump() if hasattr(res.user, "model_dump") else dict(res.user)


def sign_in(email: str, password: str) -> dict:
    """Sign in with email + password. Returns the user dict."""
    res = _client().auth.sign_in_with_password(
        {"email": email, "password": password}
    )
    if res.user is None:
        raise RuntimeError("Sign in failed — wrong email or password.")
    return res.user.model_dump() if hasattr(res.user, "model_dump") else dict(res.user)


def sign_out() -> None:
    """Sign the current user out (clears the Supabase session)."""
    try:
        _client().auth.sign_out()
    except Exception:
        # If the session was already invalid, swallow the error.
        pass


def current_user() -> Optional[dict]:
    """Return the currently authenticated user dict, or None."""
    res = _client().auth.get_user()
    if res is None or res.user is None:
        return None
    return res.user.model_dump() if hasattr(res.user, "model_dump") else dict(res.user)


def display_name_of(user: dict) -> str:
    """Pull the display_name we stored in user_metadata at sign-up."""
    meta = user.get("user_metadata") or {}
    return meta.get("display_name") or user.get("email", "user")


# --------------------------------------------------------------------------
# Audits CRUD
# --------------------------------------------------------------------------
def save_audit(file_name: str, final_report: str) -> dict:
    """
    Insert one audit row for the current user.
    Returns the inserted row.

    The user_id column is filled automatically by a Postgres DEFAULT of
    auth.uid() — see audits_schema.sql.
    """
    res = (
        _client()
        .table("audits")
        .insert({"file_name": file_name, "final_report": final_report})
        .execute()
    )
    if not res.data:
        raise RuntimeError("Failed to save audit row.")
    return res.data[0]


def list_my_audits(limit: int = 50) -> list[dict]:
    """
    Return the current user's audits, newest first.
    RLS ensures no other user's rows are visible.
    """
    res = (
        _client()
        .table("audits")
        .select("id, file_name, final_report, created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return res.data or []


def get_audit(audit_id: str) -> Optional[dict]:
    """Fetch a single audit by id (RLS limits to current user)."""
    res = (
        _client()
        .table("audits")
        .select("*")
        .eq("id", audit_id)
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None
