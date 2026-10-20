
"""PostgreSQL connection helpers for Plant Reco."""
import os
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row


def get_database_url():
    """Return the PostgreSQL URL from the environment."""
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return url


@contextmanager
def get_connection():
    """Open a PostgreSQL connection and close it afterwards."""
    with psycopg.connect(get_database_url(), row_factory=dict_row) as conn:
        yield conn
