-- Add up migration script here
CREATE TABLE IF NOT EXISTS users (
     id SERIAL PRIMARY KEY,
     username TEXT NOT NULL UNIQUE,
     created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

