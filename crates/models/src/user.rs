use chrono::prelude::*;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Deserialize, Serialize, Clone, Copy, sqlx::Type, PartialEq)]
#[sqlx(type_name = "user_role", rename_all = "lowercase")]
#[serde(rename_all = "lowercase")]
pub enum UserRole {
    Admin,
    User,
    Agent,
    SuperUser,
    Invalid,
}

impl UserRole {
    // what levels users can create - otherwise should only be created by a trusted mechanism
    pub const fn is_allowed_on_create(&self) -> bool {
        match self {
            Self::User => true,
            _ => false,
        }
    }
}


#[derive(Debug, Deserialize, Serialize, sqlx::FromRow, sqlx::Type, Clone)]
pub struct User {
    pub id: Uuid,
    pub first_name: String,
    pub last_name: String,
    pub company_name: String,
    pub email: String,
    pub password: String,
    pub roles: Vec<UserRole>,
    pub is_email_verified: bool,
    pub email_verification_token: Option<String>,
    pub email_verification_token_expires_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

