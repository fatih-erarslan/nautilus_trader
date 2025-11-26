use beclever_common::{Error, Result};

pub struct SupabaseClient {
    url: String,
    key: String,
}

impl SupabaseClient {
    pub fn new(url: String, key: String) -> Self {
        Self { url, key }
    }
}
