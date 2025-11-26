use beclever_common::{Error, Result};

pub struct E2BClient {
    api_key: String,
}

impl E2BClient {
    pub fn new(api_key: String) -> Self {
        Self { api_key }
    }
}
