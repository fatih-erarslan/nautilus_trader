use axum::{
    http::{header, HeaderValue, Request},
    middleware::Next,
    response::Response,
};

// Security headers middleware
pub async fn security_headers_middleware<B>(
    request: Request<B>,
    next: Next<B>,
) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    // X-Frame-Options: Prevent clickjacking attacks
    headers.insert(
        header::HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY"),
    );

    // X-Content-Type-Options: Prevent MIME type sniffing
    headers.insert(
        header::HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff"),
    );

    // X-XSS-Protection: Enable browser XSS protection
    headers.insert(
        header::HeaderName::from_static("x-xss-protection"),
        HeaderValue::from_static("1; mode=block"),
    );

    // Strict-Transport-Security: Enforce HTTPS
    headers.insert(
        header::HeaderName::from_static("strict-transport-security"),
        HeaderValue::from_static("max-age=31536000; includeSubDomains"),
    );

    // Content-Security-Policy: Mitigate XSS and injection attacks
    headers.insert(
        header::HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:; connect-src 'self'",
        ),
    );

    // Referrer-Policy: Control referrer information
    headers.insert(
        header::HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );

    // Permissions-Policy: Control browser features
    headers.insert(
        header::HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("geolocation=(), microphone=(), camera=()"),
    );

    // Remove X-Powered-By header if present (don't leak technology stack)
    headers.remove("x-powered-by");

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};

    #[tokio::test]
    async fn test_security_headers_added() {
        let request = Request::builder()
            .uri("/")
            .body(Body::empty())
            .unwrap();

        let next = Next::new(|_: Request<Body>| async {
            axum::response::Response::new(Body::empty())
        });

        let response = security_headers_middleware(request, next).await;
        let headers = response.headers();

        assert_eq!(
            headers.get("x-frame-options").unwrap(),
            "DENY"
        );
        assert_eq!(
            headers.get("x-content-type-options").unwrap(),
            "nosniff"
        );
        assert_eq!(
            headers.get("x-xss-protection").unwrap(),
            "1; mode=block"
        );
        assert!(headers.get("strict-transport-security").is_some());
        assert!(headers.get("content-security-policy").is_some());
        assert!(headers.get("referrer-policy").is_some());
        assert!(headers.get("permissions-policy").is_some());
    }
}
