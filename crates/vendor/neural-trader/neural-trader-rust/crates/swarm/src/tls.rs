//! TLS configuration and certificate generation for QUIC

use crate::error::{Result, SwarmError};
use rcgen::{Certificate, CertificateParams, DnType, KeyPair};
use rustls::{ClientConfig, ServerConfig};
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::sync::Arc;

/// Generate self-signed certificate for development
pub fn generate_self_signed_cert() -> Result<(Vec<CertificateDer<'static>>, PrivateKeyDer<'static>)> {
    let mut params = CertificateParams::default();
    params.distinguished_name.push(DnType::CommonName, "neural-trader-coordinator");
    params.subject_alt_names = vec![
        rcgen::SanType::DnsName("localhost".to_string()),
        rcgen::SanType::IpAddress(std::net::IpAddr::V4(std::net::Ipv4Addr::new(127, 0, 0, 1))),
        rcgen::SanType::IpAddress(std::net::IpAddr::V6(std::net::Ipv6Addr::new(0, 0, 0, 0, 0, 0, 0, 1))),
    ];

    let key_pair = KeyPair::generate()?;
    params.key_pair = Some(key_pair);

    let cert = params.self_signed(&rcgen::PKCS_ECDSA_P256_SHA256)?;

    let cert_der = CertificateDer::from(cert.der().to_vec());
    let key_der = PrivateKeyDer::try_from(cert.key_pair().serialize_der())
        .map_err(|_| SwarmError::Configuration("Invalid private key".into()))?;

    Ok((vec![cert_der], key_der))
}

/// Configure QUIC server with TLS
pub fn configure_server(
    certs: Vec<CertificateDer<'static>>,
    key: PrivateKeyDer<'static>,
) -> Result<ServerConfig> {
    let mut server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| SwarmError::Tls(e.into()))?;

    // Enable ALPN for QUIC
    server_config.alpn_protocols = vec![b"neural-trader-quic".to_vec()];

    Ok(server_config)
}

/// Configure QUIC client with TLS
pub fn configure_client() -> Result<ClientConfig> {
    let mut roots = rustls::RootCertStore::empty();

    // Add system root certificates
    for cert in rustls_native_certs::load_native_certs()? {
        roots.add(cert).ok();
    }

    let mut client_config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();

    // Enable ALPN for QUIC
    client_config.alpn_protocols = vec![b"neural-trader-quic".to_vec()];

    Ok(client_config)
}

/// Configure client for self-signed certificates (development only)
pub fn configure_client_insecure() -> Result<ClientConfig> {
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::crypto::{CryptoProvider, ring};
    use rustls::pki_types::{ServerName, UnixTime};

    // Dangerous: Skip certificate verification
    struct SkipServerVerification(Arc<CryptoProvider>);

    impl ServerCertVerifier for SkipServerVerification {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp: &[u8],
            _now: UnixTime,
        ) -> std::result::Result<ServerCertVerified, rustls::Error> {
            Ok(ServerCertVerified::assertion())
        }

        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &rustls::DigitallySignedStruct,
        ) -> std::result::Result<HandshakeSignatureValid, rustls::Error> {
            Ok(HandshakeSignatureValid::assertion())
        }

        fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
            self.0.signature_verification_algorithms.supported_schemes()
        }
    }

    let crypto_provider = Arc::new(ring::default_provider());

    let mut client_config = ClientConfig::builder_with_provider(crypto_provider.clone())
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|e| SwarmError::Configuration(e.to_string()))?
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(SkipServerVerification(crypto_provider)))
        .with_no_client_auth();

    // Enable ALPN for QUIC
    client_config.alpn_protocols = vec![b"neural-trader-quic".to_vec()];

    Ok(client_config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_self_signed_cert() {
        let result = generate_self_signed_cert();
        assert!(result.is_ok());

        let (certs, key) = result.unwrap();
        assert_eq!(certs.len(), 1);
        assert!(!key.secret_der().is_empty());
    }

    #[test]
    fn test_configure_server() {
        let (certs, key) = generate_self_signed_cert().unwrap();
        let result = configure_server(certs, key);
        assert!(result.is_ok());
    }

    #[test]
    fn test_configure_client_insecure() {
        let result = configure_client_insecure();
        assert!(result.is_ok());
    }
}
