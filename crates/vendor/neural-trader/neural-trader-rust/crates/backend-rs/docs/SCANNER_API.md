# Scanner API Documentation

## Overview

The Scanner API provides endpoints for scanning and analyzing OpenAPI specifications and REST APIs. It automatically discovers endpoints, identifies security vulnerabilities, and generates AI-powered analysis reports.

## Base URL

```
http://localhost:8080/api/scanner
```

## Endpoints

### 1. Start New Scan

**POST** `/api/scanner/scan`

Start a new API scan with specified URL and options.

**Request Body:**
```json
{
  "url": "https://api.example.com/openapi.json",
  "scan_type": "openapi",  // "openapi" | "auto"
  "options": {
    "deep_scan": true,
    "check_auth": true,
    "timeout": 30000
  }
}
```

**Response:**
```json
{
  "scan_id": "uuid-here",
  "status": "queued",
  "url": "https://api.example.com/openapi.json",
  "scan_type": "openapi",
  "created_at": "2025-11-13T12:00:00Z"
}
```

**Status Codes:**
- `200 OK` - Scan created successfully
- `400 Bad Request` - Invalid request body
- `500 Internal Server Error` - Database error

---

### 2. List All Scans

**GET** `/api/scanner/scans`

Retrieve a paginated list of all scans with optional status filtering.

**Query Parameters:**
- `page` (optional, default: 1) - Page number
- `limit` (optional, default: 20) - Items per page
- `status` (optional) - Filter by status: "queued", "running", "completed", "failed"

**Example:**
```bash
GET /api/scanner/scans?page=1&limit=10&status=completed
```

**Response:**
```json
{
  "scans": [
    {
      "id": "uuid-here",
      "url": "https://api.example.com/openapi.json",
      "scan_type": "openapi",
      "status": "completed",
      "endpoints_found": 42,
      "vulnerabilities_count": 3,
      "created_at": "2025-11-13T12:00:00Z"
    }
  ],
  "page": 1,
  "limit": 10,
  "total": 1
}
```

---

### 3. Get Scan Details

**GET** `/api/scanner/scans/:id`

Retrieve detailed information about a specific scan including all discovered endpoints and vulnerabilities.

**Path Parameters:**
- `id` - Scan UUID

**Response:**
```json
{
  "id": "uuid-here",
  "url": "https://api.example.com/openapi.json",
  "scan_type": "openapi",
  "status": "completed",
  "endpoints_found": 42,
  "vulnerabilities_count": 3,
  "scan_data": {
    "options": { "deep_scan": true },
    "endpoints": [
      {
        "path": "/api/users",
        "method": "GET",
        "authentication": "bearer",
        "parameters": []
      }
    ],
    "vulnerabilities": [
      {
        "severity": "high",
        "type": "authentication",
        "description": "Missing authentication on sensitive endpoint",
        "endpoint": "/api/admin"
      }
    ],
    "metrics": {
      "scan_duration_ms": 1234,
      "total_requests": 42
    }
  },
  "created_at": "2025-11-13T12:00:00Z",
  "updated_at": "2025-11-13T12:01:00Z",
  "completed_at": "2025-11-13T12:01:00Z"
}
```

**Status Codes:**
- `200 OK` - Scan found
- `404 Not Found` - Scan does not exist

---

### 4. Get AI-Generated Report

**GET** `/api/scanner/scans/:id/report`

Generate and retrieve an AI-powered analysis report for the scan with recommendations.

**Path Parameters:**
- `id` - Scan UUID

**Response:**
```json
{
  "scan_id": "uuid-here",
  "url": "https://api.example.com/openapi.json",
  "report_generated_at": "2025-11-13T12:05:00Z",
  "summary": {
    "total_endpoints": 42,
    "vulnerabilities": 3,
    "risk_level": "medium",
    "scan_status": "completed"
  },
  "ai_analysis": {
    "overview": "Automated security analysis completed",
    "key_findings": [
      "API endpoints discovered and catalogued",
      "Authentication mechanisms analyzed",
      "Potential security vulnerabilities identified"
    ],
    "recommendations": [
      "Review authentication flows for all endpoints",
      "Implement rate limiting on public endpoints",
      "Add input validation for all user-provided data",
      "Enable HTTPS-only communication",
      "Implement proper error handling to prevent information leakage"
    ]
  },
  "detailed_analysis": {
    "endpoints": [...],
    "vulnerabilities": [...],
    "metrics": {...}
  },
  "next_steps": [
    "Review and address high-priority vulnerabilities",
    "Implement recommended security measures",
    "Schedule regular security scans",
    "Update API documentation with security guidelines"
  ]
}
```

---

### 5. Delete Scan

**DELETE** `/api/scanner/scans/:id`

Delete a scan and all its associated data.

**Path Parameters:**
- `id` - Scan UUID

**Response:**
```json
{
  "message": "Scan deleted successfully",
  "scan_id": "uuid-here"
}
```

**Status Codes:**
- `200 OK` - Scan deleted
- `404 Not Found` - Scan does not exist
- `500 Internal Server Error` - Database error

---

### 6. Get Scanner Statistics

**GET** `/api/scanner/stats`

Retrieve overall scanner statistics.

**Response:**
```json
{
  "total_scans": 156,
  "endpoints_discovered": 4821,
  "vulnerabilities_found": 89,
  "active_scans": 3,
  "last_updated": "2025-11-13T12:05:00Z"
}
```

---

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

Common HTTP status codes:
- `200 OK` - Success
- `400 Bad Request` - Invalid input
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

---

## CORS Support

All endpoints include CORS headers allowing cross-origin requests from any domain. In production, this should be restricted to specific origins.

---

## Scan Lifecycle

1. **Queued** - Scan created and waiting to start
2. **Running** - Scan actively analyzing the API
3. **Completed** - Scan finished successfully
4. **Failed** - Scan encountered an error

---

## Usage Examples

### cURL

```bash
# Start a new scan
curl -X POST http://localhost:8080/api/scanner/scan \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://api.example.com/openapi.json",
    "scan_type": "openapi",
    "options": {"deep_scan": true}
  }'

# List all scans
curl http://localhost:8080/api/scanner/scans?limit=20

# Get scan details
curl http://localhost:8080/api/scanner/scans/YOUR-SCAN-ID

# Get AI report
curl http://localhost:8080/api/scanner/scans/YOUR-SCAN-ID/report

# Delete scan
curl -X DELETE http://localhost:8080/api/scanner/scans/YOUR-SCAN-ID

# Get statistics
curl http://localhost:8080/api/scanner/stats
```

### JavaScript/Fetch

```javascript
// Start a new scan
const response = await fetch('http://localhost:8080/api/scanner/scan', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    url: 'https://api.example.com/openapi.json',
    scan_type: 'openapi',
    options: { deep_scan: true }
  })
});
const scan = await response.json();

// Get scan details
const details = await fetch(
  `http://localhost:8080/api/scanner/scans/${scan.scan_id}`
).then(r => r.json());

// Get AI report
const report = await fetch(
  `http://localhost:8080/api/scanner/scans/${scan.scan_id}/report`
).then(r => r.json());
```

### Python

```python
import requests

# Start a new scan
response = requests.post(
    'http://localhost:8080/api/scanner/scan',
    json={
        'url': 'https://api.example.com/openapi.json',
        'scan_type': 'openapi',
        'options': {'deep_scan': True}
    }
)
scan = response.json()

# Get scan details
details = requests.get(
    f"http://localhost:8080/api/scanner/scans/{scan['scan_id']}"
).json()

# Get statistics
stats = requests.get('http://localhost:8080/api/scanner/stats').json()
```

---

## Database Schema

The scanner uses SQLite with the following table structure:

```sql
CREATE TABLE api_scans (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    scan_type TEXT NOT NULL,
    status TEXT NOT NULL,
    endpoints_found INTEGER DEFAULT 0,
    vulnerabilities_count INTEGER DEFAULT 0,
    scan_data TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);
```

---

## Future Enhancements

- Real-time scan progress updates via WebSockets
- Scheduled periodic scans
- Scan result comparison and diff
- Export reports to PDF/HTML
- Integration with CI/CD pipelines
- Webhook notifications on scan completion
- Custom vulnerability rules engine
- API authentication testing
- Performance benchmarking
