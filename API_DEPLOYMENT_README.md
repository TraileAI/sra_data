# SRA Data Processing API - Deployment Guide

## Overview

This document provides deployment instructions for the SRA Data Processing Service minimal FastAPI skeleton. This API is designed for **deployment stability** on Render.com, not as a full client API.

## API Endpoints

### 1. Root Endpoint (`/`)
- **Method**: GET
- **Purpose**: Basic service information
- **Response**: Service metadata and available endpoints

```json
{
  "service": "SRA Data Processing Service",
  "version": "1.0.0",
  "description": "Data processing service for FMP API and fundata CSV ingestion",
  "status": "running",
  "deployment": "render.com",
  "endpoints": ["/", "/health", "/status"],
  "documentation": {
    "swagger": "/docs",
    "redoc": "/redoc"
  }
}
```

### 2. Health Check Endpoint (`/health`)
- **Method**: GET
- **Purpose**: Simple health status for load balancers
- **Response Time**: < 1 second
- **Response**: Basic health status

```json
{
  "status": "healthy",
  "timestamp": "2025-09-25T15:00:00Z",
  "service": "SRA Data Processing Service",
  "version": "1.0.0"
}
```

### 3. Detailed Status Endpoint (`/status`)
- **Method**: GET
- **Purpose**: Comprehensive service monitoring
- **Response Time**: < 2 seconds
- **Response**: Detailed system and service status

```json
{
  "service": "SRA Data Processing Service",
  "version": "1.0.0",
  "status": "healthy",
  "timestamp": "2025-09-25T15:00:00Z",
  "uptime_seconds": 86400,
  "uptime_human": "1d 0h 0m 0s",
  "database": {
    "status": "connected",
    "connected": true,
    "pool_info": {
      "active_connections": 3,
      "pool_size": 10
    }
  },
  "data_services": {
    "fmp_integration": {
      "status": "available",
      "service_type": "equity_data_processing"
    },
    "fundata_processing": {
      "status": "available",
      "service_type": "csv_data_processing"
    }
  }
}
```

## Deployment Instructions

### Render.com Deployment

1. **Prerequisites**:
   - GitHub repository with the codebase
   - Render.com account

2. **Automatic Deployment**:
   ```yaml
   # render.yaml is included in the repository
   services:
     - type: web
       name: sra-data-processing
       runtime: python3
       buildCommand: pip install -r requirements.txt
       startCommand: python server.py
       plan: free
       region: oregon
       branch: main
   ```

3. **Manual Deployment Steps**:
   - Connect GitHub repository to Render.com
   - Create new Web Service
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `python server.py`
   - Deploy

4. **Environment Variables**:
   ```
   PORT=10000
   HOST=0.0.0.0
   LOG_LEVEL=info
   ACCESS_LOG=true
   TIMEOUT_KEEP_ALIVE=30
   WORKERS=1
   ```

### Docker Deployment

1. **Build Image**:
   ```bash
   docker build -t sra-data-processing .
   ```

2. **Run Container**:
   ```bash
   docker run -p 10000:10000 -e PORT=10000 sra-data-processing
   ```

3. **Docker Compose** (optional):
   ```yaml
   version: '3.8'
   services:
     api:
       build: .
       ports:
         - "10000:10000"
       environment:
         - PORT=10000
         - HOST=0.0.0.0
         - LOG_LEVEL=info
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:10000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

### Local Development

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Server**:
   ```bash
   python server.py
   ```

3. **Access API**:
   - Health Check: http://localhost:10000/health
   - Status: http://localhost:10000/status
   - Documentation: http://localhost:10000/docs

## Required Dependencies

Add to `requirements.txt`:
```
fastapi>=0.104.0
uvicorn>=0.24.0
asyncpg>=0.29.0
pydantic>=2.0.0
```

## Testing

### Manual Testing Script
```bash
python test_api_manual.py
```

### BDD Testing (with pytest-bdd)
```bash
pytest tests/features/api_skeleton.feature
```

### API Testing (with httpx/requests)
```bash
pytest tests/test_api_skeleton.py
```

## Monitoring and Health Checks

### Load Balancer Configuration
- **Health Check URL**: `/health`
- **Expected Response**: 200 OK with `{"status": "healthy"}`
- **Timeout**: 10 seconds
- **Interval**: 30 seconds

### Monitoring Endpoints
- **Basic Health**: `/health` (< 1s response)
- **Detailed Status**: `/status` (< 2s response)
- **Service Info**: `/` (< 0.5s response)

## Architecture Notes

### Purpose
This is a **minimal FastAPI skeleton** designed for:
1. **Deployment Stability**: Prevent Render.com service suspension
2. **Health Monitoring**: Basic health checks for infrastructure
3. **Service Discovery**: Basic service information endpoints

### NOT Included
- Full client API functionality
- Business logic endpoints
- Data processing endpoints
- Authentication/authorization
- Complex middleware

### Service Integration
The skeleton integrates with:
- Database infrastructure (graceful fallback)
- Data processing services (status monitoring)
- System metrics (uptime, health)

## Troubleshooting

### Common Issues

1. **Port Configuration**:
   - Render.com uses PORT environment variable
   - Default: 10000
   - Ensure server.py reads PORT from environment

2. **Import Errors**:
   - Ensure all dependencies in requirements.txt
   - Check Python path configuration
   - Verify package structure

3. **Health Check Failures**:
   - Check `/health` endpoint accessibility
   - Verify response format
   - Check response time (< 1 second)

4. **Database Connection Issues**:
   - Check DATABASE_URL environment variable
   - Verify database credentials
   - Check network connectivity

### Performance Targets
- Health endpoint: < 1 second response
- Status endpoint: < 2 seconds response
- Root endpoint: < 0.5 seconds response
- Memory usage: < 512MB
- CPU usage: < 80%

## Security Considerations

### Production Settings
1. Configure CORS origins appropriately
2. Set up proper logging levels
3. Use environment variables for secrets
4. Implement rate limiting if needed
5. Set up proper error handling

### Network Security
- Use HTTPS in production
- Configure proper firewall rules
- Limit exposed ports
- Use secure database connections

This minimal API provides essential deployment stability while maintaining simplicity and performance.