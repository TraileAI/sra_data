# Docker Integration Guide for SRA Data Service

**Standards Compliance**: This document follows architecture-documentation-standard.md for deployment considerations and provides specific guidance for integrating sra_data into existing shared Docker infrastructure.

## Overview

### Integration Strategy
This guide provides recommendations for integrating the sra_data project into the existing shared Docker image strategy used between:
- `/Users/adam/dev/buckler/idd/` (IDD project)
- `/Users/adam/dev/buckler/iam/` (IAM project)

The goal is to leverage the existing Docker patterns while accommodating the unique requirements of the sra_data background processing service.

### Current Shared Docker Infrastructure Analysis

#### IDD Project Docker Configuration
Based on analysis of `/Users/adam/dev/buckler/idd/Dockerfile`:

**Strengths**:
- Multi-stage build pattern (builder + runtime stages)
- Python 3.13-alpine base with optimized layer caching
- Non-root user security (appuser:1000)
- Comprehensive system dependencies for PostgreSQL
- Health check integration for Render deployment
- Magic-wormhole installation for inter-service communication

**Architecture Pattern**:
```dockerfile
# Stage 1: Build stage with full dependencies
FROM python:3.13-alpine AS builder
RUN apk add --no-cache gcc postgresql-dev musl-dev python3-dev build-base
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# Stage 2: Runtime stage with minimal dependencies
FROM python:3.13-alpine AS runtime
RUN apk add --no-cache postgresql-client bash
COPY --from=builder /venv /venv
COPY --from=builder /build $APP_HOME
```

#### IAM Project Docker Configuration
Based on analysis of `/Users/adam/dev/buckler/iam/Dockerfile`:

**Strengths**:
- Highly optimized multi-stage build with virtual package cleanup
- Single-layer runtime dependency installation
- Efficient COPY with proper ownership flags
- Minimal production footprint
- Production-specific requirements file (requirements-prod.txt)

**Optimization Pattern**:
```dockerfile
# Build stage with immediate cleanup
RUN apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev
RUN pip install --no-cache-dir -r requirements-prod.txt
RUN apk del .build-deps

# Production stage with efficient copying
COPY --from=builder --chown=appuser:appuser /build /app
```

#### Shared Docker Network Strategy
From IAM's `docker-compose.yml`:
- **Shared Network**: `buckler_ai_shared_network` for inter-service communication
- **Service Discovery**: Container names for internal routing
- **Port Isolation**: Each service uses different ports (8000, 8001, etc.)
- **Environment Consistency**: Common patterns for database and Redis configuration

## Recommended Integration Approach

### Option 1: Shared Base Image Strategy (Recommended)

#### Create Shared Base Dockerfile
```dockerfile
# /Users/adam/dev/buckler/shared/Dockerfile.base
FROM python:3.13-alpine AS base-builder

# Shared environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install common build dependencies in single virtual package
RUN apk add --no-cache --virtual .shared-build-deps \
    gcc \
    musl-dev \
    postgresql-dev \
    python3-dev \
    linux-headers \
    build-base

# Create shared virtual environment
WORKDIR /build
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install common dependencies
COPY shared-requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r shared-requirements.txt

# Clean up build dependencies
RUN apk del .shared-build-deps

# Runtime base
FROM python:3.13-alpine AS base-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/venv/bin:$PATH"

# Install shared runtime dependencies
RUN apk add --no-cache \
    postgresql-client \
    bash \
    curl && \
    addgroup -g 1000 -S appuser && \
    adduser -u 1000 -S -G appuser -h /app -s /bin/sh appuser

# Copy shared virtual environment
COPY --from=base-builder /venv /venv

WORKDIR /app
USER appuser
```

#### SRA Data Service Dockerfile
```dockerfile
# /Users/adam/dev/buckler/sra_data/Dockerfile
FROM base-runtime AS sra-data-builder

# Switch to root for additional dependencies
USER root

# Install sra_data specific dependencies
COPY requirements-sra-data.txt .
RUN pip install --no-cache-dir -r requirements-sra-data.txt

# Install Git LFS for fundata CSV access
RUN apk add --no-cache git git-lfs

# Copy application code
COPY --chown=appuser:appuser . /app

# Initialize Git LFS and pull CSV files
RUN git lfs install && git lfs pull

# Production stage
FROM base-runtime AS production

# Copy sra_data environment from builder
COPY --from=sra-data-builder /venv /venv
COPY --from=sra-data-builder --chown=appuser:appuser /app /app

# Create data directories
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app/data /app/logs

# Switch to application user
USER appuser

# Health check for background worker
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health')" || exit 1

# Expose port for health checks and minimal API
EXPOSE 8002

# Default command for background worker
CMD ["python", "worker.py"]
```

#### Shared Requirements Structure
```
# shared-requirements.txt (common across all services)
fastapi>=0.104.0
uvicorn>=0.24.0
asyncpg>=0.29.0
pydantic>=2.0.0
structlog>=23.0.0
httpx>=0.25.0
redis>=5.0.0
python-dateutil>=2.8.2

# requirements-sra-data.txt (sra_data specific)
pandas>=2.1.0
numpy>=1.25.0
scipy>=1.11.0
schedule>=1.2.0
tqdm>=4.66.0
```

### Option 2: Service-Specific Extension (Alternative)

#### Extend Existing IDD Base
```dockerfile
# /Users/adam/dev/buckler/sra_data/Dockerfile
FROM idd_base:latest AS sra-data-extended

# Add sra_data specific requirements
USER root
COPY requirements-sra-addon.txt .
RUN pip install --no-cache-dir -r requirements-sra-addon.txt

# Install Git LFS capability
RUN apk add --no-cache git git-lfs

# Copy sra_data application
COPY --chown=appuser:appuser . /app

# Set up Git LFS for fundata CSV files
RUN cd /app && git lfs install && git lfs pull

USER appuser

# Override health check for background service
HEALTHCHECK --interval=60s --timeout=10s --start-period=60s --retries=5 \
    CMD python -c "import requests; requests.get('http://localhost:8002/health')" || exit 1

EXPOSE 8002
CMD ["python", "worker.py"]
```

## Docker Compose Integration

### Shared Network Configuration
```yaml
# docker-compose.sra-data.yml
version: '3.8'

services:
  sra_data:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: sra_data_worker
    environment:
      # Database connection to shared PostgreSQL
      - DB_HOST=shared_postgres
      - DB_PORT=5432
      - DB_NAME=sra_data_db
      - DB_USER=sra_data_user
      - DB_PASSWORD=${SRA_DATA_DB_PASSWORD}

      # Redis connection to shared instance
      - REDIS_URL=redis://shared_redis:6379/2  # Different database number

      # FMP API configuration
      - FMP_API_KEY=${FMP_API_KEY}
      - FMP_RATE_LIMIT_PER_MINUTE=3000

      # Service configuration
      - LOG_LEVEL=INFO
      - ENVIRONMENT=production
      - PORT=8002

      # Git LFS configuration
      - GIT_LFS_SKIP_SMUDGE=0

    volumes:
      - sra_data_logs:/app/logs
      - sra_data_temp:/tmp

    ports:
      - "8002:8002"  # Health check and minimal API

    depends_on:
      shared_postgres:
        condition: service_healthy
      shared_redis:
        condition: service_healthy

    restart: unless-stopped

    networks:
      - buckler_ai_shared_network
      - sra_data_internal

    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 60s
      timeout: 10s
      retries: 5
      start_period: 60s

  # Shared database service (if not already existing)
  shared_postgres:
    image: postgres:15
    container_name: buckler_shared_postgres
    environment:
      - POSTGRES_MULTIPLE_DATABASES=idd_db,iam_db,sra_data_db
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - shared_postgres_data:/var/lib/postgresql/data
      - ./scripts/create-multiple-databases.sh:/docker-entrypoint-initdb.d/create-multiple-databases.sh
    ports:
      - "5432:5432"
    networks:
      - buckler_ai_shared_network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Shared Redis service (if not already existing)
  shared_redis:
    image: redis:7-alpine
    container_name: buckler_shared_redis
    ports:
      - "6379:6379"
    volumes:
      - shared_redis_data:/data
    networks:
      - buckler_ai_shared_network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3

networks:
  buckler_ai_shared_network:
    name: buckler_ai_shared_network
    external: true
  sra_data_internal:
    driver: bridge

volumes:
  shared_postgres_data:
    external: true
  shared_redis_data:
    external: true
  sra_data_logs:
  sra_data_temp:
```

## Implementation Recommendations

### 1. Immediate Integration (Option 1 - Recommended)

**Benefits**:
- Leverages existing optimization patterns from IAM project
- Maintains shared dependency management
- Supports future service additions easily
- Consistent security and user management

**Implementation Steps**:
1. Create shared base image with common dependencies
2. Extend base image for sra_data specific requirements
3. Configure Git LFS integration during build
4. Set up shared network connectivity
5. Configure health checks for background service monitoring

### 2. Build Strategy Considerations

#### Git LFS Integration
```bash
# Build script for handling Git LFS files
#!/bin/bash
# build-sra-data.sh

set -e

echo "Building SRA Data Docker image with Git LFS support..."

# Ensure Git LFS files are available
git lfs pull

# Build with proper context
docker build -t sra-data:latest \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --file Dockerfile .

echo "Build complete. Image: sra-data:latest"
```

#### Multi-Architecture Support
```dockerfile
# Support for different deployment architectures
ARG TARGETPLATFORM
ARG BUILDPLATFORM

FROM --platform=$BUILDPLATFORM python:3.13-alpine AS base-builder
# ... build configuration

FROM python:3.13-alpine AS runtime
# ... runtime configuration
```

### 3. Environment Configuration

#### Shared Environment Variables
```bash
# .env.shared
POSTGRES_PASSWORD=secure_shared_password
REDIS_PASSWORD=secure_redis_password
LOG_LEVEL=INFO
ENVIRONMENT=production

# Network configuration
SHARED_NETWORK_NAME=buckler_ai_shared_network
```

#### SRA Data Specific Variables
```bash
# .env.sra-data
FMP_API_KEY=your_fmp_api_key
FMP_RATE_LIMIT_PER_MINUTE=3000
FUNDATA_BASE_PATH=/app/fundata
SRA_DATA_PORT=8002
WORKER_HEALTH_CHECK_INTERVAL=60
```

## Performance Considerations

### 1. Resource Optimization
- **Memory**: Background worker optimized for 512MB-2GB usage
- **CPU**: Single-threaded with async processing
- **Storage**: Git LFS files cached locally during build
- **Network**: Minimal API surface for health checks only

### 2. Scaling Strategy
- **Horizontal Scaling**: Not recommended (single background worker design)
- **Vertical Scaling**: Increase memory/CPU allocation as needed
- **Database Scaling**: Shared PostgreSQL with connection pooling
- **Redis Scaling**: Dedicated database number for sra_data caching

### 3. Monitoring Integration
```yaml
# Add monitoring labels
services:
  sra_data:
    labels:
      - "traefik.enable=false"  # No external routing needed
      - "monitoring.service=sra-data"
      - "monitoring.type=background-worker"
      - "monitoring.health-endpoint=/health"
```

## Security Considerations

### 1. Network Security
- **Internal Only**: sra_data doesn't need external access
- **Shared Network**: Participates in buckler_ai_shared_network
- **Port Exposure**: Only health check port exposed internally

### 2. Secret Management
```yaml
# Use Docker secrets or environment files
secrets:
  fmp_api_key:
    external: true
  postgres_password:
    external: true

services:
  sra_data:
    secrets:
      - fmp_api_key
      - postgres_password
```

### 3. File System Security
```dockerfile
# Secure Git LFS handling
RUN chown -R appuser:appuser /app/fundata && \
    chmod -R 750 /app/fundata
```

## Deployment Workflow

### 1. Local Development
```bash
# Start shared infrastructure
docker-compose -f docker-compose.shared.yml up -d

# Build and start sra_data service
docker-compose -f docker-compose.sra-data.yml up --build
```

### 2. Production Deployment
```bash
# Build multi-arch images
docker buildx build --platform linux/amd64,linux/arm64 \
  -t sra-data:production --push .

# Deploy to production
docker-compose -f docker-compose.production.yml up -d
```

### 3. Health Monitoring
```bash
# Check service health
curl http://localhost:8002/health

# View logs
docker logs sra_data_worker -f

# Monitor resource usage
docker stats sra_data_worker
```

## Troubleshooting

### Common Issues

#### Git LFS Files Not Available
```bash
# Solution: Ensure LFS files are pulled
git lfs install
git lfs pull

# Verify files exist
ls -la fundata/data/*.csv
ls -la fundata/quotes/*.csv
```

#### Database Connection Issues
```bash
# Check shared network connectivity
docker network inspect buckler_ai_shared_network

# Test database connection
docker exec sra_data_worker psql -h shared_postgres -U sra_data_user -d sra_data_db -c "SELECT 1;"
```

#### Memory Issues
```yaml
# Add resource limits
services:
  sra_data:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
```

## Conclusion

This Docker integration guide provides multiple strategies for incorporating the sra_data service into the existing shared Docker infrastructure. The recommended approach (Option 1) leverages proven patterns from the IAM project while accommodating the unique requirements of background data processing.

**Key Benefits**:
- Consistent deployment patterns across all Buckler services
- Optimized resource utilization through shared base images
- Proper Git LFS integration for fundata CSV files
- Health monitoring and service discovery integration
- Security through network isolation and proper user management

**Next Steps**:
1. Implement shared base image strategy
2. Configure Docker Compose for shared network integration
3. Set up CI/CD pipelines for automated building and deployment
4. Establish monitoring and alerting for the background service
5. Document operational procedures for service management

The integration maintains the high-performance, security-focused approach established by the existing services while providing the specialized capabilities required for financial data processing.