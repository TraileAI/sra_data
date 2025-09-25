"""
Render.com Deployment Configuration Manager for SRA Data Processing System.

This module provides comprehensive deployment configuration management including:
- Advanced Render service configuration and optimization
- Auto-scaling configuration and resource monitoring
- Environment-specific deployment settings
- Service health monitoring and alerting
- Production optimization and performance tuning
- Multi-service orchestration and dependency management
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil

# Simple YAML-like functionality using JSON
def yaml_dump(data, file, **kwargs):
    """Simple YAML dump replacement using JSON."""
    json.dump(data, file, indent=kwargs.get('indent', 2))

def yaml_load(file):
    """Simple YAML load replacement using JSON."""
    return json.load(file)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Render service types."""
    WEB = "web"
    WORKER = "worker"
    CRON = "cron"
    STATIC = "static"


class PlanType(Enum):
    """Render plan types."""
    FREE = "free"
    STARTER = "starter"
    STANDARD = "standard"
    PRO = "pro"


class RuntimeType(Enum):
    """Runtime environments."""
    PYTHON3 = "python3"
    NODE = "node"
    RUBY = "ruby"
    GO = "go"
    DOCKER = "docker"


@dataclass
class EnvironmentVariable:
    """Environment variable configuration."""
    key: str
    value: Optional[str] = None
    from_service: Optional[str] = None
    sync: bool = False
    generate_value: bool = False


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    path: str = "/health"
    initial_delay_seconds: int = 60
    period_seconds: int = 10
    timeout_seconds: int = 5
    failure_threshold: int = 3


@dataclass
class AutoScalingConfig:
    """Auto-scaling configuration."""
    enabled: bool = False
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_percent: int = 70
    target_memory_percent: int = 80
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 300  # seconds


@dataclass
class BuildConfig:
    """Build configuration."""
    build_command: str
    pre_deploy_command: Optional[str] = None
    docker_file_path: Optional[str] = None
    docker_context: Optional[str] = None
    build_filter: Optional[Dict[str, str]] = None


@dataclass
class ServiceConfig:
    """Individual service configuration."""
    name: str
    type: ServiceType
    runtime: RuntimeType
    build_config: BuildConfig
    start_command: str
    plan: PlanType = PlanType.FREE
    region: str = "oregon"
    branch: str = "main"
    root_dir: str = "."
    env_vars: List[EnvironmentVariable] = None
    health_check: Optional[HealthCheckConfig] = None
    auto_scaling: Optional[AutoScalingConfig] = None
    num_instances: int = 1
    disk_gb: Optional[int] = None
    pull_request_previews_enabled: bool = True

    def __post_init__(self):
        """Initialize default values."""
        if self.env_vars is None:
            self.env_vars = []


@dataclass
class DatabaseConfig:
    """Database service configuration."""
    name: str
    database_name: str
    user: str
    plan: str = "free"
    region: str = "oregon"
    version: str = "15"  # PostgreSQL version
    ipallow_list: List[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.ipallow_list is None:
            self.ipallow_list = []


@dataclass
class DeploymentEnvironment:
    """Complete deployment environment configuration."""
    name: str
    services: List[ServiceConfig]
    databases: List[DatabaseConfig] = None
    global_env_vars: List[EnvironmentVariable] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.databases is None:
            self.databases = []
        if self.global_env_vars is None:
            self.global_env_vars = []


class DeploymentConfig:
    """
    Comprehensive Render.com deployment configuration manager.

    Provides advanced deployment configuration, optimization, and management
    for SRA Data Processing System on Render platform.
    """

    def __init__(self, config_path: str = "render.yaml", project_path: str = "."):
        """
        Initialize deployment configuration manager.

        Args:
            config_path: Path to render.yaml file
            project_path: Path to project root
        """
        self.config_path = Path(config_path)
        self.project_path = Path(project_path).resolve()
        self.environments = {}
        self.current_config = None

        # Load existing configuration if available
        if self.config_path.exists():
            self.load_config()

    def load_config(self) -> bool:
        """
        Load existing Render configuration from YAML file.

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml_load(f)

            self.current_config = config_data
            logger.info(f"Loaded configuration from {self.config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return False

    def create_production_config(self) -> DeploymentEnvironment:
        """
        Create optimized production deployment configuration.

        Returns:
            Production deployment environment configuration
        """
        # Main web service configuration
        web_service = ServiceConfig(
            name="sra-data-api",
            type=ServiceType.WEB,
            runtime=RuntimeType.PYTHON3,
            build_config=BuildConfig(
                build_command="pip install -r requirements.txt && python -m packages.sra_data.repositories.migrations init",
                pre_deploy_command="python scripts/pre_deploy_checks.py"
            ),
            start_command="python server.py",
            plan=PlanType.STANDARD,
            region="oregon",
            branch="main",
            num_instances=2,
            disk_gb=1,
            env_vars=[
                EnvironmentVariable("PORT", "10000"),
                EnvironmentVariable("HOST", "0.0.0.0"),
                EnvironmentVariable("LOG_LEVEL", "info"),
                EnvironmentVariable("ENVIRONMENT", "production"),
                EnvironmentVariable("ACCESS_LOG", "true"),
                EnvironmentVariable("TIMEOUT_KEEP_ALIVE", "30"),
                EnvironmentVariable("WORKERS", "2"),
                EnvironmentVariable("MAX_REQUESTS", "1000"),
                EnvironmentVariable("MAX_REQUESTS_JITTER", "100"),
                EnvironmentVariable("PRELOAD", "true"),
                EnvironmentVariable("WORKER_CLASS", "uvicorn.workers.UvicornWorker"),
                EnvironmentVariable("DATABASE_URL", from_service="sra-data-db"),
                EnvironmentVariable("REDIS_URL", from_service="sra-redis"),
                EnvironmentVariable("SECRET_KEY", generate_value=True),
                EnvironmentVariable("JWT_SECRET", generate_value=True)
            ],
            health_check=HealthCheckConfig(
                path="/health",
                initial_delay_seconds=120,
                period_seconds=30,
                timeout_seconds=10,
                failure_threshold=3
            ),
            auto_scaling=AutoScalingConfig(
                enabled=True,
                min_instances=1,
                max_instances=5,
                target_cpu_percent=70,
                target_memory_percent=75,
                scale_up_cooldown=300,
                scale_down_cooldown=600
            )
        )

        # Background worker service
        worker_service = ServiceConfig(
            name="sra-data-worker",
            type=ServiceType.WORKER,
            runtime=RuntimeType.PYTHON3,
            build_config=BuildConfig(
                build_command="pip install -r requirements.txt",
                pre_deploy_command="python scripts/worker_pre_deploy.py"
            ),
            start_command="python worker.py",
            plan=PlanType.STARTER,
            region="oregon",
            branch="main",
            num_instances=1,
            env_vars=[
                EnvironmentVariable("LOG_LEVEL", "info"),
                EnvironmentVariable("ENVIRONMENT", "production"),
                EnvironmentVariable("WORKER_CONCURRENCY", "4"),
                EnvironmentVariable("WORKER_TIMEOUT", "300"),
                EnvironmentVariable("DATABASE_URL", from_service="sra-data-db"),
                EnvironmentVariable("REDIS_URL", from_service="sra-redis"),
                EnvironmentVariable("SECRET_KEY", sync=True)
            ],
            auto_scaling=AutoScalingConfig(
                enabled=True,
                min_instances=1,
                max_instances=3,
                target_cpu_percent=75,
                scale_up_cooldown=600,
                scale_down_cooldown=900
            )
        )

        # Scheduled data processing job
        cron_service = ServiceConfig(
            name="sra-data-scheduler",
            type=ServiceType.CRON,
            runtime=RuntimeType.PYTHON3,
            build_config=BuildConfig(
                build_command="pip install -r requirements.txt"
            ),
            start_command="python scripts/scheduled_tasks.py",
            plan=PlanType.STARTER,
            region="oregon",
            branch="main",
            env_vars=[
                EnvironmentVariable("LOG_LEVEL", "info"),
                EnvironmentVariable("ENVIRONMENT", "production"),
                EnvironmentVariable("SCHEDULE", "0 */6 * * *"),  # Every 6 hours
                EnvironmentVariable("DATABASE_URL", from_service="sra-data-db"),
                EnvironmentVariable("REDIS_URL", from_service="sra-redis")
            ]
        )

        # Database configuration
        database = DatabaseConfig(
            name="sra-data-db",
            database_name="sra_data_prod",
            user="sra_user",
            plan="starter",
            region="oregon",
            version="15",
            ipallow_list=[]  # Allow all by default
        )

        # Redis configuration
        redis_service = ServiceConfig(
            name="sra-redis",
            type=ServiceType.WEB,  # Redis as a service
            runtime=RuntimeType.DOCKER,
            build_config=BuildConfig(
                build_command="echo 'Redis service'",
                docker_file_path="redis.dockerfile"
            ),
            start_command="redis-server",
            plan=PlanType.STARTER,
            region="oregon",
            branch="main"
        )

        return DeploymentEnvironment(
            name="production",
            services=[web_service, worker_service, cron_service, redis_service],
            databases=[database],
            global_env_vars=[
                EnvironmentVariable("TZ", "UTC"),
                EnvironmentVariable("PYTHONUNBUFFERED", "1"),
                EnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1")
            ]
        )

    def create_staging_config(self) -> DeploymentEnvironment:
        """
        Create staging deployment configuration.

        Returns:
            Staging deployment environment configuration
        """
        # Simplified staging configuration
        web_service = ServiceConfig(
            name="sra-data-staging",
            type=ServiceType.WEB,
            runtime=RuntimeType.PYTHON3,
            build_config=BuildConfig(
                build_command="pip install -r requirements.txt"
            ),
            start_command="python server.py",
            plan=PlanType.FREE,
            region="oregon",
            branch="develop",
            env_vars=[
                EnvironmentVariable("PORT", "10000"),
                EnvironmentVariable("HOST", "0.0.0.0"),
                EnvironmentVariable("LOG_LEVEL", "debug"),
                EnvironmentVariable("ENVIRONMENT", "staging"),
                EnvironmentVariable("DATABASE_URL", from_service="sra-data-staging-db"),
                EnvironmentVariable("SECRET_KEY", "staging-secret-key")
            ],
            health_check=HealthCheckConfig(
                path="/health",
                initial_delay_seconds=60,
                period_seconds=60,
                timeout_seconds=10,
                failure_threshold=5
            )
        )

        database = DatabaseConfig(
            name="sra-data-staging-db",
            database_name="sra_data_staging",
            user="sra_staging_user",
            plan="free",
            region="oregon"
        )

        return DeploymentEnvironment(
            name="staging",
            services=[web_service],
            databases=[database],
            global_env_vars=[
                EnvironmentVariable("TZ", "UTC"),
                EnvironmentVariable("PYTHONUNBUFFERED", "1")
            ]
        )

    def create_development_config(self) -> DeploymentEnvironment:
        """
        Create development deployment configuration.

        Returns:
            Development deployment environment configuration
        """
        web_service = ServiceConfig(
            name="sra-data-dev",
            type=ServiceType.WEB,
            runtime=RuntimeType.PYTHON3,
            build_config=BuildConfig(
                build_command="pip install -r requirements.txt"
            ),
            start_command="python server.py --reload",
            plan=PlanType.FREE,
            region="oregon",
            branch="develop",
            env_vars=[
                EnvironmentVariable("PORT", "10000"),
                EnvironmentVariable("HOST", "0.0.0.0"),
                EnvironmentVariable("LOG_LEVEL", "debug"),
                EnvironmentVariable("ENVIRONMENT", "development"),
                EnvironmentVariable("DEBUG", "true"),
                EnvironmentVariable("SECRET_KEY", "dev-secret-key")
            ],
            pull_request_previews_enabled=True
        )

        return DeploymentEnvironment(
            name="development",
            services=[web_service],
            global_env_vars=[
                EnvironmentVariable("TZ", "UTC"),
                EnvironmentVariable("PYTHONUNBUFFERED", "1"),
                EnvironmentVariable("DEBUG", "true")
            ]
        )

    def generate_render_yaml(self, environment: DeploymentEnvironment) -> Dict[str, Any]:
        """
        Generate render.yaml configuration from environment settings.

        Args:
            environment: Deployment environment configuration

        Returns:
            Dictionary representing render.yaml structure
        """
        config = {
            "services": [],
            "databases": []
        }

        # Add services
        for service in environment.services:
            service_config = {
                "type": service.type.value,
                "name": service.name,
                "runtime": service.runtime.value,
                "buildCommand": service.build_config.build_command,
                "startCommand": service.start_command,
                "plan": service.plan.value,
                "region": service.region,
                "branch": service.branch,
                "rootDir": service.root_dir,
                "pullRequestPreviewsEnabled": service.pull_request_previews_enabled
            }

            # Add pre-deploy command if specified
            if service.build_config.pre_deploy_command:
                service_config["preDeployCommand"] = service.build_config.pre_deploy_command

            # Add Dockerfile settings for Docker runtime
            if service.runtime == RuntimeType.DOCKER:
                if service.build_config.docker_file_path:
                    service_config["dockerfilePath"] = service.build_config.docker_file_path
                if service.build_config.docker_context:
                    service_config["dockerContext"] = service.build_config.docker_context

            # Add environment variables
            if service.env_vars or environment.global_env_vars:
                service_config["envVars"] = []

                # Global environment variables
                for env_var in environment.global_env_vars:
                    env_config = {"key": env_var.key}
                    if env_var.value:
                        env_config["value"] = env_var.value
                    elif env_var.from_service:
                        env_config["fromService"] = {
                            "type": "pserv" if "db" in env_var.from_service else "service",
                            "name": env_var.from_service,
                            "property": "connectionString" if "db" in env_var.from_service else "host"
                        }
                    elif env_var.generate_value:
                        env_config["generateValue"] = True
                    elif env_var.sync:
                        env_config["sync"] = True

                    service_config["envVars"].append(env_config)

                # Service-specific environment variables
                for env_var in service.env_vars:
                    env_config = {"key": env_var.key}
                    if env_var.value:
                        env_config["value"] = env_var.value
                    elif env_var.from_service:
                        env_config["fromService"] = {
                            "type": "pserv" if "db" in env_var.from_service else "service",
                            "name": env_var.from_service,
                            "property": "connectionString" if "db" in env_var.from_service else "host"
                        }
                    elif env_var.generate_value:
                        env_config["generateValue"] = True
                    elif env_var.sync:
                        env_config["sync"] = True

                    service_config["envVars"].append(env_config)

            # Add health check if specified
            if service.health_check:
                service_config["healthCheckPath"] = service.health_check.path

            # Add scaling settings
            if service.num_instances > 1:
                service_config["numInstances"] = service.num_instances

            if service.disk_gb:
                service_config["disk"] = {"name": f"{service.name}-disk", "sizeGB": service.disk_gb}

            config["services"].append(service_config)

        # Add databases
        for database in environment.databases:
            db_config = {
                "name": database.name,
                "databaseName": database.database_name,
                "user": database.user,
                "plan": database.plan,
                "region": database.region,
                "postgresMajorVersion": database.version
            }

            if database.ipallow_list:
                db_config["ipAllowList"] = database.ipallow_list

            config["databases"].append(db_config)

        return config

    def save_config(self, environment: DeploymentEnvironment, output_path: Optional[str] = None) -> bool:
        """
        Save deployment configuration to render.yaml file.

        Args:
            environment: Deployment environment to save
            output_path: Optional custom output path

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_dict = self.generate_render_yaml(environment)
            output_file = Path(output_path) if output_path else self.config_path

            with open(output_file, 'w') as f:
                yaml_dump(config_dict, f, indent=2)

            logger.info(f"Configuration saved to {output_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False

    def create_docker_files(self) -> bool:
        """
        Create necessary Docker files for deployment.

        Returns:
            True if created successfully, False otherwise
        """
        try:
            # Create main Dockerfile
            dockerfile_content = """FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app \\
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:10000/health')"

# Default command
CMD ["python", "server.py"]
"""

            with open(self.project_path / "Dockerfile", 'w') as f:
                f.write(dockerfile_content)

            # Create Redis Dockerfile
            redis_dockerfile_content = """FROM redis:7-alpine

# Copy custom Redis configuration
COPY redis.conf /usr/local/etc/redis/redis.conf

# Expose Redis port
EXPOSE 6379

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD redis-cli ping | grep PONG

# Start Redis with custom config
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]
"""

            with open(self.project_path / "redis.dockerfile", 'w') as f:
                f.write(redis_dockerfile_content)

            # Create Redis configuration
            redis_config = """# Redis configuration for Render deployment
bind 0.0.0.0
port 6379
timeout 300
tcp-keepalive 60
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
"""

            with open(self.project_path / "redis.conf", 'w') as f:
                f.write(redis_config)

            # Create .dockerignore
            dockerignore_content = """.git
.gitignore
README.md
Dockerfile
.dockerignore
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env
pip-log.txt
pip-delete-this-directory.txt
.tox
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.DS_Store
.mypy_cache
.pytest_cache
.hypothesis
tests/
"""

            with open(self.project_path / ".dockerignore", 'w') as f:
                f.write(dockerignore_content)

            logger.info("Docker files created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create Docker files: {e}")
            return False

    def create_deployment_scripts(self) -> bool:
        """
        Create deployment and maintenance scripts.

        Returns:
            True if created successfully, False otherwise
        """
        try:
            scripts_dir = self.project_path / "scripts"
            scripts_dir.mkdir(exist_ok=True)

            # Pre-deploy checks script
            pre_deploy_script = """#!/usr/bin/env python3
\"\"\"
Pre-deployment checks for SRA Data Processing System.

Validates environment, database connectivity, and system health
before deployment proceeds.
\"\"\"

import os
import sys
import logging
import asyncio
import psutil
from pathlib import Path

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))

from sra_data.repositories.database import DatabaseManager
from sra_data.infrastructure.git_lfs_manager import GitLFSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def check_database_connection():
    \"\"\"Check database connectivity.\"\"\"
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def check_system_resources():
    \"\"\"Check system resource availability.\"\"\"
    try:
        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            logger.warning(f"⚠️ High memory usage: {memory.percent}%")
        else:
            logger.info(f"✅ Memory usage: {memory.percent}%")

        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            logger.error(f"❌ Low disk space: {disk.percent}% used")
            return False
        else:
            logger.info(f"✅ Disk usage: {disk.percent}%")

        return True
    except Exception as e:
        logger.error(f"❌ System resource check failed: {e}")
        return False


def check_git_lfs():
    \"\"\"Check Git LFS status.\"\"\"
    try:
        lfs_manager = GitLFSManager()
        health = lfs_manager.health_check()

        if health["overall_health"] in ["excellent", "good"]:
            logger.info(f"✅ Git LFS health: {health['overall_health']}")
            return True
        else:
            logger.warning(f"⚠️ Git LFS health: {health['overall_health']}")
            for issue in health['issues']:
                logger.warning(f"  - {issue}")
            return True  # Don't fail deployment for LFS issues
    except Exception as e:
        logger.error(f"❌ Git LFS check failed: {e}")
        return True  # Don't fail deployment for LFS issues


def check_environment_variables():
    \"\"\"Check required environment variables.\"\"\"
    required_vars = [
        'DATABASE_URL',
        'SECRET_KEY',
        'LOG_LEVEL'
    ]

    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    else:
        logger.info("✅ All required environment variables present")
        return True


async def main():
    \"\"\"Run all pre-deployment checks.\"\"\"
    logger.info("Starting pre-deployment checks...")

    checks = [
        ("Environment variables", check_environment_variables()),
        ("System resources", check_system_resources()),
        ("Git LFS", check_git_lfs()),
        ("Database connection", await check_database_connection())
    ]

    failed_checks = []
    for check_name, result in checks:
        if not result:
            failed_checks.append(check_name)

    if failed_checks:
        logger.error(f"❌ Pre-deployment checks failed: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        logger.info("✅ All pre-deployment checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
"""

            with open(scripts_dir / "pre_deploy_checks.py", 'w') as f:
                f.write(pre_deploy_script)

            # Worker pre-deploy script
            worker_pre_deploy_script = """#!/usr/bin/env python3
\"\"\"
Worker service pre-deployment checks.
\"\"\"

import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_worker_config():
    \"\"\"Check worker configuration.\"\"\"
    worker_vars = [
        'WORKER_CONCURRENCY',
        'WORKER_TIMEOUT',
        'DATABASE_URL'
    ]

    missing_vars = []
    for var in worker_vars:
        if not os.getenv(var):
            missing_vars.append(var)

    if missing_vars:
        logger.error(f"❌ Missing worker variables: {', '.join(missing_vars)}")
        return False
    else:
        logger.info("✅ Worker configuration valid")
        return True


def main():
    \"\"\"Run worker pre-deployment checks.\"\"\"
    logger.info("Starting worker pre-deployment checks...")

    if check_worker_config():
        logger.info("✅ Worker pre-deployment checks passed!")
        sys.exit(0)
    else:
        logger.error("❌ Worker pre-deployment checks failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""

            with open(scripts_dir / "worker_pre_deploy.py", 'w') as f:
                f.write(worker_pre_deploy_script)

            # Scheduled tasks script
            scheduled_tasks_script = """#!/usr/bin/env python3
\"\"\"
Scheduled tasks for data processing and maintenance.
\"\"\"

import os
import sys
import logging
import asyncio
from datetime import datetime

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages'))

from sra_data.repositories.database import DatabaseManager
from sra_data.repositories.performance_optimizer import PerformanceOptimizer
from sra_data.infrastructure.git_lfs_manager import GitLFSManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def database_maintenance():
    \"\"\"Perform database maintenance tasks.\"\"\"
    try:
        db_manager = DatabaseManager()
        await db_manager.initialize()

        optimizer = PerformanceOptimizer(db_manager)

        # Run optimization
        result = await optimizer.optimize_database()
        logger.info(f"Database optimization completed: {len(result.get('optimizations_applied', []))} optimizations applied")

        # Update statistics
        await optimizer.update_statistics()
        logger.info("Database statistics updated")

    except Exception as e:
        logger.error(f"Database maintenance failed: {e}")


def lfs_cleanup():
    \"\"\"Perform Git LFS cleanup.\"\"\"
    try:
        lfs_manager = GitLFSManager()
        result = lfs_manager.optimize_repository()
        logger.info(f"LFS cleanup completed: {len(result.get('actions_taken', []))} actions taken")
    except Exception as e:
        logger.error(f"LFS cleanup failed: {e}")


async def main():
    \"\"\"Run scheduled maintenance tasks.\"\"\"
    logger.info(f"Starting scheduled maintenance at {datetime.utcnow()}")

    # Run maintenance tasks
    await database_maintenance()
    lfs_cleanup()

    logger.info("Scheduled maintenance completed")


if __name__ == "__main__":
    asyncio.run(main())
"""

            with open(scripts_dir / "scheduled_tasks.py", 'w') as f:
                f.write(scheduled_tasks_script)

            # Make scripts executable
            for script_file in scripts_dir.glob("*.py"):
                script_file.chmod(0o755)

            logger.info("Deployment scripts created successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to create deployment scripts: {e}")
            return False

    def optimize_for_production(self, environment: DeploymentEnvironment) -> DeploymentEnvironment:
        """
        Apply production optimizations to deployment configuration.

        Args:
            environment: Environment to optimize

        Returns:
            Optimized deployment environment
        """
        optimized_env = environment

        for service in optimized_env.services:
            if service.type == ServiceType.WEB:
                # Optimize web service settings
                service.env_vars.extend([
                    EnvironmentVariable("PYTHONOPTIMIZE", "2"),  # Enable optimizations
                    EnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1"),  # Don't write .pyc files
                    EnvironmentVariable("PYTHONUNBUFFERED", "1"),  # Unbuffered output
                    EnvironmentVariable("WORKER_CONNECTIONS", "1000"),
                    EnvironmentVariable("MAX_WORKER_CONNECTIONS", "1000"),
                    EnvironmentVariable("BACKLOG", "2048")
                ])

                # Ensure health checks are configured
                if not service.health_check:
                    service.health_check = HealthCheckConfig()

                # Configure auto-scaling for production workloads
                if not service.auto_scaling:
                    service.auto_scaling = AutoScalingConfig(enabled=True)

            elif service.type == ServiceType.WORKER:
                # Optimize worker service settings
                service.env_vars.extend([
                    EnvironmentVariable("PYTHONOPTIMIZE", "2"),
                    EnvironmentVariable("CELERY_OPTIMIZATION", "fair"),
                    EnvironmentVariable("WORKER_PREFETCH_MULTIPLIER", "1")
                ])

        logger.info("Applied production optimizations to deployment configuration")
        return optimized_env

    def validate_config(self, environment: DeploymentEnvironment) -> Dict[str, List[str]]:
        """
        Validate deployment configuration for common issues.

        Args:
            environment: Environment to validate

        Returns:
            Dictionary of validation results with warnings and errors
        """
        validation_results = {
            "errors": [],
            "warnings": [],
            "suggestions": []
        }

        # Check for required services
        service_types = [s.type for s in environment.services]
        if ServiceType.WEB not in service_types:
            validation_results["errors"].append("No web service defined")

        # Check environment variables
        for service in environment.services:
            if service.type == ServiceType.WEB:
                required_vars = ["PORT", "HOST"]
                service_var_keys = [ev.key for ev in service.env_vars]

                missing_vars = [var for var in required_vars if var not in service_var_keys]
                if missing_vars:
                    validation_results["errors"].append(
                        f"Service {service.name} missing required variables: {', '.join(missing_vars)}"
                    )

            # Check for database URL
            if not any(ev.key == "DATABASE_URL" for ev in service.env_vars):
                validation_results["warnings"].append(
                    f"Service {service.name} has no DATABASE_URL configured"
                )

            # Check for secrets
            if service.type == ServiceType.WEB and not any(ev.key == "SECRET_KEY" for ev in service.env_vars):
                validation_results["warnings"].append(
                    f"Service {service.name} has no SECRET_KEY configured"
                )

        # Check resource allocations
        for service in environment.services:
            if service.plan == PlanType.FREE and service.num_instances > 1:
                validation_results["warnings"].append(
                    f"Service {service.name} uses free plan but requests {service.num_instances} instances"
                )

            if not service.health_check and service.type == ServiceType.WEB:
                validation_results["suggestions"].append(
                    f"Consider adding health check to service {service.name}"
                )

        return validation_results

    def get_deployment_status(self) -> Dict[str, Any]:
        """
        Get current deployment status and metrics.

        Returns:
            Dictionary containing deployment status information
        """
        status = {
            "config_loaded": self.current_config is not None,
            "environments_available": list(self.environments.keys()),
            "last_updated": None,
            "validation_status": "unknown"
        }

        if self.config_path.exists():
            status["last_updated"] = self.config_path.stat().st_mtime

        return status


# Example usage and testing functions
def main():
    """Example usage of DeploymentConfig."""
    # Initialize deployment manager
    deploy_config = DeploymentConfig()

    # Create production configuration
    print("Creating production configuration...")
    prod_env = deploy_config.create_production_config()

    # Optimize for production
    prod_env = deploy_config.optimize_for_production(prod_env)

    # Validate configuration
    validation = deploy_config.validate_config(prod_env)
    if validation["errors"]:
        print("❌ Configuration errors:")
        for error in validation["errors"]:
            print(f"  - {error}")
    else:
        print("✅ Configuration validation passed")

    if validation["warnings"]:
        print("⚠️ Warnings:")
        for warning in validation["warnings"]:
            print(f"  - {warning}")

    # Generate and save configuration
    print("\\nGenerating render.yaml...")
    if deploy_config.save_config(prod_env, "render.prod.yaml"):
        print("✅ Production configuration saved to render.prod.yaml")

    # Create staging configuration
    print("\\nCreating staging configuration...")
    staging_env = deploy_config.create_staging_config()
    if deploy_config.save_config(staging_env, "render.staging.yaml"):
        print("✅ Staging configuration saved to render.staging.yaml")

    # Create Docker files
    print("\\nCreating Docker files...")
    if deploy_config.create_docker_files():
        print("✅ Docker files created successfully")

    # Create deployment scripts
    print("\\nCreating deployment scripts...")
    if deploy_config.create_deployment_scripts():
        print("✅ Deployment scripts created successfully")

    # Get status
    status = deploy_config.get_deployment_status()
    print(f"\\nDeployment status: {status}")


if __name__ == "__main__":
    main()