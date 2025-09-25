"""
Environment Variable Management and Secrets Handling for SRA Data Processing System.

This module provides comprehensive environment variable management including:
- Secure configuration loading and validation
- Environment-specific settings management
- Secrets handling and encryption
- Configuration validation and verification
- Environment isolation and security
- Production-ready secrets management
"""

import os
import json
import logging
import base64
import hashlib
from typing import Dict, List, Optional, Any, Union, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import tempfile
import shutil
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnvironmentType(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class VariableType(Enum):
    """Variable types for classification."""
    SECRET = "secret"
    CONFIG = "config"
    DATABASE = "database"
    SERVICE = "service"
    FEATURE_FLAG = "feature_flag"
    SYSTEM = "system"


@dataclass
class EnvironmentVariable:
    """Environment variable definition."""
    key: str
    value: str
    var_type: VariableType
    required: bool = True
    description: str = ""
    encrypted: bool = False
    environments: Set[EnvironmentType] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.environments is None:
            self.environments = set(EnvironmentType)


@dataclass
class EnvironmentSchema:
    """Environment variable schema definition."""
    name: str
    environment_type: EnvironmentType
    variables: List[EnvironmentVariable]
    inherits_from: Optional[str] = None

    def __post_init__(self):
        """Initialize default values."""
        if not self.variables:
            self.variables = []


@dataclass
class ValidationResult:
    """Environment validation result."""
    is_valid: bool
    missing_variables: List[str]
    invalid_variables: List[str]
    warnings: List[str]
    errors: List[str]


class EncryptionManager:
    """
    Handles encryption and decryption of sensitive environment variables.
    """

    def __init__(self, master_key: Optional[str] = None):
        """
        Initialize encryption manager.

        Args:
            master_key: Master key for encryption. If None, generates from environment
        """
        self.master_key = master_key or os.getenv('ENCRYPTION_MASTER_KEY')
        if not self.master_key:
            # Generate a master key for development (not recommended for production)
            self.master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            logger.warning("Using generated master key. Set ENCRYPTION_MASTER_KEY for production!")

        # Derive encryption key
        self.fernet = self._create_fernet(self.master_key)

    def _create_fernet(self, master_key: str) -> Fernet:
        """Create Fernet cipher from master key."""
        # Use PBKDF2 to derive key from master key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'sra_data_salt',  # In production, use random salt per environment
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    def encrypt_value(self, value: str) -> str:
        """
        Encrypt a string value.

        Args:
            value: Value to encrypt

        Returns:
            Encrypted value as base64 string
        """
        try:
            encrypted = self.fernet.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            raise

    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt a base64 encrypted value.

        Args:
            encrypted_value: Base64 encrypted value

        Returns:
            Decrypted string value
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self.fernet.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            raise

    def is_encrypted(self, value: str) -> bool:
        """
        Check if a value appears to be encrypted.

        Args:
            value: Value to check

        Returns:
            True if value appears encrypted
        """
        try:
            # Try to decode as base64 and check if it could be encrypted
            base64.urlsafe_b64decode(value.encode())
            return len(value) > 50 and '=' in value  # Simple heuristic
        except:
            return False


class EnvironmentManager:
    """
    Comprehensive environment variable management system.

    Provides secure configuration loading, validation, and management
    for different deployment environments.
    """

    def __init__(self,
                 config_dir: str = "config",
                 environment: Optional[EnvironmentType] = None,
                 encryption_enabled: bool = True):
        """
        Initialize environment manager.

        Args:
            config_dir: Directory containing environment configuration files
            environment: Current environment type
            encryption_enabled: Whether to enable encryption for secrets
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        self.current_environment = environment or self._detect_environment()
        self.encryption_enabled = encryption_enabled

        # Initialize encryption manager
        if encryption_enabled:
            self.encryption_manager = EncryptionManager()
        else:
            self.encryption_manager = None

        # Environment schemas
        self.schemas: Dict[str, EnvironmentSchema] = {}

        # Loaded configuration
        self.configuration: Dict[str, str] = {}

        # Load default schemas
        self._create_default_schemas()

    def _detect_environment(self) -> EnvironmentType:
        """
        Detect current environment from environment variables.

        Returns:
            Detected environment type
        """
        env_name = os.getenv('ENVIRONMENT', '').lower()

        if env_name in ['prod', 'production']:
            return EnvironmentType.PRODUCTION
        elif env_name in ['stage', 'staging']:
            return EnvironmentType.STAGING
        elif env_name in ['test', 'testing']:
            return EnvironmentType.TESTING
        else:
            return EnvironmentType.DEVELOPMENT

    def _create_default_schemas(self):
        """Create default environment schemas."""

        # Base variables for all environments
        base_variables = [
            EnvironmentVariable("LOG_LEVEL", "info", VariableType.CONFIG,
                              description="Logging level"),
            EnvironmentVariable("PYTHONUNBUFFERED", "1", VariableType.SYSTEM,
                              description="Python unbuffered output"),
            EnvironmentVariable("PYTHONDONTWRITEBYTECODE", "1", VariableType.SYSTEM,
                              description="Don't write Python bytecode files"),
            EnvironmentVariable("TZ", "UTC", VariableType.SYSTEM,
                              description="Timezone setting"),
        ]

        # Database variables
        database_variables = [
            EnvironmentVariable("DATABASE_URL", "", VariableType.DATABASE,
                              encrypted=True, description="Database connection URL"),
            EnvironmentVariable("DATABASE_POOL_SIZE", "10", VariableType.CONFIG,
                              required=False, description="Database connection pool size"),
            EnvironmentVariable("DATABASE_MAX_OVERFLOW", "20", VariableType.CONFIG,
                              required=False, description="Database max overflow connections"),
        ]

        # API variables
        api_variables = [
            EnvironmentVariable("PORT", "10000", VariableType.CONFIG,
                              description="API server port"),
            EnvironmentVariable("HOST", "0.0.0.0", VariableType.CONFIG,
                              description="API server host"),
            EnvironmentVariable("SECRET_KEY", "", VariableType.SECRET,
                              encrypted=True, description="Application secret key"),
        ]

        # Service variables
        service_variables = [
            EnvironmentVariable("REDIS_URL", "", VariableType.SERVICE,
                              required=False, encrypted=True,
                              description="Redis connection URL"),
            EnvironmentVariable("CACHE_TTL", "3600", VariableType.CONFIG,
                              required=False, description="Cache TTL in seconds"),
        ]

        # Development environment schema
        dev_variables = base_variables + database_variables + api_variables + service_variables + [
            EnvironmentVariable("DEBUG", "true", VariableType.FEATURE_FLAG,
                              environments={EnvironmentType.DEVELOPMENT},
                              description="Enable debug mode"),
            EnvironmentVariable("RELOAD", "true", VariableType.FEATURE_FLAG,
                              environments={EnvironmentType.DEVELOPMENT},
                              description="Enable auto-reload"),
        ]

        self.schemas["development"] = EnvironmentSchema(
            "development", EnvironmentType.DEVELOPMENT, dev_variables
        )

        # Testing environment schema
        test_variables = [var for var in dev_variables if var.key != "DEBUG"] + [
            EnvironmentVariable("TEST_DATABASE_URL", "", VariableType.DATABASE,
                              environments={EnvironmentType.TESTING},
                              encrypted=True, description="Test database URL"),
        ]

        self.schemas["testing"] = EnvironmentSchema(
            "testing", EnvironmentType.TESTING, test_variables,
            inherits_from="development"
        )

        # Staging environment schema
        staging_variables = [var for var in base_variables + database_variables + api_variables + service_variables] + [
            EnvironmentVariable("JWT_SECRET", "", VariableType.SECRET,
                              environments={EnvironmentType.STAGING, EnvironmentType.PRODUCTION},
                              encrypted=True, description="JWT signing secret"),
        ]

        self.schemas["staging"] = EnvironmentSchema(
            "staging", EnvironmentType.STAGING, staging_variables
        )

        # Production environment schema
        prod_variables = staging_variables + [
            EnvironmentVariable("SENTRY_DSN", "", VariableType.SERVICE,
                              required=False, encrypted=True,
                              environments={EnvironmentType.PRODUCTION},
                              description="Sentry error tracking DSN"),
            EnvironmentVariable("WORKERS", "2", VariableType.CONFIG,
                              environments={EnvironmentType.PRODUCTION},
                              description="Number of worker processes"),
            EnvironmentVariable("MAX_REQUESTS", "1000", VariableType.CONFIG,
                              environments={EnvironmentType.PRODUCTION},
                              required=False, description="Max requests per worker"),
        ]

        self.schemas["production"] = EnvironmentSchema(
            "production", EnvironmentType.PRODUCTION, prod_variables,
            inherits_from="staging"
        )

    def create_environment_file(self, environment: EnvironmentType,
                              template: bool = False) -> bool:
        """
        Create environment file for specific environment.

        Args:
            environment: Environment type
            template: Whether to create template with empty values

        Returns:
            True if created successfully
        """
        try:
            env_name = environment.value
            schema = self.schemas.get(env_name)

            if not schema:
                logger.error(f"No schema found for environment: {env_name}")
                return False

            env_file = self.config_dir / f".env.{env_name}"

            with open(env_file, 'w') as f:
                f.write(f"# Environment configuration for {env_name.upper()}\n")
                f.write(f"# Generated by SRA Data Processing Environment Manager\n\n")

                f.write("# System Configuration\n")
                f.write(f"ENVIRONMENT={env_name}\n\n")

                # Group variables by type
                by_type = {}
                for var in schema.variables:
                    if var.var_type not in by_type:
                        by_type[var.var_type] = []
                    by_type[var.var_type].append(var)

                # Write variables by type
                for var_type in VariableType:
                    if var_type not in by_type:
                        continue

                    f.write(f"# {var_type.value.replace('_', ' ').title()} Variables\n")

                    for var in by_type[var_type]:
                        # Skip if not applicable to this environment
                        if var.environments and environment not in var.environments:
                            continue

                        # Add description as comment
                        if var.description:
                            f.write(f"# {var.description}\n")

                        # Write variable
                        value = "" if template else var.value

                        # Encrypt secrets if enabled
                        if (not template and value and var.encrypted and
                            self.encryption_enabled and self.encryption_manager):
                            try:
                                value = self.encryption_manager.encrypt_value(value)
                                f.write(f"# ENCRYPTED: {var.key}\n")
                            except Exception as e:
                                logger.warning(f"Failed to encrypt {var.key}: {e}")

                        required_marker = " # REQUIRED" if var.required else ""
                        f.write(f"{var.key}={value}{required_marker}\n")

                    f.write("\n")

            logger.info(f"Environment file created: {env_file}")
            return True

        except Exception as e:
            logger.error(f"Failed to create environment file: {e}")
            return False

    def load_environment(self, environment: Optional[EnvironmentType] = None) -> bool:
        """
        Load environment configuration.

        Args:
            environment: Environment to load (uses current if None)

        Returns:
            True if loaded successfully
        """
        try:
            env = environment or self.current_environment
            env_name = env.value

            # Load from multiple sources in order of precedence
            sources = [
                self.config_dir / f".env.{env_name}",  # Environment-specific
                self.config_dir / ".env.local",        # Local overrides
                self.config_dir / ".env"               # Base configuration
            ]

            loaded_vars = {}

            for source in sources:
                if source.exists():
                    env_vars = self._load_env_file(source)
                    loaded_vars.update(env_vars)
                    logger.debug(f"Loaded {len(env_vars)} variables from {source}")

            # Apply to environment
            for key, value in loaded_vars.items():
                # Decrypt if encrypted
                if (self.encryption_enabled and self.encryption_manager and
                    value and self.encryption_manager.is_encrypted(value)):
                    try:
                        value = self.encryption_manager.decrypt_value(value)
                    except Exception as e:
                        logger.warning(f"Failed to decrypt {key}: {e}")

                os.environ[key] = value
                self.configuration[key] = value

            logger.info(f"Loaded environment configuration for {env_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load environment: {e}")
            return False

    def _load_env_file(self, file_path: Path) -> Dict[str, str]:
        """
        Load variables from .env file.

        Args:
            file_path: Path to .env file

        Returns:
            Dictionary of environment variables
        """
        variables = {}

        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    # Parse KEY=VALUE
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()

                        # Remove quotes if present
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]

                        variables[key] = value
                    else:
                        logger.warning(f"Invalid line in {file_path}:{line_num}: {line}")

        except Exception as e:
            logger.error(f"Failed to load env file {file_path}: {e}")

        return variables

    def validate_environment(self, environment: Optional[EnvironmentType] = None) -> ValidationResult:
        """
        Validate current environment configuration.

        Args:
            environment: Environment to validate (uses current if None)

        Returns:
            Validation result
        """
        env = environment or self.current_environment
        schema = self.schemas.get(env.value)

        if not schema:
            return ValidationResult(
                is_valid=False,
                missing_variables=[],
                invalid_variables=[],
                warnings=[],
                errors=[f"No schema found for environment: {env.value}"]
            )

        missing_vars = []
        invalid_vars = []
        warnings = []
        errors = []

        # Check required variables
        for var in schema.variables:
            # Skip if not applicable to this environment
            if var.environments and env not in var.environments:
                continue

            current_value = os.getenv(var.key)

            if var.required and not current_value:
                missing_vars.append(var.key)
            elif current_value:
                # Validate specific variable types
                if var.var_type == VariableType.DATABASE:
                    if not self._validate_database_url(current_value):
                        invalid_vars.append(var.key)
                elif var.var_type == VariableType.SECRET:
                    if len(current_value) < 32:
                        warnings.append(f"{var.key} appears to be a weak secret")

        # Check for unexpected variables
        current_vars = set(os.environ.keys())
        expected_vars = {var.key for var in schema.variables
                        if not var.environments or env in var.environments}

        unexpected_vars = current_vars - expected_vars - {
            'PATH', 'HOME', 'USER', 'SHELL', 'PWD', 'OLDPWD', 'TERM',
            'PYTHON_VERSION', 'PIP_VERSION', 'VIRTUAL_ENV'
        }

        if unexpected_vars:
            warnings.append(f"Unexpected environment variables: {', '.join(sorted(unexpected_vars))}")

        is_valid = not missing_vars and not invalid_vars and not errors

        return ValidationResult(
            is_valid=is_valid,
            missing_variables=missing_vars,
            invalid_variables=invalid_vars,
            warnings=warnings,
            errors=errors
        )

    def _validate_database_url(self, url: str) -> bool:
        """
        Validate database URL format.

        Args:
            url: Database URL to validate

        Returns:
            True if valid
        """
        # Basic validation - should start with supported scheme
        valid_schemes = ['postgresql://', 'postgres://', 'sqlite:///', 'mysql://']
        return any(url.startswith(scheme) for scheme in valid_schemes)

    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get summary of current configuration.

        Returns:
            Configuration summary
        """
        schema = self.schemas.get(self.current_environment.value)

        summary = {
            "environment": self.current_environment.value,
            "encryption_enabled": self.encryption_enabled,
            "schema_loaded": schema is not None,
            "variables_loaded": len(self.configuration),
            "configuration_status": "unknown"
        }

        if schema:
            applicable_vars = [
                var for var in schema.variables
                if not var.environments or self.current_environment in var.environments
            ]

            summary.update({
                "total_variables": len(applicable_vars),
                "required_variables": len([var for var in applicable_vars if var.required]),
                "secret_variables": len([var for var in applicable_vars
                                       if var.var_type == VariableType.SECRET]),
                "variable_types": {
                    vtype.value: len([var for var in applicable_vars
                                    if var.var_type == vtype])
                    for vtype in VariableType
                }
            })

            # Validate current configuration
            validation = self.validate_environment()
            summary["configuration_status"] = "valid" if validation.is_valid else "invalid"
            summary["validation_summary"] = {
                "missing_variables": len(validation.missing_variables),
                "invalid_variables": len(validation.invalid_variables),
                "warnings": len(validation.warnings),
                "errors": len(validation.errors)
            }

        return summary

    def export_configuration(self, output_path: str,
                           include_secrets: bool = False) -> bool:
        """
        Export current configuration to file.

        Args:
            output_path: Output file path
            include_secrets: Whether to include secret values

        Returns:
            True if exported successfully
        """
        try:
            config_data = {
                "environment": self.current_environment.value,
                "timestamp": None,
                "configuration": {}
            }

            schema = self.schemas.get(self.current_environment.value)
            if not schema:
                logger.error(f"No schema for environment: {self.current_environment.value}")
                return False

            for var in schema.variables:
                # Skip if not applicable to current environment
                if var.environments and self.current_environment not in var.environments:
                    continue

                value = os.getenv(var.key, "")

                # Handle secrets
                if var.var_type == VariableType.SECRET and not include_secrets:
                    value = "[REDACTED]" if value else "[NOT_SET]"

                config_data["configuration"][var.key] = {
                    "value": value,
                    "type": var.var_type.value,
                    "required": var.required,
                    "description": var.description,
                    "encrypted": var.encrypted
                }

            with open(output_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            logger.info(f"Configuration exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return False

    def rotate_secrets(self) -> bool:
        """
        Rotate encrypted secrets with new encryption key.

        Returns:
            True if rotated successfully
        """
        if not self.encryption_enabled or not self.encryption_manager:
            logger.warning("Encryption not enabled, cannot rotate secrets")
            return False

        try:
            # Create new encryption manager with new key
            new_master_key = base64.urlsafe_b64encode(os.urandom(32)).decode()
            new_encryption_manager = EncryptionManager(new_master_key)

            schema = self.schemas.get(self.current_environment.value)
            if not schema:
                return False

            rotated_count = 0

            for var in schema.variables:
                if not var.encrypted:
                    continue

                current_value = os.getenv(var.key)
                if not current_value:
                    continue

                # Decrypt with old key and encrypt with new key
                if self.encryption_manager.is_encrypted(current_value):
                    try:
                        decrypted = self.encryption_manager.decrypt_value(current_value)
                        new_encrypted = new_encryption_manager.encrypt_value(decrypted)

                        # Update environment variable
                        os.environ[var.key] = new_encrypted
                        self.configuration[var.key] = new_encrypted

                        rotated_count += 1
                    except Exception as e:
                        logger.error(f"Failed to rotate secret {var.key}: {e}")

            # Update encryption manager
            self.encryption_manager = new_encryption_manager

            logger.info(f"Rotated {rotated_count} secrets")
            return True

        except Exception as e:
            logger.error(f"Failed to rotate secrets: {e}")
            return False


# Example usage and testing functions
def main():
    """Example usage of EnvironmentManager."""
    # Initialize environment manager
    env_manager = EnvironmentManager(config_dir="config")

    print(f"Current environment: {env_manager.current_environment.value}")

    # Create environment files
    print("\\nCreating environment configuration files...")
    for env_type in EnvironmentType:
        if env_manager.create_environment_file(env_type, template=True):
            print(f"✅ Created template for {env_type.value}")

    # Load current environment
    print("\\nLoading environment configuration...")
    if env_manager.load_environment():
        print("✅ Environment loaded successfully")

    # Validate configuration
    print("\\nValidating environment configuration...")
    validation = env_manager.validate_environment()

    if validation.is_valid:
        print("✅ Environment configuration is valid")
    else:
        print("❌ Environment configuration has issues:")
        for error in validation.errors:
            print(f"  Error: {error}")
        for missing in validation.missing_variables:
            print(f"  Missing: {missing}")
        for invalid in validation.invalid_variables:
            print(f"  Invalid: {invalid}")

    if validation.warnings:
        print("⚠️ Warnings:")
        for warning in validation.warnings:
            print(f"  - {warning}")

    # Get configuration summary
    print("\\nConfiguration summary:")
    summary = env_manager.get_configuration_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Export configuration
    print("\\nExporting configuration...")
    if env_manager.export_configuration("current_config.json"):
        print("✅ Configuration exported to current_config.json")


if __name__ == "__main__":
    main()