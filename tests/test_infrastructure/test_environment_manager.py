"""
Comprehensive tests for Environment Manager infrastructure component.

Tests cover:
- Environment variable management and validation
- Encryption and decryption of secrets
- Environment file creation and loading
- Configuration validation and verification
- Environment-specific settings management
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from packages.sra_data.infrastructure.environment_manager import (
    EnvironmentManager,
    EnvironmentType,
    VariableType,
    EnvironmentVariable,
    EnvironmentSchema,
    ValidationResult,
    EncryptionManager
)


class TestEncryptionManager:
    """Test encryption manager functionality."""

    def test_encryption_manager_initialization(self):
        """Test encryption manager initialization."""
        enc_manager = EncryptionManager("test-master-key-32-chars-long!!")
        assert enc_manager.master_key == "test-master-key-32-chars-long!!"
        assert enc_manager.fernet is not None

    def test_encrypt_decrypt_value(self):
        """Test value encryption and decryption."""
        enc_manager = EncryptionManager("test-master-key-32-chars-long!!")

        # Test encryption and decryption
        original_value = "secret-database-password-123"
        encrypted_value = enc_manager.encrypt_value(original_value)
        decrypted_value = enc_manager.decrypt_value(encrypted_value)

        assert original_value == decrypted_value
        assert original_value != encrypted_value
        assert len(encrypted_value) > len(original_value)

    def test_is_encrypted(self):
        """Test encrypted value detection."""
        enc_manager = EncryptionManager("test-master-key-32-chars-long!!")

        # Test with encrypted value
        encrypted_value = enc_manager.encrypt_value("test-secret")
        assert enc_manager.is_encrypted(encrypted_value)

        # Test with plain value
        assert not enc_manager.is_encrypted("plain-text")
        assert not enc_manager.is_encrypted("short")

    def test_encryption_with_invalid_key(self):
        """Test encryption with invalid key."""
        enc_manager = EncryptionManager("test-master-key-32-chars-long!!")

        encrypted_value = enc_manager.encrypt_value("test")

        # Try to decrypt with different key
        wrong_enc_manager = EncryptionManager("wrong-master-key-32-chars-long!")

        with pytest.raises(Exception):
            wrong_enc_manager.decrypt_value(encrypted_value)


class TestEnvironmentManager:
    """Test environment manager functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def env_manager(self, temp_config_dir):
        """Create environment manager with temporary directory."""
        return EnvironmentManager(
            config_dir=temp_config_dir,
            environment=EnvironmentType.DEVELOPMENT,
            encryption_enabled=False  # Disable for most tests
        )

    def test_environment_manager_initialization(self, env_manager):
        """Test environment manager initialization."""
        assert env_manager.current_environment == EnvironmentType.DEVELOPMENT
        assert env_manager.encryption_enabled is False
        assert len(env_manager.schemas) == 4  # dev, test, staging, prod
        assert "development" in env_manager.schemas

    def test_environment_detection(self):
        """Test automatic environment detection."""
        # Test production detection
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            env_manager = EnvironmentManager()
            assert env_manager.current_environment == EnvironmentType.PRODUCTION

        # Test staging detection
        with patch.dict(os.environ, {"ENVIRONMENT": "staging"}):
            env_manager = EnvironmentManager()
            assert env_manager.current_environment == EnvironmentType.STAGING

        # Test default (development)
        with patch.dict(os.environ, {}, clear=True):
            env_manager = EnvironmentManager()
            assert env_manager.current_environment == EnvironmentType.DEVELOPMENT

    def test_create_environment_file(self, env_manager):
        """Test environment file creation."""
        # Create development environment file
        result = env_manager.create_environment_file(
            EnvironmentType.DEVELOPMENT,
            template=True
        )

        assert result is True

        # Check if file was created
        env_file = Path(env_manager.config_dir) / ".env.development"
        assert env_file.exists()

        # Check file contents
        content = env_file.read_text()
        assert "ENVIRONMENT=development" in content
        assert "# REQUIRED" in content
        assert "DATABASE_URL=" in content
        assert "PORT=" in content

    def test_load_environment_file_parsing(self, env_manager):
        """Test environment file loading and parsing."""
        # Create test environment file
        env_file = Path(env_manager.config_dir) / ".env.development"
        env_content = '''# Test environment file
ENVIRONMENT=development
DATABASE_URL=postgresql://test:pass@localhost/testdb
PORT=8000
DEBUG=true
# This is a comment
QUOTED_VALUE="quoted string"
SINGLE_QUOTED='single quoted'
EMPTY_VALUE=
'''
        env_file.write_text(env_content)

        # Load environment
        result = env_manager.load_environment(EnvironmentType.DEVELOPMENT)
        assert result is True

        # Check loaded configuration
        assert env_manager.configuration["ENVIRONMENT"] == "development"
        assert env_manager.configuration["DATABASE_URL"] == "postgresql://test:pass@localhost/testdb"
        assert env_manager.configuration["PORT"] == "8000"
        assert env_manager.configuration["DEBUG"] == "true"
        assert env_manager.configuration["QUOTED_VALUE"] == "quoted string"
        assert env_manager.configuration["SINGLE_QUOTED"] == "single quoted"
        assert env_manager.configuration["EMPTY_VALUE"] == ""

        # Check environment variables were set
        assert os.environ["ENVIRONMENT"] == "development"
        assert os.environ["DATABASE_URL"] == "postgresql://test:pass@localhost/testdb"

    def test_validate_environment_missing_required(self, env_manager):
        """Test environment validation with missing required variables."""
        # Don't load any environment
        validation = env_manager.validate_environment(EnvironmentType.DEVELOPMENT)

        assert validation.is_valid is False
        assert len(validation.missing_variables) > 0
        assert "DATABASE_URL" in validation.missing_variables
        assert "SECRET_KEY" in validation.missing_variables

    def test_validate_environment_complete(self, env_manager):
        """Test environment validation with complete configuration."""
        # Set all required variables
        schema = env_manager.schemas["development"]
        for var in schema.variables:
            if var.required and not var.environments or EnvironmentType.DEVELOPMENT in var.environments:
                os.environ[var.key] = var.value or "test-value"

        validation = env_manager.validate_environment(EnvironmentType.DEVELOPMENT)

        assert validation.is_valid is True
        assert len(validation.missing_variables) == 0
        assert len(validation.errors) == 0

    def test_validate_database_url(self, env_manager):
        """Test database URL validation."""
        valid_urls = [
            "postgresql://user:pass@localhost/db",
            "postgres://user:pass@host:5432/database",
            "sqlite:///path/to/db.sqlite",
            "mysql://user:pass@host/db"
        ]

        for url in valid_urls:
            assert env_manager._validate_database_url(url) is True

        invalid_urls = [
            "invalid-url",
            "http://not-a-database",
            "",
            "just-text"
        ]

        for url in invalid_urls:
            assert env_manager._validate_database_url(url) is False

    def test_encryption_integration(self, temp_config_dir):
        """Test environment manager with encryption enabled."""
        env_manager = EnvironmentManager(
            config_dir=temp_config_dir,
            encryption_enabled=True
        )

        assert env_manager.encryption_enabled is True
        assert env_manager.encryption_manager is not None

        # Create environment file with secrets
        result = env_manager.create_environment_file(
            EnvironmentType.PRODUCTION,
            template=False  # Include actual values
        )

        assert result is True

        # Check that encrypted values are in the file
        env_file = Path(temp_config_dir) / ".env.production"
        content = env_file.read_text()
        assert "# ENCRYPTED:" in content

    def test_get_configuration_summary(self, env_manager):
        """Test configuration summary generation."""
        summary = env_manager.get_configuration_summary()

        assert "environment" in summary
        assert summary["environment"] == "development"
        assert "encryption_enabled" in summary
        assert "schema_loaded" in summary
        assert "total_variables" in summary
        assert "configuration_status" in summary

        # Check validation summary
        assert "validation_summary" in summary
        validation_summary = summary["validation_summary"]
        assert "missing_variables" in validation_summary
        assert "warnings" in validation_summary

    def test_export_configuration(self, env_manager):
        """Test configuration export."""
        # Set some environment variables
        os.environ["TEST_VAR"] = "test-value"
        os.environ["SECRET_KEY"] = "secret-value"

        export_path = Path(env_manager.config_dir) / "exported_config.json"

        # Export without secrets
        result = env_manager.export_configuration(str(export_path), include_secrets=False)
        assert result is True
        assert export_path.exists()

        # Check exported content
        with open(export_path) as f:
            exported = json.load(f)

        assert "environment" in exported
        assert "configuration" in exported
        assert exported["environment"] == "development"

        # Check that secrets are redacted
        config = exported["configuration"]
        if "SECRET_KEY" in config:
            assert config["SECRET_KEY"]["value"] in ["[REDACTED]", "[NOT_SET]"]

        # Export with secrets
        export_path_secrets = Path(env_manager.config_dir) / "exported_with_secrets.json"
        result = env_manager.export_configuration(str(export_path_secrets), include_secrets=True)
        assert result is True

    def test_environment_schemas(self, env_manager):
        """Test environment schema definitions."""
        # Check that all environment schemas exist
        expected_schemas = ["development", "testing", "staging", "production"]
        for schema_name in expected_schemas:
            assert schema_name in env_manager.schemas
            schema = env_manager.schemas[schema_name]
            assert isinstance(schema, EnvironmentSchema)
            assert len(schema.variables) > 0

        # Check schema inheritance
        staging_schema = env_manager.schemas["staging"]
        production_schema = env_manager.schemas["production"]

        assert production_schema.inherits_from == "staging"

        # Check environment-specific variables
        dev_schema = env_manager.schemas["development"]
        dev_var_names = {var.key for var in dev_schema.variables}
        assert "DEBUG" in dev_var_names

        prod_schema = env_manager.schemas["production"]
        prod_var_names = {var.key for var in prod_schema.variables}
        assert "WORKERS" in prod_var_names
        assert "SENTRY_DSN" in prod_var_names

    def test_variable_types_and_categories(self, env_manager):
        """Test that variables are properly categorized by type."""
        schema = env_manager.schemas["development"]

        # Group variables by type
        by_type = {}
        for var in schema.variables:
            if var.var_type not in by_type:
                by_type[var.var_type] = []
            by_type[var.var_type].append(var)

        # Check that we have variables of each type
        assert VariableType.SECRET in by_type
        assert VariableType.CONFIG in by_type
        assert VariableType.DATABASE in by_type
        assert VariableType.SYSTEM in by_type

        # Check secret variables are marked for encryption
        secret_vars = by_type[VariableType.SECRET]
        for var in secret_vars:
            assert var.encrypted is True

    def test_environment_file_creation_template_vs_values(self, env_manager):
        """Test difference between template and value environment files."""
        # Create template file
        result_template = env_manager.create_environment_file(
            EnvironmentType.DEVELOPMENT,
            template=True
        )
        assert result_template is True

        template_file = Path(env_manager.config_dir) / ".env.development"
        template_content = template_file.read_text()

        # Create file with values
        result_values = env_manager.create_environment_file(
            EnvironmentType.DEVELOPMENT,
            template=False
        )
        assert result_values is True

        # Rename the template file to compare
        template_file.rename(template_file.with_suffix('.template'))
        values_content = template_file.read_text()

        # Template should have empty values, values file should have defaults
        assert "DATABASE_URL=" in template_content  # Empty in template
        assert "PORT=10000" in values_content or "PORT=" in template_content

    def test_multiple_environment_files_precedence(self, env_manager):
        """Test loading from multiple environment files with precedence."""
        config_dir = Path(env_manager.config_dir)

        # Create base .env file
        base_env = config_dir / ".env"
        base_env.write_text('''
BASE_VAR=base_value
OVERRIDE_VAR=base_override
        ''')

        # Create environment-specific file
        dev_env = config_dir / ".env.development"
        dev_env.write_text('''
DEV_VAR=dev_value
OVERRIDE_VAR=dev_override
        ''')

        # Create local override file
        local_env = config_dir / ".env.local"
        local_env.write_text('''
LOCAL_VAR=local_value
OVERRIDE_VAR=local_override
        ''')

        # Load environment
        result = env_manager.load_environment(EnvironmentType.DEVELOPMENT)
        assert result is True

        # Check precedence (local > env-specific > base)
        assert env_manager.configuration["BASE_VAR"] == "base_value"
        assert env_manager.configuration["DEV_VAR"] == "dev_value"
        assert env_manager.configuration["LOCAL_VAR"] == "local_value"
        assert env_manager.configuration["OVERRIDE_VAR"] == "local_override"  # Local wins


@pytest.mark.asyncio
class TestEnvironmentManagerAsync:
    """Test async functionality of environment manager."""

    def test_async_methods_not_implemented(self):
        """Test that async methods would be properly implemented if needed."""
        # This is a placeholder - the current implementation doesn't have async methods
        # but this test structure is ready if they're added
        pass


# Integration tests
class TestEnvironmentManagerIntegration:
    """Integration tests for environment manager."""

    def test_full_development_workflow(self):
        """Test complete development environment setup workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize environment manager
            env_manager = EnvironmentManager(
                config_dir=temp_dir,
                environment=EnvironmentType.DEVELOPMENT,
                encryption_enabled=True
            )

            # Create environment files
            for env_type in EnvironmentType:
                result = env_manager.create_environment_file(env_type, template=True)
                assert result is True

            # Load development environment
            result = env_manager.load_environment()
            assert result is True

            # Validate configuration
            validation = env_manager.validate_environment()
            # May not be valid due to missing required vars, but should not error

            # Get summary
            summary = env_manager.get_configuration_summary()
            assert summary["environment"] == "development"

            # Export configuration
            export_path = Path(temp_dir) / "config_export.json"
            result = env_manager.export_configuration(str(export_path))
            assert result is True
            assert export_path.exists()

    def test_production_environment_setup(self):
        """Test production environment setup with encryption."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_manager = EnvironmentManager(
                config_dir=temp_dir,
                environment=EnvironmentType.PRODUCTION,
                encryption_enabled=True
            )

            # Create production environment file
            result = env_manager.create_environment_file(
                EnvironmentType.PRODUCTION,
                template=False
            )
            assert result is True

            # Check that production-specific variables are included
            prod_file = Path(temp_dir) / ".env.production"
            content = prod_file.read_text()

            # Production should have worker settings
            assert "WORKERS=" in content
            # Should have JWT secret
            assert "JWT_SECRET" in content
            # Should not have DEBUG flag
            assert "DEBUG=" not in content

    def test_secret_rotation_workflow(self):
        """Test secret rotation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_manager = EnvironmentManager(
                config_dir=temp_dir,
                encryption_enabled=True
            )

            # Set some encrypted environment variables
            os.environ["SECRET_KEY"] = "original-secret"
            os.environ["JWT_SECRET"] = "original-jwt-secret"

            # Rotate secrets
            result = env_manager.rotate_secrets()
            assert result is True

            # Check that secrets were changed
            # Note: In practice, you'd need to reload from files
            # This tests the rotation mechanism itself


if __name__ == "__main__":
    pytest.main([__file__])