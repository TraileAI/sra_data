"""
Comprehensive tests for Git LFS Manager.

Tests cover:
- LFS initialization and configuration
- File tracking and management
- Repository optimization
- Health monitoring
- Status reporting
- Error handling and edge cases
"""

import pytest
import tempfile
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add packages to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'packages'))

from sra_data.infrastructure.git_lfs_manager import GitLFSManager, LFSFileInfo, LFSStats


class TestGitLFSManager:
    """Test suite for GitLFSManager class."""

    @pytest.fixture
    def temp_repo(self):
        """Create a temporary git repository for testing."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True)

        yield repo_path

        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def lfs_manager(self, temp_repo):
        """Create GitLFSManager instance for testing."""
        return GitLFSManager(repo_path=str(temp_repo))

    @pytest.fixture
    def sample_csv_file(self, temp_repo):
        """Create a sample CSV file for testing."""
        csv_file = temp_repo / "test_data.csv"
        content = "col1,col2,col3\\n" + "data,data,data\\n" * 1000  # Make it reasonably large
        csv_file.write_text(content)
        return csv_file

    def test_initialization(self, lfs_manager, temp_repo):
        """Test GitLFSManager initialization."""
        assert lfs_manager.repo_path == temp_repo
        assert lfs_manager.lfs_config_path == temp_repo / ".gitattributes"
        assert "*.csv" in lfs_manager.lfs_patterns
        assert lfs_manager.size_threshold_mb == 10

    def test_is_lfs_initialized_false(self, lfs_manager):
        """Test LFS initialization check when not initialized."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(1, "git lfs version")
            assert not lfs_manager._is_lfs_initialized()

    def test_is_lfs_initialized_true(self, lfs_manager):
        """Test LFS initialization check when initialized."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert lfs_manager._is_lfs_initialized()

    @patch("subprocess.run")
    def test_initialize_lfs_success(self, mock_run, lfs_manager):
        """Test successful LFS initialization."""
        # Mock LFS not initialized initially
        mock_run.side_effect = [
            subprocess.CalledProcessError(1, "git lfs version"),  # Not initialized
            MagicMock(returncode=0),  # Install success
        ]

        with patch.object(lfs_manager, '_configure_lfs_tracking'):
            result = lfs_manager.initialize_lfs()
            assert result is True

    @patch("subprocess.run")
    def test_initialize_lfs_already_initialized(self, mock_run, lfs_manager):
        """Test LFS initialization when already initialized."""
        mock_run.return_value = MagicMock(returncode=0)

        result = lfs_manager.initialize_lfs()
        assert result is True

    def test_configure_lfs_tracking_new_file(self, lfs_manager, temp_repo):
        """Test configuring LFS tracking patterns in new .gitattributes."""
        lfs_manager._configure_lfs_tracking()

        assert lfs_manager.lfs_config_path.exists()
        content = lfs_manager.lfs_config_path.read_text()
        assert "*.csv filter=lfs diff=lfs merge=lfs -text" in content

    def test_configure_lfs_tracking_existing_file(self, lfs_manager, temp_repo):
        """Test configuring LFS tracking patterns in existing .gitattributes."""
        # Create existing .gitattributes with some content
        lfs_manager.lfs_config_path.write_text("*.txt filter=lfs diff=lfs merge=lfs -text\\n")

        lfs_manager._configure_lfs_tracking()

        content = lfs_manager.lfs_config_path.read_text()
        assert "*.txt filter=lfs diff=lfs merge=lfs -text" in content
        assert "*.csv filter=lfs diff=lfs merge=lfs -text" in content

    def test_find_large_files_empty(self, lfs_manager):
        """Test finding large files when none exist."""
        large_files = lfs_manager._find_large_files()
        assert large_files == []

    def test_find_large_files_with_large_csv(self, lfs_manager, temp_repo):
        """Test finding large CSV files."""
        # Create a large CSV file (>10MB)
        large_csv = temp_repo / "large_data.csv"
        content = "col1,col2,col3\\n" + "data,data,data\\n" * 500000  # Make it large
        large_csv.write_text(content)

        # Mock os.path.getsize to return a large size
        with patch("os.path.getsize", return_value=15 * 1024 * 1024):  # 15MB
            large_files = lfs_manager._find_large_files()
            assert len(large_files) > 0
            assert str(large_csv) in large_files

    def test_is_in_repo_true(self, lfs_manager, temp_repo):
        """Test checking if file is in repository."""
        test_file = temp_repo / "test.csv"
        test_file.write_text("test")
        assert lfs_manager._is_in_repo(str(test_file))

    def test_is_in_repo_false_outside(self, lfs_manager):
        """Test checking if file outside repository."""
        assert not lfs_manager._is_in_repo("/tmp/test.csv")

    def test_is_in_repo_false_git_dir(self, lfs_manager, temp_repo):
        """Test checking if file in .git directory."""
        git_file = temp_repo / ".git" / "config"
        assert not lfs_manager._is_in_repo(str(git_file))

    @patch("subprocess.run")
    def test_is_already_tracked_true(self, mock_run, lfs_manager):
        """Test checking if file is already tracked by LFS."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test.csv: filter: lfs"
        )
        assert lfs_manager._is_already_tracked("test.csv")

    @patch("subprocess.run")
    def test_is_already_tracked_false(self, mock_run, lfs_manager):
        """Test checking if file is not tracked by LFS."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="test.csv: filter: unspecified"
        )
        assert not lfs_manager._is_already_tracked("test.csv")

    @patch("subprocess.run")
    def test_track_file_with_lfs_success(self, mock_run, lfs_manager, temp_repo):
        """Test tracking file with LFS successfully."""
        test_file = temp_repo / "test.csv"
        test_file.write_text("test")

        mock_run.return_value = MagicMock(returncode=0)
        result = lfs_manager._track_file_with_lfs(str(test_file))
        assert result is True

    @patch("subprocess.run")
    def test_track_file_with_lfs_failure(self, mock_run, lfs_manager, temp_repo):
        """Test tracking file with LFS failure."""
        test_file = temp_repo / "test.csv"
        test_file.write_text("test")

        mock_run.side_effect = subprocess.CalledProcessError(1, "git lfs track")
        result = lfs_manager._track_file_with_lfs(str(test_file))
        assert result is False

    def test_get_tracked_patterns_empty(self, lfs_manager):
        """Test getting tracked patterns from empty .gitattributes."""
        patterns = lfs_manager._get_tracked_patterns()
        assert patterns == []

    def test_get_tracked_patterns_with_content(self, lfs_manager, temp_repo):
        """Test getting tracked patterns from .gitattributes with content."""
        content = """*.csv filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
# Comment line
"""
        lfs_manager.lfs_config_path.write_text(content)

        patterns = lfs_manager._get_tracked_patterns()
        assert "*.csv" in patterns
        assert "*.parquet" in patterns
        assert len(patterns) == 2

    @patch("subprocess.run")
    def test_get_lfs_files_empty(self, mock_run, lfs_manager):
        """Test getting LFS files when none exist."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=""
        )

        lfs_files = lfs_manager._get_lfs_files()
        assert lfs_files == []

    @patch("subprocess.run")
    def test_get_lfs_files_with_data(self, mock_run, lfs_manager):
        """Test getting LFS files with data."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="abc123def 1024 data.csv\\nxyz789uvw 2048 large.parquet\\n"
        )

        lfs_files = lfs_manager._get_lfs_files()
        assert len(lfs_files) == 2

        assert lfs_files[0].path == "data.csv"
        assert lfs_files[0].size == 1024
        assert lfs_files[0].lfs_oid == "abc123def"
        assert lfs_files[0].is_lfs is True

        assert lfs_files[1].path == "large.parquet"
        assert lfs_files[1].size == 2048

    def test_get_repository_stats_empty(self, lfs_manager):
        """Test getting repository stats when no LFS files exist."""
        with patch.object(lfs_manager, '_get_lfs_files', return_value=[]):
            stats = lfs_manager._get_repository_stats()

            assert isinstance(stats, LFSStats)
            assert stats.total_lfs_files == 0
            assert stats.total_lfs_size == 0
            assert stats.largest_file is None
            assert stats.largest_size == 0
            assert stats.average_file_size == 0.0

    def test_get_repository_stats_with_files(self, lfs_manager):
        """Test getting repository stats with LFS files."""
        mock_files = [
            LFSFileInfo("file1.csv", 1000, True, "oid1"),
            LFSFileInfo("file2.csv", 2000, True, "oid2"),
            LFSFileInfo("file3.csv", 3000, True, "oid3")
        ]

        with patch.object(lfs_manager, '_get_lfs_files', return_value=mock_files):
            stats = lfs_manager._get_repository_stats()

            assert stats.total_lfs_files == 3
            assert stats.total_lfs_size == 6000
            assert stats.largest_file == "file3.csv"
            assert stats.largest_size == 3000
            assert stats.average_file_size == 2000.0

    def test_calculate_health_score_minimal(self, lfs_manager):
        """Test calculating health score with minimal setup."""
        status = {"is_initialized": False}
        score = lfs_manager._calculate_health_score(status)
        assert score == 0.0

    def test_calculate_health_score_full(self, lfs_manager):
        """Test calculating health score with full setup."""
        mock_stats = LFSStats(
            total_lfs_files=5,
            total_lfs_size=50000,
            largest_file="test.csv",
            largest_size=10000,
            average_file_size=10000.0,
            compression_ratio=0.8
        )

        status = {
            "is_initialized": True,
            "tracked_patterns": ["*.csv"],
            "lfs_files": [LFSFileInfo("test.csv", 1000, True)],
            "repository_stats": mock_stats
        }

        score = lfs_manager._calculate_health_score(status)
        assert score == 1.0

    def test_get_lfs_status_success(self, lfs_manager):
        """Test getting LFS status successfully."""
        with patch.object(lfs_manager, '_is_lfs_initialized', return_value=True), \\
             patch.object(lfs_manager, '_get_tracked_patterns', return_value=["*.csv"]), \\
             patch.object(lfs_manager, '_get_lfs_files', return_value=[]), \\
             patch.object(lfs_manager, '_get_repository_stats') as mock_stats, \\
             patch.object(lfs_manager, '_get_storage_usage', return_value={}):

            mock_stats.return_value = LFSStats(0, 0, None, 0, 0.0, 0.0)

            status = lfs_manager.get_lfs_status()

            assert "is_initialized" in status
            assert "tracked_patterns" in status
            assert "lfs_files" in status
            assert "health_score" in status
            assert status["is_initialized"] is True

    def test_get_lfs_status_error(self, lfs_manager):
        """Test getting LFS status with error."""
        with patch.object(lfs_manager, '_is_lfs_initialized', side_effect=Exception("Test error")):
            status = lfs_manager.get_lfs_status()
            assert "error" in status

    @patch("subprocess.run")
    def test_cleanup_lfs_cache_success(self, mock_run, lfs_manager):
        """Test LFS cache cleanup success."""
        mock_run.return_value = MagicMock(returncode=0)
        result = lfs_manager._cleanup_lfs_cache()
        assert result is True

    @patch("subprocess.run")
    def test_cleanup_lfs_cache_failure(self, mock_run, lfs_manager):
        """Test LFS cache cleanup failure."""
        mock_run.side_effect = subprocess.CalledProcessError(1, "git lfs prune")
        result = lfs_manager._cleanup_lfs_cache()
        assert result is False

    def test_optimize_gitattributes_no_file(self, lfs_manager):
        """Test optimizing .gitattributes when file doesn't exist."""
        result = lfs_manager._optimize_gitattributes()
        assert result is False

    def test_optimize_gitattributes_with_duplicates(self, lfs_manager, temp_repo):
        """Test optimizing .gitattributes with duplicate patterns."""
        content = """*.csv filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
# Comment
"""
        lfs_manager.lfs_config_path.write_text(content)

        result = lfs_manager._optimize_gitattributes()
        assert result is True

        # Check that duplicates were removed
        optimized_content = lfs_manager.lfs_config_path.read_text()
        csv_count = optimized_content.count("*.csv filter=lfs")
        assert csv_count == 1

    def test_generate_recommendations_no_lfs_files(self, lfs_manager):
        """Test generating recommendations when no LFS files exist."""
        mock_status = {
            "repository_stats": LFSStats(0, 0, None, 0, 0.0, 0.0)
        }

        with patch.object(lfs_manager, 'get_lfs_status', return_value=mock_status):
            recommendations = lfs_manager._generate_recommendations()
            assert any("No LFS files found" in rec for rec in recommendations)

    def test_generate_recommendations_large_files(self, lfs_manager):
        """Test generating recommendations for large files."""
        mock_status = {
            "repository_stats": LFSStats(
                total_lfs_files=5,
                total_lfs_size=2 * 1024 * 1024 * 1024,  # 2GB
                largest_file="large.csv",
                largest_size=500 * 1024 * 1024,  # 500MB
                average_file_size=200 * 1024 * 1024,  # 200MB average
                compression_ratio=0.8
            )
        }

        with patch.object(lfs_manager, 'get_lfs_status', return_value=mock_status):
            recommendations = lfs_manager._generate_recommendations()
            assert any("Average file size is large" in rec for rec in recommendations)
            assert any("Large LFS storage usage" in rec for rec in recommendations)

    def test_health_check_all_good(self, lfs_manager):
        """Test health check when everything is good."""
        with patch.object(lfs_manager, '_is_lfs_initialized', return_value=True), \\
             patch.object(lfs_manager, '_find_large_files', return_value=[]), \\
             patch.object(lfs_manager, '_is_already_tracked', return_value=True):

            # Mock .gitattributes exists
            lfs_manager.lfs_config_path.write_text("*.csv filter=lfs diff=lfs merge=lfs -text\\n")

            health = lfs_manager.health_check()

            assert health["overall_health"] == "excellent"
            assert health["score"] == 1.0
            assert len(health["issues"]) == 0

    def test_health_check_lfs_not_initialized(self, lfs_manager):
        """Test health check when LFS not initialized."""
        with patch.object(lfs_manager, '_is_lfs_initialized', return_value=False):
            health = lfs_manager.health_check()

            assert "Git LFS not initialized" in health["issues"]
            assert "Run git lfs install" in health["recommendations"]

    def test_health_check_missing_gitattributes(self, lfs_manager):
        """Test health check when .gitattributes is missing."""
        with patch.object(lfs_manager, '_is_lfs_initialized', return_value=True), \\
             patch.object(lfs_manager, '_find_large_files', return_value=[]):

            health = lfs_manager.health_check()

            assert ".gitattributes file missing" in health["issues"]
            assert "Create .gitattributes" in health["recommendations"]

    def test_health_check_untracked_large_files(self, lfs_manager):
        """Test health check with untracked large files."""
        with patch.object(lfs_manager, '_is_lfs_initialized', return_value=True), \\
             patch.object(lfs_manager, '_find_large_files', return_value=["large.csv"]), \\
             patch.object(lfs_manager, '_is_already_tracked', return_value=False):

            # Mock .gitattributes exists
            lfs_manager.lfs_config_path.write_text("*.csv filter=lfs diff=lfs merge=lfs -text\\n")

            health = lfs_manager.health_check()

            assert "large files not tracked by LFS" in health["issues"][0]
            assert "Track large files with Git LFS" in health["recommendations"]

    def test_health_check_error(self, lfs_manager):
        """Test health check with error."""
        with patch.object(lfs_manager, '_is_lfs_initialized', side_effect=Exception("Test error")):
            health = lfs_manager.health_check()

            assert health["overall_health"] == "error"
            assert "Health check error" in health["issues"][0]

    def test_track_large_files_success(self, lfs_manager, temp_repo):
        """Test tracking large files successfully."""
        # Create a large file
        large_file = temp_repo / "large.csv"
        large_file.write_text("test data")

        with patch.object(lfs_manager, '_find_large_files', return_value=[str(large_file)]), \\
             patch.object(lfs_manager, '_should_track_file', return_value=True), \\
             patch.object(lfs_manager, '_track_file_with_lfs', return_value=True):

            tracked = lfs_manager.track_large_files()
            assert str(large_file) in tracked

    def test_track_large_files_with_failures(self, lfs_manager, temp_repo):
        """Test tracking large files with some failures."""
        large_file1 = temp_repo / "large1.csv"
        large_file2 = temp_repo / "large2.csv"

        with patch.object(lfs_manager, '_find_large_files', return_value=[str(large_file1), str(large_file2)]), \\
             patch.object(lfs_manager, '_should_track_file', return_value=True), \\
             patch.object(lfs_manager, '_track_file_with_lfs', side_effect=[True, False]):

            tracked = lfs_manager.track_large_files()
            assert len(tracked) == 1
            assert str(large_file1) in tracked

    def test_migrate_existing_files_success(self, lfs_manager, temp_repo):
        """Test migrating existing files to LFS."""
        # Create files
        file1 = temp_repo / "file1.csv"
        file2 = temp_repo / "file2.csv"
        file1.write_text("data" * 1000)
        file2.write_text("data" * 2000)

        with patch.object(lfs_manager, '_find_large_files', return_value=[str(file1), str(file2)]), \\
             patch.object(lfs_manager, '_is_already_tracked', return_value=False), \\
             patch.object(lfs_manager, '_track_file_with_lfs', return_value=True):

            result = lfs_manager.migrate_existing_files()

            assert len(result["migrated_files"]) == 2
            assert str(file1) in result["migrated_files"]
            assert str(file2) in result["migrated_files"]
            assert result["total_size_migrated"] > 0

    def test_migrate_existing_files_with_failures(self, lfs_manager, temp_repo):
        """Test migrating existing files with failures."""
        file1 = temp_repo / "file1.csv"
        file1.write_text("data")

        with patch.object(lfs_manager, '_find_large_files', return_value=[str(file1)]), \\
             patch.object(lfs_manager, '_is_already_tracked', return_value=False), \\
             patch.object(lfs_manager, '_track_file_with_lfs', return_value=False):

            result = lfs_manager.migrate_existing_files()

            assert len(result["failed_files"]) == 1
            assert str(file1) in result["failed_files"]

    def test_optimize_repository_success(self, lfs_manager):
        """Test repository optimization success."""
        with patch.object(lfs_manager, '_cleanup_lfs_cache', return_value=True), \\
             patch.object(lfs_manager, '_prune_lfs_objects', return_value=True), \\
             patch.object(lfs_manager, '_optimize_gitattributes', return_value=True), \\
             patch.object(lfs_manager, '_generate_recommendations', return_value=["test rec"]):

            result = lfs_manager.optimize_repository()

            assert "LFS cache cleanup" in result["actions_taken"]
            assert "LFS object pruning" in result["actions_taken"]
            assert ".gitattributes optimization" in result["actions_taken"]
            assert len(result["recommendations"]) > 0

    def test_optimize_repository_error(self, lfs_manager):
        """Test repository optimization with error."""
        with patch.object(lfs_manager, '_cleanup_lfs_cache', side_effect=Exception("Test error")):
            result = lfs_manager.optimize_repository()
            assert "error" in result


class TestLFSDataClasses:
    """Test LFS-related data classes."""

    def test_lfs_file_info_creation(self):
        """Test LFSFileInfo creation."""
        file_info = LFSFileInfo(
            path="test.csv",
            size=1024,
            is_lfs=True,
            lfs_oid="abc123",
            compressed_size=512
        )

        assert file_info.path == "test.csv"
        assert file_info.size == 1024
        assert file_info.is_lfs is True
        assert file_info.lfs_oid == "abc123"
        assert file_info.compressed_size == 512

    def test_lfs_stats_creation(self):
        """Test LFSStats creation."""
        stats = LFSStats(
            total_lfs_files=5,
            total_lfs_size=10240,
            largest_file="large.csv",
            largest_size=5120,
            average_file_size=2048.0,
            compression_ratio=0.8
        )

        assert stats.total_lfs_files == 5
        assert stats.total_lfs_size == 10240
        assert stats.largest_file == "large.csv"
        assert stats.largest_size == 5120
        assert stats.average_file_size == 2048.0
        assert stats.compression_ratio == 0.8


# Integration tests
class TestGitLFSIntegration:
    """Integration tests for Git LFS Manager."""

    def test_full_lfs_workflow(self, temp_repo):
        """Test complete LFS workflow from initialization to health check."""
        lfs_manager = GitLFSManager(str(temp_repo))

        # Create a mock large CSV file
        large_csv = temp_repo / "large_data.csv"
        large_csv.write_text("col1,col2\\n" + "data,data\\n" * 1000)

        # Mock the file as large
        with patch("os.path.getsize", return_value=15 * 1024 * 1024):  # 15MB
            # Mock LFS commands to avoid actual Git LFS dependency in tests
            with patch("subprocess.run") as mock_run:
                # Configure mock responses
                mock_run.side_effect = [
                    subprocess.CalledProcessError(1, "git lfs version"),  # Not initialized
                    MagicMock(returncode=0),  # Install success
                    MagicMock(returncode=0),  # Track success
                ]

                # Initialize and track files
                init_success = lfs_manager.initialize_lfs()
                assert init_success

                # Track large files
                with patch.object(lfs_manager, '_is_already_tracked', return_value=False):
                    tracked_files = lfs_manager.track_large_files()
                    # Would track if not mocked

                # Health check
                with patch.object(lfs_manager, '_is_lfs_initialized', return_value=True):
                    health = lfs_manager.health_check()
                    # Health check would pass basic checks
                    assert "checks" in health

    def test_error_handling_robustness(self, temp_repo):
        """Test error handling robustness."""
        lfs_manager = GitLFSManager(str(temp_repo))

        # Test with various error conditions
        with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "test")):
            # Should handle errors gracefully
            status = lfs_manager.get_lfs_status()
            assert isinstance(status, dict)

            health = lfs_manager.health_check()
            assert health["overall_health"] in ["poor", "error"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])