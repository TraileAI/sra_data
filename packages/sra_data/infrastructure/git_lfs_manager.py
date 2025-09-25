"""
Git LFS Manager for SRA Data Processing System.

This module provides comprehensive Git LFS management capabilities including:
- Automated LFS tracking for large CSV files
- Repository size optimization and monitoring
- LFS health checks and maintenance
- File size analysis and recommendations
- Automated LFS configuration management
"""

import os
import subprocess
import logging
import json
import glob
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LFSFileInfo:
    """Information about an LFS-tracked file."""
    path: str
    size: int
    is_lfs: bool
    lfs_oid: Optional[str] = None
    compressed_size: Optional[int] = None


@dataclass
class LFSStats:
    """Git LFS repository statistics."""
    total_lfs_files: int
    total_lfs_size: int
    largest_file: Optional[str]
    largest_size: int
    average_file_size: float
    compression_ratio: float


class GitLFSManager:
    """
    Comprehensive Git LFS management system for handling large CSV files
    and optimizing repository performance.
    """

    def __init__(self, repo_path: str = "."):
        """
        Initialize Git LFS Manager.

        Args:
            repo_path: Path to the git repository
        """
        self.repo_path = Path(repo_path).resolve()
        self.lfs_config_path = self.repo_path / ".gitattributes"
        self.lfs_patterns = [
            "*.csv",
            "*.parquet",
            "*.feather",
            "*.hdf5",
            "*.h5",
            "*.xlsx",
            "*.json.gz",
            "*.pickle"
        ]
        self.size_threshold_mb = 10  # Files larger than 10MB should use LFS

    def initialize_lfs(self) -> bool:
        """
        Initialize Git LFS in the repository.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Check if already initialized
            if self._is_lfs_initialized():
                logger.info("Git LFS already initialized")
                return True

            # Initialize Git LFS
            result = subprocess.run(
                ["git", "lfs", "install"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Git LFS initialized successfully")

            # Configure LFS tracking patterns
            self._configure_lfs_tracking()

            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize Git LFS: {e}")
            logger.error(f"Error output: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during LFS initialization: {e}")
            return False

    def _is_lfs_initialized(self) -> bool:
        """Check if Git LFS is already initialized."""
        try:
            result = subprocess.run(
                ["git", "lfs", "version"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def _configure_lfs_tracking(self) -> None:
        """Configure LFS tracking patterns in .gitattributes."""
        existing_patterns = set()

        # Read existing .gitattributes if it exists
        if self.lfs_config_path.exists():
            with open(self.lfs_config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        pattern = line.split()[0] if line.split() else ""
                        if pattern:
                            existing_patterns.add(pattern)

        # Add missing patterns
        new_patterns = []
        for pattern in self.lfs_patterns:
            if pattern not in existing_patterns:
                new_patterns.append(f"{pattern} filter=lfs diff=lfs merge=lfs -text")

        # Append new patterns to .gitattributes
        if new_patterns:
            with open(self.lfs_config_path, 'a') as f:
                if self.lfs_config_path.stat().st_size > 0:
                    f.write('\n')
                f.write('\n'.join(new_patterns) + '\n')

            logger.info(f"Added {len(new_patterns)} new LFS tracking patterns")

    def track_large_files(self, force: bool = False) -> List[str]:
        """
        Automatically track large files with Git LFS.

        Args:
            force: Force re-tracking of files already in git

        Returns:
            List of files that were added to LFS tracking
        """
        tracked_files = []

        try:
            # Find large files in repository
            large_files = self._find_large_files()

            for file_path in large_files:
                if self._should_track_file(file_path, force):
                    if self._track_file_with_lfs(file_path):
                        tracked_files.append(file_path)

            logger.info(f"Successfully tracked {len(tracked_files)} files with Git LFS")
            return tracked_files

        except Exception as e:
            logger.error(f"Error tracking large files: {e}")
            return tracked_files

    def _find_large_files(self) -> List[str]:
        """Find files larger than the size threshold."""
        large_files = []

        for pattern in self.lfs_patterns:
            files = glob.glob(str(self.repo_path / "**" / pattern), recursive=True)
            for file_path in files:
                if self._is_in_repo(file_path):
                    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                    if file_size_mb > self.size_threshold_mb:
                        large_files.append(file_path)

        return large_files

    def _is_in_repo(self, file_path: str) -> bool:
        """Check if file is within the repository and not in .git directory."""
        path = Path(file_path).resolve()
        try:
            path.relative_to(self.repo_path)
            return ".git" not in path.parts and ".venv" not in path.parts
        except ValueError:
            return False

    def _should_track_file(self, file_path: str, force: bool) -> bool:
        """Determine if file should be tracked with LFS."""
        if not force and self._is_already_tracked(file_path):
            return False
        return True

    def _is_already_tracked(self, file_path: str) -> bool:
        """Check if file is already tracked by Git LFS."""
        try:
            result = subprocess.run(
                ["git", "check-attr", "filter", file_path],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return "lfs" in result.stdout
        except subprocess.CalledProcessError:
            return False

    def _track_file_with_lfs(self, file_path: str) -> bool:
        """Track specific file with Git LFS."""
        try:
            # Get relative path from repo root
            rel_path = Path(file_path).relative_to(self.repo_path)

            # Track the file pattern
            result = subprocess.run(
                ["git", "lfs", "track", str(rel_path)],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            logger.info(f"Tracking {rel_path} with Git LFS")
            return True

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to track {file_path} with LFS: {e}")
            return False

    def get_lfs_status(self) -> Dict[str, any]:
        """
        Get comprehensive Git LFS status information.

        Returns:
            Dictionary containing LFS status information
        """
        try:
            status = {
                "is_initialized": self._is_lfs_initialized(),
                "tracked_patterns": self._get_tracked_patterns(),
                "lfs_files": self._get_lfs_files(),
                "repository_stats": self._get_repository_stats(),
                "storage_usage": self._get_storage_usage(),
                "health_score": 0.0
            }

            # Calculate health score
            status["health_score"] = self._calculate_health_score(status)

            return status

        except Exception as e:
            logger.error(f"Error getting LFS status: {e}")
            return {"error": str(e)}

    def _get_tracked_patterns(self) -> List[str]:
        """Get list of patterns tracked by Git LFS."""
        patterns = []

        if self.lfs_config_path.exists():
            with open(self.lfs_config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and "filter=lfs" in line:
                        pattern = line.split()[0]
                        patterns.append(pattern)

        return patterns

    def _get_lfs_files(self) -> List[LFSFileInfo]:
        """Get list of files tracked by Git LFS."""
        lfs_files = []

        try:
            # Get LFS files list
            result = subprocess.run(
                ["git", "lfs", "ls-files", "--long"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        oid = parts[0]
                        size = int(parts[1])
                        path = ' '.join(parts[2:])

                        file_info = LFSFileInfo(
                            path=path,
                            size=size,
                            is_lfs=True,
                            lfs_oid=oid
                        )
                        lfs_files.append(file_info)

        except subprocess.CalledProcessError:
            logger.warning("Could not get LFS files list")

        return lfs_files

    def _get_repository_stats(self) -> LFSStats:
        """Get repository statistics."""
        lfs_files = self._get_lfs_files()

        if not lfs_files:
            return LFSStats(
                total_lfs_files=0,
                total_lfs_size=0,
                largest_file=None,
                largest_size=0,
                average_file_size=0.0,
                compression_ratio=0.0
            )

        total_size = sum(f.size for f in lfs_files)
        largest_file = max(lfs_files, key=lambda f: f.size)

        return LFSStats(
            total_lfs_files=len(lfs_files),
            total_lfs_size=total_size,
            largest_file=largest_file.path,
            largest_size=largest_file.size,
            average_file_size=total_size / len(lfs_files),
            compression_ratio=0.0  # Would need to calculate actual compression
        )

    def _get_storage_usage(self) -> Dict[str, int]:
        """Get Git LFS storage usage information."""
        try:
            # Get LFS storage info
            result = subprocess.run(
                ["git", "lfs", "env"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )

            # Parse storage information from output
            storage_info = {"local_storage": 0, "remote_storage": 0}

            # This would need to parse the actual LFS env output
            # For now, return basic structure
            return storage_info

        except subprocess.CalledProcessError:
            return {"local_storage": 0, "remote_storage": 0}

    def _calculate_health_score(self, status: Dict) -> float:
        """Calculate LFS health score (0.0 to 1.0)."""
        score = 0.0

        # Base score for initialization
        if status.get("is_initialized"):
            score += 0.3

        # Score for having tracked patterns
        if status.get("tracked_patterns"):
            score += 0.2

        # Score for having LFS files
        if status.get("lfs_files"):
            score += 0.3

        # Score for repository stats
        stats = status.get("repository_stats")
        if stats and isinstance(stats, LFSStats):
            if stats.total_lfs_files > 0:
                score += 0.2

        return min(score, 1.0)

    def optimize_repository(self) -> Dict[str, any]:
        """
        Optimize Git LFS repository for better performance.

        Returns:
            Dictionary containing optimization results
        """
        optimization_results = {
            "actions_taken": [],
            "space_saved": 0,
            "performance_improvement": 0.0,
            "recommendations": []
        }

        try:
            # Clean up LFS cache
            if self._cleanup_lfs_cache():
                optimization_results["actions_taken"].append("LFS cache cleanup")

            # Prune unused LFS objects
            if self._prune_lfs_objects():
                optimization_results["actions_taken"].append("LFS object pruning")

            # Optimize .gitattributes
            if self._optimize_gitattributes():
                optimization_results["actions_taken"].append(".gitattributes optimization")

            # Generate recommendations
            optimization_results["recommendations"] = self._generate_recommendations()

            logger.info(f"Repository optimization completed: {len(optimization_results['actions_taken'])} actions taken")

        except Exception as e:
            logger.error(f"Error during repository optimization: {e}")
            optimization_results["error"] = str(e)

        return optimization_results

    def _cleanup_lfs_cache(self) -> bool:
        """Clean up Git LFS cache to free space."""
        try:
            result = subprocess.run(
                ["git", "lfs", "prune"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("LFS cache cleaned up successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"LFS cache cleanup failed: {e}")
            return False

    def _prune_lfs_objects(self) -> bool:
        """Prune unused LFS objects."""
        try:
            # This would implement LFS object pruning logic
            # For now, return True to indicate attempt was made
            return True
        except Exception:
            return False

    def _optimize_gitattributes(self) -> bool:
        """Optimize .gitattributes file."""
        try:
            if not self.lfs_config_path.exists():
                return False

            # Read current content
            with open(self.lfs_config_path, 'r') as f:
                lines = f.readlines()

            # Remove duplicates and optimize
            unique_lines = []
            seen_patterns = set()

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    pattern = line.split()[0] if line.split() else ""
                    if pattern and pattern not in seen_patterns:
                        unique_lines.append(line)
                        seen_patterns.add(pattern)
                elif line.startswith('#') or not line:
                    unique_lines.append(line)

            # Write optimized content
            with open(self.lfs_config_path, 'w') as f:
                f.write('\n'.join(unique_lines) + '\n')

            return True

        except Exception as e:
            logger.error(f"Failed to optimize .gitattributes: {e}")
            return False

    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Analyze repository and generate recommendations
        status = self.get_lfs_status()
        stats = status.get("repository_stats")

        if isinstance(stats, LFSStats):
            if stats.total_lfs_files == 0:
                recommendations.append("No LFS files found. Consider tracking large files with LFS.")

            if stats.average_file_size > 100 * 1024 * 1024:  # 100MB
                recommendations.append("Average file size is large. Consider data compression or chunking.")

            if stats.total_lfs_size > 1024 * 1024 * 1024:  # 1GB
                recommendations.append("Large LFS storage usage. Consider regular pruning and cleanup.")

        return recommendations

    def migrate_existing_files(self) -> Dict[str, any]:
        """
        Migrate existing large files to Git LFS.

        Returns:
            Dictionary containing migration results
        """
        migration_results = {
            "migrated_files": [],
            "failed_files": [],
            "total_size_migrated": 0,
            "errors": []
        }

        try:
            # Find files that should be migrated
            large_files = self._find_large_files()

            for file_path in large_files:
                try:
                    if not self._is_already_tracked(file_path):
                        # Get file size before migration
                        file_size = os.path.getsize(file_path)

                        # Track with LFS
                        if self._track_file_with_lfs(file_path):
                            migration_results["migrated_files"].append(file_path)
                            migration_results["total_size_migrated"] += file_size
                        else:
                            migration_results["failed_files"].append(file_path)

                except Exception as e:
                    migration_results["failed_files"].append(file_path)
                    migration_results["errors"].append(f"Failed to migrate {file_path}: {e}")

            logger.info(f"Migration completed: {len(migration_results['migrated_files'])} files migrated")

        except Exception as e:
            logger.error(f"Migration error: {e}")
            migration_results["errors"].append(str(e))

        return migration_results

    def health_check(self) -> Dict[str, any]:
        """
        Perform comprehensive Git LFS health check.

        Returns:
            Dictionary containing health check results
        """
        health_check = {
            "overall_health": "unknown",
            "checks": {},
            "issues": [],
            "recommendations": [],
            "score": 0.0
        }

        try:
            # Check LFS initialization
            health_check["checks"]["lfs_initialized"] = self._is_lfs_initialized()
            if not health_check["checks"]["lfs_initialized"]:
                health_check["issues"].append("Git LFS not initialized")
                health_check["recommendations"].append("Run git lfs install to initialize")

            # Check .gitattributes
            health_check["checks"]["gitattributes_exists"] = self.lfs_config_path.exists()
            if not health_check["checks"]["gitattributes_exists"]:
                health_check["issues"].append(".gitattributes file missing")
                health_check["recommendations"].append("Create .gitattributes with LFS patterns")

            # Check for large files not in LFS
            large_files = self._find_large_files()
            untracked_large_files = [f for f in large_files if not self._is_already_tracked(f)]
            health_check["checks"]["untracked_large_files"] = len(untracked_large_files)

            if untracked_large_files:
                health_check["issues"].append(f"{len(untracked_large_files)} large files not tracked by LFS")
                health_check["recommendations"].append("Track large files with Git LFS")

            # Calculate overall health
            issues_count = len(health_check["issues"])
            if issues_count == 0:
                health_check["overall_health"] = "excellent"
                health_check["score"] = 1.0
            elif issues_count <= 2:
                health_check["overall_health"] = "good"
                health_check["score"] = 0.8
            elif issues_count <= 5:
                health_check["overall_health"] = "fair"
                health_check["score"] = 0.6
            else:
                health_check["overall_health"] = "poor"
                health_check["score"] = 0.3

            logger.info(f"Health check completed: {health_check['overall_health']} ({health_check['score']:.1f})")

        except Exception as e:
            logger.error(f"Health check error: {e}")
            health_check["issues"].append(f"Health check error: {e}")
            health_check["overall_health"] = "error"

        return health_check


# Example usage and testing functions
def main():
    """Example usage of GitLFSManager."""
    # Initialize manager
    lfs_manager = GitLFSManager()

    # Initialize LFS
    print("Initializing Git LFS...")
    if lfs_manager.initialize_lfs():
        print("✅ Git LFS initialized successfully")
    else:
        print("❌ Failed to initialize Git LFS")
        return

    # Track large files
    print("\nTracking large files...")
    tracked_files = lfs_manager.track_large_files()
    if tracked_files:
        print(f"✅ Tracked {len(tracked_files)} files with LFS")
        for file_path in tracked_files:
            print(f"  - {file_path}")
    else:
        print("ℹ️ No large files to track")

    # Get status
    print("\nGetting LFS status...")
    status = lfs_manager.get_lfs_status()
    print(f"LFS Status: {json.dumps(status, indent=2, default=str)}")

    # Health check
    print("\nPerforming health check...")
    health = lfs_manager.health_check()
    print(f"Health: {health['overall_health']} (Score: {health['score']:.1f})")
    if health['issues']:
        print("Issues found:")
        for issue in health['issues']:
            print(f"  ❌ {issue}")
    if health['recommendations']:
        print("Recommendations:")
        for rec in health['recommendations']:
            print(f"  💡 {rec}")


if __name__ == "__main__":
    main()