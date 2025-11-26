"""Advanced Model Version Control System with Git-like Functionality."""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import shutil
import difflib
from collections import defaultdict

logger = logging.getLogger(__name__)


class ChangeType(Enum):
    """Types of changes in model versions."""
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"
    RESTORED = "restored"
    MERGED = "merged"
    BRANCHED = "branched"


@dataclass
class ModelChange:
    """Represents a change in model version."""
    change_type: ChangeType
    timestamp: datetime
    author: str
    description: str
    affected_parameters: List[str]
    old_values: Dict[str, Any]
    new_values: Dict[str, Any]
    performance_delta: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'change_type': self.change_type.value,
            'timestamp': self.timestamp.isoformat(),
            'author': self.author,
            'description': self.description,
            'affected_parameters': self.affected_parameters,
            'old_values': self.old_values,
            'new_values': self.new_values,
            'performance_delta': self.performance_delta
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelChange':
        """Create from dictionary."""
        data = data.copy()
        data['change_type'] = ChangeType(data['change_type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ModelVersion:
    """Represents a specific version of a model."""
    version_id: str
    version_number: str
    model_id: str
    parent_version: Optional[str]
    created_at: datetime
    author: str
    commit_message: str
    changes: List[ModelChange]
    model_hash: str
    parameters_hash: str
    performance_metrics: Dict[str, float]
    tags: List[str]
    branch: str = "main"
    is_stable: bool = False
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'version_id': self.version_id,
            'version_number': self.version_number,
            'model_id': self.model_id,
            'parent_version': self.parent_version,
            'created_at': self.created_at.isoformat(),
            'author': self.author,
            'commit_message': self.commit_message,
            'changes': [change.to_dict() for change in self.changes],
            'model_hash': self.model_hash,
            'parameters_hash': self.parameters_hash,
            'performance_metrics': self.performance_metrics,
            'tags': self.tags,
            'branch': self.branch,
            'is_stable': self.is_stable
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        """Create from dictionary."""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['changes'] = [ModelChange.from_dict(change) for change in data['changes']]
        return cls(**data)


class ModelVersionControl:
    """Git-like version control system for ML models."""
    
    def __init__(self, base_path: str = "model_management/models/versions"):
        """
        Initialize version control system.
        
        Args:
            base_path: Base directory for version storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Version control files
        self.versions_file = self.base_path / "versions.json"
        self.branches_file = self.base_path / "branches.json"
        self.tags_file = self.base_path / "tags.json"
        
        # Load existing data
        self.versions = self._load_versions()
        self.branches = self._load_branches()
        self.tags = self._load_tags()
        
        logger.info(f"Model version control initialized at {self.base_path}")
    
    def _load_versions(self) -> Dict[str, ModelVersion]:
        """Load version history from disk."""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                return {
                    version_id: ModelVersion.from_dict(version_data)
                    for version_id, version_data in data.items()
                }
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
        return {}
    
    def _load_branches(self) -> Dict[str, str]:
        """Load branch information."""
        if self.branches_file.exists():
            try:
                with open(self.branches_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load branches: {e}")
        return {"main": None}  # Main branch with no commits yet
    
    def _load_tags(self) -> Dict[str, str]:
        """Load tag information."""
        if self.tags_file.exists():
            try:
                with open(self.tags_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load tags: {e}")
        return {}
    
    def _save_versions(self):
        """Save version history to disk."""
        try:
            data = {
                version_id: version.to_dict()
                for version_id, version in self.versions.items()
            }
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    def _save_branches(self):
        """Save branch information."""
        try:
            with open(self.branches_file, 'w') as f:
                json.dump(self.branches, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save branches: {e}")
    
    def _save_tags(self):
        """Save tag information."""
        try:
            with open(self.tags_file, 'w') as f:
                json.dump(self.tags, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tags: {e}")
    
    def _calculate_hash(self, data: Any) -> str:
        """Calculate hash of data."""
        content = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_version_id(self, model_id: str) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{model_id}_{timestamp}_{len(self.versions)}"
    
    def _get_next_version_number(self, model_id: str, branch: str = "main") -> str:
        """Get next version number for model."""
        # Get existing versions for this model and branch
        model_versions = [
            v for v in self.versions.values()
            if v.model_id == model_id and v.branch == branch
        ]
        
        if not model_versions:
            return "1.0.0"
        
        # Parse highest version number
        version_numbers = []
        for version in model_versions:
            try:
                parts = version.version_number.split('.')
                version_numbers.append(tuple(int(p) for p in parts))
            except ValueError:
                continue
        
        if not version_numbers:
            return "1.0.0"
        
        # Increment patch version
        latest = max(version_numbers)
        return f"{latest[0]}.{latest[1]}.{latest[2] + 1}"
    
    def _compare_parameters(self, old_params: Dict, new_params: Dict) -> Tuple[List[str], Dict, Dict]:
        """Compare parameter sets and identify changes."""
        affected_parameters = []
        old_values = {}
        new_values = {}
        
        all_keys = set(old_params.keys()) | set(new_params.keys())
        
        for key in all_keys:
            old_val = old_params.get(key)
            new_val = new_params.get(key)
            
            if old_val != new_val:
                affected_parameters.append(key)
                old_values[key] = old_val
                new_values[key] = new_val
        
        return affected_parameters, old_values, new_values
    
    def commit_version(self, model_id: str, model_data: Any, parameters: Dict,
                      performance_metrics: Dict[str, float], commit_message: str,
                      author: str = "AI Trading System", branch: str = "main") -> str:
        """
        Commit a new version of a model.
        
        Args:
            model_id: Model identifier
            model_data: Model data/parameters
            parameters: Model parameters
            performance_metrics: Performance metrics
            commit_message: Commit message
            author: Author of the commit
            branch: Branch name
            
        Returns:
            Version ID
        """
        version_id = self._generate_version_id(model_id)
        version_number = self._get_next_version_number(model_id, branch)
        
        # Calculate hashes
        model_hash = self._calculate_hash(model_data)
        parameters_hash = self._calculate_hash(parameters)
        
        # Find parent version
        parent_version = self.branches.get(branch)
        
        # Determine changes
        changes = []
        if parent_version and parent_version in self.versions:
            parent = self.versions[parent_version]
            
            # Compare parameters
            affected_params, old_vals, new_vals = self._compare_parameters(
                parent.performance_metrics, performance_metrics
            )
            
            # Calculate performance delta
            performance_delta = {}
            for metric, new_val in performance_metrics.items():
                old_val = parent.performance_metrics.get(metric, 0)
                performance_delta[metric] = new_val - old_val
            
            change = ModelChange(
                change_type=ChangeType.MODIFIED,
                timestamp=datetime.now(),
                author=author,
                description=commit_message,
                affected_parameters=affected_params,
                old_values=old_vals,
                new_values=new_vals,
                performance_delta=performance_delta
            )
            changes.append(change)
        else:
            # First commit
            change = ModelChange(
                change_type=ChangeType.CREATED,
                timestamp=datetime.now(),
                author=author,
                description=commit_message,
                affected_parameters=list(parameters.keys()),
                old_values={},
                new_values=parameters,
                performance_delta=performance_metrics
            )
            changes.append(change)
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            version_number=version_number,
            model_id=model_id,
            parent_version=parent_version,
            created_at=datetime.now(),
            author=author,
            commit_message=commit_message,
            changes=changes,
            model_hash=model_hash,
            parameters_hash=parameters_hash,
            performance_metrics=performance_metrics,
            tags=[],
            branch=branch
        )
        
        # Store version
        self.versions[version_id] = version
        self.branches[branch] = version_id
        
        # Save to disk
        self._save_versions()
        self._save_branches()
        
        logger.info(f"Committed version {version_number} for model {model_id}")
        return version_id
    
    def create_branch(self, branch_name: str, from_version: str = None) -> bool:
        """
        Create a new branch.
        
        Args:
            branch_name: Name of the new branch
            from_version: Version to branch from (current HEAD if None)
            
        Returns:
            True if successful
        """
        if branch_name in self.branches:
            logger.warning(f"Branch {branch_name} already exists")
            return False
        
        if from_version is None:
            from_version = self.branches.get("main")
        
        if from_version and from_version not in self.versions:
            logger.error(f"Version {from_version} not found")
            return False
        
        self.branches[branch_name] = from_version
        self._save_branches()
        
        logger.info(f"Created branch {branch_name} from {from_version}")
        return True
    
    def merge_branch(self, source_branch: str, target_branch: str = "main",
                    commit_message: str = None) -> Optional[str]:
        """
        Merge one branch into another.
        
        Args:
            source_branch: Source branch to merge
            target_branch: Target branch
            commit_message: Merge commit message
            
        Returns:
            Version ID of merge commit
        """
        if source_branch not in self.branches or target_branch not in self.branches:
            logger.error("Source or target branch not found")
            return None
        
        source_version_id = self.branches[source_branch]
        target_version_id = self.branches[target_branch]
        
        if not source_version_id or not target_version_id:
            logger.error("Branch has no commits")
            return None
        
        source_version = self.versions[source_version_id]
        target_version = self.versions[target_version_id]
        
        # Create merge commit
        if commit_message is None:
            commit_message = f"Merge {source_branch} into {target_branch}"
        
        merge_version_id = self._generate_version_id(source_version.model_id)
        version_number = self._get_next_version_number(source_version.model_id, target_branch)
        
        # Combine performance metrics (weighted average)
        combined_metrics = {}
        for metric in set(source_version.performance_metrics.keys()) | set(target_version.performance_metrics.keys()):
            source_val = source_version.performance_metrics.get(metric, 0)
            target_val = target_version.performance_metrics.get(metric, 0)
            combined_metrics[metric] = (source_val + target_val) / 2
        
        merge_change = ModelChange(
            change_type=ChangeType.MERGED,
            timestamp=datetime.now(),
            author="System",
            description=commit_message,
            affected_parameters=[],
            old_values={},
            new_values={},
            performance_delta={}
        )
        
        merge_version = ModelVersion(
            version_id=merge_version_id,
            version_number=version_number,
            model_id=source_version.model_id,
            parent_version=target_version_id,
            created_at=datetime.now(),
            author="System",
            commit_message=commit_message,
            changes=[merge_change],
            model_hash=source_version.model_hash,
            parameters_hash=source_version.parameters_hash,
            performance_metrics=combined_metrics,
            tags=[],
            branch=target_branch
        )
        
        self.versions[merge_version_id] = merge_version
        self.branches[target_branch] = merge_version_id
        
        self._save_versions()
        self._save_branches()
        
        logger.info(f"Merged {source_branch} into {target_branch}")
        return merge_version_id
    
    def tag_version(self, version_id: str, tag_name: str) -> bool:
        """
        Tag a specific version.
        
        Args:
            version_id: Version to tag
            tag_name: Tag name
            
        Returns:
            True if successful
        """
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        if tag_name in self.tags:
            logger.warning(f"Tag {tag_name} already exists")
            return False
        
        self.tags[tag_name] = version_id
        self.versions[version_id].tags.append(tag_name)
        
        self._save_tags()
        self._save_versions()
        
        logger.info(f"Tagged version {version_id} as {tag_name}")
        return True
    
    def get_version_history(self, model_id: str, branch: str = None) -> List[ModelVersion]:
        """
        Get version history for a model.
        
        Args:
            model_id: Model identifier
            branch: Branch to filter by
            
        Returns:
            List of versions ordered by creation time
        """
        versions = [
            v for v in self.versions.values()
            if v.model_id == model_id and (branch is None or v.branch == branch)
        ]
        
        return sorted(versions, key=lambda v: v.created_at, reverse=True)
    
    def get_version_diff(self, version1_id: str, version2_id: str) -> Dict:
        """
        Get differences between two versions.
        
        Args:
            version1_id: First version ID
            version2_id: Second version ID
            
        Returns:
            Dictionary of differences
        """
        if version1_id not in self.versions or version2_id not in self.versions:
            raise ValueError("One or both versions not found")
        
        v1 = self.versions[version1_id]
        v2 = self.versions[version2_id]
        
        # Compare performance metrics
        metric_diffs = {}
        all_metrics = set(v1.performance_metrics.keys()) | set(v2.performance_metrics.keys())
        
        for metric in all_metrics:
            val1 = v1.performance_metrics.get(metric, 0)
            val2 = v2.performance_metrics.get(metric, 0)
            metric_diffs[metric] = {
                'v1_value': val1,
                'v2_value': val2,
                'difference': val2 - val1,
                'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
            }
        
        return {
            'version1': {
                'id': version1_id,
                'number': v1.version_number,
                'created_at': v1.created_at.isoformat()
            },
            'version2': {
                'id': version2_id,
                'number': v2.version_number,
                'created_at': v2.created_at.isoformat()
            },
            'performance_metrics_diff': metric_diffs,
            'hash_diff': {
                'model_hash_changed': v1.model_hash != v2.model_hash,
                'parameters_hash_changed': v1.parameters_hash != v2.parameters_hash
            }
        }
    
    def rollback_to_version(self, model_id: str, version_id: str, branch: str = "main") -> bool:
        """
        Rollback model to a specific version.
        
        Args:
            model_id: Model identifier
            version_id: Version to rollback to
            branch: Branch to rollback
            
        Returns:
            True if successful
        """
        if version_id not in self.versions:
            logger.error(f"Version {version_id} not found")
            return False
        
        rollback_version = self.versions[version_id]
        if rollback_version.model_id != model_id:
            logger.error("Version does not belong to specified model")
            return False
        
        # Create rollback commit
        commit_message = f"Rollback to version {rollback_version.version_number}"
        
        rollback_change = ModelChange(
            change_type=ChangeType.RESTORED,
            timestamp=datetime.now(),
            author="System",
            description=commit_message,
            affected_parameters=[],
            old_values={},
            new_values={},
            performance_delta={}
        )
        
        new_version_id = self._generate_version_id(model_id)
        version_number = self._get_next_version_number(model_id, branch)
        
        new_version = ModelVersion(
            version_id=new_version_id,
            version_number=version_number,
            model_id=model_id,
            parent_version=self.branches[branch],
            created_at=datetime.now(),
            author="System",
            commit_message=commit_message,
            changes=[rollback_change],
            model_hash=rollback_version.model_hash,
            parameters_hash=rollback_version.parameters_hash,
            performance_metrics=rollback_version.performance_metrics.copy(),
            tags=[],
            branch=branch
        )
        
        self.versions[new_version_id] = new_version
        self.branches[branch] = new_version_id
        
        self._save_versions()
        self._save_branches()
        
        logger.info(f"Rolled back model {model_id} to version {rollback_version.version_number}")
        return True
    
    def get_performance_trend(self, model_id: str, metric: str, 
                            branch: str = "main", days: int = 30) -> List[Dict]:
        """
        Get performance trend for a metric over time.
        
        Args:
            model_id: Model identifier
            metric: Performance metric name
            branch: Branch to analyze
            days: Number of days to look back
            
        Returns:
            List of performance data points
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_versions = [
            v for v in self.versions.values()
            if (v.model_id == model_id and v.branch == branch and 
                v.created_at >= cutoff_date and metric in v.performance_metrics)
        ]
        
        # Sort by creation time
        relevant_versions.sort(key=lambda v: v.created_at)
        
        trend_data = []
        for version in relevant_versions:
            trend_data.append({
                'version_id': version.version_id,
                'version_number': version.version_number,
                'timestamp': version.created_at.isoformat(),
                'value': version.performance_metrics[metric],
                'commit_message': version.commit_message
            })
        
        return trend_data
    
    def get_version_statistics(self) -> Dict:
        """Get statistics about version control system."""
        total_versions = len(self.versions)
        total_models = len(set(v.model_id for v in self.versions.values()))
        total_branches = len(self.branches)
        total_tags = len(self.tags)
        
        # Author statistics
        authors = defaultdict(int)
        for version in self.versions.values():
            authors[version.author] += 1
        
        # Branch statistics
        branch_stats = {}
        for branch, head_version_id in self.branches.items():
            branch_versions = [v for v in self.versions.values() if v.branch == branch]
            branch_stats[branch] = {
                'head_version': head_version_id,
                'total_commits': len(branch_versions),
                'latest_commit': max(branch_versions, key=lambda v: v.created_at).created_at.isoformat() if branch_versions else None
            }
        
        return {
            'total_versions': total_versions,
            'total_models': total_models,
            'total_branches': total_branches,
            'total_tags': total_tags,
            'authors': dict(authors),
            'branch_statistics': branch_stats,
            'storage_path': str(self.base_path)
        }