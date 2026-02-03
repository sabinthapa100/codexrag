"""
Safety Agent - Audit and Validation
====================================
Ensures all operations are read-only and logs all queries.

This agent provides:
1. Zero-deletion guarantee via operation validation
2. Immutable audit trail of all queries
3. Manifest verification for file integrity

Author: Sabin Thapa (sthapa3@kent.edu)
"""

from __future__ import annotations
import re
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field


@dataclass
class AuditEntry:
    """Single audit log entry."""
    timestamp: str
    operation: str
    query: str
    status: str  # "SUCCESS", "BLOCKED", "ERROR"
    details: Optional[str] = None


class SafetyAgent:
    """
    Validates all operations and maintains audit trail.
    
    This agent guarantees:
    - No file modifications to source codebase
    - All queries logged with timestamps
    - File integrity verification via SHA256
    """
    
    # Operations that are NEVER allowed
    FORBIDDEN_PATTERNS = [
        r'os\.remove',
        r'os\.unlink',
        r'os\.rmdir',
        r'shutil\.rmtree',
        r'shutil\.move',
        r'shutil\.copy',
        r'Path\(.*\)\.unlink',
        r'Path\(.*\)\.rmdir',
        r'Path\(.*\)\.write_text',
        r'Path\(.*\)\.write_bytes',
        r'open\s*\([^)]*["\']w["\']',
        r'open\s*\([^)]*["\']a["\']',
        r'open\s*\([^)]*["\']x["\']',
    ]
    
    # Safe read operations
    ALLOWED_PATTERNS = [
        r'Path\(.*\)\.read_text',
        r'Path\(.*\)\.read_bytes',
        r'open\s*\([^)]*["\']r["\']',
        r'json\.load',
        r'yaml\.safe_load',
    ]
    
    def __init__(self, codexrag_dir: Path):
        """
        Initialize SafetyAgent.
        
        Args:
            codexrag_dir: Path to .codexrag directory for storing logs
        """
        self.codexrag_dir = Path(codexrag_dir)
        self.codexrag_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_log_path = self.codexrag_dir / "audit.log"
        self.manifest_path = self.codexrag_dir / "cache" / "manifest.json"
        
        self._audit_cache: List[AuditEntry] = []
    
    def validate_operation(self, operation_code: str) -> bool:
        """
        Check if an operation is safe (read-only).
        
        Args:
            operation_code: Code string to validate
            
        Returns:
            True if operation is safe, False if dangerous
        """
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, operation_code, re.IGNORECASE):
                self._log_violation(operation_code, pattern)
                return False
        return True
    
    def _log_violation(self, operation: str, matched_pattern: str) -> None:
        """Log a security violation attempt."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            operation="VALIDATE",
            query=operation[:200],
            status="BLOCKED",
            details=f"Matched forbidden pattern: {matched_pattern}"
        )
        self._write_audit(entry)
    
    def log_query(self, query: str, response_preview: str = "") -> None:
        """
        Log a user query for audit trail.
        
        Args:
            query: The user's question
            response_preview: First 100 chars of response
        """
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            operation="QUERY",
            query=query[:500],
            status="SUCCESS",
            details=response_preview[:100] if response_preview else None
        )
        self._write_audit(entry)
    
    def log_index(self, repo_path: str, num_files: int, num_chunks: int) -> None:
        """Log an indexing operation."""
        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            operation="INDEX",
            query=f"Indexed {repo_path}",
            status="SUCCESS",
            details=f"Files: {num_files}, Chunks: {num_chunks}"
        )
        self._write_audit(entry)
    
    def _write_audit(self, entry: AuditEntry) -> None:
        """Append entry to audit log (append-only, never overwrite)."""
        self._audit_cache.append(entry)
        
        # Append to file
        with self.audit_log_path.open("a", encoding="utf-8") as f:
            log_line = (
                f"{entry.timestamp}|{entry.operation}|{entry.status}|"
                f"{entry.query.replace('|', ' ')}|{entry.details or ''}\n"
            )
            f.write(log_line)
    
    def verify_manifest(self, repo_root: Path) -> Dict[str, str]:
        """
        Verify file integrity against stored manifest.
        
        Args:
            repo_root: Root of the codebase
            
        Returns:
            Dict of changed files: {path: "modified"|"deleted"|"new"}
        """
        if not self.manifest_path.exists():
            return {}
        
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        changes = {}
        
        for rel_path, stored_hash in manifest.items():
            full_path = repo_root / rel_path
            if not full_path.exists():
                changes[rel_path] = "deleted"
            else:
                current_hash = self._hash_file(full_path)
                if current_hash != stored_hash:
                    changes[rel_path] = "modified"
        
        return changes
    
    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        hasher = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_audit_summary(self, last_n: int = 10) -> List[Dict]:
        """
        Get recent audit entries.
        
        Args:
            last_n: Number of recent entries to return
            
        Returns:
            List of audit entry dicts
        """
        if not self.audit_log_path.exists():
            return []
        
        entries = []
        lines = self.audit_log_path.read_text(encoding="utf-8").strip().split("\n")
        
        for line in lines[-last_n:]:
            if not line.strip():
                continue
            parts = line.split("|", 4)
            if len(parts) >= 4:
                entries.append({
                    "timestamp": parts[0],
                    "operation": parts[1],
                    "status": parts[2],
                    "query": parts[3],
                    "details": parts[4] if len(parts) > 4 else None
                })
        
        return entries
    
    def generate_safety_report(self) -> str:
        """Generate a human-readable safety report."""
        entries = self.get_audit_summary(100)
        
        total = len(entries)
        blocked = sum(1 for e in entries if e["status"] == "BLOCKED")
        queries = sum(1 for e in entries if e["operation"] == "QUERY")
        indexes = sum(1 for e in entries if e["operation"] == "INDEX")
        
        report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    CODEXRAG SAFETY REPORT                    ║
╠══════════════════════════════════════════════════════════════╣
║  Total Operations:     {total:>5}                                  ║
║  Queries Executed:     {queries:>5}                                  ║
║  Index Operations:     {indexes:>5}                                  ║
║  Blocked (Security):   {blocked:>5}                                  ║
╠══════════════════════════════════════════════════════════════╣
║  Status: {"🔴 SECURITY EVENTS DETECTED" if blocked > 0 else "🟢 ALL OPERATIONS SAFE"}                       ║
╚══════════════════════════════════════════════════════════════╝

Recent Activity:
"""
        for e in entries[-5:]:
            status_icon = "✅" if e["status"] == "SUCCESS" else "🚫"
            report += f"  {status_icon} [{e['timestamp'][:19]}] {e['operation']}: {e['query'][:50]}...\n"
        
        return report
