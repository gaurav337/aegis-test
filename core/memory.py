import sqlite3
import hashlib
import uuid
import numpy as np
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Optional SOTA Dependency: pip install sqlite-vec
try:
    import sqlite_vec
    HAS_VEC = True
except ImportError:
    HAS_VEC = False

class MemorySystem:
    def __init__(self, db_path="data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self._init_db()

    def _get_connection(self):
        """Get connection with WAL mode enabled for concurrency."""
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        if HAS_VEC:
            sqlite_vec.load(conn)
        return conn

    def _init_db(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # 1. Main Cases Table (tool_scores_vector REMOVED - dead storage)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cases (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                file_hash TEXT UNIQUE,
                file_type TEXT,
                verdict TEXT,
                confidence REAL,
                ensemble_score REAL,
                tool_scores_json TEXT,
                reasoning TEXT,
                feedback_label TEXT,
                metadata_json TEXT
            )
        """)
        
        # 2. FTS5 Virtual Table for Reasoning Search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS cases_fts USING fts5(
                reasoning,
                content='cases',
                content_rowid='rowid'
            )
        """)
        
        # 3. FTS5 Sync Triggers
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS cases_ai AFTER INSERT ON cases BEGIN
                INSERT INTO cases_fts(rowid, reasoning) VALUES (new.rowid, new.reasoning);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS cases_ad AFTER DELETE ON cases BEGIN
                INSERT INTO cases_fts(cases_fts, rowid, reasoning) 
                VALUES('delete', old.rowid, old.reasoning);
            END
        """)
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS cases_au AFTER UPDATE OF reasoning ON cases BEGIN
                INSERT INTO cases_fts(cases_fts, rowid, reasoning) 
                VALUES('delete', old.rowid, old.reasoning);
                INSERT INTO cases_fts(rowid, reasoning) 
                VALUES(new.rowid, new.reasoning);
            END
        """)
        
        # 4. Global Stats Cache
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS global_stats (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                sample_count INTEGER DEFAULT 0,
                tool_keys_json TEXT,
                mean_vector BLOB,
                m2_matrix BLOB,
                last_updated TEXT
            )
        """)
        
        # 5. Proper JSON field indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_dataset 
            ON cases(json_extract(metadata_json, '$.dataset'))
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meta_method 
            ON cases(json_extract(metadata_json, '$.method'))
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hash ON cases(file_hash)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback ON cases(feedback_label)")
        conn.commit()
        conn.close()

    def _compute_file_hash(self, path: str) -> str:
        """Full SHA256 (64 chars)."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _to_vector(self, scores: Dict[str, float], keys: List[str]) -> np.ndarray:
        """Ensures consistent vector ordering. Missing keys padded with 0.0."""
        return np.array([scores.get(k, 0.0) for k in keys], dtype=np.float64)

    def _to_vector_with_mask(self, scores: Dict[str, float], keys: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Returns vector and boolean mask of present tools."""
        values = []
        mask = []
        for k in keys:
            if k in scores:
                values.append(scores[k])
                mask.append(True)
            else:
                values.append(0.0)  # Placeholder
                mask.append(False)
        return np.array(values, dtype=np.float64), np.array(mask, dtype=bool)

    def _get_canonical_keys(self, scores: Dict[str, float]) -> List[str]:
        return sorted(scores.keys())

    def _validate_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Sanitize tool scores - NaN/Inf protection ONLY."""
        clean = {}
        for key, val in scores.items():
            if not isinstance(val, (int, float)):
                raise ValueError(f"Non-numeric score for {key}: {type(val)}")
            if not np.isfinite(val):
                raise ValueError(f"Non-finite score for {key}: {val}")
            clean[key] = float(val)
        return clean

    def _ensure_psd(self, matrix: np.ndarray) -> np.ndarray:
        """Project matrix to nearest PSD via eigenvalue clipping."""
        matrix = (matrix + matrix.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _rebuild_global_stats(self, conn, keys: List[str]):
        """Recalculate global stats from scratch."""
        cursor = conn.cursor()
        cursor.execute("SELECT tool_scores_json FROM cases")
        
        n = 0
        mean = None
        M2 = None
        
        for row in cursor:
            if row[0] is None:
                continue
            scores = json.loads(row[0])
            vec = self._to_vector(scores, keys)
            n += 1
            
            if mean is None:
                mean = vec.copy()
                M2 = np.zeros((len(keys), len(keys)), dtype=np.float64)
            else:
                delta = vec - mean
                mean = mean + delta / n
                delta2 = vec - mean
                M2 = M2 + np.outer(delta, delta2)
        
        if n == 0:
            cursor.execute("""
                INSERT OR REPLACE INTO global_stats 
                (id, sample_count, tool_keys_json, mean_vector, 
                 m2_matrix, last_updated)
                VALUES (1, 0, ?, NULL, NULL, ?)
            """, (json.dumps(keys), 
                  datetime.now(timezone.utc).isoformat()))
            return
        
        if M2 is None:
            M2 = np.zeros((len(keys), len(keys)), dtype=np.float64)
        
        cursor.execute("""
            INSERT OR REPLACE INTO global_stats 
            (id, sample_count, tool_keys_json, mean_vector, m2_matrix, last_updated)
            VALUES (1, ?, ?, ?, ?, ?)
        """, (n, json.dumps(keys), mean.tobytes(), M2.tobytes(), 
              datetime.now(timezone.utc).isoformat()))

    def _reverse_welford(self, conn, scores: Dict[str, float]) -> bool:
        """Returns True if rebuilt, False if normal reverse completed."""
        cursor = conn.cursor()
        keys = self._get_canonical_keys(scores)
        current_vec = self._to_vector(scores, keys)
        
        cursor.execute("""
            SELECT sample_count, tool_keys_json, mean_vector, m2_matrix 
            FROM global_stats WHERE id = 1
        """)
        row = cursor.fetchone()
        
        if row is None or row[0] <= 1:
            self._rebuild_global_stats(conn, keys)
            return True
        
        stored_count = row[0]
        stored_keys = json.loads(row[1])
        
        if stored_keys != keys:
            self._rebuild_global_stats(conn, keys)
            return True
        
        old_mean = np.frombuffer(row[2], dtype=np.float64)
        old_M2 = np.frombuffer(row[3], dtype=np.float64).reshape(len(keys), len(keys))
        
        new_n = stored_count - 1
        delta = current_vec - old_mean
        new_mean = (old_mean * stored_count - current_vec) / new_n
        delta2 = current_vec - new_mean
        
        new_M2 = old_M2 - np.outer(delta2, delta)
        
        diag = np.diag(new_M2)
        if np.any(diag < -1e-10):
            self._rebuild_global_stats(conn, keys)
            return True
        
        np.fill_diagonal(new_M2, np.maximum(diag, 0.0))
        new_M2 = self._ensure_psd(new_M2)
        
        cursor.execute("""
            INSERT OR REPLACE INTO global_stats 
            (id, sample_count, tool_keys_json, mean_vector, m2_matrix, last_updated)
            VALUES (1, ?, ?, ?, ?, ?)
        """, (new_n, json.dumps(keys), new_mean.tobytes(), new_M2.tobytes(),
              datetime.now(timezone.utc).isoformat()))
        return False

    def _update_global_stats(self, conn, scores: Dict[str, float]):
        """Proper Welford's algorithm, no internal commit."""
        cursor = conn.cursor()
        keys = self._get_canonical_keys(scores)
        current_vec = self._to_vector(scores, keys)
        
        cursor.execute("""
            SELECT sample_count, tool_keys_json, mean_vector, m2_matrix 
            FROM global_stats WHERE id = 1
        """)
        row = cursor.fetchone()
        
        if row is None or row[0] == 0 or row[3] is None:
            d = len(keys)
            M2_init = np.zeros((d, d), dtype=np.float64)
            cursor.execute("""
                INSERT OR REPLACE INTO global_stats 
                (id, sample_count, tool_keys_json, mean_vector, m2_matrix, last_updated)
                VALUES (1, 1, ?, ?, ?, ?)
            """, (json.dumps(keys), current_vec.tobytes(), M2_init.tobytes(), 
                  datetime.now(timezone.utc).isoformat()))
            return
        
        stored_count = row[0]
        stored_keys = json.loads(row[1])
        
        if stored_keys != keys:
            self._rebuild_global_stats(conn, keys)
            return
        
        old_mean = np.frombuffer(row[2], dtype=np.float64)
        old_M2 = np.frombuffer(row[3], dtype=np.float64).reshape(len(keys), len(keys))
        
        new_n = stored_count + 1
        delta = current_vec - old_mean
        new_mean = old_mean + delta / new_n
        delta2 = current_vec - new_mean
        new_M2 = old_M2 + np.outer(delta, delta2)
        
        cursor.execute("""
            INSERT OR REPLACE INTO global_stats 
            (id, sample_count, tool_keys_json, mean_vector, m2_matrix, last_updated)
            VALUES (1, ?, ?, ?, ?, ?)
        """, (new_n, json.dumps(keys), new_mean.tobytes(), new_M2.tobytes(),
              datetime.now(timezone.utc).isoformat()))

    def _get_covariance(self, M2: np.ndarray, n: int) -> np.ndarray:
        """Compute covariance from M2 only when needed."""
        if n < 2:
            return np.eye(M2.shape[0])
        cov = M2 / (n - 1)
        cov += np.eye(M2.shape[0]) * 1e-6
        return cov

    def store_case(self, *, file_path: str = None, file_hash: str = None,
                   file_type: str, verdict: str, confidence: float, 
                   ensemble: float, tool_scores_dict: dict, reasoning: str,
                   metadata: Optional[Dict[str, Any]] = None):
        """
        FIX #1: CORRECT parameter name AND colon (metadata: not meta)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            tool_scores_dict = self._validate_scores(tool_scores_dict)
            
            if file_path and file_hash:
                raise ValueError("Provide file_path OR file_hash, not both")
            if file_path:
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"Cannot hash: {file_path}")
                final_hash = self._compute_file_hash(file_path)
            elif file_hash:
                final_hash = file_hash
            else:
                raise ValueError("Must provide either file_path or file_hash")
            
            timestamp = datetime.now(timezone.utc).isoformat()
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor.execute(
                "SELECT id, feedback_label, tool_scores_json FROM cases WHERE file_hash = ?", 
                (final_hash,)
            )
            existing = cursor.fetchone()
            
            if existing:
                old_scores = json.loads(existing[2])
                old_keys = self._get_canonical_keys(old_scores)
                new_keys = self._get_canonical_keys(tool_scores_dict)
                
                cursor.execute("""
                    UPDATE cases SET 
                        file_type=?, verdict=?, confidence=?, ensemble_score=?,
                        tool_scores_json=?,
                        reasoning=?, timestamp=?, metadata_json=?
                    WHERE file_hash=?
                """, (file_type, verdict, confidence, ensemble, json.dumps(tool_scores_dict),
                      reasoning, timestamp, metadata_json, final_hash))
                
                if old_keys != new_keys:
                    self._rebuild_global_stats(conn, new_keys)
                else:
                    was_rebuilt = self._reverse_welford(conn, old_scores)
                    if not was_rebuilt:
                        self._update_global_stats(conn, tool_scores_dict)
            else:
                case_id = str(uuid.uuid4())
                cursor.execute("""
                    INSERT INTO cases 
                    (id, timestamp, file_hash, file_type, verdict, confidence, 
                     ensemble_score, tool_scores_json,
                     reasoning, feedback_label, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
                """, (case_id, timestamp, final_hash, file_type, verdict, 
                      confidence, ensemble, json.dumps(tool_scores_dict), 
                      reasoning, metadata_json))
                self._update_global_stats(conn, tool_scores_dict)
            
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def store_cases_batch(self, cases: List[Dict]) -> Dict[str, int]:
        """Bulk insert for dataset processing."""
        if not cases:
            return {"inserted": 0, "skipped": 0, "total": 0}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            all_keys_set = set()
            cursor.execute("SELECT tool_keys_json FROM global_stats WHERE id = 1")
            row = cursor.fetchone()
            if row and row[0]:
                existing_keys = json.loads(row[0])
                all_keys_set.update(existing_keys)
            
            for case in cases:
                case["tool_scores_dict"] = self._validate_scores(case["tool_scores_dict"])
                all_keys_set.update(case["tool_scores_dict"].keys())
            all_keys = sorted(all_keys_set)
            
            insert_batch = []
            for case in cases:
                case_id = str(uuid.uuid4())
                timestamp = datetime.now(timezone.utc).isoformat()
                metadata_json = json.dumps(case.get("metadata")) if case.get("metadata") else None
                insert_batch.append((
                    case_id, timestamp, case["file_hash"], case["file_type"],
                    case["verdict"], case["confidence"], case["ensemble"],
                    json.dumps(case["tool_scores_dict"]),
                    case["reasoning"], metadata_json
                ))
                
            cursor.executemany("""
                INSERT OR IGNORE INTO cases 
                (id, timestamp, file_hash, file_type, verdict, confidence,
                 ensemble_score, tool_scores_json,
                 reasoning, feedback_label, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
            """, insert_batch)
            
            # rowcount under executemany tells us total successful inserts
            inserted = cursor.rowcount if cursor.rowcount >= 0 else 0
            skipped = len(cases) - inserted
            
            self._rebuild_global_stats(conn, all_keys)
            conn.commit()
            
            return {"inserted": inserted, "skipped": skipped, "total": len(cases)}
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def store_feedback(self, file_hash: str, actual_label: str) -> bool:
        """Returns confirmation and raises error if not found."""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE cases SET feedback_label = ? WHERE file_hash = ?
            """, (actual_label, file_hash))
            conn.commit()
            updated = cursor.rowcount > 0
            if not updated:
                raise ValueError(f"No case found with hash: {file_hash}")
            return updated
        finally:
            conn.close()

    def query_similar_history(self, current_tool_scores: dict, top_k: int = 3, 
                              keyword_filter: Optional[str] = None,
                              metadata_filter: Optional[Dict[str, str]] = None) -> List[Dict]:
        """Query similar historical cases with Mahalanobis distance."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT sample_count, tool_keys_json, mean_vector, m2_matrix 
                FROM global_stats WHERE id = 1
            """)
            stats_row = cursor.fetchone()
            
            if not stats_row or stats_row[0] == 0:
                return []
            
            stored_count = stats_row[0]
            stored_keys = json.loads(stats_row[1])
            dim = len(stored_keys)
            
            if stats_row[3] is None:
                use_mahalanobis = False
                global_cov_common = np.eye(dim)
            else:
                global_M2 = np.frombuffer(stats_row[3], dtype=np.float64)
                expected_size = dim * dim
                if len(global_M2) != expected_size:
                    return []
                global_M2 = global_M2.reshape(dim, dim)
                use_mahalanobis = stored_count >= 2
                
                if use_mahalanobis:
                    global_cov = self._get_covariance(global_M2, stored_count)
                else:
                    global_cov = np.eye(dim)
            
            current_keys = self._get_canonical_keys(current_tool_scores)
            common_keys = sorted(set(current_keys) & set(stored_keys))
            
            if len(common_keys) < 1:
                return []
            
            current_vec = self._to_vector(current_tool_scores, common_keys)
            
            key_indices = [stored_keys.index(k) for k in common_keys]
            if use_mahalanobis:
                global_cov_common = self._ensure_psd(
                    global_cov[np.ix_(key_indices, key_indices)]
                )
            else:
                global_cov_common = np.eye(len(common_keys))
            
            cursor.execute(
                "SELECT COUNT(*) FROM cases WHERE feedback_label IS NOT NULL"
            )
            total = cursor.fetchone()[0]
            
            params = []
            base_query = """
                SELECT c.rowid, c.timestamp, c.tool_scores_json, c.feedback_label, c.reasoning 
                FROM cases c
            """
            
            if keyword_filter:
                base_query += " JOIN cases_fts f ON c.rowid = f.rowid"
            
            base_query += " WHERE c.feedback_label IS NOT NULL"
            
            if keyword_filter:
                base_query += " AND f.reasoning MATCH ?"
                params.append(keyword_filter)
            
            if metadata_filter:
                _SAFE_KEY = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
                for key, value in metadata_filter.items():
                    if not _SAFE_KEY.match(key):
                        raise ValueError(
                            f"Invalid metadata key: {key!r}. "
                            f"Keys must be alphanumeric with underscores."
                        )
                    base_query += f" AND json_extract(c.metadata_json, '$.{key}') = ?"
                    params.append(value)
            
            if total <= top_k * 50:
                base_query += " ORDER BY c.rowid DESC"
            else:
                base_query += " ORDER BY c.rowid DESC LIMIT ?"
                params.append(top_k * 50)
            
            cursor.execute(base_query, params)
            rows = cursor.fetchall()
        finally:
            conn.close()
        
        if not rows:
            return []

        matches = []
        now = datetime.now(timezone.utc)
        
        try:
            global_cov_inv_fast = np.linalg.inv(global_cov_common)
        except np.linalg.LinAlgError:
            global_cov_inv_fast = np.eye(len(common_keys))

        for row in rows:
            db_rowid, ts_str, scores_json, label, reasoning = row
            hist_scores = json.loads(scores_json)
            hist_vec, hist_mask = self._to_vector_with_mask(hist_scores, common_keys)
            
            if not np.any(hist_mask):
                continue
            
            idx = np.where(hist_mask)[0]
            delta_sub = current_vec[idx] - hist_vec[idx]
            
            if len(idx) == len(common_keys):
                cov_sub_inv = global_cov_inv_fast
            else:
                cov_sub = global_cov_common[np.ix_(idx, idx)]
                try:
                    cov_sub_inv = np.linalg.inv(cov_sub)
                except np.linalg.LinAlgError:
                    cov_sub_inv = np.eye(len(idx))
            
            inner = delta_sub @ cov_sub_inv @ delta_sub
            
            if inner < 0 or not np.isfinite(inner):
                dist = np.linalg.norm(delta_sub)
            else:
                dist = np.sqrt(inner)
            
            case_time = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            days_diff = max((now - case_time).days, 0)
            decay_penalty = min(np.exp(0.005 * days_diff), 10.0)
            final_score = dist * decay_penalty
            
            matches.append({
                "distance": final_score,
                "feedback_label": label,
                "tool_scores": hist_scores,
                "reasoning": reasoning,
                "age_days": days_diff
            })
            
        matches.sort(key=lambda x: x["distance"])
        return matches[:top_k]

    def vacuum_database(self):
        """Maintenance: Reclaim space after bulk deletes/updates."""
        conn = self._get_connection()
        try:
            conn.execute("VACUUM")
            conn.commit()
        finally:
            conn.close()