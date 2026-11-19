import json
import sqlite3
import time
from pathlib import Path


class JobLedger:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (job_key TEXT PRIMARY KEY, sample_id TEXT, status TEXT, attempts INTEGER DEFAULT 0, metadata TEXT, updated REAL)')
        self.db.commit()

    def close(self):
        self.db.close()

    def get(self, key):
        r = self.db.execute('SELECT * FROM jobs WHERE job_key=?', (key,)).fetchone()
        if r is None:
            return None
        result = dict(r)
        result['metadata'] = json.loads(result['metadata'] or '{}')
        return result

    def ensure(self, key, sample_id):
        self.db.execute('INSERT OR IGNORE INTO jobs(job_key,sample_id,status,metadata,updated) VALUES (?,?,?,?,?)',
                        (key, sample_id, 'pending', '{}', time.time()))
        self.db.commit()

    def set(self, key, status, metadata=None, attempt=False):
        old = self.get(key)
        data = old['metadata']
        data.update(metadata or {})
        self.db.execute('UPDATE jobs SET status=?, metadata=?, attempts=attempts+?, updated=? WHERE job_key=?',
                        (status, json.dumps(data, ensure_ascii=False), int(attempt), time.time(), key))
        self.db.commit()

    def recover_inflight(self):
        # Only called under an exclusive API process lock; not against an active worker.
        self.db.execute("UPDATE jobs SET status='ambiguous' WHERE status='inflight'")
        self.db.commit()

    def rows(self):
        return [self.get(r[0]) for r in self.db.execute('SELECT job_key FROM jobs ORDER BY updated')]


class RunLock:
    """Cross-platform OS lock; released automatically after a crashed process."""
    def __init__(self, path):
        self.path = Path(path)

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.file.seek(0)
        self.file.write(b'0')
        self.file.flush()
        self.file.seek(0)
        try:
            import msvcrt
            msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
        except ImportError:
            import fcntl
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError('已有API阶段运行，不能同时修改任务账本') from None
        return self

    def __exit__(self, *args):
        self.file.close()
