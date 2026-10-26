r"""Install-root layout and the current.json state file. Stdlib only: launch.pyw loads this file by path.

<install_root>/
    app/             程式（pyproject.toml、uv.lock、packaging/wheels、src/、launch.pyw）
    envs/<lock_id>/  基礎環境（launch.pyw 建立，設定模式從這裡啟動；首次安裝或 uv.lock 變了才用）
    envs/<lock_id>-<flavor>-<n>/  每次安裝都裝進一個新資料夾；current.json 的 env 欄位指向正在用的那個
    uv-cache/        uv 快取
    logs/
    current.json     目前用哪個環境（write_current 是唯一會寫的狀態）
    uv.exe
    python/          內嵌 Python

status() 只看 current.json 與環境的 python 在不在，不檢查 torch 本身，那是自我檢查的工作。
current.json 的 env 缺少時（舊的開發安裝）就是 envs/<lock_id>；retired 列出切換後等著刪除的舊環境資料夾（env_cleanup 負責刪）。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

SCHEMA = 1
CURRENT_FILE = "current.json"
READY = "READY"
NEED_SETUP = "NEED_SETUP"


def lock_sha256(lock_path) -> str:
    """uv.lock 位元組的 SHA256（小寫十六進位）。"""
    digest = hashlib.sha256()
    with open(lock_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_id(sha: str) -> str:
    return sha[:8]


def app_dir(install_root) -> Path:
    return Path(install_root) / "app"


def cache_dir(install_root) -> Path:
    return Path(install_root) / "uv-cache"


def logs_dir(install_root) -> Path:
    return Path(install_root) / "logs"


def envs_dir(install_root) -> Path:
    return Path(install_root) / "envs"


def env_dir(install_root, lock_sha: str) -> Path:
    """The base env folder for a lock (also the legacy single-env layout)."""
    return envs_dir(install_root) / lock_id(lock_sha)


_ENV_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def valid_env_name(name) -> bool:
    """One plain folder name under envs/: no separators, no '..', nothing that could point elsewhere."""
    return isinstance(name, str) and _ENV_NAME.fullmatch(name) is not None and ".." not in name


def python_in_env(env_dir) -> Path:
    env_dir = Path(env_dir)
    return env_dir / "Scripts" / "python.exe" if sys.platform == "win32" else env_dir / "bin" / "python"


@dataclass(frozen=True)
class Install:
    """Where everything for the shipped uv.lock lives. locate() hashes the lock once; pass this on instead of re-hashing."""
    root: Path
    lock_sha256: str

    @property
    def app(self) -> Path:
        return app_dir(self.root)

    @property
    def lock(self) -> Path:
        return self.app / "uv.lock"

    @property
    def env(self) -> Path:
        """The base env launch.pyw builds for this lock; the env in use is current_env_dir()."""
        return env_dir(self.root, self.lock_sha256)

    @property
    def python(self) -> Path:
        return python_in_env(self.env)


def locate(install_root) -> Install:
    """Raises OSError when app/uv.lock is missing or unreadable."""
    root = Path(install_root)
    return Install(root, lock_sha256(app_dir(root) / "uv.lock"))


@dataclass(frozen=True, kw_only=True)
class Current:
    schema: int = SCHEMA
    lock_sha256: str
    flavor: str | None              # None：只有基礎環境（還沒裝 torch）
    verified: bool
    updated_at: str                 # ISO 8601 UTC
    env: str | None = None          # envs/ 底下的資料夾名稱；None：舊格式，就是 envs/<lock_id>
    retired: tuple[str, ...] = ()   # 已經換掉、等著刪除的舊環境資料夾名稱


def current_env_dir(install_root, current: Current) -> Path:
    return envs_dir(install_root) / (current.env or lock_id(current.lock_sha256))


def _current_path(install_root) -> Path:
    return Path(install_root) / CURRENT_FILE


def read_current(install_root) -> Current | None:
    """讀 current.json；不存在、壞掉、欄位缺漏或 schema 不認得都回傳 None，不拋例外。"""
    try:
        with open(_current_path(install_root), "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError, RecursionError):                       # ValueError 含 JSONDecodeError、UnicodeDecodeError
        return None
    if not isinstance(data, dict):
        return None
    try:
        schema, sha, flavor, verified, updated_at = (data[k] for k in ("schema", "lock_sha256", "flavor", "verified", "updated_at"))
    except KeyError:
        return None
    if type(schema) is not int or schema != SCHEMA:                     # type() 比對：排除 True == 1
        return None
    if not isinstance(sha, str) or not (flavor is None or isinstance(flavor, str)):
        return None
    if not isinstance(verified, bool) or not isinstance(updated_at, str):
        return None
    env, retired = data.get("env"), data.get("retired", [])            # 兩個都是後來加的：缺少時用預設值
    if not (env is None or valid_env_name(env)):
        return None
    if not isinstance(retired, list) or not all(valid_env_name(name) for name in retired):
        return None
    return Current(schema=schema, lock_sha256=sha, flavor=flavor, verified=verified, updated_at=updated_at,
                   env=env, retired=tuple(dict.fromkeys(retired)))


def write_current(install_root, current: Current) -> None:
    """原子寫入：同資料夾的暫存檔 + os.replace；中途失敗時舊的 current.json 不受影響。"""
    root = Path(install_root)
    root.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=root, prefix=CURRENT_FILE + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(asdict(current), fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, _current_path(root))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def status(install_root, shipped_lock_path) -> str:
    """READY：current 有效、對應目前的 lock、已選 flavor 且驗證過，環境的 python 存在；其他一律 NEED_SETUP。"""
    current = read_current(install_root)
    if current is None or current.flavor is None or not current.verified:
        return NEED_SETUP
    try:
        sha = lock_sha256(shipped_lock_path)
    except OSError:
        return NEED_SETUP
    if current.lock_sha256 != sha:
        return NEED_SETUP
    if not python_in_env(current_env_dir(install_root, current)).is_file():
        return NEED_SETUP
    return READY
