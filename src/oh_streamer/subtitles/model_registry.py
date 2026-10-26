"""Tested model registrations. Catalogs never contain credentials."""
from __future__ import annotations
import json
import os
from pathlib import Path
import sys
import threading
import uuid
import copy
from . import runtime

_LOCK = threading.RLock()

# Versioned product plan, not an assertion about current remote availability.
SERVICE_PLANS = {
    'groq': dict(base_url='https://api.groq.com/openai/v1', api_family='groq',
                 models={'asr': 'whisper-large-v3-turbo', 'translation': 'qwen/qwen3.8-27b'}),
    'azure-translator': dict(base_url='https://api.cognitive.microsofttranslator.com',
                            api_family='azure-translator', models={'translation': 'azure-translator-lines'}),
}


LOCAL_HOSTS = frozenset({'localhost', '127.0.0.1', '::1'})


def is_local_base_url(base_url) -> bool:
    """True for loopback endpoints (Ollama / LM Studio), which need no API key. Cloud hosts never match."""
    from urllib.parse import urlsplit
    try:
        return (urlsplit(str(base_url or '')).hostname or '').lower() in LOCAL_HOSTS
    except ValueError:
        return False


def is_cloud_model(row) -> bool:
    """True when using this model/connection sends data to a remote host.

    Local runtimes and loopback openai-compatible servers (Ollama / LM Studio) are not cloud:
    they need neither the cloud-consent toggle nor an API key. Unknown rows count as cloud.
    """
    if not row:
        return False
    if row.get('provider') == 'local':
        return False
    return not is_local_base_url(row.get('base_url'))


def list_connections(settings=None):
    with _LOCK:
        return [_safe_with_auth(c) for c in _document(settings)['connections']]


def connection_api_key(connection_id, settings=None):
    """Runtime-only secret lookup; never expose through the bridge.

    Resolve the key from a frozen binding's stable connection ID while keeping
    its saved endpoint/model metadata. Key rotation does not rewrite old jobs.
    """
    with _LOCK:
        c = next((c for c in _document(settings)['connections'] if c['id'] == connection_id), None)
        if c is None:
            raise ValueError('Unknown subtitle connection')
        return model_api_key(c)


def _cleanup_credential(token, settings=None):
    """Best-effort keyring cleanup, retaining tokens needed by recovery data."""
    if os.name == 'nt' or not token:
        return
    try:
        from .configuration import references
        with _LOCK:
            if references(_document(settings), token):
                return
            import keyring
            keyring.delete_password('oh_streamer.subtitles', token)
    except Exception:
        pass


def test_save_connection(spec, settings=None):
    """Validate capabilities separately and commit connection + bindings in one replace.

    Existing IDs are retained. A partial validation never deletes successful bindings;
    validation describes the *current* key, so stale success is never advertised.
    No exception text from adapters is returned to the frontend.
    """
    from datetime import datetime, timezone
    from .model_adapters import validate_spec, test_model
    from .combos import is_cloud_enabled
    candidate_credential = None
    try:
        with _LOCK:
            doc = _document(settings)
            original_doc = copy.deepcopy(doc)
            cid = spec.get('id')
            old = next((c for c in doc['connections'] if c['id'] == cid), None)
            if cid and old is None:
                return dict(ok=False, code='unknown_connection', error='找不到服務連線')
            family = spec.get('api_family', (old or {}).get('api_family', ''))
            template = SERVICE_PLANS.get(family, {})
            base = spec.get('base_url', (old or {}).get('base_url', template.get('base_url')))
            requested = spec.get('models') or template.get('models')
            if not isinstance(requested, dict) or not requested:
                raise ValueError()
            c = dict(old or {}, id=cid or 'connection-' + uuid.uuid4().hex,
                     provider='openai-compatible', api_family=family, base_url=base,
                     azure_region=spec.get('azure_region', (old or {}).get('azure_region', '')))
            # Validate the whole request before making calls or storing any secret.
            candidates = [validate_spec(dict(kind=kind, provider=c['provider'], base_url=base,
                             remote_model=remote, api_family=family, azure_region=c['azure_region']))
                          for kind, remote in requested.items()]
            if family == 'azure-translator' and any(r['kind'] != 'translation' for r in candidates):
                raise ValueError()
            c['base_url'] = candidates[0]['base_url']
            c['azure_region'] = candidates[0].get('azure_region', '')
            if 'label' in spec:
                label = spec['label']
                if not isinstance(label, str) or not label.strip() or len(label.strip()) > 40:
                    return dict(ok=False, code='invalid_label', error='連線名稱請填寫 1～40 個字')
                c['label'] = label.strip()
            if is_cloud_model(c) and not is_cloud_enabled(settings):
                return dict(ok=False, code='cloud_disabled', error='雲端字幕功能未開啟')
            raw = str(spec.get('api_key') or '').strip()
            if 'api_key' in spec:
                c.pop('credential', None)
                if raw:
                    c['credential'] = _secret(raw)
                    candidate_credential = c['credential']
            key = model_api_key(c)
            if not key and not is_local_base_url(c['base_url']):
                return dict(ok=False, code='missing_credential', error='請設定金鑰或環境變數')
            if key and any(other['id'] != c['id'] and other.get('api_family') == family
                           and other.get('base_url', '').rstrip('/') == c['base_url'].rstrip('/')
                           and model_api_key(other) == key for other in doc['connections']):
                return dict(ok=False, code='duplicate_connection', error='此金鑰已有連線，請編輯原有連線')
        validation = copy.deepcopy(c.get('validation', {}))
        # Credential/endpoint changes invalidate capabilities omitted by the caller.
        if old and any(c.get(k) != old.get(k) for k in ('credential', 'base_url', 'azure_region')):
            validation = {k: dict(status='unverified', checked_at=None) for k in validation}
        results = {}
        for row in candidates:
            kind = row['kind']
            checked = datetime.now(timezone.utc).isoformat()
            try:
                test_model(row, api_key=key,
                           test_audio_path=spec.get('test_audio_path', '__builtin__') if kind == 'asr' else None,
                           settings=settings)
                verified = kind != 'asr' or spec.get('test_audio_path') not in (None, '', '__builtin__')
                results[kind] = dict(status='verified' if verified else 'reachable', checked_at=checked,
                                     remote_model=row['remote_model'])
                binding = next((m for m in doc['models'] if m.get('connection_id') == c['id']
                                and m['kind'] == kind and m.get('remote_model') == row['remote_model']), None)
                if binding is None:
                    doc['models'].append(dict(row, id='user-' + uuid.uuid4().hex,
                        connection_id=c['id'], builtin=False, removable=True, tested=verified))
            except Exception as exc:
                results[kind] = dict(status='failed', checked_at=checked, remote_model=row['remote_model'],
                                     error=_format_test_error(exc, row))
        validation.update(results)
        with _LOCK:
            if _document(settings) != original_doc:
                return dict(ok=False, code='concurrent_update', error='服務設定已變更，請重新驗證')
            c['validation'] = validation
            doc['connections'] = [x for x in doc['connections'] if x['id'] != c['id']] + [c]
            runtime.atomic_write_json(registry_path(settings), doc)
            _cleanup_credential((old or {}).get('credential'), settings)
            return dict(ok=all(r['status'] == 'verified' for r in results.values()),
                        code='saved' if all(r['status'] == 'verified' for r in results.values()) else 'partial_validation',
                        connection=_safe_with_auth(c),
                        models=[_safe(m) for m in doc['models'] if m.get('connection_id') == c['id']])
    except Exception:
        return dict(ok=False, code='invalid_connection', error='服務設定無效或無法安全儲存')
    finally:
        _cleanup_credential(candidate_credential, settings)


def remove_connection(connection_id, settings=None, *, remove_models=False):
    with _LOCK:
        doc = _document(settings)
        if not remove_models and any(m.get('connection_id') == connection_id for m in doc['models']):
            return dict(ok=False, code='connection_in_use', error='服務連線仍被模型引用')
        if not any(c['id'] == connection_id for c in doc['connections']):
            return dict(ok=False, code='unknown_connection', error='找不到服務連線')
        old = next(c for c in doc['connections'] if c['id'] == connection_id)
        if remove_models:
            doc['models'] = [m for m in doc['models'] if m.get('connection_id') != connection_id]
        doc['connections'] = [c for c in doc['connections'] if c['id'] != connection_id]
        runtime.atomic_write_json(registry_path(settings), doc)
        _cleanup_credential(old.get('credential'), settings)
        return dict(ok=True)

def registry_path(settings=None):
    if settings and settings.get('model_registry_path'):
        return Path(settings['model_registry_path'])
    if os.name == 'nt':
        root = Path(os.environ.get('LOCALAPPDATA', Path.home() / 'AppData/Local'))
    elif sys.platform == 'darwin':
        root = Path.home() / 'Library/Application Support'
    else:
        root = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share'))
    return root / 'oh_streamer/subtitle_models.json'

def _consolidate_connections(doc, settings=None):
    connections = doc.get('connections', [])
    if len(connections) <= 1:
        return False
    seen = {}
    new_conns = []
    models_map = {}
    modified = False
    for c in connections:
        key = (c.get('api_family'), model_api_key(c), c.get('base_url', ''))
        if key in seen and key[1]:
            primary = seen[key]
            for role, val in c.get('validation', {}).items():
                if role not in primary.get('validation', {}) or primary['validation'][role].get('status') != 'verified':
                    primary.setdefault('validation', {})[role] = val
            models_map[c['id']] = primary['id']
            modified = True
        else:
            seen[key] = c
            new_conns.append(c)
    if modified:
        doc['connections'] = new_conns
        for m in doc.get('models', []):
            if m.get('connection_id') in models_map:
                m['connection_id'] = models_map[m['connection_id']]
        path = registry_path(settings)
        runtime.atomic_write_json(path, doc)
    return modified

def _document(settings=None):
    path = registry_path(settings)
    if not path.exists():
        return dict(schema_version=2, connections=[], models=[])
    value = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(value, list):
        # One connection per old model: endpoint equality is not account identity.
        doc = dict(schema_version=2, connections=[], models=[], legacy_backup=copy.deepcopy(value))
        for old in value:
            row = dict(old)
            if row.get('provider') != 'local':
                cid = 'connection-' + str(uuid.uuid5(uuid.NAMESPACE_URL, row['id']))
                connection = dict(id=cid, provider=row['provider'], env_model_id=row['id'],
                                  validation={row['kind']: dict(status='legacy_tested' if row.get('tested') else 'unverified', checked_at=None)})
                connection.update({k: row[k] for k in ('api_family', 'base_url', 'azure_region') if k in row})
                if row.get('credential'):
                    connection['credential'] = row.pop('credential')
                doc['connections'].append(connection)
                row['connection_id'] = cid
            doc['models'].append(row)
        _consolidate_connections(doc, settings)
        runtime.atomic_write_json(path, doc)
        return doc
    if not isinstance(value, dict) or value.get('schema_version') != 2:
        raise ValueError('Invalid subtitle model registry')
    _consolidate_connections(value, settings)
    return value


def _read(settings=None):
    with _LOCK:
        doc = _document(settings)
        connections = {c['id']: c for c in doc['connections']}
        rows = []
        for model in doc['models']:
            row = dict(model)
            if row.get('connection_id'):
                c = connections[row['connection_id']]
                row.update({k: c[k] for k in ('base_url', 'azure_region', 'api_family', 'credential', 'env_model_id') if k in c})
                row['capability_validation'] = copy.deepcopy(c.get('validation', {}).get(row['kind'],
                                                          dict(status='legacy_tested' if row.get('tested') else 'unverified', checked_at=None)))
            rows.append(row)
        return rows


def _write_rows(rows, settings=None, validation_updates=None):
    doc = _document(settings)
    connections = {c['id']: c for c in doc['connections']}
    models = []
    for original in rows:
        row = dict(original)
        row.pop('capability_validation', None)
        if row.get('provider') != 'local':
            cid = row.get('connection_id') or 'connection-' + uuid.uuid4().hex
            c = connections.setdefault(cid, dict(id=cid, validation={}, env_model_id=row['id']))
            for key in ('provider', 'api_family', 'base_url', 'azure_region'):
                if key in row:
                    c[key] = row[key]
            if row.get('credential'):
                c['credential'] = row.pop('credential')
            else:
                c.pop('credential', None)
            row.pop('env_model_id', None)
            row['connection_id'] = cid
        models.append(row)
    for cid, validation in (validation_updates or {}).items():
        connections[cid]['validation'] = validation
    doc.update(models=models, connections=list(connections.values()))
    runtime.atomic_write_json(registry_path(settings), doc)

_SECRET_FIELDS = {'credential', 'api_key', 'secret', 'token', 'env_model_id'}

def _safe(model):
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items() if k not in _SECRET_FIELDS}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value
    return clean(model)

def _auth_source(model) -> str:
    if model.get('provider') == 'local':
        return 'local'
    if model.get('credential'):
        return 'dpapi' if os.name == 'nt' else 'keyring'
    if _env_key_for_model(model):
        return 'env'
    if is_local_base_url(model.get('base_url')):
        return 'keyless'
    return 'none'

def _safe_with_auth(model):
    out = _safe(model)
    out['auth_source'] = _auth_source(model)
    return out

def list_models(settings=None):
    from .combos import TRANSLATION_PROFILES
    asr_labels = {'kotoba': 'Kotoba-Whisper v2.0', 'turbo': 'Whisper large-v3-turbo',
                  'whisper-large-v3': 'Whisper large-v3', 'qwen3-asr-1.7b': 'Qwen3-ASR 1.7B'}
    translation_labels = {'sakura-v1': 'Sakura 7B v1.0',
                          'sakura-galtransl-7b-v3.7': 'Sakura-GalTransl 7B v3.7',
                          'sakura-galtransl-14b-v3.8': 'Sakura-GalTransl 14B v3.8', 'qwen3': 'Qwen3 8B'}
    models = [dict(id=runtime.model_dir_name(alias), label=asr_labels.get(alias, runtime.model_dir_name(alias)), kind='asr', provider='local', recipe=alias,
                   builtin=True, removable=False, auth_source='local') for alias in runtime.MODEL_SPECS if alias != 'kotoba-v2.0']
    models.append(dict(id='__dual__', label='雙模型融合 · whisper-ja-1.5B + anime-whisper', kind='asr', provider='local', recipe='__dual__',
                       builtin=True, removable=False, auth_source='local'))
    models += [dict(id=alias, label=translation_labels[alias], kind='translation', provider='local', recipe=alias,
                    builtin=True, removable=False, auth_source='local') for alias in TRANSLATION_PROFILES]
    from .combos import TRANSLATION_FILES
    from .environment import runtime_paths
    model_root = Path(runtime_paths(settings)['models'])
    for row in models:
        if row.get('recipe') == '__dual__':
            row['model_path'] = str(model_root)
        else:
            row['model_path'] = str(model_root / runtime.model_dir_name(row['recipe'])) if row['kind'] == 'asr' else str(model_root / TRANSLATION_FILES[row['recipe']])
    models += [_safe_with_auth(m) for m in _read(settings)]
    for row in models:
        exists = False
        if row['provider'] == 'local':
            if row.get('recipe') == '__dual__':
                path = Path(row.get('model_path', ''))
                p1 = path / runtime.model_dir_name('whisper-ja-1.5B')
                p2 = path / runtime.model_dir_name('anime-whisper')
                exists = ((p1 / 'config.json').is_file() and any(p.is_file() for pat in ('*.safetensors', '*.bin') for p in p1.glob(pat)) and
                          (p2 / 'config.json').is_file() and any(p.is_file() for pat in ('*.safetensors', '*.bin') for p in p2.glob(pat)))
            else:
                path = Path(row.get('model_path',''))
                if row['kind'] == 'translation':
                    exists = path.is_file()
                else:
                    exists = (path / 'config.json').is_file() and any(p.is_file() for pattern in ('*.safetensors','*.bin') for p in path.glob(pattern))
        validation = row.get('capability_validation', {})
        cloud_status = 'tested' if validation.get('status', 'legacy_tested') in ('verified', 'legacy_tested') else 'unverified'
        if validation.get('remote_model', row.get('remote_model')) != row.get('remote_model'):
            cloud_status = 'unverified'
        row['status'] = ('ready' if exists else 'missing') if row['provider'] == 'local' else cloud_status
        row['status_label'] = {'ready':'模型檔案可用','missing':'找不到模型檔案','tested':'已通過推論測試；連線狀態依執行時確認',
                               'unverified':'目前連線尚未通過此模型能力驗證'}[row['status']]
    return models

def get_model(model_id, settings=None):
    for row in _read(settings):
        if row['id'] == model_id:
            return dict(row)
    raise ValueError('Unknown registered subtitle model')

def _secret(value, decrypt=False):
    """DPAPI user-scoped protection on Windows; OS keyring on other platforms."""
    if os.name == 'nt':
        import base64
        import ctypes
        from ctypes import wintypes
        class Blob(ctypes.Structure):
            _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_byte))]
        raw = base64.b64decode(value) if decrypt else value.encode('utf-8')
        buf = ctypes.create_string_buffer(raw)
        source = Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_byte)))
        target = Blob()
        crypt = ctypes.WinDLL('crypt32', use_last_error=True)
        method = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
        if not method(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
            raise RuntimeError('OS credential protection failed')
        try:
            result = ctypes.string_at(target.data, target.size)
            return result.decode('utf-8') if decrypt else base64.b64encode(result).decode('ascii')
        finally:
            ctypes.WinDLL('kernel32').LocalFree(target.data)
    import keyring
    if decrypt:
        result = keyring.get_password('oh_streamer.subtitles', value)
        if result is None:
            raise RuntimeError('Model credential is unavailable')
        return result
    token = uuid.uuid4().hex
    keyring.set_password('oh_streamer.subtitles', token, value)
    return token

def _env_key_for_model(model) -> str:
    """Read-only environment variable fallback for CI and local testing without touching disk."""
    import re
    model_id = str(model.get('env_model_id') or model.get('id') or '').strip()
    if model_id:
        norm_id = re.sub(r'[^A-Za-z0-9]+', '_', model_id).strip('_').upper()
        if norm_id:
            val = os.environ.get(f'OH_STREAMER_SUBTITLE_KEY_{norm_id}', '').strip()
            if val:
                return val
    api_family = str(model.get('api_family') or '').strip()
    if api_family:
        norm_fam = re.sub(r'[^A-Za-z0-9]+', '_', api_family).strip('_').upper()
        if norm_fam:
            val = os.environ.get(f'OH_STREAMER_SUBTITLE_KEY_{norm_fam}', '').strip()
            if val:
                return val
    return os.environ.get('OH_STREAMER_SUBTITLE_API_KEY', '').strip()

def model_api_key(model):
    if model.get('credential'):
        return _secret(model['credential'], True)
    return _env_key_for_model(model)

def _format_test_error(exc: Exception, row: dict | None = None) -> str:
    from .providers.http import ProviderError
    remote = str((row or {}).get('remote_model') or '')
    if isinstance(exc, ProviderError):
        status_part = f'（HTTP {exc.http_status}）' if exc.http_status else ''
        if exc.kind == 'auth':
            return f'模型推論測試未通過：API 金鑰無效或無權限{status_part}；請檢查金鑰是否正確且前後無多餘空白。'
        if exc.kind in ('rate_limit', 'quota_daily'):
            return f'模型推論測試未通過：已達 API 速率或額度上限{status_part}；請稍後再試。'
        if exc.kind == 'bad_response' and exc.http_status:
            return f'模型推論測試未通過：遠端服務拒絕請求{status_part}；請確認遠端模型名稱（{remote}）與 API 基底網址是否正確。'
        if exc.kind == 'bad_response':
            return f'模型推論測試未通過：模型（{remote}）回傳格式不符合字幕要求。'
        if exc.kind in ('network', 'timeout'):
            return '模型推論測試未通過：無法連線至遠端 API 伺服器或連線逾時；請檢查網路與 API 基底網址。'
    return '模型推論測試未通過；請檢查模型檔案、字幕執行環境、API 連線、金鑰與含語音的測試音訊。'

def test_and_add_model(spec, settings=None):
    from .model_adapters import validate_spec, test_model
    try:
        row = validate_spec(spec, settings=settings)
    except Exception:
        return {'ok': False, 'error': '模型設定不受支援；請檢查模型用途、相容配方、檔案路徑與 API 網址。'}
    try:
        # Test the actual adapter, never just availability or file existence.
        effective_key = str(spec.get('api_key') or '').strip() or _env_key_for_model(row)
        test_model(row, api_key=effective_key, test_audio_path=spec.get('test_audio_path'), settings=settings)
        row.update(id='user-' + uuid.uuid4().hex, builtin=False, removable=True, tested=True)
        if spec.get('api_key'):
            row['credential'] = _secret(str(spec['api_key']).strip())
        with _LOCK:
            rows = _read(settings)
            existing_idx = next(
                (
                    i for i, r in enumerate(rows)
                    if r.get('kind') == row.get('kind')
                    and r.get('provider') == row.get('provider')
                    and (
                        (row.get('provider') == 'openai-compatible' and r.get('base_url') == row.get('base_url') and r.get('remote_model') == row.get('remote_model')
                         and model_api_key(r) == effective_key)
                        or (row.get('provider') == 'local' and r.get('recipe') == row.get('recipe') and r.get('model_path') == row.get('model_path'))
                    )
                ),
                None,
            )
            if existing_idx is not None:
                row['id'] = rows[existing_idx]['id']
                if rows[existing_idx].get('connection_id'):
                    row['connection_id'] = rows[existing_idx]['connection_id']
                if not row.get('credential') and rows[existing_idx].get('credential'):
                    row['credential'] = rows[existing_idx]['credential']
                rows[existing_idx] = row
            else:
                rows.append(row)
            updates = None
            if row.get('provider') != 'local':
                from datetime import datetime, timezone
                cid = row.setdefault('connection_id', 'connection-' + uuid.uuid4().hex)
                c = next((c for c in _document(settings)['connections'] if c['id'] == cid), {})
                validation = copy.deepcopy(c.get('validation', {}))
                verified = True
                validation[row['kind']] = dict(status='verified', checked_at=datetime.now(timezone.utc).isoformat(),
                                               remote_model=row.get('remote_model'))
                updates = {cid: validation}
            _write_rows(rows, settings, validation_updates=updates)
        return {'ok': True, 'model': _safe_with_auth(row)}
    except Exception as exc:
        # Remote bodies, URLs, and library errors may echo credentials.
        return {'ok': False, 'error': _format_test_error(exc, row)}

def update_model_key(model_id, api_key, settings=None):
    """Test a new API key (or env fallback) for an existing registered cloud model and update its encrypted key."""
    from .model_adapters import test_model
    with _LOCK:
        rows = _read(settings)
        target_idx = next((i for i, r in enumerate(rows) if r.get('id') == model_id), None)
        if target_idx is None:
            return {'ok': False, 'error': '找不到指定的已登錄雲端模型。'}
        row = dict(rows[target_idx])
        if row.get('provider') == 'local':
            return {'ok': False, 'error': '本機模型不需要設定 API 金鑰。'}
    raw_key = str(api_key or '').strip()
    effective_key = raw_key or _env_key_for_model(row)
    if not effective_key and not is_cloud_model(row):
        effective_key = ''  # loopback servers are keyless
    elif not effective_key:
        return {'ok': False, 'error': '請輸入新的 API 金鑰，或先設定系統環境變數。'}
    candidate_credential = None
    try:
        audio_arg = '__builtin__' if row.get('kind') == 'asr' else None
        test_model(row, api_key=effective_key, test_audio_path=audio_arg, settings=settings)
        with _LOCK:
            rows = _read(settings)
            target_idx = next((i for i, r in enumerate(rows) if r.get('id') == model_id), None)
            if target_idx is None:
                return {'ok': False, 'error': '找不到指定的已登錄雲端模型。'}
            if rows[target_idx] != row:
                return dict(ok=False, code='concurrent_update', error='服務設定已變更，請重新驗證')
            if raw_key:
                rows[target_idx]['credential'] = _secret(raw_key)
                candidate_credential = rows[target_idx]['credential']
            else:
                rows[target_idx].pop('credential', None)
            # Shared connection rotation must update every hydrated sibling before writing.
            cid = rows[target_idx].get('connection_id')
            new_cred = rows[target_idx].get('credential')
            for sibling in rows:
                if cid and sibling.get('connection_id') == cid:
                    sibling.pop('credential', None)
                    if raw_key:
                        sibling['credential'] = new_cred
            updates = None
            if cid:
                from datetime import datetime, timezone
                doc = _document(settings)
                c = next(c for c in doc['connections'] if c['id'] == cid)
                validation = {kind: dict(status='unverified', checked_at=None) for kind in c.get('validation', {})}
                validation[row['kind']] = dict(status='reachable' if row['kind'] == 'asr' else 'verified', checked_at=datetime.now(timezone.utc).isoformat(),
                                               remote_model=row.get('remote_model'))
                updates = {cid: validation}
            _write_rows(rows, settings, validation_updates=updates)
            _cleanup_credential(row.get('credential'), settings)
            return {'ok': True, 'model': _safe_with_auth(rows[target_idx])}
    except Exception as exc:
        return {'ok': False, 'error': _format_test_error(exc, row)}
    finally:
        _cleanup_credential(candidate_credential, settings)

def remove_model(model_id, settings=None):
    from .configuration import references
    if references(settings or {}, model_id, include_history=False):
        return dict(ok=False, code='model_in_use', error='模型仍被字幕設定引用')
    with _LOCK:
        rows = _read(settings)
        found = next((r for r in rows if r['id'] == model_id), None)
        if found is None:
            return {'ok': False, 'error': 'Only registered user models can be removed.'}
        _write_rows([r for r in rows if r['id'] != model_id], settings)
        if os.name != 'nt' and found.get('credential') and not found.get('connection_id'):
            try:
                import keyring
                keyring.delete_password('oh_streamer.subtitles', found['credential'])
            except Exception:
                pass
        return {'ok': True}

def resolve_model(model_id, kind=None, settings=None):
    if str(model_id).startswith('user-'):
        row = get_model(model_id, settings)
    else:
        row = next((m for m in list_models(settings) if m['id'] == model_id), None)
        if row is None:
            raise ValueError('Unknown subtitle model')
    if kind and row['kind'] != kind:
        raise ValueError('Subtitle model kind does not match')
    return row

def model_identity(model, settings=None):
    import hashlib
    row = resolve_model(model, settings=settings) if isinstance(model,str) else model
    from .environment import model_file, model_ref
    identity = {k:v for k,v in model_ref(row, settings).items() if k not in (_SECRET_FIELDS | {'label','status','status_label','auth_source','connection_id','capability_validation'})}
    if row.get('provider') == 'local' and (row.get('model_path') or row.get('model_name')):
        path = Path(model_file(row, settings))
        if row['kind'] == 'asr':
            if row.get('recipe') == '__dual__':
                from .runtime import local_model_fingerprint, model_dir_name
                p1 = path / model_dir_name('whisper-ja-1.5B')
                p2 = path / model_dir_name('anime-whisper')
                try:
                    identity['fingerprint'] = [local_model_fingerprint(p1), local_model_fingerprint(p2)]
                except OSError:
                    identity['fingerprint'] = 'missing'
            else:
                identity['fingerprint'] = runtime.local_model_fingerprint(path)
        else:
            stat = path.stat()
            identity['fingerprint'] = [stat.st_size,stat.st_mtime_ns]
    return hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
