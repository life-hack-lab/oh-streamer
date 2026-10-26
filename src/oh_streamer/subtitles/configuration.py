"""Versioned subtitle editor contract. Resolution is read-only, never inference.

S1 saves this alongside legacy runtime rows; S2 consumes resolved flow and S3
consumes fallback policies. Merely choosing a mode grants no cloud consent.
"""
from __future__ import annotations

import copy

VERSION = 1
PLAN_VERSION = 'cloud-first-1'
MODES = ('cloud', 'local', 'custom')
FORBIDDEN = {'api_key', 'credential', 'secret', 'token', 'legacy_backup'}


class ConfigurationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)

    @property
    def message(self):
        """User-facing zh-TW text; `code` (and str()) stay the stable machine-readable identifier."""
        return MESSAGES.get(self.code, f'字幕配置無效（{self.code}）')


MESSAGES = {
    'configuration_not_ready': '字幕方案尚未設定完成：請到「設定 → 字幕」完成雲端連線並允許外連，或改用本機方案。',
    'translation_not_configured': '尚未選擇翻譯模型：請到「設定 → 字幕」選擇翻譯模型。',
}


def references(value, reference, *, include_history=True):
    if isinstance(value, dict):
        return any(references(child, reference, include_history=include_history)
                   for key, child in value.items() if include_history or key not in ('migration', 'legacy_backup'))
    if isinstance(value, list):
        return any(references(child, reference, include_history=include_history) for child in value)
    return value == reference


def assert_secret_free(value):
    if isinstance(value, dict):
        if FORBIDDEN.intersection(value):
            raise ConfigurationError('secret_in_configuration')
        for child in value.values():
            assert_secret_free(child)
    elif isinstance(value, list):
        for child in value:
            assert_secret_free(child)


def _profile(primary=None):
    primary = primary or {}
    asr, trans = primary.get('asr', {}), primary.get('translation', {})
    return dict(asr=asr.get('model'),
                translation=(trans.get('refine_model_id') if trans.get('mode') == 'tiered' else None) or trans.get('model_id') or trans.get('profile'),
                processing=primary.get('cloud_strategy', 'smart_horizon'),
                whisper_policy='auto', whisper_models=['whisper-large-v3-turbo', 'whisper-large-v3'],
                fallback=dict(asr=[], translation=[], max_attempts=4, retries=1, cooldown_seconds=60),
                legacy_primary=copy.deepcopy(primary))


def migrate(settings, primary, *, fresh=False):
    """Pure, idempotent migration; original runtime settings remain intact."""
    if settings.get('subtitle_profile') is not None:
        return normalize(settings['subtitle_profile'])
    local = _profile()
    local.update(asr='whisper-ja-1.5B', translation='sakura-galtransl-7b-v3.7')
    cloud = _profile()
    # Unconfigured service placeholders intentionally do not select account models.
    profiles = dict(cloud=cloud, local=local, custom=_profile())
    current = _profile(primary)
    current['processing'] = settings.get('subtitle_cloud_strategy', current['processing'])
    def remote(model):
        return isinstance(model, str) and model.startswith('user-')
    # User registrations can be local; without resolving the registry use custom.
    mode = 'custom' if any(remote(current[k]) for k in ('asr', 'translation')) else 'local'
    profiles[mode] = current
    value = dict(schema_version=VERSION, plan_version=PLAN_VERSION, active_mode='cloud' if fresh else mode,
                 profiles=profiles, migration=dict(source='legacy', original_primary=copy.deepcopy(primary)))
    return normalize(value)


def drop_draft(value):
    """D2: the draft -> refine flow is gone. Strip every trace of it from saved settings, profiles and frozen
    plans in place (recursively); a former draft model is simply dropped, never turned into a fallback. A legacy
    `translation.mode == 'tiered'` becomes 'standard' using its refine model. Idempotent."""
    if isinstance(value, dict):
        if isinstance(value.get('models'), dict):
            value['models'].pop('draft', None)
        if value.get('flow') == 'draft_refine':
            value['flow'] = 'direct'
        if 'legacy_primary' in value and 'asr' in value and 'translation' in value:
            value.pop('draft', None)                       # a saved profile
        translation = value.get('translation')
        if isinstance(translation, dict) and ('baseline_model_id' in translation or translation.get('mode') == 'tiered'):
            if translation.get('mode') == 'tiered':
                translation['mode'] = 'standard'
                refine = translation.get('refine_model_id')
                if refine:
                    translation['model_id'] = refine
            translation.pop('baseline_model_id', None)
            translation.pop('refine_model_id', None)
        for child in value.values():
            drop_draft(child)
    elif isinstance(value, list):
        for child in value:
            drop_draft(child)
    return value


def normalize(value):
    assert_secret_free(value)
    if isinstance(value, dict):
        value = drop_draft(copy.deepcopy(value))
    if not isinstance(value, dict) or value.get('schema_version') != VERSION:
        raise ConfigurationError('unsupported_schema')
    if set(value) != {'schema_version', 'plan_version', 'active_mode', 'profiles', 'migration'}:
        raise ConfigurationError('invalid_configuration')
    if value.get('active_mode') not in MODES or set(value.get('profiles', {})) != set(MODES):
        raise ConfigurationError('invalid_mode')
    out = copy.deepcopy(value)
    for mode, profile in out['profiles'].items():
        optional = {'connection_pool'} if mode == 'cloud' else set()
        if not isinstance(profile, dict) or set(profile) - optional != set(_profile()):
            raise ConfigurationError('invalid_profile')
        if 'connection_pool' in profile:
            pool = profile['connection_pool']
            if (not isinstance(pool, dict) or set(pool) != {'connections', 'auto_switch'}
                    or type(pool['auto_switch']) is not bool or not isinstance(pool['connections'], list)
                    or len(pool['connections']) > 10):
                raise ConfigurationError('invalid_connection_pool')
            seen = set()
            for row in pool['connections']:
                if (not isinstance(row, dict) or set(row) != {'id', 'enabled'}
                        or not isinstance(row['id'], str) or not row['id']
                        or type(row['enabled']) is not bool or row['id'] in seen):
                    raise ConfigurationError('invalid_connection_pool')
                seen.add(row['id'])
        for role in ('asr', 'translation'):
            if profile[role] is not None and (not isinstance(profile[role], str) or not profile[role]):
                raise ConfigurationError('invalid_model_reference')
        if profile['processing'] not in ('smart_horizon', 'full_preload'):
            raise ConfigurationError('invalid_processing')
        if profile['whisper_policy'] not in ('auto', 'fixed'):
            raise ConfigurationError('invalid_whisper_policy')
        allowed_whisper = {'whisper-large-v3-turbo', 'whisper-large-v3'}
        if not isinstance(profile['whisper_models'], list) or not set(profile['whisper_models']) <= allowed_whisper:
            raise ConfigurationError('invalid_whisper_models')
        policy = profile['fallback']
        if not isinstance(policy, dict) or set(policy) != {'asr', 'translation', 'max_attempts', 'retries', 'cooldown_seconds'}:
            raise ConfigurationError('invalid_fallback')
        for role in ('asr', 'translation'):
            ids = policy[role]
            if not isinstance(ids, list) or any(not isinstance(x, str) or not x for x in ids):
                raise ConfigurationError('invalid_fallback')
            policy[role] = list(dict.fromkeys(x for x in ids if x != profile[role]))
        for key, low, high in (('max_attempts', 1, 12), ('retries', 0, 3), ('cooldown_seconds', 1, 3600)):
            if type(policy[key]) is not int or not low <= policy[key] <= high:
                raise ConfigurationError('invalid_attempt_budget')
        if not isinstance(profile['legacy_primary'], dict):
            raise ConfigurationError('invalid_legacy_primary')
    if out.get('plan_version') != PLAN_VERSION:
        raise ConfigurationError('unsupported_plan')
    return out


def project_connection_pool(value, settings=None):
    """Project an explicit cloud connection pool onto the existing model fallback contract.

    Older profiles keep their exact bindings until the user first edits the pool.
    Each role uses only a currently validated model of that connection; nothing
    falls back to another account outside the explicitly enabled pool.
    """
    from .model_registry import list_connections, _read
    config = normalize(value)
    cloud = config['profiles']['cloud']
    pool = cloud.get('connection_pool')
    if pool is None:
        return config
    cloud['legacy_primary'] = {}
    connections = {c['id']: c for c in list_connections(settings)}
    rows = _read(settings)
    for entry in pool['connections']:
        connection = connections.get(entry['id'])
        if connection is None or connection.get('api_family') != 'groq':
            raise ConfigurationError('unknown_pool_connection')
    enabled = [c for c in pool['connections'] if c['enabled']]
    if not pool['auto_switch']:
        enabled = enabled[:1]
    for role in ('asr', 'translation'):
        candidates = []
        for entry in enabled:
            validation = connections[entry['id']].get('validation', {}).get(role, {})
            if validation.get('status') not in ('verified', 'reachable', 'legacy_tested'):
                continue
            model = next((m for m in rows if m.get('connection_id') == entry['id'] and m['kind'] == role
                          and (not validation.get('remote_model') or m.get('remote_model') == validation['remote_model'])), None)
            if model:
                candidates.append(model['id'])
        cloud[role] = candidates[0] if candidates else None
        cloud['fallback'][role] = candidates[1:]
    # The first ASR candidate can also have one same-connection Whisper sibling.
    cloud['fallback']['max_attempts'] = max(cloud['fallback']['max_attempts'], len(enabled) + 1)
    return config


def resolve(value, settings=None):
    """Freeze concrete bindings and ordered candidates, excluding secrets/status.

    Missing references produce readiness errors rather than choosing the first model.
    No cloud calls, downloads or fallback execution happen here.
    """
    from .model_registry import resolve_model
    from .combos import is_cloud_enabled
    config = project_connection_pool(value, settings)
    mode = config['active_mode']
    profile = config['profiles'][mode]
    errors = []
    fields = ('id', 'kind', 'provider', 'connection_id', 'base_url', 'api_family',
              'azure_region', 'remote_model', 'recipe', 'model_path')

    def binding(model_id, kind):
        if model_id is None:
            return None
        try:
            model = resolve_model(model_id, kind, settings)
        except (ValueError, KeyError):
            errors.append(dict(code='unknown_model', model_id=model_id, kind=kind))
            return None
        from .model_registry import is_cloud_model
        cloud = is_cloud_model(model)
        if cloud and mode == 'local':
            errors.append(dict(code='local_mode_cloud_candidate', model_id=model_id))
        if cloud and not is_cloud_enabled(settings):
            errors.append(dict(code='cloud_disabled', model_id=model_id))
        validation = model.get('capability_validation', {})
        if cloud and (validation.get('status') not in ('verified', 'reachable', 'legacy_tested')
                      or validation.get('remote_model', model.get('remote_model')) != model.get('remote_model')):
            errors.append(dict(code='capability_unverified', model_id=model_id, kind=kind))
        from .environment import model_ref
        return model_ref({key: copy.deepcopy(model[key]) for key in fields if key in model}, settings)

    resolved = {role: binding(profile[role], 'asr' if role == 'asr' else 'translation')
                for role in ('asr', 'translation')}
    fallbacks = {role: [binding(mid, role) for mid in profile['fallback'][role]] for role in ('asr', 'translation')}
    for role in ('asr', 'translation'):
        if profile[role] is None:
            errors.append(dict(code='not_configured', role=role))
    return dict(schema_version=VERSION, plan_version=PLAN_VERSION, mode=mode,
                flow='direct', models=resolved,
                fallback=fallbacks, policy=copy.deepcopy(profile['fallback']),
                processing=profile['processing'], whisper_policy=profile['whisper_policy'],
                whisper_models=copy.deepcopy(profile['whisper_models']) if profile['whisper_policy'] == 'auto' else [],
                ready=not errors, errors=errors)


RUNTIME_CONTRACT = 'cloud-first-1'


def runtime_plan(settings):
    """Only explicitly activated snapshots override legacy jobs."""
    if (settings or {}).get('subtitle_runtime_contract') == RUNTIME_CONTRACT:
        return settings['subtitle_resolved']
    return None


def runtime_uses_cloud(settings):
    plan = runtime_plan(settings)
    from .model_registry import is_cloud_model
    return bool(plan and any(is_cloud_model(row) for row in
        [*plan['models'].values(), *plan.get('fallback', {}).get('asr', []),
         *plan.get('fallback', {}).get('translation', [])]))


def runtime_primary(settings, primary):
    """Project the frozen contract into the existing stage pipeline."""
    plan = runtime_plan(settings)
    if not plan:
        return copy.deepcopy(primary)
    if not plan['ready']:
        raise ConfigurationError('configuration_not_ready')
    profile = (settings.get('subtitle_profile') or {}).get('profiles', {}).get(plan['mode'], {})
    saved = profile.get('legacy_primary')
    if saved:
        from .combos import normalize_combo
        row = normalize_combo({**saved, 'id': 'primary'})
    else:
        row = copy.deepcopy(primary)
    asr, trans = (plan['models'][r] for r in ('asr', 'translation'))
    row.setdefault('asr', {}).update(model=asr['id'],
        mode='dual' if (saved and saved.get('asr', {}).get('mode') == 'dual') or asr['id'] == '__dual__' else 'diagnostic')
    translation = row.setdefault('translation', {})
    translation.update(model_id=trans['id'], mode='standard')
    translation.pop('baseline_model_id', None)
    translation.pop('refine_model_id', None)
    row['cloud_strategy'] = plan['processing']
    return row


def frozen_binding(settings, model_id, kind):
    plan = runtime_plan(settings)
    if plan:
        for binding in [*plan['models'].values(), *plan.get('fallback', {}).get(kind, [])]:
            if binding and binding['id'] == model_id and binding['kind'] == kind:
                return copy.deepcopy(binding)
    return None


def translation_flow(config=None, settings=None):
    """Return explicit (model, None, options) (no refinement stage, D2), never guess accounts."""
    plan = runtime_plan(settings)
    options = copy.deepcopy((config or {}).get('translation') or {})
    if plan:
        if not plan['ready']:
            raise ConfigurationError('configuration_not_ready')
        return plan['models']['translation']['id'], None, options
    if options.get('model_id'):
        return options['model_id'], None, options
    raise ConfigurationError('translation_not_configured')


# --------------------------------------------------------------------------------------------------
# M6a: one frozen plan per subtitle track (primary + comparison rows)
# --------------------------------------------------------------------------------------------------

PRIMARY_TRACK = 'primary'

TRACK_ERROR_MESSAGES = {
    'unknown_model': '找不到模型 {model_id}',
    'cloud_disabled': '尚未允許雲端外連（{model_id}）',
    'capability_unverified': '雲端模型尚未通過驗證（{model_id}）',
    'not_configured': '尚未選擇{role_name}模型',
    'local_mode_cloud_candidate': '本機方案不可使用雲端模型（{model_id}）',
    'unsupported_correction': '比較字幕暫不支援 ASR 校正，請把校正設為「無」',
    'unsupported_decode': '比較字幕暫不支援自訂解碼參數',
    'unsupported_segmentation': '比較字幕只支援逐句（utterance）分段',
    'invalid_combo': '比較字幕設定無效：{detail}',
    'invalid_configuration': '比較字幕設定無效（{detail}）',
}


def track_error_message(error):
    """zh-TW text of one track readiness error (resolve() error dict or an M6a combo check)."""
    error = dict(error or {})
    role = error.get('role')
    error.setdefault('role_name', {'asr': '語音辨識', 'translation': '翻譯'}.get(role, role or ''))
    error.setdefault('model_id', '')
    error.setdefault('detail', '')
    template = TRACK_ERROR_MESSAGES.get(error.get('code'))
    if template is None:
        return '字幕設定無效（%s）' % error.get('code')
    try:
        return template.format(**error)
    except (KeyError, IndexError):
        return template


def comparison_rows(settings):
    """[(row id, normalized combo | None, error | None)] of the enabled comparison rows of a queue snapshot
    (`subtitle_configs`[1:], else `subtitle_comparison_rows` when `subtitle_comparison_enabled`)."""
    from .combos import ComboError, normalize_combo
    settings = settings or {}
    configs = settings.get('subtitle_configs')
    if isinstance(configs, list):
        raw_rows = configs[1:]
    elif settings.get('subtitle_comparison_enabled'):
        raw_rows = settings.get('subtitle_comparison_rows') or []
    else:
        raw_rows = []
    out = []
    for raw in raw_rows:
        if not isinstance(raw, dict) or not isinstance(raw.get('id'), str) or not raw['id'] \
                or raw['id'] == PRIMARY_TRACK or '/' in raw['id']:
            continue
        try:
            out.append((raw['id'], normalize_combo({k: v for k, v in raw.items() if k != 'source'}), None))
        except (ComboError, ValueError, TypeError) as exc:
            out.append((raw['id'], None, dict(code='invalid_combo', detail=str(exc))))
    return out


def _combo_asr_model(combo):
    """Model registry id of a combo's ASR: dual -> '__dual__', a built-in alias -> its folder-name id."""
    asr = combo.get('asr') or {}
    if asr.get('mode') == 'dual':
        return '__dual__'
    model = asr.get('model')
    if isinstance(model, str) and model.startswith('user-'):
        return model
    from .runtime import model_dir_name
    return model_dir_name(model) if model else None


def _combo_unsupported(combo):
    from .combos import ASR_DEFAULT_DECODE
    errors = []
    asr = combo.get('asr') or {}
    if (combo.get('correction') or {}).get('mode', 'none') != 'none':
        errors.append(dict(code='unsupported_correction'))
    if (asr.get('decode') or {}) != dict(ASR_DEFAULT_DECODE.get(asr.get('model'), {})):
        errors.append(dict(code='unsupported_decode'))
    if asr.get('segmentation', 'utterance') != 'utterance':
        errors.append(dict(code='unsupported_segmentation'))
    return errors


def _empty_plan(mode='custom', processing='smart_horizon', errors=()):
    return dict(schema_version=VERSION, plan_version=PLAN_VERSION, mode=mode, flow='direct',
                models=dict(asr=None, translation=None), fallback=dict(asr=[], translation=[]),
                policy={}, processing=processing, whisper_policy='auto', whisper_models=[], ready=False,
                errors=list(errors))


def comparison_plan(settings, combo, *, processing=None):
    """Frozen plan of one comparison row (design 6.1 M1: no fallback). Same migrate + resolve as the primary,
    with the row's ASR / translation models. Extra keys: track, combo_id, label, combo, reasons (zh-TW)."""
    settings = settings or {}
    errors = _combo_unsupported(combo)
    strategy = processing or settings.get('subtitle_cloud_strategy', 'smart_horizon')
    try:
        value = migrate({'subtitle_cloud_strategy': strategy}, combo)
        profile = value['profiles'][value['active_mode']]
        profile['asr'] = _combo_asr_model(combo)
        profile['fallback'] = dict(profile['fallback'], asr=[], translation=[])
        plan = resolve(value, settings)
    except ConfigurationError as exc:
        plan = _empty_plan(processing=strategy, errors=[dict(code='invalid_configuration', detail=exc.code)])
    plan['fallback'] = dict(asr=[], translation=[])           # comparison rows never fall back
    plan['errors'] = errors + list(plan.get('errors') or [])
    plan['ready'] = not plan['errors']
    plan.update(track=combo['id'], combo_id=combo['id'], label=combo.get('label') or combo['id'],
                combo=copy.deepcopy(combo), reasons=[track_error_message(e) for e in plan['errors']])
    return plan


def track_runtime_plans(settings):
    """{track id: frozen plan}: 'primary' = runtime_plan(settings) (None for legacy jobs), then one plan per
    enabled comparison row (track id = combo id) in editor order. A comparison plan with ready=False carries
    `errors` / `reasons`; the scheduler gives that track no work and the primary is unaffected.

    A queue snapshot carries `subtitle_track_plans` (frozen at enqueue time, `subtitle_jobs.freeze_track_plans`);
    when present it is returned as is, so later edits of the global profile or rows cannot change a queued job."""
    frozen = (settings or {}).get('subtitle_track_plans')
    if isinstance(frozen, dict) and PRIMARY_TRACK in frozen:
        return copy.deepcopy(frozen)
    primary = runtime_plan(settings)
    plans = {PRIMARY_TRACK: primary}
    processing = (primary or {}).get('processing')
    for track_id, combo, error in comparison_rows(settings):
        if track_id in plans:
            continue
        if combo is None:
            plan = _empty_plan(errors=[error])
            plan.update(track=track_id, combo_id=track_id, label=track_id, combo=None,
                        reasons=[track_error_message(error)])
        else:
            plan = comparison_plan(settings, combo, processing=processing)
        plans[track_id] = plan
    return plans
