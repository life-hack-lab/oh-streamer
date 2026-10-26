"""M4a: one-candidate-at-a-time fallback (CandidateChain / decide / run_candidate) and the slice-plan side
(TaskState.tried, owner dispatch, derived per-candidate reasons). Pure CPU: fake providers, temp Ledger."""
import json
from pathlib import Path
import tempfile
import unittest

from oh_streamer.subtitles import fallback as fb
from oh_streamer.subtitles import slice_plan as sp
from oh_streamer.subtitles.providers.http import ProviderError


def row(name, kind='asr', local=False, groq=False, model=None, connection=None):
    out = dict(id='user-' + name, kind=kind, provider='local' if local else 'openai-compatible',
               connection_id=connection or ('connection-' + name), remote_model=model or name,
               base_url='https://api.groq.com/openai/v1' if groq else 'https://example.org/v1',
               recipe='whisper' if kind == 'asr' else 'general-llm', model_path='m-' + name)
    if groq:
        out['api_family'] = 'groq'
    return out


def frozen(asr, translation=None, asr_fallback=(), mt_fallback=(), **extra):
    plan = dict(models=dict(asr=asr, translation=translation or row('mt', 'translation')),
                fallback=dict(asr=list(asr_fallback), translation=list(mt_fallback)),
                policy=dict(max_attempts=4, retries=1, cooldown_seconds=60), whisper_policy='fixed',
                whisper_models=[])
    plan.update(extra)
    return plan


CLOUD_LOCAL = frozen(row('groq'), asr_fallback=[row('gpu', local=True)])          # cloud primary + local backup
LOCAL_CLOUD = frozen(row('gpu', local=True), asr_fallback=[row('groq')])          # local primary + cloud backup
NOW = 1000.0


def err(kind, retry_after=None, **attrs):
    e = ProviderError(kind, kind + ' happened', retryable=True, retry_after=retry_after)
    for k, v in attrs.items():
        setattr(e, k, v)
    return e


def one_slice_plan():
    return sp.build_slice_plan(300.0)


class ChainTests(unittest.TestCase):
    def test_chain_order_executor_and_dedupe(self):
        plan = frozen(row('groq'), asr_fallback=[row('groq'), row('gpu', local=True), row('azure')])
        chain = fb.CandidateChain(plan, 'asr')
        self.assertEqual([c.id for c in chain], ['user-groq', 'user-gpu', 'user-azure'])
        self.assertEqual([c.executor for c in chain], ['cloud', 'local', 'cloud'])
        self.assertEqual(fb.CandidateChain(plan, 'mt').role, 'translation')

    def test_groq_sibling_auto_and_fixed(self):
        primary = row('g', groq=True, model='whisper-large-v3')
        auto = frozen(primary, whisper_policy='auto', whisper_models=['whisper-large-v3-turbo'])
        ids = [c.id for c in fb.CandidateChain(auto, 'asr')]
        self.assertEqual(ids, ['user-g', 'user-g#whisper-large-v3-turbo'])
        self.assertTrue(fb.CandidateChain(auto, 'asr').candidates[1].sibling)
        # fixed: the sibling on the same connection is not a fallback; another connection is
        sib = dict(primary, id='user-sib', remote_model='whisper-large-v3-turbo')
        fixed = frozen(primary, asr_fallback=[sib, dict(sib, id='user-other', connection_id='c2')])
        self.assertEqual([c.id for c in fb.CandidateChain(fixed, 'asr')], ['user-g', 'user-other'])

    def test_fresh_work_starts_at_head(self):
        pick = fb.next_candidate(CLOUD_LOCAL, 'asr', sp.TaskState(), NOW)
        self.assertEqual((pick.candidate.id, pick.retry_at, pick.executor), ('user-groq', None, 'cloud'))

    def test_fresh_work_skips_cooling_primary(self):
        cooling = {'user-groq': NOW + 60}
        chain = fb.CandidateChain(CLOUD_LOCAL, 'asr', cooldown=lambda c: cooling.get(c.id))
        pick = chain.next(sp.TaskState(), NOW)
        self.assertEqual((pick.candidate.id, pick.executor), ('user-gpu', 'local'))
        # short cooldown: wait on the head instead of handing work away
        cooling['user-groq'] = NOW + 5
        pick = chain.next(sp.TaskState(), NOW)
        self.assertEqual((pick.candidate.id, pick.retry_at), ('user-groq', NOW + 5))
        # everything cooling long: the one available soonest
        only = frozen(row('groq'))
        cooling['user-groq'] = NOW + 90
        pick = fb.CandidateChain(only, 'asr', cooldown=lambda c: cooling.get(c.id)).next(None, NOW)
        self.assertEqual((pick.candidate.id, pick.retry_at), ('user-groq', NOW + 90))

    def test_ledger_cooldown_reads_real_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = [NOW]
            ledger = fb.Ledger(Path(tmp) / 'l.db', clock=lambda: clock[0])
            chain0 = fb.CandidateChain(CLOUD_LOCAL, 'asr')
            head = chain0.candidates[0]
            ck = fb.candidate_key(ledger.account(head.row, 'k'), head.row)
            ledger.fail(ck, err('rate_limit', 45), 45)
            chain = fb.CandidateChain(CLOUD_LOCAL, 'asr', cooldown=fb.ledger_cooldown(ledger, lambda r: 'k'))
            self.assertEqual(chain.next(None, NOW).candidate.id, 'user-gpu')
            self.assertEqual(ledger.peek(ck)[1], 'rate_limit')
            clock[0] = NOW + 46
            self.assertIsNone(ledger.peek(ck))


class DecideTests(unittest.TestCase):
    def decide(self, plan, state, cid, error, now=NOW, **kw):
        chain = fb.CandidateChain(plan, 'asr', **kw)
        return chain.decide(state, chain.get(cid), error, now)

    def test_short_429_defers_on_same_candidate(self):
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('rate_limit', 5))
        self.assertEqual((d.action, d.candidate.id, d.retry_at, d.attempt), (fb.WAIT_QUOTA, 'user-groq', NOW + 5, 0))
        self.assertIsNone(d.wait_executor)
        # admission pacing (no request sent) is the same kind of wait
        d = self.decide(CLOUD_LOCAL, sp.TaskState(attempt=1), 'user-groq', err('rate_limit', 3, not_sent=True))
        self.assertEqual((d.action, d.attempt), (fb.WAIT_QUOTA, 1))

    def test_long_429_hands_to_local_with_reclaim(self):
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('rate_limit', 47))
        self.assertEqual((d.action, d.candidate.id, d.wait_executor), (fb.SWITCH, 'user-gpu', 'local'))
        self.assertTrue(d.handoff)
        self.assertEqual((d.reclaim_at, d.reclaim_executor), (NOW + 47, 'cloud'))
        self.assertEqual(d.tried['user-groq']['retry_at'], NOW + 47)
        # 429 without Retry-After uses the short default (20 s): stays on the candidate
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('rate_limit'))
        self.assertEqual((d.action, d.retry_at), (fb.WAIT_QUOTA, NOW + fb.RATE_LIMIT_DEFAULT_COOLDOWN))

    def test_three_sent_429s_in_a_row_switch_even_when_short(self):
        state, actions = sp.TaskState(), []
        for _ in range(3):
            d = self.decide(CLOUD_LOCAL, state, 'user-groq', err('rate_limit', 5))
            actions.append(d.action)
            state = sp.TaskState(tried=d.tried)
        self.assertEqual(actions, [fb.WAIT_QUOTA, fb.WAIT_QUOTA, fb.SWITCH])
        self.assertEqual((d.candidate.id, d.reclaim_executor), ('user-gpu', 'cloud'))
        self.assertIsNotNone(d.reclaim_at)

    def test_not_sent_pacing_neither_counts_nor_resets_the_429_streak(self):
        state, actions = sp.TaskState(), []
        for e in (err('rate_limit', 5), err('rate_limit', 3, not_sent=True), err('rate_limit', 5),
                  err('rate_limit', 3, not_sent=True), err('rate_limit', 5)):
            d = self.decide(CLOUD_LOCAL, state, 'user-groq', e)
            actions.append(d.action)
            state = sp.TaskState(tried=d.tried)
        self.assertEqual(actions, [fb.WAIT_QUOTA] * 4 + [fb.SWITCH])

    def test_progress_restarts_the_429_streak(self):
        state = sp.TaskState()
        for _ in range(2):
            d = self.decide(CLOUD_LOCAL, state, 'user-groq', err('rate_limit', 5))
            state = sp.TaskState(tried=d.tried)
        d = self.decide(CLOUD_LOCAL, state, 'user-groq', err('rate_limit', 5, progressed=True))
        self.assertEqual((d.action, d.tried['user-groq']['short_429']), (fb.WAIT_QUOTA, 1))

    def test_single_candidate_keeps_waiting_after_many_429s(self):
        state = sp.TaskState()
        for _ in range(5):
            d = self.decide(frozen(row('groq')), state, 'user-groq', err('rate_limit', 5))
            self.assertEqual(d.action, fb.WAIT_QUOTA)
            state = sp.TaskState(tried=d.tried)

    def test_long_quota_without_other_candidate_waits(self):
        d = self.decide(frozen(row('groq')), sp.TaskState(), 'user-groq', err('rate_limit', 300))
        self.assertEqual((d.action, d.retry_at), (fb.WAIT_QUOTA, NOW + 300))

    def test_quota_daily_switches_even_when_short(self):
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('quota_daily', 10))
        self.assertEqual((d.action, d.candidate.id, d.reclaim_at), (fb.SWITCH, 'user-gpu', NOW + 10))
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('quota_daily'))
        self.assertEqual(d.reclaim_at, NOW + fb.DAILY_QUOTA_DEFAULT_WAIT)

    def test_cloud_to_cloud_switch_needs_no_reservation(self):
        plan = frozen(row('groq'), asr_fallback=[row('azure')])
        d = self.decide(plan, sp.TaskState(), 'user-groq', err('quota_daily'))
        self.assertEqual((d.action, d.candidate.id, d.wait_executor, d.reclaim_at), (fb.SWITCH, 'user-azure', None, None))

    def test_local_error_retries_then_cloud_without_reclaim(self):
        plan, state = sp.build_slice_plan(300.0), sp.TaskState()
        actions = []
        for _ in range(3):
            chain = fb.CandidateChain(LOCAL_CLOUD, 'asr')
            d = chain.decide(plan.slices[0].get('asr'), chain.get('user-gpu'), RuntimeError('cuda boom'), NOW,
                             kind='error')
            fb.apply_decision(plan, 0, 'asr', d)
            actions.append(d.action)
        self.assertEqual(actions, [fb.RETRY, fb.RETRY, fb.SWITCH])
        st = plan.slices[0].get('asr')
        self.assertEqual((st.candidate, st.wait_executor, st.attempt, st.reclaim_at), ('user-groq', 'cloud', 0, None))
        self.assertEqual(st.tried['user-gpu']['kind'], 'error')
        self.assertNotIn('retry_at', st.tried['user-gpu'])
        # retry backoff 5 s then 10 s
        chain = fb.CandidateChain(LOCAL_CLOUD, 'asr')
        self.assertEqual(chain.decide(state, chain.get('user-gpu'), None, NOW, kind='oom').retry_at, NOW + 5)

    def test_retryable_cloud_errors_retry_to_limit(self):
        st = sp.TaskState(candidate='user-groq', attempt=2)
        d = self.decide(CLOUD_LOCAL, st, 'user-groq', err('server'))
        self.assertEqual((d.action, d.candidate.id), (fb.SWITCH, 'user-gpu'))
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('network', 12))
        self.assertEqual((d.action, d.retry_at, d.attempt), (fb.RETRY, NOW + 12, 1))

    def test_auth_switches_immediately(self):
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('auth'))
        self.assertEqual((d.action, d.candidate.id), (fb.SWITCH, 'user-gpu'))

    def test_content_blocked_does_not_switch(self):
        d = self.decide(CLOUD_LOCAL, sp.TaskState(), 'user-groq', err('content_blocked'))
        self.assertEqual((d.action, d.fail_kind), (fb.FAILED, 'content_blocked'))

    def test_exhausted_failed_with_every_reason(self):
        plan = sp.build_slice_plan(300.0)
        chain = fb.CandidateChain(CLOUD_LOCAL, 'asr')
        fb.apply_decision(plan, 0, 'asr', chain.decide(sp.TaskState(), chain.get('user-groq'), err('auth'), NOW))
        st = plan.slices[0].get('asr')
        chain = fb.CandidateChain(CLOUD_LOCAL, 'asr', max_attempts=1)
        d = chain.decide(st, chain.get('user-gpu'), None, NOW + 1, kind='oom')
        self.assertEqual((d.action, d.fail_kind), (fb.FAILED, 'exhausted'))
        fb.apply_decision(plan, 0, 'asr', d)
        status = sp.derive_status(plan, now=NOW + 2)
        self.assertTrue(status.stuck and status.blocked_playback)
        failure = status.failures[0]
        self.assertTrue(failure.exhausted)
        self.assertEqual([(c.id, c.kind, c.executor) for c in failure.candidates],
                         [('user-groq', 'auth', 'cloud'), ('user-gpu', 'oom', 'local')])
        self.assertIn('user-groq: auth', failure.detail)
        # retry button clears everything
        plan.reset_failed()
        self.assertEqual(plan.slices[0].get('asr'), sp.TaskState())

    def test_quota_left_candidate_returns_instead_of_exhausting(self):
        # cloud quota -> local -> local errors: the cloud candidate recovers later, so wait for it
        st = sp.TaskState(candidate='user-gpu', tried={'user-groq': dict(kind='rate_limit', at=NOW,
                                                                           retry_at=NOW + 100)})
        chain = fb.CandidateChain(CLOUD_LOCAL, 'asr', max_attempts=1)
        d = chain.decide(st, chain.get('user-gpu'), None, NOW + 10, kind='error')
        self.assertEqual((d.action, d.candidate.id, d.retry_at, d.wait_executor),
                         (fb.SWITCH, 'user-groq', NOW + 100, 'cloud'))

    def test_backup_added_later_is_used(self):
        # the chain is read from the plan at decision time: same state, same error, backup appended later
        st = sp.TaskState(candidate='user-groq', attempt=2)
        before = frozen(row('groq'))
        self.assertEqual(fb.decide_after_failure(before, 'asr', st, 'user-groq', err('server'), NOW).fail_kind,
                         'exhausted')
        after = frozen(row('groq'), asr_fallback=[row('azure'), row('gpu', local=True)])
        d = fb.decide_after_failure(after, 'asr', st, 'user-groq', err('server'), NOW)
        self.assertEqual((d.action, d.candidate.id), (fb.SWITCH, 'user-azure'))
        # pending work that already left its candidates picks the new backup up too
        st = sp.TaskState(candidate='user-gpu', tried={'user-groq': dict(kind='auth', at=NOW),
                                                       'user-gpu': dict(kind='error', at=NOW)})
        self.assertEqual(fb.next_candidate(frozen(row('groq'), asr_fallback=[row('gpu', local=True)]), 'asr',
                                           st, NOW).candidate.id, 'user-gpu')   # still its current candidate
        st.candidate = None
        self.assertTrue(fb.next_candidate(frozen(row('groq'), asr_fallback=[row('gpu', local=True)]), 'asr',
                                          st, NOW).exhausted)
        self.assertEqual(fb.next_candidate(after, 'asr', st, NOW).candidate.id, 'user-azure')
        # ids, not positions: a backup inserted in front of an already-tried one does not shift its record
        st2 = sp.TaskState(candidate='user-azure', tried={'user-groq': dict(kind='auth', at=NOW),
                                                           'user-azure': dict(kind='auth', at=NOW)})
        reordered = frozen(row('groq'), asr_fallback=[row('new'), row('azure')])
        self.assertEqual(fb.next_candidate(reordered, 'asr', st2, NOW).candidate.id, 'user-new')

    def test_removed_candidate_id(self):
        st = sp.TaskState(candidate='user-gone')
        d = fb.decide_after_failure(CLOUD_LOCAL, 'asr', st, 'user-gone', err('auth'), NOW)
        self.assertEqual((d.action, d.candidate.id), (fb.SWITCH, 'user-groq'))


class ReclaimAndOwnerTests(unittest.TestCase):
    def handed_off(self):
        plan = sp.build_slice_plan(600.0, first_slice_seconds=None)   # two slices
        chain = fb.CandidateChain(CLOUD_LOCAL, 'asr')
        fb.apply_decision(plan, 0, 'asr', chain.decide(sp.TaskState(), chain.get('user-groq'),
                                                         err('rate_limit', 47), NOW))
        return plan

    def test_cloud_recovers_and_takes_pending_work_back(self):
        plan = self.handed_off()
        st = plan.slices[0].get('asr')
        self.assertEqual((st.wait_executor, st.reclaim_at, st.reclaim_executor), ('local', NOW + 47, 'cloud'))
        local, cloud = sp.Executor('local'), sp.Executor('cloud')
        # stored reservation path (no ctx.owner)
        self.assertEqual(sp.task_owner(st, NOW + 1), 'local')
        self.assertEqual(sp.task_owner(st, NOW + 47), 'cloud')
        before = sp.SchedulerContext(now=NOW + 1)
        self.assertEqual(sp.select_next_task(plan, before, local).slice_index, 0)
        self.assertEqual(sp.select_next_task(plan, before, cloud).slice_index, 1)
        after = sp.SchedulerContext(now=NOW + 50)
        self.assertEqual(sp.select_next_task(plan, after, cloud).slice_index, 0)
        # chain path agrees
        self.assertEqual(fb.CandidateChain(CLOUD_LOCAL, 'asr').owner(st, NOW + 1), 'local')
        self.assertEqual(fb.CandidateChain(CLOUD_LOCAL, 'asr').owner(st, NOW + 50), 'cloud')
        status = sp.derive_status(plan, now=NOW + 1)
        self.assertEqual((status.waiting_executor, status.reclaim_at), ('local', NOW + 47))
        self.assertEqual([c.id for c in status.blocking_tried], ['user-groq'])
        # the cloud takes it: running on its own candidate again, attempt reset
        plan.mark_running(0, 'asr', 'cloud', candidate='user-groq')
        st = plan.slices[0].get('asr')
        self.assertEqual((st.status, st.candidate, st.attempt), (sp.RUNNING, 'user-groq', 0))
        self.assertIn('user-groq', st.tried)

    def test_local_started_keeps_it(self):
        plan = self.handed_off()
        plan.mark_running(0, 'asr', 'local', candidate='user-gpu')
        cloud = sp.Executor('cloud')
        task = sp.select_next_task(plan, sp.SchedulerContext(now=NOW + 100), cloud)
        self.assertEqual(task.slice_index, 1)          # slice 0 is running on local, not stolen
        plan.release(0, 'asr')                          # crash: back to pending, bookkeeping kept
        self.assertEqual(plan.slices[0].get('asr').candidate, 'user-gpu')

    def test_error_handoff_is_not_reclaimed(self):
        plan = sp.build_slice_plan(300.0)
        chain = fb.CandidateChain(LOCAL_CLOUD, 'asr', max_attempts=1)
        fb.apply_decision(plan, 0, 'asr', chain.decide(sp.TaskState(), chain.get('user-gpu'), None, NOW,
                                                         kind='error'))
        st = plan.slices[0].get('asr')
        for t in (NOW + 1, NOW + 10_000):
            self.assertEqual(sp.task_owner(st, t), 'cloud')
            self.assertEqual(fb.CandidateChain(LOCAL_CLOUD, 'asr').owner(st, t), 'cloud')

    def test_selector_owner_cloud_primary_local_backup(self):
        plan = sp.build_slice_plan(600.0, first_slice_seconds=None)
        ctx = sp.SchedulerContext(now=NOW, owner=fb.owner_function(CLOUD_LOCAL, now=NOW))
        self.assertIsNone(sp.select_next_task(plan, ctx, sp.Executor('local')))
        self.assertEqual(sp.select_next_task(plan, ctx, sp.Executor('cloud')).slice_index, 0)
        # same with only the static default owner
        ctx = sp.SchedulerContext(now=NOW, default_owner='cloud')
        self.assertIsNone(sp.select_next_task(plan, ctx, sp.Executor('local')))

    def test_selector_owner_local_primary_cloud_backup_d1(self):
        plan = sp.build_slice_plan(600.0, first_slice_seconds=None)
        ctx = sp.SchedulerContext(now=NOW, owner=fb.owner_function(LOCAL_CLOUD, now=NOW))
        self.assertIsNone(sp.select_next_task(plan, ctx, sp.Executor('cloud')))   # no spill-over while busy
        self.assertEqual(sp.select_next_task(plan, ctx, sp.Executor('local')).slice_index, 0)

    def test_selector_owner_skips_cooling_cloud_for_fresh_work(self):
        plan = sp.build_slice_plan(300.0)
        cooling = lambda c: NOW + 60 if c.id == 'user-groq' else None
        ctx = sp.SchedulerContext(now=NOW, owner=fb.owner_function(CLOUD_LOCAL, cooldown=cooling, now=NOW))
        self.assertIsNone(sp.select_next_task(plan, ctx, sp.Executor('cloud')))
        self.assertEqual(sp.select_next_task(plan, ctx, sp.Executor('local')).stage, 'asr')

    def test_mt_role_uses_translation_chain(self):
        plan = sp.build_slice_plan(300.0)
        plan.mark_done(0, 'asr', 'user-groq')
        both = frozen(row('groq'), translation=row('sakura', 'translation', local=True))
        ctx = sp.SchedulerContext(now=NOW, owner=fb.owner_function(both, now=NOW))
        self.assertIsNone(sp.select_next_task(plan, ctx, sp.Executor('cloud')))
        self.assertEqual(sp.select_next_task(plan, ctx, sp.Executor('local')).stage, 'mt')

    def test_no_owner_information_keeps_old_behaviour(self):
        plan = sp.build_slice_plan(300.0)
        for name in ('cloud', 'local'):
            self.assertEqual(sp.select_next_task(plan, sp.SchedulerContext(now=NOW), sp.Executor(name)).stage, 'asr')


class SerializationTests(unittest.TestCase):
    def test_old_json_loads(self):
        old = {"version": 1, "duration": 300.0, "media_id": "m", "tracks": ["primary"], "slice_seconds": 300.0,
               "first_slice_seconds": 90.0, "strategy": "smart_horizon", "horizon_minutes": 30,
               "lazy": {"cuts": [0.0, 300.0], "closed": True, "failures": 0},
               "slices": [{"index": 0, "start": 0.0, "end": 300.0, "tasks": {
                   "asr/primary": {"status": "failed", "fail_kind": "auth", "fail_at": 5.0, "attempt": 1,
                                   "model_id": "user-groq"},
                   "mt/primary": {"status": "pending", "wait_executor": "local", "attempt": 2}}}]}
        plan = sp.SlicePlan.from_dict(json.loads(json.dumps(old)))
        st = plan.slices[0].get('asr')
        self.assertEqual((st.fail_kind, st.candidate, st.tried), ('auth', None, {}))
        self.assertEqual(sp.derive_status(plan).failures[0].candidates, [])
        mt = plan.slices[0].get('mt')
        self.assertEqual((mt.wait_executor, mt.attempt, mt.reclaim_at), ('local', 2, None))
        # an old attempt count belongs to whatever candidate runs first
        chain = fb.CandidateChain(CLOUD_LOCAL, 'asr')
        self.assertEqual(chain.decide(sp.TaskState(attempt=2), chain.get('user-groq'), err('server'), NOW).action,
                         fb.SWITCH)

    def test_new_fields_round_trip_and_stay_compact(self):
        st = sp.TaskState(candidate='user-gpu', tried={'user-groq': dict(kind='rate_limit', at=1.0, retry_at=9.0)},
                          wait_executor='local', reclaim_at=9.0, reclaim_executor='cloud')
        data = json.loads(json.dumps(st.to_dict()))
        self.assertEqual(sp.TaskState.from_dict(data), st)
        self.assertEqual(sp.TaskState().to_dict(), {"status": "pending"})

    def test_transitions_keep_bookkeeping(self):
        plan = sp.build_slice_plan(300.0)
        tried = {'user-groq': dict(kind='auth', at=1.0)}
        plan.slices[0].set('asr', 'primary', sp.TaskState(candidate='user-gpu', tried=tried, attempt=1))
        plan.defer(0, 'asr', NOW + 5, attempt=2)
        self.assertEqual(plan.slices[0].get('asr').tried, tried)
        plan.wait_for(0, 'asr', 'local')
        plan.mark_running(0, 'asr', 'local')
        self.assertEqual((plan.slices[0].get('asr').candidate, plan.slices[0].get('asr').attempt), ('user-gpu', 2))
        self.assertEqual(plan.release_running(), 1)
        plan.mark_failed(0, 'asr', 'error', at=NOW)
        self.assertEqual(plan.slices[0].get('asr').tried, tried)
        plan.reset_failed()
        plan.mark_done(0, 'asr', 'user-gpu')
        self.assertEqual(plan.slices[0].get('asr').tried, {})


class FakeLedger:
    """Records Ledger calls; claim / reserve behaviour is scripted."""
    def __init__(self, blocked=None, admission=None):
        self.blocked, self.admission = blocked, admission
        self.calls = []

    def account(self, row, key):
        return 'acct'

    def claim(self, key):
        self.calls.append(('claim', key))
        return self.blocked

    def reserve(self, account, *, tokens=0, audio=0):
        self.calls.append(('reserve', tokens, audio))
        if self.admission:
            raise self.admission

    def fail(self, key, error, cooldown):
        self.calls.append(('fail', error.kind, cooldown))

    def success(self, key):
        self.calls.append(('success',))

    def release_probe(self, key):
        self.calls.append(('release',))


class RunCandidateTests(unittest.TestCase):
    def setUp(self):
        self.groq = fb.CandidateChain(frozen(row('g', groq=True)), 'asr').candidates[0]
        self.local = fb.CandidateChain(LOCAL_CLOUD, 'asr').candidates[0]

    def run_with(self, ledger, operation, candidate=None):
        return fb.run_candidate(candidate or self.groq, operation, lambda r: 'key', ledger=ledger,
                                plan=frozen(row('g')), tokens=128, audio=30)

    def raising(self, error):
        def operation(row, key):
            raise error
        return operation

    def test_success(self):
        ledger = FakeLedger()
        self.assertEqual(self.run_with(ledger, lambda row, key: (row['id'], key)), ('user-g', 'key'))
        self.assertEqual([c[0] for c in ledger.calls], ['claim', 'reserve', 'success', 'release'])
        ledger = FakeLedger()
        self.run_with(ledger, lambda row, key: 1, candidate=self.local)
        self.assertNotIn('reserve', [c[0] for c in ledger.calls])      # pacing is Groq-only

    def test_429_cooldown_is_retry_after_not_60(self):
        ledger = FakeLedger()
        with self.assertRaises(ProviderError):
            self.run_with(ledger, self.raising(err('rate_limit', 4)))
        self.assertIn(('fail', 'rate_limit', 4.0), ledger.calls)
        ledger = FakeLedger()
        with self.assertRaises(ProviderError):
            self.run_with(ledger, self.raising(err('rate_limit')))
        self.assertIn(('fail', 'rate_limit', fb.RATE_LIMIT_DEFAULT_COOLDOWN), ledger.calls)

    def test_other_kinds_cooldown(self):
        for kind, expected in (('quota_daily', 60.0), ('auth', 60.0)):
            ledger = FakeLedger()
            with self.assertRaises(ProviderError):
                self.run_with(ledger, self.raising(err(kind)))
            self.assertIn(('fail', kind, expected), ledger.calls)
        for kind in ('network', 'server', 'content_blocked'):
            ledger = FakeLedger()
            with self.assertRaises(ProviderError):
                self.run_with(ledger, self.raising(err(kind)))
            self.assertFalse([c for c in ledger.calls if c[0] == 'fail'], kind)
            self.assertEqual(ledger.calls[-1], ('release',))

    def test_real_ledger_429_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = [NOW]
            ledger = fb.Ledger(Path(tmp) / 'l.db', clock=lambda: clock[0])
            with self.assertRaises(ProviderError):
                fb.run_candidate(self.local, self.raising(err('rate_limit', 4)), lambda r: 'k', ledger=ledger)
            ck = fb.candidate_key(ledger.account(self.local.row, 'k'), self.local.row)
            self.assertEqual(ledger.peek(ck)[0], NOW + 4)
            clock[0] = NOW + 4.5
            self.assertEqual(fb.run_candidate(self.local, lambda r, k: 'ok', lambda r: 'k', ledger=ledger), 'ok')
            self.assertIsNone(ledger.peek(ck))

    def test_admission_and_cooldown_waits_do_not_sleep_or_send(self):
        sent = []
        ledger = FakeLedger(admission=err('rate_limit', 2.5))
        with self.assertRaises(ProviderError) as raised:
            self.run_with(ledger, lambda row, key: sent.append(1))
        self.assertTrue(raised.exception.not_sent)
        self.assertEqual((raised.exception.retry_after, sent), (2.5, []))
        self.assertFalse([c for c in ledger.calls if c[0] == 'fail'])
        ledger = FakeLedger(blocked=ProviderError('rate_limit', 'cooling', retryable=True, retry_after=40))
        with self.assertRaises(ProviderError) as raised:
            self.run_with(ledger, lambda row, key: sent.append(1))
        self.assertTrue(raised.exception.not_sent)
        self.assertEqual(sent, [])
        # decide turns them into waits / handoffs
        chain = fb.CandidateChain(CLOUD_LOCAL, 'asr')
        d = chain.decide(sp.TaskState(), chain.get('user-groq'), raised.exception, NOW)
        self.assertEqual((d.action, d.wait_executor), (fb.SWITCH, 'local'))

    def test_credentials_error_propagates(self):
        def credentials(row):
            raise ProviderError('auth', 'no key')
        with self.assertRaises(ProviderError) as raised:
            fb.run_candidate(self.groq, lambda r, k: 1, credentials, ledger=FakeLedger())
        self.assertFalse(getattr(raised.exception, 'not_sent', False))


if __name__ == '__main__':
    unittest.main()
