"""CPU coverage of sentence-before-translation rules and original-span filtering."""
import sys
from pathlib import Path
import unittest
import json
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from oh_streamer.subtitles.core import Segment, parse_srt
from oh_streamer.subtitles.sentences import (split_sentences, sentence_cues, allocate_durations,
    filter_short, wrap_two_lines, render_sentence_srt)

try:
    import opencc  # noqa: F401
    HAVE_OPENCC = True
except ImportError:
    HAVE_OPENCC = False
needs_opencc = unittest.skipUnless(HAVE_OPENCC, "OpenCC (opencc) is not installed in this venv")


class SentenceTests(unittest.TestCase):
    def test_vad_rescue_preserves_baseline_and_ignores_overlapping_candidates(self):
        from oh_streamer.subtitles.asr import rescue_missing_intervals
        baseline = [dict(start=1., end=4.), dict(start=8., end=10.)]
        candidates = [dict(start=.9, end=4.2), dict(start=5., end=7.),
                      dict(start=6., end=7.5), dict(start=10., end=11.),
                      dict(start=12., end=12.)]
        result = rescue_missing_intervals(baseline, candidates)
        self.assertEqual(result, [baseline[0], candidates[1], baseline[1], candidates[3]])
        self.assertEqual(baseline, [dict(start=1., end=4.), dict(start=8., end=10.)])

    def test_vad_independent_chunks_use_global_offsets_and_can_be_disabled(self):
        from contextlib import nullcontext
        from oh_streamer.subtitles import asr
        from unittest.mock import MagicMock
        model = MagicMock()
        audio = [0.] * (3*16000)
        torch = MagicMock()
        torch.from_numpy.side_effect = lambda a: a
        torch.inference_mode.side_effect = nullcontext
        def detect(waveform, *args, **kwargs):
            return ([dict(start=0., end=.5)] if len(waveform)==len(audio)
                    else [dict(start=.1, end=.4)])
        with patch.object(asr, '_load_vad', return_value=(model, detect, 'mock')):
            spans, _, _ = asr._detect_speech_intervals(audio, torch=torch, rescue_chunk_seconds=1.)
            self.assertEqual(spans, [dict(start=0., end=.5), dict(start=1.1, end=1.4),
                                     dict(start=2.1, end=2.4)])
            self.assertEqual(model.reset_states.call_count, 3)
            spans, _, _ = asr._detect_speech_intervals(audio, torch=torch, rescue_chunk_seconds=0.)
            self.assertEqual(spans, [dict(start=0., end=.5)])

    def test_repetition_with_dialogue_is_hard_dropped_by_f7(self):
        from oh_streamer.subtitles.core import clean_segments
        text = 'いやいやいやいや、社長の人柄が私を動かしたんですよ。'
        result = clean_segments([Segment(0, 4.3, text)])
        self.assertEqual(result, [])
        result = clean_segments([Segment(0, 4, 'はい、ギギギギ、仕事して、あぷあぷあぷ。続けて。')])
        self.assertEqual(result, [])

    def test_repetition_only_noise_dropped(self):
        from oh_streamer.subtitles.core import clean_segments
        for text in ('ギギギギ…', 'あぷあぷあぷ…', 'はっはっはっはっ!', 'はっはっはっは!'):
            self.assertEqual(clean_segments([Segment(0, 3, text)], show_interjections=True), [])

    def test_unbalanced_wrap_stays_single_line(self):
        for text in ('いやー、中田さんの熱意には本当に感動しましたよ',
                     '夜はさ、美味しいとこ連れていってあげますよ'):
            self.assertEqual(wrap_two_lines(text), text)
        wrapped = wrap_two_lines('這是第一個句子，這是第二個句子，還有第三個句子')
        lengths = [len(line) for line in wrapped.splitlines()]
        self.assertEqual(len(lengths), 2)
        self.assertGreaterEqual(min(lengths) * 3, sum(lengths))

    def test_short_filter_default_is_one_second_whitelist(self):
        from oh_streamer.subtitles import cli
        import inspect
        defaults = inspect.signature(cli.translate_file).parameters
        self.assertEqual(defaults['short_threshold'].default, 1.0)
        self.assertEqual(defaults['short_policy'].default, 'whitelist')
        rows = [Segment(0, .999, '梅津'), Segment(1, 2, '梅津'), Segment(2, 2.5, 'はい')]
        self.assertEqual(filter_short(rows, 1.0)[0], rows[1:])

    @needs_opencc

    def test_cli_translates_sentences_before_render_and_cache_replay(self):
        from oh_streamer.subtitles import cli
        calls=[]
        class Client:
            metrics={}
            def __init__(self,*args,**kwargs): pass
            def __enter__(self): return self
            def __exit__(self,*args): pass
            def translate(self,messages,**kwargs):
                calls.append(messages)
                return '1. 第一個句子\n2. 第二個句子'
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            source,model=root/'source.json',root/'model.gguf'
            source.write_text(json.dumps({'segments':[dict(start=1,end=5,text='仕事です。次の仕事です。')]}),encoding='utf-8')
            model.write_bytes(b'test')
            with patch('oh_streamer.subtitles.llm.LlamaServerClient',Client), \
                 patch.object(cli,'assert_gpu_safe',return_value={}) as safety, \
                 patch.object(cli,'NvmlMemorySampler') as memory:
                memory.return_value.__enter__.return_value.to_dict.return_value={}
                result=cli.translate_file(source,root,model='sakura',model_path=model,
                                          llama_server='unused',sentence_mode=True)
                self.assertEqual(len(result['segments']),2)
                self.assertTrue(calls[0][1]['content'].endswith('1. 仕事です。\n2. 次の仕事です。'))
                ja=parse_srt((root/'ja.srt').read_text('utf-8'))
                zh=parse_srt((root/'sakura.zh-TW.srt').read_text('utf-8'))
                self.assertEqual([(s.start,s.end) for s in ja],[(s.start,s.end) for s in zh])
                (root/'sakura.zh-TW.srt').unlink()
                (root/'ja.srt').unlink()
                cli.translate_file(source,root,model='sakura',model_path=model,
                                   llama_server='unused',sentence_mode=True)
                self.assertEqual(safety.call_count,1)
                self.assertEqual(len(calls),1)
                self.assertEqual(len(parse_srt((root/'sakura.zh-TW.srt').read_text('utf-8'))),2)
                self.assertEqual(len(parse_srt((root/'ja.srt').read_text('utf-8'))),2)

    def test_punctuation_and_leading_ellipsis(self):
        self.assertEqual(split_sentences('…はい。何？えっ?!そう…終わり'),
                         ['…はい。','何？','えっ?!','そう…','終わり'])

    def test_proportional_time_inside_parent(self):
        cues = sentence_cues(Segment(10,16,'あいう。えお。'))
        self.assertEqual(len(cues),2)
        self.assertEqual((cues[0].start,cues[-1].end),(10,16))
        self.assertAlmostEqual(cues[0].end,13.6)
        self.assertEqual(cues[0].end,cues[1].start)

    def test_bounded_allocation(self):
        durations=allocate_durations([1,100,2],10)
        self.assertAlmostEqual(sum(durations),10)
        self.assertTrue(all(.7 <= d <= 7 for d in durations))
        with self.assertRaises(ValueError): allocate_durations([1,2],.9)

    def test_long_comma_sentence(self):
        cues=sentence_cues(Segment(0,14,'仕事ですよ、次の仕事ですよ、最後の仕事ですよ'))
        self.assertTrue(all(.7-1e-6 <= s.end-s.start <= 7+1e-6 for s in cues))
        self.assertEqual(''.join(s.text for s in cues),'仕事ですよ、次の仕事ですよ、最後の仕事ですよ')

    def test_tiny_parent_and_many_sentences(self):
        cues=sentence_cues(Segment(0,.5,'はい。え？'))
        self.assertEqual(len(cues),1)
        self.assertEqual(cues[0].end,.5)
        self.assertIn('parent_shorter_than_minimum',cues[0].flags)

    def test_indivisible_long_exception(self):
        cue=sentence_cues(Segment(0,9,'漢字連続'))[0]
        self.assertIn('indivisible_long_sentence',cue.flags)

    def test_two_lines_never_cut_han_word(self):
        text='這邊還有很多東西需要搬到另外一個房間'
        self.assertEqual(wrap_two_lines(text),text)
        text='這是第一個句子，這是第二個句子，還有第三個句子'
        wrapped=wrap_two_lines(text)
        self.assertLessEqual(len(wrapped.splitlines()),2)
        self.assertEqual(wrapped.replace('\n',''),text)
        self.assertIn('，\n',wrapped)

    def test_filter_original_span_strict_threshold_and_responses(self):
        rows=[Segment(0,.9,'…余の?'),Segment(1,1.5,'はい'),Segment(2,3,'部屋だから'),Segment(4,8,'燿ちゃん')]
        keep,drop=filter_short(rows,1.)
        self.assertEqual(drop,[rows[0]])
        self.assertEqual(len(keep),3)
        self.assertEqual(filter_short(rows,1.,'keep'),(rows,[]))
        self.assertEqual(len(filter_short(rows,1.,'drop')[1]),2)

    def test_round_trip_keeps_grid(self):
        cues=sentence_cues(Segment(10,16,'はい。これは仕事です。'))
        actual=parse_srt(render_sentence_srt(cues))
        self.assertEqual(len(actual),len(cues))
        for a,b in zip(actual,cues):
            self.assertAlmostEqual(a.start,b.start,places=3)
            self.assertAlmostEqual(a.end,b.end,places=3)

if __name__ == '__main__': unittest.main()
