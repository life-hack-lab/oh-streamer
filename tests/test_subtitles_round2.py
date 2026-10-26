"""CPU-only regression coverage for round-two subtitle recipes."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from oh_streamer.subtitles.asr import utterance_groups
from oh_streamer.subtitles.glossary import load_glossary, sakura_messages, parse_lines, SYSTEM, PREFIX
from oh_streamer.subtitles.llm import build_translation_prompt
import tempfile
from oh_streamer.subtitles.postprocess import taiwan_text, display_segments, render_fixed_srt
from oh_streamer.subtitles.core import Segment, parse_srt
from oh_streamer.subtitles.suspicion import mark_disagreement
from oh_streamer.subtitles.postprocess import on_reference_timing


class UtteranceTests(unittest.TestCase):
    def test_short_spans_keep_vad_edges(self):
        import numpy as np
        spans = [{'start': 1., 'end': 3.}, {'start': 5., 'end': 8.}]
        self.assertEqual(utterance_groups(spans, np.ones(160000)), [(1., 3.), (5., 8.)])

    def test_long_span_uses_longest_silence_and_bounds_duration(self):
        import numpy as np
        audio = np.ones(40 * 16000, dtype=np.float32)
        audio[10*16000:12*16000] = 0
        groups = utterance_groups([{'start': 0., 'end': 40.}], audio)
        self.assertAlmostEqual(groups[0][1], 11., places=2)
        self.assertTrue(all(0 < b-a <= 15 for a,b in groups))
        self.assertEqual(groups[-1][1], 40)
        self.assertTrue(all(a[1] == b[0] for a,b in zip(groups, groups[1:])))

    def test_no_silence_hard_cut(self):
        import numpy as np
        self.assertEqual(utterance_groups([{'start': 0., 'end': 31.}], np.ones(32*16000)),
                         [(0., 15.), (15., 30.), (30., 31.)])


class GlossaryTests(unittest.TestCase):
    def test_text_json_and_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'glossary.txt'
            path.write_text('# comment\n酒井->酒井 #人名\n',encoding='utf-8')
            self.assertEqual(load_glossary(path),[dict(src='酒井',dst='酒井',info='人名')])
            path = Path(folder)/'glossary.json'
            path.write_text('{"部長":"部长"}',encoding='utf-8')
            self.assertEqual(load_glossary(path),[dict(src='部長',dst='部长')])
            path.write_text('[{"src":"部長"}]',encoding='utf-8')
            with self.assertRaises(ValueError): load_glossary(path)

    def test_official_format_is_not_wrapped_twice(self):
        terms=[dict(src='部長',dst='部长',info='不可互换')]
        messages=sakura_messages('1. 部長',terms)
        self.assertEqual(messages[0]['content'],SYSTEM)
        self.assertEqual(messages[1]['content'],PREFIX+'部長->部长 #不可互换\n将下面的日文文本根据对应关系和备注翻译成中文：1. 部長')
        self.assertEqual(build_translation_prompt('sakura',messages),messages)
        messages[0]['content']+='改动'
        with self.assertRaises(ValueError): build_translation_prompt('sakura',messages)

    def test_line_count_validation(self):
        self.assertEqual(parse_lines('一\n二',(4,5)),'4. 一\n5. 二')
        with self.assertRaises(ValueError): parse_lines('一',(4,5))


class PostprocessTests(unittest.TestCase):
    def test_reference_cue_grid_preserves_edges_and_all_translation_text(self):
        rows=[dict(start=1,end=8,text='這句翻譯有比較長的文字。'),dict(start=9,end=10,text='好的')]
        reference=[Segment(1,4,'舊一'),Segment(4,8,'舊二'),Segment(9,10,'舊三')]
        result=on_reference_timing(rows,reference)
        self.assertEqual([(s.start,s.end) for s in result],[(s.start,s.end) for s in reference])
        self.assertEqual(''.join(s.text for s in result),'這句翻譯有比較長的文字好的')
    def test_terms_and_local_exceptions(self):
        self.assertEqual(taiwan_text('去酒店找酒店小姐結賬，簽合同看視頻。'),
                         '去飯店找酒店小姐結帳，簽合約看影片')
        self.assertEqual(taiwan_text('合同法與合同制，合同。'),'合同法與合同制，合約')
        self.assertEqual(taiwan_text('OK，TOKEN'), '好，TOKEN')
        self.assertEqual(taiwan_text('甲OK乙'), '甲OK乙')
        rules=[dict(src='甲',dst='乙',match='exact')]
        self.assertEqual(taiwan_text('甲',rules),'乙')
        self.assertEqual(taiwan_text('甲方',rules),'甲方')

    def test_punctuation_and_internal_period(self):
        self.assertEqual(taiwan_text('是。真的?!,\n好。'),'是。真的？！\n好')

    def test_default_hides_failure_and_explicit_japanese_never_leaks_marker(self):
        rows=[dict(start=0,end=2,text='[原文] こんにちは',text_ja='こんにちは',preserved_original=True),
              dict(start=3,end=5,text='你好。')]
        self.assertEqual([s.text for s in display_segments(rows)],['你好'])
        self.assertEqual([s.text for s in display_segments(rows,'japanese')],['こんにちは','你好'])
        self.assertEqual(display_segments([dict(start=0,end=1,text='[原文] 酒井')]),[])

    def test_japanese_no_word_cut_and_fixed_timing(self):
        text='だから'*20
        cues=[Segment(1,9,text)]
        output=render_fixed_srt(cues,japanese=True)
        self.assertIn(text,output)
        self.assertEqual(parse_srt(output),cues)
        zh=parse_srt(render_fixed_srt([Segment(1,9,'中'*50)]))
        self.assertEqual((zh[0].start,zh[0].end),(1,9))
        self.assertTrue(all(len(line)<=18 for line in zh[0].text.splitlines()))


class SuspicionTests(unittest.TestCase):
    def test_exact_pairing_normalization_and_disagreement(self):
        rows=[dict(start=0,end=2,text='はい！'),dict(start=3,end=4,text='謎の文字')]
        second=[dict(start=0,end=2,text='はい'),dict(start=3,end=4,text='おはよう')]
        result=mark_disagreement(rows,second)
        self.assertFalse(result[0]['suspected_asr'])
        self.assertTrue(result[1]['suspected_asr'])
        second[1]['start']=3.1
        self.assertIsNone(mark_disagreement(rows,second)[1]['asr_similarity'])

    def test_display_policy_is_opt_in(self):
        row=dict(start=0,end=2,text='可能錯誤',suspected_asr=True)
        self.assertEqual(len(display_segments([row])),1)
        self.assertEqual(display_segments([row],suspicious='hide'),[])
        self.assertIn('可能聽錯',display_segments([row],suspicious='mark')[0].text)


if __name__ == '__main__':
    unittest.main()
