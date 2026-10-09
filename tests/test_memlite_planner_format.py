import types
import unittest
import torch
from g05.utils.memlite_planner_format import planner_only_format_constants


class Helper:
    def _sample(self, logits, prev_tokens=None, **kwargs):
        return torch.full((logits.shape[0],), ord('D'),dtype=torch.long)


class Text:
    def process(self, text, **kwargs):
        return list(map(ord,text)),None,None


def processor():
    return types.SimpleNamespace(modality_processors={'text':Text()},tokenizer=types.SimpleNamespace(
        decode=lambda ids, **kw:''.join(map(chr,ids))))


class FormatTests(unittest.TestCase):
    def test_constants_only_then_free_decision_and_restore(self):
        helper=Helper();logits=torch.zeros(2,256)
        with planner_only_format_constants(helper,processor(),batch_size=2,hl_end_id=255,enabled=True) as receipt:
            a=[]
            for _ in 'Previous outcome: UNKNOWN|':
                a.append(helper._sample(logits).tolist())
            self.assertEqual(''.join(chr(x[0]) for x in a),'Previous outcome: UNKNOWN|')
            self.assertEqual(helper._sample(logits).tolist(),[ord('D')]*2)
            self.assertEqual(receipt['forced_tokens'],len(a)*2)
        self.assertNotIn('_sample',helper.__dict__)

    def test_exception_restores_override(self):
        helper=Helper(); prior=lambda *a,**kw:None;helper._sample=prior
        with self.assertRaises(RuntimeError):
            with planner_only_format_constants(helper,processor(),batch_size=1,hl_end_id=255,enabled=True):
                raise RuntimeError('bad request')
        self.assertIs(helper._sample,prior)

    def test_disabled_does_not_change_helper(self):
        helper=Helper()
        with planner_only_format_constants(helper,processor(),batch_size=1,hl_end_id=255,enabled=False):
            self.assertNotIn('_sample',helper.__dict__)


if __name__=='__main__':unittest.main()
