import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/vlm_sft'))
import visual_grounding as grounding


class GroundingContractTests(unittest.TestCase):
    def test_exact_contract(self):
        with self.assertRaisesRegex(ValueError,'Duplicate JSON key'):
            grounding.parse_answer('{"visibility":"absent","visibility":"present","boxes":[[1,2,3,4]]}')
        for value in ({'visibility':'present','boxes':[[1,2,999,1000]]},
                      {'visibility':'absent','boxes':[]},{'visibility':'uncertain','boxes':[]}):
            self.assertEqual(grounding.parse_answer(grounding.canonical_target(value)),value)
        invalid=[{'visibility':'present','boxes':[]},{'visibility':'absent','boxes':[[1,2,3,4]]},
                 {'visibility':'uncertain','boxes':[],'action':'move'},
                 {'visibility':'present','boxes':[[0,0,0,100]]},
                 {'visibility':'present','boxes':[[0,0,1001,100]]},
                 {'visibility':'present','boxes':[[False,0,100,100]]},
                 {'visibility':'present','boxes':[[0.,0,100,100]]},
                 {'visibility':'present','boxes':[[0,0,100,100]]*2}]
        for value in invalid:
            with self.subTest(value=value),self.assertRaises(ValueError):grounding.parse_answer(json.dumps(value))

    def test_special_tokens_missing_eos_and_trailing_tokens_are_not_hidden(self):
        class Tokenizer:
            eos_token_id=1;pad_token_id=0;unk_token_id=2;all_special_ids=[0,1,2]
            def __len__(self):return 1000
            def convert_tokens_to_ids(self,s):return 1
            def convert_ids_to_tokens(self,i):return '<|im_end|>'
            def decode(self,tokens,skip_special_tokens=False):
                assert skip_special_tokens is False
                return ''.join('<special>' if i<3 else chr(i-10) for i in tokens)
        processor=SimpleNamespace(tokenizer=Tokenizer());ids=[ord(c)+10 for c in '{"visibility":"absent","boxes":[]}']
        good=grounding.decode_response(processor,ids+[1,0,0]);self.assertTrue(good['format_valid'])
        for bad in ([2]+ids+[1],ids,ids+[1,44]):
            value=grounding.decode_response(processor,bad);self.assertFalse(value['format_valid'])
            self.assertEqual(value['generated_token_ids'],bad)

    def test_only_current_image_and_query_enter_prefix(self):
        for n,m in ((720,640),(480,480)):
            msg,receipt=grounding.messages('the radio',Image.new('RGB',(n,n)))
            self.assertEqual(msg[1]['content'][0]['text'],'CURRENT_RAW')
            self.assertEqual(msg[1]['content'][1]['image'].size,(m,m))
            self.assertEqual(receipt['size'],[m,m])
            self.assertEqual(msg[1]['content'][-1]['text'],'Query: the radio')
        with self.assertRaises(ValueError):grounding.messages('task0 success',Image.new('RGB',(720,720)))

    def test_real_tensor_mask_prefix_and_eos_for_variable_box_lengths(self):
        import torch
        from modeling import collate,supervised_loss
        class Tokenizer:
            eos_token_id=1;unk_token_id=0;pad_token_id=0
            def convert_tokens_to_ids(self,text):return 1 if text=='<|im_end|>' else 0
            def convert_ids_to_tokens(self,idx):return '<|im_end|>' if idx==1 else 'other'
            def encode(self,text,**_):return [ord(c)+10 for c in text]
        class Processor:
            tokenizer=Tokenizer()
            def apply_chat_template(self,msg,**kw):
                self.last=msg;assert kw['enable_thinking'] is False
                n=3 if msg[1]['content'][1]['image'].width==640 else 5
                return {'input_ids':torch.tensor([[4]*n]),'attention_mask':torch.ones((1,n),dtype=torch.long),
                        'mm_token_type_ids':torch.ones((1,n),dtype=torch.long),'pixel_values':torch.ones((2,3)),
                        'image_grid_thw':torch.tensor([[1,2,2]])}
        processor=Processor();items=[]
        for size,target in ((720,{'visibility':'present','boxes':[[100,200,400,600],[700,200,900,500]]}),
                            (480,{'visibility':'uncertain','boxes':[]})):
            row={'query':'the radio','target':json.dumps(target),'task':999,'frame':12345,'history':['SECRET_ACTION']}
            image=Image.new('RGB',(size,size));prefix,_=grounding.encode(processor,row,image)
            full,_=grounding.encode(processor,row,image,supervised=True);n=prefix['input_ids'].shape[1]
            self.assertTrue(torch.equal(prefix['input_ids'],full['input_ids'][:,:n]))
            self.assertTrue((full['labels'][:,:n]==-100).all());self.assertEqual(int(full['labels'][0,-1]),1)
            self.assertNotIn('12345',str(processor.last));self.assertNotIn('SECRET_ACTION',str(processor.last))
            self.assertTrue((full['mm_token_type_ids'][:,n:]==0).all());items.append(full)
            with self.assertRaises(ValueError):grounding.encode(processor,row,image,supervised=True,blind=True)
        batch=collate(items,0)
        self.assertTrue((batch['labels'][batch['attention_mask']==0]==-100).all())
        class Model:
            def __call__(self,input_ids,labels=None,logits_to_keep=None,**_):
                logits=torch.sin(torch.arange(input_ids.numel()*160).reshape(*input_ids.shape,160).float()/100)
                loss=None if labels is None else torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,160),labels[:,1:].reshape(-1),ignore_index=-100)
                return SimpleNamespace(logits=logits[:,-logits_to_keep:] if logits_to_keep else logits,loss=loss)
        model=Model();self.assertTrue(torch.allclose(model(**batch).loss,supervised_loss(model,batch),atol=1e-6))


if __name__=='__main__':unittest.main()
