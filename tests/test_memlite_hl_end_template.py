"""MEM-Lite high-level end-token rendering regressions."""

from types import SimpleNamespace

from g05.data_processor.processor.samples_builder import MEMLiteHighLevelBuilder, IntentActionBuilder
from g05.models.g05.io.input_preprocessor import (
    BuiltinTextProcessor,
    InputPreprocessor,
    TOKEN_INDEX,
)


class _Tokenizer:
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return 97 if token == "<HL_END>" else self.unk_token_id

    def __call__(self, text, add_special_tokens=False):
        if text == "<HL_END>":
            ids = [97]
        else:
            ids = [11 + index for index, _ in enumerate(str(text))]
        return {"input_ids": ids}


def _minimal_preprocessor():
    processor = object.__new__(InputPreprocessor)
    processor.tokenizer = _Tokenizer()
    processor.token_manager = SimpleNamespace(
        resolve_template=lambda template: template,
        resolve=lambda _raw_key: None,
    )
    processor.control_tokens = {"eoc": "EOC", "eov": "EOV"}
    processor.pred_eov = False
    processor.modality_processors = {"text": BuiltinTextProcessor()}
    return processor


def test_hl_end_is_preserved_as_a_registered_static_token_not_placeholder():
    processor = _minimal_preprocessor()
    segments = processor._parse_template("<EOC><intent_text>|<HL_END><EOV>")
    end_segments = [segment for segment in segments if segment.content == "<HL_END>"]
    assert len(end_segments) == 1
    assert end_segments[0].type == "static"


def test_high_level_end_token_is_an_unmasked_text_target_after_eoc():
    processor = _minimal_preprocessor()
    segments = processor._parse_template("<EOC><intent_text>|<HL_END><EOV>")
    prefix = processor._parse_control_tokens(segments, control_flag="return_prefix")
    end_segments = [segment for segment in prefix if segment.content == "<HL_END>"]
    assert len(end_segments) == 1
    assert end_segments[0].type == "dynamic"
    assert end_segments[0].sample_key == ""
    assert end_segments[0].processor_key == "text"
    assert end_segments[0].masked is False

    _, labels, attention, _ = processor._preprocess_single_sample(
        prefix,
        {"intent": "Intent: open drawer"},
        processor.tokenizer,
        loss_on_static_text=False,
        training=True,
    )
    assert 97 in labels
    assert attention[labels.index(97)] == TOKEN_INDEX.PRED_TEXT_TOKEN_INDEX


def test_mixed_memlite_templates_keep_hl_end_only_on_high_branch():
    high_template = MEMLiteHighLevelBuilder(num_input_images=1, image_sizes={"cam": (224, 224)}).template
    low_template = IntentActionBuilder(num_input_images=1, image_sizes={"cam": (224, 224)}).template
    assert "<HL_END>" in high_template
    assert "<HL_END>" not in low_template
