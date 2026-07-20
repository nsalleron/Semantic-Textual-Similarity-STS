"""Test the BERT fine-tuning pipeline and checkpoint loader.

Heavy: downloads ``bert-base-uncased`` and runs a tiny training loop. Marked
``slow``/``download``. Run explicitly with::

    pytest -m slow tests/test_bert_finetuned.py
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.download]

pytest.importorskip("sentence_transformers")
pytest.importorskip("transformers")

TINY_DATASET = [
    {"sentence1": "a man is playing guitar", "sentence2": "a person plays a guitar",
     "similarity_score": 4.5},
    {"sentence1": "a dog runs in the field", "sentence2": "the stock market fell",
     "similarity_score": 0.5},
    {"sentence1": "she is cooking dinner", "sentence2": "a woman prepares a meal",
     "similarity_score": 4.0},
    {"sentence1": "the cat sleeps", "sentence2": "a rocket launched into space",
     "similarity_score": 0.2},
]


def test_train_save_load_predict(tmp_path):
    from transformers import BertTokenizer

    from api.models.bert_finetuned_model import BertFinetunedModel
    from scripts.train_bert import train

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = train(TINY_DATASET, tokenizer, epochs=1, learning_rate=1e-5, batch_size=2)

    out = tmp_path / "bert_finetuned"
    model.save(str(out))

    loaded = BertFinetunedModel.load(out)
    assert loaded.embedding_dim == 768

    sim = loaded.predict("a man is playing guitar", "a person plays a guitar")
    assert 0.0 <= sim <= 1.0


def test_load_missing_checkpoint_raises(tmp_path):
    from api.models.bert_finetuned_model import BertFinetunedModel

    with pytest.raises(FileNotFoundError):
        BertFinetunedModel.load(tmp_path / "does_not_exist")
