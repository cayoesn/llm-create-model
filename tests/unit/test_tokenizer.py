import os

import pytest

from app.model.tokenizer import Tokenizer


def test_tokenizer_build_vocab():
    tokenizer = Tokenizer()
    text = "abc"
    tokenizer.build_vocab(text)
    assert tokenizer.vocab_size == 3
    assert tokenizer.stoi == {"a": 0, "b": 1, "c": 2}
    assert tokenizer.itos == {0: "a", 1: "b", 2: "c"}


def test_tokenizer_encode_decode():
    tokenizer = Tokenizer()
    tokenizer.build_vocab("hello world")
    encoded = tokenizer.encode("hello")
    assert isinstance(encoded, list)
    assert all(isinstance(x, int) for x in encoded)

    decoded = tokenizer.decode(encoded)
    assert decoded == "hello"


def test_tokenizer_invalid_char():
    tokenizer = Tokenizer()
    tokenizer.build_vocab("abc")
    with pytest.raises(ValueError):
        tokenizer.encode("d")


def test_tokenizer_save_load_vocab(tmp_path):
    tokenizer = Tokenizer()
    tokenizer.build_vocab("abc")
    vocab_path = os.path.join(tmp_path, "vocab.json")
    tokenizer.save_vocab(vocab_path)

    new_tokenizer = Tokenizer()
    new_tokenizer.load_vocab(vocab_path)
    assert new_tokenizer.vocab_size == 3
    assert new_tokenizer.stoi == tokenizer.stoi
    assert new_tokenizer.encode("abc") == [0, 1, 2]
