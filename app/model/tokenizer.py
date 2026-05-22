import json
import os


class Tokenizer:
    """
    A custom character-level tokenizer.

    This tokenizer maps each unique character in a text to a unique integer ID.
    It supports encoding strings into a list of integers and decoding a list of
    integers back into a string.
    """

    def __init__(self) -> None:
        self.stoi: dict[str, int] = {}
        self.itos: dict[int, str] = {}
        self.vocab_size: int = 0

    def build_vocab(self, text: str) -> None:
        """
        Builds the vocabulary from the provided text.

        Args:
            text: The text to build the vocabulary from.
        """
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = dict(enumerate(chars))

    def encode(self, s: str) -> list[int]:
        """
        Encodes a string into a list of character IDs.

        Args:
            s: The string to encode.

        Returns:
            A list of integers representing the character IDs.

        Raises:
            ValueError: If a character in the string is not in the vocabulary.
        """
        try:
            return [self.stoi[c] for c in s]
        except KeyError as e:
            raise ValueError(f"Character {e} not in vocabulary.")

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of character IDs into a string.

        Args:
            l: A list of character IDs.

        Returns:
            The decoded string.

        Raises:
            ValueError: If an ID is not in the vocabulary.
        """
        try:
            return "".join(self.itos[i] for i in ids)
        except KeyError as e:
            raise ValueError(f"ID {e} not in vocabulary.")

    def save_vocab(self, path: str) -> None:
        """
        Saves the vocabulary to a JSON file.

        Args:
            path: The path to save the vocabulary to.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.stoi, f, ensure_ascii=False, indent=2)

    def load_vocab(self, path: str) -> None:
        """
        Loads the vocabulary from a JSON file.

        Args:
            path: The path to load the vocabulary from.
        """
        with open(path, encoding="utf-8") as f:
            self.stoi = json.load(f)
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
