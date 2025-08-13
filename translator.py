
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import Optional

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-en-hi"

class Translator:
    def __init__(self, model_name: str = DEFAULT_MODEL, src_lang: Optional[str] = None, tgt_lang: Optional[str] = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("translation", model=self.model, tokenizer=self.tokenizer, truncation=True)
        self.src = src_lang
        self.tgt = tgt_lang

    def translate(self, text: str) -> str:
        if not text or not text.strip():
            return text
        out = self.pipe(text, clean_up_tokenization_spaces=True, max_length=512)
        return out[0]["translation_text"]
