
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

paths = [str(x) for x in Path("./data/train/").glob("**/*.txt")]

tokenizer = ByteLevelBPETokenizer(lowercase=True, unicode_normalizer='nfkc')

special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
tokenizer.train(files=paths, vocab_size=52_000, min_frequency=2, special_tokens=special_tokens)

tokenizer.save_model("./data/result/")

tokenizer = ByteLevelBPETokenizer("./data/result/vocab.json", "./data/result/merges.txt")

tokenizer._tokenizer.post_processor = BertProcessing(("</s>", tokenizer.token_to_id("</s>")),
                                                     ("<s>", tokenizer.token_to_id("<s>")))

tokenizer.enable_truncation(max_length=512)

print(tokenizer.encode("allinllachu manan allinlla huk wasipita mañana Perú").tokens)
