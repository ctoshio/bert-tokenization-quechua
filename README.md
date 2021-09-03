# bert-tokenization-quechua
El bert-tokenization-quechua fue entrenado con corpus solamente en quechua sureño (collao y chanka). Para el tokenizador se uso el enfoque de Byte-level BPE con un vocabulario de 52000 tokens de subpalabras.

## Acerca del tokenizador
|Modulo| Descarga |
|------|----------|
| Tokenizer | [merges.txt](https://drive.google.com/file/d/1PrM9LMJ9Pmrc8yqKBT1OMRPXD1urkJ1r/view?usp=sharing), [vocab.json](https://drive.google.com/file/d/1i6L13u5P9HVzzmKsNZxe_wICteulIWY5/view?usp=sharing) |

## Usabilidad
Una vez descargado merges.txt y vocab.json , le direccionamos la ruta especifica.
```python
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing


tokenizer = ByteLevelBPETokenizer(
    "./vocab.json",
    "./merges.txt",
)
```
```python
tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
``` 
```python
tokenizer.encode("allinllachu manan allinlla huk wasipita").tokens
```
    ['<s>',
    'allin',
    'llachu',
    'Ġmanan',
    'Ġallinlla',
    'Ġhuk',
    'Ġwasipi',
    'ta',
    '</s>']

