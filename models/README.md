# Modelli dell'OCR

`text-detection.onnx` e `text-recognition.onnx` sono i modelli pre-addestrati del motore OCR
[Ocrs](https://github.com/robertknight/ocrs), creati da **Robert Knight** e addestrati con il
progetto [ocrs-models](https://github.com/robertknight/ocrs-models) sul dataset
[HierText](https://github.com/google-research-datasets/hiertext) e su dati sintetici.

- Origine: <https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.onnx> e
  <https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.onnx>, i modelli che scarica
  lo stesso programma `ocrs`; scheda dei modelli: <https://huggingface.co/robertknight/ocrs>
- Licenza: [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/),
  testo completo in [`LICENSE`](LICENSE). I modelli sono forniti «così come sono», senza garanzie.
- Modifiche: nessuna. I file sono identici agli originali e vengono inclusi così come sono
  nell'eseguibile di DocPDF al momento della compilazione.

| File | SHA-256 |
|---|---|
| `text-detection.onnx` | `a917b23dbd9524b465df7e922641b3ff2981623df4ded5a0234004ef2fee7cfe` |
| `text-recognition.onnx` | `86c145c2edb96c8caed5b1ebb8f44d706408922211451c309c625157dd6061c5` |

La licenza CC BY-SA 4.0 vale solo per questi due file; il resto di DocPDF è sotto licenza MIT
(vedi [`../LICENSE`](../LICENSE)).
