"""haf_torch — the HaF/AHA design mirrored in PyTorch with a SmolVLM backbone.

Same architecture contract as src/haf (JAX, PaliGemma/pi0.5) but backbone-swapped:
  VLM (SmolVLM-256M, pretrained, fine-tuned) -> pooled joint V-L hidden state
    -> flow-matching action expert (continuous action chunk)   [BC objective]
    -> auxiliary head(s) (AHA)                                  [recoverability-selected]
Config flags/weights and the KI (stop-grad) toggle mirror HAFConfig so experiments transfer.
"""
