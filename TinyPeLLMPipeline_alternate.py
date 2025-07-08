from typing import Optional, List

class TinyPeLLMPipeline:
    """
    Custom pipeline for TinyPeLLM - compatible with transformers.pipeline
    """

    def __init__(self, model, tokenizer=None, device: Optional[int] = None, task: Optional[str] = None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.task = task

        if device is not None and hasattr(self.model, "to"):
            self.model = self.model.to(device)

    def __call__(self, text: str, max_length: int = 100, num_return_sequences: int = 1,
                 temperature: float = 1.0, top_p: float = 1.0, do_sample: bool = True,
                 **kwargs) -> List[str]:
        """
        Generate text using the model.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            **kwargs
        )
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
