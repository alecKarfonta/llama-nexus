"""
Custom lm-eval model wrapper for llama-cpp-python that loads GGUF models directly.
This enables running lm-eval benchmarks on GGUF models without needing an API server.
"""

import os
from typing import List, Optional, Tuple
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
import logging

logger = logging.getLogger(__name__)


@register_model("gguf-local")
class GGUFLocalLM(LM):
    """
    An lm-eval model wrapper for llama-cpp-python that loads GGUF models directly.
    
    Model args:
        model_path: Path to the GGUF model file
        n_ctx: Context size (default: 2048)
        n_gpu_layers: Number of layers to offload to GPU (-1 for all, default: -1)
        n_batch: Batch size for prompt processing (default: 512)
    """
    
    def __init__(
        self,
        model_path: str = None,
        pretrained: str = None,  # alias for model_path
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,
        n_batch: int = 512,
        **kwargs
    ):
        super().__init__()
        
        # Support both model_path and pretrained args
        self.model_path = model_path or pretrained
        if not self.model_path:
            raise ValueError("Must provide model_path or pretrained argument")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        
        logger.info(f"Loading GGUF model from {self.model_path}")
        logger.info(f"Config: n_ctx={n_ctx}, n_gpu_layers={n_gpu_layers}, n_batch={n_batch}")
        
        # Import and load llama-cpp-python
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python is required for gguf-local model. "
                "Install with: pip install llama-cpp-python"
            )
        
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            logits_all=True,  # Required for loglikelihood
            verbose=True,
        )
        
        logger.info(f"Model loaded successfully. Vocab size: {self.llm.n_vocab()}")
    
    @property
    def eot_token_id(self):
        """Return the end-of-text token ID."""
        return self.llm.token_eos()
    
    @property
    def max_length(self) -> int:
        """Return the maximum context length."""
        return self.n_ctx
    
    @property
    def max_gen_toks(self) -> int:
        """Return maximum generation tokens."""
        return 256
    
    @property
    def batch_size(self) -> int:
        """Return batch size (llama-cpp processes one at a time)."""
        return 1
    
    @property
    def device(self) -> str:
        """Return device string."""
        return "cuda" if self.n_gpu_layers != 0 else "cpu"
    
    def tok_encode(self, string: str) -> List[int]:
        """Tokenize a string."""
        return self.llm.tokenize(string.encode("utf-8"), add_bos=False)
    
    def tok_decode(self, tokens: List[int]) -> str:
        """Decode tokens to string."""
        return self.llm.detokenize(tokens).decode("utf-8", errors="replace")
    
    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
    ) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood for each request's continuation tokens.
        
        Args:
            requests: List of ((context_str, continuation_str), context_tokens, continuation_tokens)
            
        Returns:
            List of (loglikelihood, is_greedy) tuples
        """
        from tqdm import tqdm
        import numpy as np
        
        results = []
        
        for request in tqdm(requests, disable=disable_tqdm, desc="Processing loglikelihood"):
            (context_str, continuation_str), context_tokens, continuation_tokens = request
            
            # Combine context and continuation
            all_tokens = context_tokens + continuation_tokens
            
            # Truncate if too long
            if len(all_tokens) > self.n_ctx:
                # Keep as much context as possible while ensuring continuation fits
                excess = len(all_tokens) - self.n_ctx
                context_tokens = context_tokens[excess:]
                all_tokens = context_tokens + continuation_tokens
            
            if len(all_tokens) == 0:
                results.append((0.0, False))
                continue
            
            try:
                # Evaluate the full sequence
                self.llm.reset()
                self.llm.eval(all_tokens)
                
                # Get logits for all positions
                logprobs_sum = 0.0
                is_greedy = True
                
                # We need logprobs for the continuation tokens
                # The logprob of token at position i is from the logits at position i-1
                context_len = len(context_tokens)
                
                for i, token in enumerate(continuation_tokens):
                    pos = context_len + i
                    if pos == 0:
                        continue  # Skip first position (no prior context)
                    
                    # Get the logits from the previous position
                    logits = self.llm._temp_logits if hasattr(self.llm, '_temp_logits') else None
                    
                    if logits is None:
                        # Use the scores from llama-cpp-python
                        scores = self.llm.scores[pos - 1]
                        
                        # Apply softmax to get log probabilities
                        max_score = np.max(scores)
                        exp_scores = np.exp(scores - max_score)
                        log_sum_exp = np.log(np.sum(exp_scores)) + max_score
                        
                        # Get log probability of the actual token
                        token_logprob = scores[token] - log_sum_exp
                        logprobs_sum += token_logprob
                        
                        # Check if greedy (argmax matches)
                        if np.argmax(scores) != token:
                            is_greedy = False
                
                results.append((float(logprobs_sum), is_greedy))
                
            except Exception as e:
                logger.warning(f"Error computing loglikelihood: {e}")
                results.append((0.0, False))
        
        return results
    
    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """Compute log-likelihoods for requests."""
        new_requests = []
        for request in requests:
            context, continuation = request.args
            context_tokens = self.tok_encode(context) if context else []
            continuation_tokens = self.tok_encode(continuation)
            new_requests.append(((context, continuation), context_tokens, continuation_tokens))
        
        return self._loglikelihood_tokens(new_requests)
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        """Compute rolling log-likelihoods."""
        results = []
        for request in requests:
            text = request.args[0]
            tokens = self.tok_encode(text)
            
            if len(tokens) == 0:
                results.append(0.0)
                continue
            
            # Process tokens and compute log-likelihood
            try:
                self.llm.reset()
                self.llm.eval(tokens)
                
                import numpy as np
                logprobs_sum = 0.0
                
                for i in range(1, len(tokens)):
                    scores = self.llm.scores[i - 1]
                    max_score = np.max(scores)
                    exp_scores = np.exp(scores - max_score)
                    log_sum_exp = np.log(np.sum(exp_scores)) + max_score
                    token_logprob = scores[tokens[i]] - log_sum_exp
                    logprobs_sum += token_logprob
                
                results.append(float(logprobs_sum))
            except Exception as e:
                logger.warning(f"Error in loglikelihood_rolling: {e}")
                results.append(0.0)
        
        return results
    
    def generate_until(self, requests) -> List[str]:
        """Generate text until a stop condition."""
        results = []
        
        for request in requests:
            context = request.args[0]
            gen_kwargs = request.args[1] if len(request.args) > 1 else {}
            
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.0)
            
            # Tokenize context
            context_tokens = self.tok_encode(context)
            
            # Truncate if necessary
            if len(context_tokens) >= self.n_ctx - 1:
                context_tokens = context_tokens[-(self.n_ctx - max_gen_toks - 1):]
            
            try:
                # Generate using llama-cpp-python
                output = self.llm.create_completion(
                    prompt=context,
                    max_tokens=max_gen_toks,
                    temperature=temperature if temperature > 0 else 0.001,
                    stop=until if until else None,
                    echo=False,
                )
                
                generated_text = output["choices"][0]["text"]
                results.append(generated_text)
                
            except Exception as e:
                logger.warning(f"Error in generate_until: {e}")
                results.append("")
        
        return results


# Make sure the model is registered when this module is imported
logger.info("Registered gguf-local model for lm-eval")
