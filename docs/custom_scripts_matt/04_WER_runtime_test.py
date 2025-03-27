import torch
import numpy as np
import jiwer
from pyctcdecode import build_ctcdecoder

"""
This script replicates the relevant "CTC logic" from your runtime_analysis_pass:
    - We have a 'ctc_head' that transforms encoder outputs -> logits
    - We take log_softmax and decode with beam search (pyctcdecode)
    - We compute WER with a 'safe_wer' function, same as your code

We demonstrate how passing [pred], [ref] vs pred, ref to jiwer.wer
produces different WER values.
"""

# -------------------------------
# 1. Example "ctc_head" mock
# -------------------------------
class MockCTCHead(torch.nn.Module):
    """
    A fake linear layer that simulates your ctc_head, transforming
    encoder features -> (time, vocab_size) logits.
    """
    def __init__(self, in_dim, vocab_size):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, vocab_size)

    def forward(self, x):
        # x shape: [batch_size, time, in_dim]
        # We'll flatten to apply a single linear layer for demonstration
        batch_size, time_steps, feat_dim = x.shape
        x = x.reshape(batch_size * time_steps, feat_dim)
        out = self.linear(x)
        out = out.view(batch_size, time_steps, -1)
        return out

# -------------------------------
# 2. Same "safe_wer" logic
# -------------------------------
def safe_wer(pred, ref, verbose=False):
    if pred.strip() == "" and ref.strip() != "":
        return 1.0
    elif pred.strip() == "" and ref.strip() == "":
        return 0.0
    else:
        measures = jiwer.compute_measures(ref, pred)
        if verbose:
            print(f"  Substitutions (S): {measures['substitutions']}")
            print(f"  Deletions     (D): {measures['deletions']}")
            print(f"  Insertions    (I): {measures['insertions']}")
            print(f"  Hits          (H): {measures['hits']}")
            print(f"  WER           : {measures['wer']:.4f}")
        return measures["wer"]


def safe_wer_incorrect(pred, ref):
    """
    Demonstrates what happens if we pass [pred], [ref] to jiwer.wer
    (the common mistake that yields weird WERs).
    """
    return jiwer.wer([ref], [pred])  # WRONG usage

# -------------------------------
# 3. Main test function
# -------------------------------
def test_runtime_like_logic():
    # a) Suppose we have 13 tokens in the reference (like your example).
    ref_text = "the bear shook his shaggy sides and then a well known voice replied"
    # b) Suppose the model only predicts "replied"
    #    We'll show how that gets recognized with beam search.

    # Step 1: We'll define a mini vocab with indexes:
    #    0 = [PAD], 1 = 'the', 2 = 'bear', 3 = 'shook', ...
    # For simplicity, we won't EXACTLY match your real vocab.
    # We'll just ensure "replied" is in the vocab at index 8, for instance.
    vocab_list = [
        "[PAD]", "the", "bear", "shook", "his", "shaggy", 
        "sides", "and", "then", "a", "well", "known", 
        "voice", "replied"   # index 13
    ]
    decoder = build_ctcdecoder(vocab_list)

    # Step 2: Mock a single "batch" of data
    #    We'll say our "encoder output" is shape [batch=1, time=3, in_dim=4].
    encoder_output = torch.randn(1, 3, 4)

    # Step 3: Our "ctc_head" transforms [1,3,4] -> [1,3,vocab_size=14].
    ctc_head = MockCTCHead(in_dim=4, vocab_size=len(vocab_list))
    logits = ctc_head(encoder_output)  # shape [1, 3, 14]

    # Step 4: Convert to log-softmax and decode with pyctcdecode
    #    This is exactly like your line:
    #      sample_logits = torch.from_numpy(preds_np[i])
    #      sample_log_probs = sample_logits.log_softmax(dim=-1).cpu().numpy()
    #      transcription = decoder.decode(...)




    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()
    transcription = decoder.decode(log_probs, beam_width=10)  # same as your code
    transcription = transcription.lower()

    # Step 5: Compare the predicted text vs. reference text
    #    We'll just override the predicted text to "replied" for demonstration
    #    or we can see what the random model guessed.
    #    Let's forcibly do: predicted_text = "replied"
    #    But for demonstration, let's show both the random decode and forced "replied".
    random_decode = transcription
    forced_decode = "replied"

    # Step 6: We'll compute WER using your safe_wer logic
    correct_wer_random = safe_wer(random_decode, ref_text)
    correct_wer_forced = safe_wer(forced_decode, ref_text)
    # We'll also compute the incorrect version
    incorrect_wer_forced = safe_wer_incorrect(forced_decode, ref_text)

    # Step 7: Print results
    print("==== RUNTIME-LIKE TEST ====")
    print(f"Reference:  '{ref_text}'\n")
    print(f"Random decode from CTC:  '{random_decode}'  (just from random weights!)")
    print(f"Forced decode example:   '{forced_decode}'\n")
    print(f"Correct WER (random vs ref): {correct_wer_random:.4f}")
    print(f"Correct WER (forced vs ref): {correct_wer_forced:.4f}")
    print(f"INCORRECT WER (forced vs ref) [passing lists]: {incorrect_wer_forced}\n")
    print("Note how the incorrect usage often yields strange integer values like 4\n\n")

    print("WER with random numbers decoded")
    correct_wer_random = safe_wer(random_decode, ref_text, verbose=True)

    print("WER with prediction (relied)")
    correct_wer_forced = safe_wer(forced_decode, ref_text, verbose=True)

# -------------------------------
# 4. Run the test
# -------------------------------
if __name__ == "__main__":
    test_runtime_like_logic()

