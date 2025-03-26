from chop.tools import get_tokenized_dataset
checkpoint = "facebook/wav2vec2-base-960h"
dataset_name = "nyalpatel/condensed_librispeech_asr"


dataset, tokenizer = get_tokenized_dataset(
    dataset=dataset_name,
    checkpoint=checkpoint,
    return_tokenizer=True,
)