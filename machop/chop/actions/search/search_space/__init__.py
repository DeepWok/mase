from .search_space_quantise import MixedPrecisionSearchSpace
from .llm_quantise import LLMMixedPrecisionSearchSpace

search_space_map = {
    "mixed_precision": MixedPrecisionSearchSpace,
    "llm_mixed_precision": LLMMixedPrecisionSearchSpace,
}
