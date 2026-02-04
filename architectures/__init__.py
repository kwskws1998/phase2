# Architecture definitions package
# Contains model architecture implementations

from .ministral_3_3b_instruct import (
    Mistral3Config,
    Mistral3ForConditionalGeneration,
    Ministral3TokenConfig,
)

# Optional: mHC variant
try:
    from .ministral_3_3b_instruct_mHC import (
        Mistral3Config_mHC,
        Mistral3ForConditionalGeneration_mHC,
    )
except ImportError:
    pass
