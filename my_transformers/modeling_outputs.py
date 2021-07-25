import transformers
from transformers.modeling_outputs import *

@dataclass
class MyTokenClassifierOutput(TokenClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    seq_loss: Optional[torch.FloatTensor] = None
    pool_loss: Optional[torch.FloatTensor] = None
    seq_logits: torch.FloatTensor = None
    pool_logits: torch.FloatTensor = None
