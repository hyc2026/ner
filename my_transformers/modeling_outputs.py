import transformers
from transformers.modeling_outputs import *

@dataclass
class MyTokenClassifierOutput(TokenClassifierOutput):
    loss: Optional[torch.FloatTensor] = None
    ner_loss: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    ner_logits: torch.FloatTensor = None
    cls_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
