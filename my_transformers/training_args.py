from transformers.training_args import *
from transformers import TrainingArguments

class MyTrainingArguments(TrainingArguments):
    label_names: Optional[List[str]] = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )