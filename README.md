# 授信报告要素结构化

授信调查报告是银行在对客户授信之前对客户的组织架构、股东情况、关联企业、公司治理情况、财务状况、项目本身生产运营情况、贷款用途、偿还贷款能力以及贷款收益等项目进行综合考察分析后，对客户进行信用等级评定，进而提出授信额度建议的书面材料。

授信报告往往文本篇幅长，内容过，靠人类逐字阅读来获取信息效率低，因此从授信报告中提取结构化的信息往往能节省大量的人力物力成本。

## 数据集

数据集中标注了实体的位置，实体之间的关系以及对实体和关系的描述。

```json
[
    {
        "id": 1,
        "text": "阿里巴巴公司是一所电子商务公司，其股东之一马云曾经是一名英语教师，占股约7.6%",
        "cls_label": "基本信息",
        "spo_list": [
            {
                "order": 1,
                "s": {
                    "text": "马云",
                    "start": 21,
                    "end": 23,
                    "type": "PER",
                },
                "p": "股东",
                "o": {
                    "text": "阿里巴巴公司",
                    "start": 0,
                    "end": 6,
                    "type": "ORG",
                }
            },
        ],
        "desc_list": [
            {
                "order": 1,
                "text": "一所电子商务公司",
                "start": 7,
                "end": 15,
                "type": "S"
            },
            {
                "order": 1,
                "text": "曾经是一名英语教师",
                "start": 23,
                "end": 32,
                "type": "P"
            },
            {
                "order": 1,
                "text": "占股约7.6%",
                "start": 33,
                "end": 40,
                "type": "O"
            },
        ]
    },
]
```

### 数据预处理

在不丢失信息的情况下对输入语句进行切割，以适应transformer的输入长度的限定。长度不能设定为512，因为tokenizer会加入一些标签。切割时不能切断实体以及描述，一般情况下在标点符号处切割。

将数据集分为训练集，验证集和测试集，以保证“主体 - 关系 - 客体”的分布均匀。（对于数量少的“主体 - 关系 - 客体”关系仅出现在训练集中）

## Transformers库的修改与使用

### 定制化Transformers

transformers库对模型进行了耦合封装，但是利用面向对象的方式我们也可以解耦进行自定义的配置。具体方法是在项目中构建一个my_transformers目录：

```
.my_transformers
|-- __init__.py
|-- data
|   `-- data_collator.py
|-- file_utils.py
|-- modeling_outputs.py
|-- models
|   |-- bert
|   |   |-- configuration_bert.py
|   |   `-- modeling_bert.py
|   `-- layers
|       `-- crf.py
|-- trainer.py
`-- training_args.py
```

摘出自己需要修改的函数，如modeling_outputs.py中的TokenClassifierOutput，需导入transformers.modeling_outputs中的全部配置，并用MyTokenClassifierOutput继承TokenClassifierOutput，添加自己的修改。

```python
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
```

其余的文件类似，具体见[joint_model](https://github.com/hyc2026/transformers-joint-model)，若有私有的变量和函数需要复制出来，否则无法访问。

### Transformers的训练过程

transformers库使用trainer进行训练的大致过程：

1. 使用Trainer类声明一个trainer对象

   ```python
   trainer = MyTrainer(
       model=model, # BertForTokenClassification
       args=training_args, # TrainingArguments
       train_dataset=train_dataset,
       eval_dataset=eval_dataset,
       tokenizer=tokenizer, # 将文本输入转换成模型能够理解的数字形式: input_ids
       					 # 用于区分多个句子: token_type_ids
       					 # 遮蔽掉补全位置的无用信息: attention_mask
       data_collator=data_collator, # 将label处理为与input_ids对应的形式
       compute_metrics=compute_metrics, # 计算precision, accuracy, f1等
   )
   ```

2. 使用train方法开始训练

   ```python
   train_result = trainer.train(resume_from_checkpoint=checkpoint)
   ```

3. 调用get_train_dataloader()方法进行数据整理(返回一个DataLoader对象，内部使用data_collator)

   ```python
   train_dataloader = self.get_train_dataloader()
   return DataLoader(
       train_dataset,
       batch_size=self.args.train_batch_size,
       collate_fn=self.data_collator,
       num_workers=self.args.dataloader_num_workers,
       pin_memory=self.args.dataloader_pin_memory,
   )
   ```

4. 调用training_step()对输入进行训练，并返回损失

   ```python
   tr_loss += self.training_step(model, inputs)
   ```

5. 在training_step()内部调用compute_loss()函数计算损失，compute_loss()内部调用模型并获取模型的输出

   ```python
   outputs = model(**inputs)
   
   # class BertForTokenClassification
   def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
       outputs = self.bert(
           input_ids,
           attention_mask=attention_mask,
           token_type_ids=token_type_ids,
           position_ids=position_ids,
           head_mask=head_mask,
           inputs_embeds=inputs_embeds,
           output_attentions=output_attentions,
           output_hidden_states=output_hidden_states,
       )
       sequence_output = outputs[0] 
       sequence_output = self.dropout(sequence_output)
       logits = self.classifier(sequence_output)
   	loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
   	return TokenClassifierOutput(
           loss=loss,
           logits=logits,
           hidden_states=outputs.hidden_states,
           attentions=outputs.attentions,
       )
   ```

## 预训练

## 主客体抽取

需要抽取出`人`和`组织`两种实体，因此共需要五种标签["O", "B_PER", "I_PER", "B_ORG", "I_ORG"]，同时在该步骤预测出文本段的分类标签，因此使用transformers库的TokenClassification和TextClassification两个模型联合训练。

输入数据格式

```json
{
    "id": 1,
    "tokens": ["阿", "里", "巴", "巴", "公", "司", "是", "一", "所", "电", "子", "商", "务", "公", "司", "，", "其", "股", "东", "之", "一", "马", "云", "曾", "经", "是", "一", "名", "英", "语", "教", "师", "，", "占", "股", "约", "7", ".", "6", "%"],
    "ner_tags": ["B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-PER", "I-PER", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"],
    "label": "基本信息",
}
```

在[run_ner.py](https://github.com/huggingface/transformers/blob/master/examples/pytorch/token-classification/run_ner.py)的基础上进行更改，增加整个句子的class标签，并在`MyBertConfig`中增加cls_label2id和id2cls_label两个字段：

```python
cls_label_list = raw_datasets["train"].unique("label")
cls_label_list.sort()
cls_label_to_id = {v: i for i, v in enumerate(cls_label_list)}
if cls_label_to_id is not None:
    config.cls_label2id = cls_label_to_id
    config.id2cls_label = {id: label for label, id in config.cls_label2id.items()}
```

更改data_collator，将label字段加载到数据集中。

更改模型，将MyBertForBertClassification和MyBertForTextClassification模型进行融合。将outputs的整层用于实体抽取，将`[CLS]`标签的输出用于文本分类

```python
def forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    label=None,
):
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    ner_output = outputs[0]
    cls_output = outputs[0][:, 0, :]

    ner_output = self.dropout(ner_output)
    ner_logits = self.classifier(ner_output)
    cls_output = self.dropout1(cls_output)
    cls_logits = self.classifier1(cls_output)

    ner_loss = None
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = ner_logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            ner_loss = loss_fct(active_logits, active_labels)
        else:
            ner_loss = loss_fct(ner_logits.view(-1, self.num_labels), labels.view(-1))

    cls_loss = None
    if label is not None:
        self.config.problem_type = "single_label_classification"
        loss_fct = CrossEntropyLoss()
        cls_loss = loss_fct(cls_logits.view(-1, self.num_label), label.view(-1))

    gama = 0.5
    loss=gama * ner_loss + (1 - gama) * cls_loss,
    if not return_dict:
        output = (ner_logits, cls_output,) + outputs[2:]
        return ((loss, ner_loss, cls_loss) + output) if loss is not None else output
    
    return MyTokenClassifierOutput(
        loss=loss,
        ner_loss=ner_loss,
        cls_loss=cls_loss,
        ner_logits=ner_logits,
        cls_logits=cls_logits,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
```

## 关系及描述抽取

已经抽取出了实体，将一条数据中抽取到的实体两两组对用于预测实体之间的关系，具体的，将实体的位置进行掩码处理，共有四种掩码: [S-ORG]、[S-PER]、[O-ORG]和[O-PER]。需要对tokenizer进行修改，用来将新加入的特殊token作为整体生成id。

```python
tokenizer = AutoTokenizer.from_pretrained(...)
additional_special_tokens_list = ["[S-ORG]", "[O-ORG]", "[S-PER]", "[O-PER]"]
tokenizer.add_special_tokens({'additional_special_tokens': additional_special_tokens_list})

model = BertForTokenClassification.from_pretrained(...)
model.resize_token_embeddings(len(tokenizer))
```

若假设共需预测n种关系，则共需要$2\times(2+n)+1$种标签：["B_S", "I_S", "B_O", "I_O", "B_P1", "I-P1", ..., "O"]

输入数据格式

```json
{
    "id": 1,
    "tokens": ["[O-ORG]", "是", "一", "所", "电", "子", "商", "务", "公", "司", "，", "其", "股", "东", "之", "一", "[S-PER]", "曾", "经", "是", "一", "名", "英", "语", "教", "师", "，", "占", "股", "约", "7", ".", "6", "%"],
    "ner_tags": ["O", "O", "B-O", "I-O", "I-O", "I-O", "I-O", "I-O", "I-O", "I-O", "O", "O", "O", "O", "O", "O", "O", "B-S", "I-S", "I-S", "I-S", "I-S", "I-S", "I-S", "I-S", "I-S", "O", "B-股东", "I-股东", "I-股东", "I-股东", "I-股东", "I-股东", "I-股东"],
}
```
