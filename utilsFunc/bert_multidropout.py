# bert模型文件
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertConfig, BertModel
import logging

logger = logging.getLogger(__name__)

class Bert(nn.Module):
    def __init__(self, config):
        super(Bert, self).__init__()
        # BertConfig 这是一个配置类，继承 PretrainedConfig 类，用于model的配置
        model_config = BertConfig.from_pretrained(
            config.config_file,  # 模型config文件位置。【按理说是一个model，这里是BertConfig类所以是config？】
            num_labels=config.num_labels,  # 分类数
            finetuning_task=config.task,  # 自定义一个名字
        )
        # Bert模型类，继承torch.nn.Module，实例化对象时使用from_pretrained()函数初始化模型权重，参数config用于配置模型参数
        self.bert = BertModel.from_pretrained(
            config.model_name_or_path,  # 模型位置
            config=model_config,
        )
        if config.requires_grad:  # 如果是False，难道默认grad是False的？
            for param in self.bert.parameters():
                param.requires_grad = True

        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # dropout率
        self.multi_drop = config.multi_drop
        self.classifier = nn.Linear(self.hidden_size, config.num_labels)  # 最后一层全连接层，后面接softmax

    # BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=tensor(xx),pooled_output=tensor(),grad_fn=<TanhBackward0>), hidden_states=None, past_key_values=None, attentions=None, cross_attentions=None )
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        # 预训练模型走一遍  得到两个部分：last_hidden_state+pooled_output
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 只取类别部分 [batchsize, hiddensize]
        pooled_output = outputs[1]

        out = None
        loss = 0
        # multi-sample-dropout技术！！
        for i in range(self.multi_drop):
            out = self.dropout(pooled_output)  # dropout  [batchsize, hiddensize]
            out = self.classifier(out)  # 全连接层【多次共享权重，这样算是吗？】 [batchsize, num_labels]
            if labels is not None:  # out是 float类型
                if i == 0:  # 第一次  不应该先做softmax在计算loss函数吗？ RuntimeError: expected scalar type Long but found Int
                    loss = F.cross_entropy(out, labels) / self.multi_drop
                else:
                    loss += F.cross_entropy(out, labels) / self.multi_drop

        # if self.loss_method in ['binary']:
        #     # sigmoid函数对每一个元素运用函数
        #     out = torch.sigmoid(out).flatten()  # 二分类通常是sigmoid激活函数；多分类是softmax
        # else:
        #     # softmax依赖于多个元素  out的维度（bs*num_labels） 我们希望在一行的num_labels上做平均，所以dim=1
        #     out = torch.softmax(out,dim=1)  # 多分类【不知道对不对】

        return out, loss


