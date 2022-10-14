import torch
import torch.nn.functional as f
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from typing import Optional
from dataclasses import dataclass, field
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    ## NOTE ## mod start
    ver: str = field(
        default="orig",
        metadata={"help": "orig/pen/pen_cay impose unitary constrain or not."},
    )
    penalty: Optional[float] = field(
        default=0.003,
        metadata={
            "help": "Regularizer strength to match the distance of the predifined embedding."
        },
    )
    ## NOTE ## mod end

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

if __name__ == '__main__':
    for i in range(1):
        # model_path = f'/media/h/Ritsu/scratchpad/mlm/base/orig_0/checkpoint-{(i+1)*50000}'
        model_path = 'google/bert_uncased_L-12_H-768_A-12'
        # model_path = f'/media/h/Ritsu/scratchpad/mlm/base/emb_cay_0'
        model_args = ModelArguments()
        config_kwargs = {
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        }

        config = AutoConfig.from_pretrained(model_path, **config_kwargs)
        config.ver = 'emb'
        config.device = 'cpu'
        penalty = 1
        model = AutoModelForMaskedLM.from_config(config)
        model.bert.embeddings.word_embeddings.weight.data.copy_(torch.nn.functional.normalize(model.bert.embeddings.word_embeddings.weight.data))
        target_emb = model.bert.embeddings.word_embeddings.weight.data.to('cpu')
        source_emb = model.bert.embeddings.word_embeddings_syno.weight.data.to('cpu')
        target_mean = target_emb.norm(dim=1).mean()
        source_mean = source_emb.norm(dim=1).mean()
        print(f"target_mean={target_mean:.6f} source_mean={source_mean:.6f}")
    exit()
    size = 'tiny'
    models = {  
        'orig':f'/media/h/Ritsu/scratchpad/mlm/{size}/orig_0',
        # 'orig':f'google/bert_uncased_L-12_H-768_A-12', 
        'pen_cay':f'/media/h/Ritsu/scratchpad/mlm/{size}/pen_cay_0'
    }
    model_args = ModelArguments()
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    trim = 'orig'
    config = AutoConfig.from_pretrained(models[trim], **config_kwargs)
    config.ver = trim
    config.device = 'cpu'
    penalty = 1
    model = AutoModelForMaskedLM.from_config(config)
    if 'pen' in config.ver:
        with torch.no_grad():
            w = model.bert.embeddings.word_embeddings_syno.weight.cpu()
            # w = torch.randn_like(w)
            # w = f.normalize(w, p=2, dim=1)
            print(torch.norm(w,dim=1))
            distance = torch.linalg.norm(w@w.T-model.bert.embeddings.syno_dis.cpu())
            loss_weight = penalty*distance
    else:
        with torch.no_grad():
            w = model.bert.embeddings.word_embeddings.weight.cpu()
            # w = torch.randn_like(w)
            # w = f.normalize(w, p=2, dim=1)
    print(w.shape)
    norm = torch.norm(w,dim=1)
    print(f"mean: {norm.mean():.5f}, std: {norm.std():.5f}")
    # print(loss_weight)

