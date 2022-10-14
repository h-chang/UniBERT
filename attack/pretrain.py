import os, random

def pretrain(size, ver, seed, testing, i):
        from_scratch = True
        max_seq = 128 #128, 512
        
        bert_model = {               # function of size only
                'tiny':'google/bert_uncased_L-2_H-128_A-2',
                'mini':'google/bert_uncased_L-4_H-256_A-4',
                'small':'google/bert_uncased_L-4_H-512_A-8',
                'medium':'google/bert_uncased_L-8_H-512_A-8',
                # 'base':'google/bert_uncased_L-12_H-768_A-12',
                'base':'bert-base-uncased',
                'large':'bert-large-uncased',
        }
        # intermediate_size = ?????
        ## NOTE CHECK if ROBERTA is CASED
        reberta_model = {
                'base':'roberta-base',
                'large':'roberta-large',
        }
        model = bert_model[size]
        batch_size = {          # function of size only
                'tiny':256, # max_seq_128{pen=240,cay&orig=256,emb=200}; max_seq_512{pen=60} 
                'mini':100,
                'small':100,
                'medium':100,
                'base':16, #LEN=512(pen=14); LEN=128(pen=100) 100
        } 
        
        command = (
                f'/home/h/dev/env/anaconda3/envs/textattack/bin/python '
                f'/home/h/dev/pathfind/attack/run_mlm.py '
                f'--ver {ver} '
                f'--dataset_name bookcorpus '
                f'--model_name_or_path {model} '
                f'--cache_dir /media/h/Ritsu/scratchpad/huggingface/models '
                f'--output_dir /media/h/Ritsu/scratchpad/mlm/{size}/{ver} '
                f'--per_device_train_batch_size {batch_size[size]} '
                f'--per_device_eval_batch_size {batch_size[size]} '
                f'--evaluation_strategy steps ' #steps
                f'--num_train_epochs {0.001 if testing else 5} ' #5
                f'--max_eval_samples {1000 if testing else 10000} ' #10000
                f'--eval_steps    {10 if testing else 10000} '  #10000
                f'--logging_steps  {10 if testing else 1000} ' #1000
                f'--save_steps {10 if testing else 10000}  ' #25000,50000
                f'--dataset_config_name plain_text '
                f'--dataloader_pin_memory True ' #true
                f'--dataloader_num_workers 16 ' #16 too high will make the system laggy
                f'--preprocessing_num_workers 46 ' #46
                f'--mlm_probability 0.15 ' #0.15
                f'--learning_rate {0 if testing else 1e-4} ' #1e-4, set lr to 0 to generate orig model
                f'--adam_beta1 0.9 ' #0.9
                f'--adam_beta2 0.999 ' #0.999
                f'--weight_decay 0.01 ' #0.01
                f'--lr_scheduler_type linear ' #linear
                f'--warmup_ratio 0.01 ' #0.01
                f'--seed {seed} '
                f'--report_to tensorboard '
                f'--do_train ' 
                f'--do_eval '
                f'--overwrite_output_dir '
                f'--penalty 1 ' #0.003
                f'--max_seq_length {max_seq} ' #128, 512
                f'--from_scratch {from_scratch} '
        )
        return command

if __name__ == '__main__':
        testing = True
        sizes = ['tiny'] #['tiny','mini','small','medium','base']
        vers = ['cay'] #['orig','pen','pen_cay']
        pretrain_repeat = 1
        ### pretraining
        gpu = 1
        for size in sizes:
                for i in range(pretrain_repeat):
                        for ver in vers:
                                ver_pretrain = f"{ver}_{i}"
                                seed = i
                                command = pretrain(size, ver_pretrain, seed, testing, i)
                                os.environ["CUDA_VISIBLE_DEVICES"]=f'{gpu}'
                                os.system(command)
                                # gpu = (gpu+1)%2
