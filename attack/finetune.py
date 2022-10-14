import os, datastat, json

def train(task,label,seed,size,ver_pretrain,ver_finetune,testing,eval_split,margin):
    batch_size = {      # function of size only
        'tiny':128,     # LEN=128{reg=1024} LEN=256(pen=240, cay,reg=256 emb=200) LEN=512(pen=60, orig=256)
        'mini':100,
        'small':100,
        'medium':100,
        'base':160,      # LEN=512(pen=14); LEN=128(pen=128) 
    } 
    pretrain_model = f'/media/h/Ritsu/scratchpad/mlm/{size}/{ver_pretrain}'
    finetune_model = f"/home/h/dev/pathfind/attack/run/finetune/{task}/{size}/{ver_finetune}"
    os.makedirs(finetune_model, exist_ok=True)
    config = {
        'model-name-or-path':           f'{pretrain_model}',        # Using distilbert, cased version, from `transformers`
        'log-to-tb':                    f'',
        'tb-log-dir':                   f'{finetune_model}',
        'logging-interval-step':        f'{100}',
        'ver':                          f'{ver_finetune}',
        'dataset':                      f'{task}',                  # On the SNLI dataset
        'model-num-labels':             f'{len(label)}',            # That has 3 labels
        'filter-train-by-labels':       f'{" ".join(label)}',       # And filter -1 label from train and test
        'filter-eval-by-labels':        f'{" ".join(label)}',
        'per-device-train-batch-size':  f'{batch_size[size]}',      # And batch size of 128
        'output-dir':                   f'{finetune_model}',
        'random-seed':                  f'{seed}',
        'model-max-length':             f'{128}',                   # With a maximum sequence length of 128
        'num-epochs':                   f'{1 if testing else 5}',  # 25
        'checkpoint-interval-epochs':   f'{1 if testing else 5}',   # 5
        'penalty':                      f'{1.0}',
        'margin':                       f'{margin}',
    }
    if eval_split:
        config['dataset-eval-split'] = eval_split
    with open(f'{finetune_model}/sim_config.json', 'w') as outfile:
        json.dump(config, outfile)
    settings = ' '.join([f'--{setting} {config[setting]}' for setting in config])
    command = f'textattack train {settings}'  
    return command

def attack(task,label,seed,size,ver_finetune,finetune_epoch,testing,eval_split,recipe,margin):
    batch_size = {      # function of size only
        'tiny':60,      # LEN=256(pen=240, cay,reg=256 emb=200) LEN=512(pen=60)
        'mini':100,
        'small':100,
        'medium':100,
        'base':160,      # LEN=512(pen=14); LEN=128(pen=128) 
    } 
    finetune_model = f"/home/h/dev/pathfind/attack/run/finetune/{task}/{size}/{ver_finetune}/checkpoint-epoch-{finetune_epoch}"
    attack_result_dir = f"/home/h/dev/pathfind/attack/run/attack/{task}/{size}/{ver_finetune}"
    attack_sim_config = f"{attack_result_dir}/result_{recipe}_epoch_{finetune_epoch}.json"
    attack_result_txt = f"{attack_result_dir}/result_{recipe}_epoch_{finetune_epoch}.txt"
    attack_result_csv = f"{attack_result_dir}/result_{recipe}_epoch_{finetune_epoch}.csv"
    config = {
        'recipe':                       f'{recipe}',        
        'num-examples':                 f'{10 if testing else 1000}',
        'num-workers-per-device':       f'{64}',
        'model-batch-size':             f'{batch_size[size]}',
        'model':                        f'{finetune_model}',
        'dataset-from-huggingface':     f'{task}',                  
        'dataset-split':                f'{eval_split if eval_split else "test"}',          
        'filter-by-labels':             f'{" ".join(label)}',       
        'log-to-txt':                   f'{attack_result_txt}',
        'log-to-csv':                   f'{attack_result_csv}',      
        'random-seed':                  f'{seed}',
        'shuffle':                      f'',       
        'ver':                          f'{ver_finetune}',
        'penalty':                      f'{1.0}',
        'margin':                       f'{margin}',    
    }
    os.makedirs(attack_result_dir, exist_ok=True)
    with open(f'{attack_sim_config}', 'w') as outfile:
        json.dump(config, outfile)
    settings = ' '.join([f'--{setting} {config[setting]}' for setting in config])
    command = f'textattack attack {settings}'  
    return command

def fintune_and_attack(size,df,ver_pretrain,ver_finetune,seed,gpu,testing,recipes,tsp):
    # i = int(ver_finetune.split('_')[-1])
    # margins = [100]
    # margin = margins[i]
    margin = 100
    for _,row in df.iterrows():
        os.environ["CUDA_VISIBLE_DEVICES"]=f'{gpu}'
        command_prefix = f'TS_SOCKET=/tmp/gpu{gpu} tsp ' if tsp else ''
        gpu = (gpu+1)%2
        task = row['name']
        label = list(filter(lambda v: v !='-1',row['label']))
        if task == "multi_nli":
            eval_split = 'validation_matched'
        else:
            # automatically detect which eval split to use for finetuning and set to test split for attack
            eval_split = None
        ##
        ## finetune ##
        ##
        if 1:
            command = train(task,label,seed,size,ver_pretrain,ver_finetune,testing,eval_split,margin)
            os.system(f"{command_prefix}{command}")
        ##
        ## attack ##
        ##
        if 1:
            if testing:
                epochs = [5]
            else:
                epochs = [5]
            for recipe in recipes:
                for finetune_epoch in epochs:
                    command = attack(task,label,seed,size,ver_finetune,finetune_epoch,testing,eval_split,recipe,margin)
                    os.system(f"{command_prefix} {command}")
    return gpu

if __name__ == "__main__":
    sizes = ['base']
    testing = True
    # finetune settings
    finetune_repeat = 1
    finetune_vers = ['roberta'] #'orig', 'emb_cay','pen_cay'
    datasets = ['rotten_tomatoes'] #['yelp_polarity','ag_news','snli','imdb','rotten_tomatoes','multi_nli']
    recipes = ['textfooler'] #['a2t','alzantot','bae','bert-attack','checklist','clare','deepwordbug','faster-alzantot','hotflip','iga','input-reduction','kuleshov','pruthi','pso','pwws','textbugger','textfooler']
    df = datastat.get_stats(datasets)
    gpu = 0
    for size in sizes:
        for i in range(finetune_repeat):
            seed = i
            for ver in finetune_vers:
                ver_pretrain = f"{ver}_0"
                ver_finetune = f"{ver}_{i}"
                tsp = False
                fintune_and_attack(size,df,ver_pretrain,ver_finetune,seed,gpu,testing,recipes,tsp)

