import hanlp,os,sys
from hanlp.pretrained import pos
from load_update_config import read_config, update_config
root_path = os.path.join(os.path.dirname(__file__),'..')
sys.path.append(root_path)
config = read_config(os.path.join(root_path,'config/config.yaml'))
mtl_task_model = os.path.join(root_path, 'model/{}'.format(config['model_name']['mtl_task_model']))
stl_task_model_tok = os.path.join(root_path, 'model/{}'.format(config['model_name']['stl_task_model_tok']))
stl_task_model_ner = os.path.join(root_path, 'model/{}'.format(config['model_name']['stl_task_model_ner']))
multi_task = config['multi_task_or_single_task']['multi_task']
if multi_task == True:
    Hanlp = hanlp.load(mtl_task_model)
    output = Hanlp([input()], tasks='ner/msra')
    print(output)
else:
    HanLP_tok = hanlp.load(stl_task_model_tok)
    HanLP_ner = hanlp.load(stl_task_model_ner)
    HanLP = hanlp.pipeline()\
        .append(HanLP_tok, output_key='tok')\
        .append(HanLP_ner, output_key='ner')
    output = HanLP([input()])
    print(output)
