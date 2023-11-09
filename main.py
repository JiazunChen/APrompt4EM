import logging
import os.path

from args import parse_args, parse_em_args
from data import PromptEMData
from train import self_training
from utils import set_seed, set_logger, read_entities, read_ground_truth_few_shot, read_ground_truth,process_ground_truth_few_shot
import numpy as np
import  datetime


def load_my_dataset(args,data_type):

    if data_type in ["google-amazon", "rel-text", "semi-heter", "semi-homo", "semi-rel", "semi-text-c", "semi-text-w", "geo-heter"]:
        data = PromptEMData(data_type)
        data.left_entities, data.right_entities = read_entities(data_type, args)
        if args.natural:
            if data_type in ["rel-text",]:
                _, data.right_entities = np.load(f'./data/natural/{data_type.replace("-", "_")}_nlp.npy',allow_pickle=True).tolist()
            else:
                data.left_entities, data.right_entities =   np.load(f'./data/natural/{data_type.replace("-","_")}_nlp.npy',allow_pickle=True).tolist()
            if args.pos_emd and data_type not in ["rel-text","semi-text-w","semi-text-c"]: #Not structured data without col pos
                data.leftpos, data.rightpos = np.load(f'./data/natural/{data_type.replace("-","_")}_nlp_pos.npy', allow_pickle=True).tolist()
            else:
                data.leftpos, data.rightpos = None, None
                args.pos_emd = False

            if args.gpt_aug:
                data.left_entities, data.right_entities = np.load(f'./data/natural/{data_type.replace("-", "_")}_nlp_gpt.npy',allow_pickle=True).tolist()
                data.leftpos, data.rightpos = None, None
                args.pos_emd = False
        else:
            data.leftpos, data.rightpos = None, None
            args.pos_emd = False
        data.train_pairs, data.train_y, \
        data.train_un_pairs, data.train_un_y = read_ground_truth_few_shot(f"data/{data_type}", ["train"], k=args.k,
                                                                          seed=args.seed,
                                                                          return_un_y=True)

        data.valid_pairs, data.valid_y = read_ground_truth(f"data/{data_type}", ["valid"])
        data.test_pairs, data.test_y = read_ground_truth(f"data/{data_type}", ["test"])
        data.read_all_ground_truth(f"data/{data_type}")
    elif data_type == 'wdc':
         data = PromptEMData("google-amazon")
         if args.natural:
            wdc = np.load('./data/natural/wdc_entities_nlp.npy', allow_pickle=True).tolist()
            if args.pos_emd:
                pos_map = np.load('./wdc_entities_position.npy', allow_pickle=True).tolist()
                data.leftpos, data.rightpos = pos_map['left'], pos_map['right']
            else:
                data.leftpos, data.rightpos = None, None

            if args.gpt_aug:
                 data.left_entities, data.right_entities = np.load(f'./data/natural/wdc_nlp_gpt.npy',
                                                                   allow_pickle=True).tolist()
                 data.leftpos, data.rightpos = None, None
                 args.pos_emd = False

         else:
            wdc = np.load('./data/natural/wdc_entities_col.npy', allow_pickle=True).tolist()
            data.leftpos, data.rightpos = None, None
            args.pos_emd = False
         # #./80pair/wdc_entities_nlp_new.npy #./wdc-info/newwdc_add_chat_9_22.npy
         #print(wdc['left'].keys())
         pairs = np.load('./data/natural/wdc_newpairs.npy', allow_pickle=True).tolist()
         data.left_entities = wdc['left']
         data.right_entities = wdc['right']
         data.train_pairs, data.train_y = process_ground_truth_few_shot(pairs['wdcproducts80cc20rnd000un_train_small.json.gz'],k=args.k)
         data.valid_pairs, data.valid_y = pairs['wdcproducts80cc20rnd000un_valid_small.json.gz']
         data.test_pairs, data.test_y =  pairs['wdcproducts80cc20rnd100un_gs.json.gz']
         data.train_un_pairs, data.train_un_y = [],[]
    else:
        print('no name')
    logging.info(f"left size: {len(data.left_entities)}, right size: {len(data.right_entities)}")
    logging.info(f"labeled train size: {len(data.train_pairs)}")
    logging.info(f"unlabeled train size: {len(data.train_un_pairs)}")
    logging.info(f"valid size: {len(data.valid_pairs)}")
    logging.info(f"test size: {len(data.test_pairs)}")
    # for checking pseudo label acc
    return data


pos_size_map ={
    'semi-text-w': 9,
    'semi-text-c': 9,
    'wdc': 6,
    'google-amazon':4,
    'geo-heter':7,
    'rel-text':5,
    'semi-homo':5,
    'semi-heter':10,
    'semi-rel':10,
}

def init_default_args(args):
    args.natural = True
    args.use_valid = True
    args.num_att_layers = 1
    args.pos_size = pos_size_map[data_type]
    args.pos_emd = False
    args.template_no = 3
    args.learning_rate = 2e-5
    args.one_hot = False
    args.normal = True
    args.last_layer = True
    args.warm_up_ratio = -1
    args.grouped_parameters = False
    args.template_learning_rate = 5e-5
    args.weight_decay = 0.01
    args.layerwise_learning_rate_decay = 0.95
    args.early_stop = 20
    args.num_att_layers = 1
    args.pe_pos = False
    args.gpt_aug = False
    args.maxlength = 512
    args.query_size = 4
    args.orthogonal_loss = True
    args.ort_ratio = 1
    args.head_num = 1
    args.k = 0.1
    if args.data_type == 'wdc':
        args.query_size = 16
        args.num_att_layers = 3
    if args.data_type=='google-amazon':
        args.query_size = 1
        args.grouped_parameters = True
        args.head_num=2
    if args.data_type == 'geo-heter':
        args.head_num = 4
        args.query_size = 2
        args.pe_pos =  True
    if args.data_type == 'rel-text':
        args.grouped_parameters = True
        args.head_num =  1
        args.query_size = 2
    if args.data_type == 'semi-homo':
        args.query_size =2
        args.head_num = 4
        args.pe_pos = True
    if args.data_type == 'semi-text-w':
        args.query_size = 1
        args.pe_pos =  True
    if args.data_type == 'semi-text-c':
        args.query_size = 4
        args.num_att_layers = 2
    if  args.data_type == 'semi-rel':
        args.query_size = 4
        args.head_num = 1
    if  args.data_type == 'semi-heter':
        args.query_size = 4
        args.head_num = 1

if __name__ == '__main__':
    if not os.path.exists('./result_model'):
        os.mkdir('./result_model')
    if not os.path.exists('./log'):
        os.mkdir('./log')
        
    common_args = parse_args()
    set_logger("APrompt4EM")
    tasks = [common_args.data_name]
    if common_args.data_name == "all":
        tasks = pos_size_map.keys()
    for data_type in  tasks:
        args = parse_em_args(common_args, data_type)
        args.data_type = data_type
        if  args.default:
            init_default_args(args)
        now = datetime.datetime.now()
 
        args.model_file = './result_model/' + now.strftime('%Y%m%d%H%M%s')
        if not os.path.exists(args.model_file):
            os.mkdir(args.model_file)
        args.log()
        with open(args.save_file, 'a') as f:
            print('args :', args, file=f)
        data = load_my_dataset(args, data_type)
        set_seed(common_args.seed)
        #continue
        self_training(args, data)

