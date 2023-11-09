import  os
datename = 'all'
query_sizes = [1,2,4,8]
num_att_layers = [1,2,3]
head_nums  = [1,2,4,8]
'''
 
for seed in range(3):
    cmd = f'python main.py --natural --normal --last_layer --orthogonal_loss --use_valid --seed {seed}'
    os.system('python main.py --default')
    for query_size in query_sizes:
        os.system(cmd+f' --query_size {query_size}')
        print(cmd+f' --query_size {query_size}')

    for num_att_layer in num_att_layers:
        os.system(cmd+f' --num_att_layer {num_att_layer}')
        print(cmd+f' --num_att_layer {num_att_layer}')

    for head_num in head_nums:
        os.system(cmd+f' --head_num {head_num}')
        print(cmd+f' --head_num {head_num}')

    os.system(cmd + f' --grouped_parameters  ')
    print(cmd + f' --grouped_parameters  ')

    os.system(cmd + f' --pe_pos ')
    print(cmd + f' --pe_pos')
'''
### gpt_augment
datenames = ['wdc','rel-text','semi-text-w','semi-text-c']
query_sizes = [1,2,4,8]
num_att_layers = [1,2,3]
head_nums = [1,2,4,8]
for datename in datenames:
    for seed in range(1):
        cmd = f'python main.py --natural --normal --last_layer --orthogonal_loss --gpt_aug --seed {seed} --data_name {datename}'
        for query_size in query_sizes:
            os.system(cmd + f' --query_size {query_size}')
            print(cmd+f' --query_size {query_size}')
        for num_att_layer in num_att_layers:
            os.system(cmd + f' --num_att_layer {num_att_layer}')
            print(cmd + f' --num_att_layer {num_att_layer}')
        for head_num in head_nums:
            os.system(cmd + f' --head_num {head_num}')
            print(cmd + f' --head_num {head_num}')