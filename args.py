import argparse
import logging


class APrompt4EMArgs:
    def __init__(self, args, data_name: str) -> None:
        allowed_data_names_for_gpt_aug = ['wdc', 'rel-text', 'semi-text-w', 'semi-text-c']

        # Check if gpt_aug is True and data_name is not in the allowed list
        if args.gpt_aug and data_name not in allowed_data_names_for_gpt_aug:
            raise ValueError(f"gpt_aug can only be True if data_name is one of {allowed_data_names_for_gpt_aug}."
                             f" Received data_name: '{data_name}' with gpt_aug set to {args.gpt_aug}.")
        self.seed = args.seed
        self.device = args.device
        self.model_name_or_path = args.model_name_or_path
        self.model_type = self.model_name_or_path.split("/")[-1].split("-")[0]
        self.batch_size = args.batch_size
        self.text_summarize = args.text_summarize
        self.learning_rate = args.lr
        self.max_length = args.max_length
        self.add_token = args.add_token
        self.data_name = data_name
        self.template_no = args.template_no
        self.k = args.k
        self.num_iter = args.num_iter
        self.save_model = args.save_model
        self.teacher_epochs = args.teacher_epochs
        self.one_word=args.one_word
        # New arguments added here
        self.save_file = args.save_file
        self.head_num = args.head_num
        self.num_att_layers = args.num_att_layers
        self.pos_size = args.pos_size
        self.warm_up_ratio = args.warm_up_ratio
        self.template_learning_rate = args.template_learning_rate
        self.weight_decay = args.weight_decay
        self.layerwise_learning_rate_decay = args.layerwise_learning_rate_decay
        self.early_stop = args.early_stop
        self.maxlength = args.maxlength
        self.query_size = args.query_size
        self.ort_ratio = args.ort_ratio

        # Boolean arguments
        self.pos_emd = getattr(args, 'pos_emd', False)
        self.one_hot = getattr(args, 'one_hot', False)
        self.normal = getattr(args, 'normal', False)
        self.last_layer = getattr(args, 'last_layer', False)
        self.grouped_parameters = getattr(args, 'grouped_parameters', False)
        self.pe_pos = getattr(args, 'pe_pos', False)
        self.gpt_aug = getattr(args, 'gpt_aug', False)
        self.use_valid = getattr(args, 'use_valid', False)
        self.orthogonal_loss = getattr(args, 'orthogonal_loss', False)
        self.default = getattr(args, 'default', False)
        self.natural = getattr(args, 'natural', False)



    def __str__(self) -> str:
        return f"[{', '.join((f'{k}:{v}' for (k, v) in self.__dict__.items()))}]"

    def log(self):
        logging.info("====APrompt4EM Args====")
        for (k, v) in self.__dict__.items():
            logging.info(f"{k}: {v}")


def int_or_float(value):
    try:
        value = int(value)
        return value
    except ValueError:
        try:
            value = float(value)
            return value
        except ValueError:
            return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--batch_size", "-bs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-5, help="(teacher) lr")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--add_token", default=True)
    parser.add_argument("--data_name", "-d", type=str,
                        choices=["google-amazon", "rel-text", "semi-heter", "semi-homo", "semi-rel", "semi-text-c",
                                 "semi-text-w","geo-heter",'wdc', "all"], default="all")
    parser.add_argument("--template_no", "-tn", type=int, default=3, choices=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--self_training", "-st", action="store_true", default=False)
    parser.add_argument("--dynamic_dataset", "-dd", type=int, default=-1,
                        help="-1 means that dd is off, otherwise it means the frequency of dd.")
    parser.add_argument("--num_iter", "-ni", type=int, default=1)
    parser.add_argument("--k", "-k", type=int_or_float, default=0.1)
    parser.add_argument("--pseudo_label_method", "-pm", type=str, default="uncertainty",
                        choices=["uncertainty", "confidence", "unfold_fold"])
    parser.add_argument("--mc_dropout_pass", "-mdp", type=int, default=10)
    parser.add_argument("--uncertainty_ratio", "-ur", type=float, default=0.1)
    parser.add_argument("--el2n_ratio", "-er", type=float, default=0.1)
    parser.add_argument("--confidence_ratio", "-cr", type=float, default=0.1)
    parser.add_argument("--text_summarize", "-ts", action="store_true")
    parser.add_argument("--save_model", "-save", action="store_true", default=False)
    parser.add_argument("--only_plm", "-op", action="store_true", default=False)
    parser.add_argument("--teacher_epochs", "-te", type=int, default=30)
    parser.add_argument("--student_epochs", "-se", type=int, default=30)
    parser.add_argument("--test_pseudo_label", "-tpl", type=str, default="")
    parser.add_argument("--one_word","-ow",action="store_true",default=False)
    #### args for aprompt4em
    parser.add_argument('--save_file', type=str, default='res.txt', help='File to save results')
    parser.add_argument('--head_num', type=int, default=1, help='Number of heads in the model')
    parser.add_argument('--num_att_layers', type=int, default=1, help='Number of attention layers')
    parser.add_argument('--pos_size', type=int, default=10 , help='Size of positional encoding')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate for the model')
    parser.add_argument('--warm_up_ratio', type=float, default=-1, help='Warm up ratio for learning rate')
    parser.add_argument('--template_learning_rate', type=float, default=5e-5, help='Learning rate for the template')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay rate')
    parser.add_argument('--layerwise_learning_rate_decay', type=float, default=0.95,
                        help='Layerwise learning rate decay')
    parser.add_argument('--early_stop', type=int, default=20, help='Early stopping criteria')
    parser.add_argument('--maxlength', type=int, default=512, help='Maximum length of sequences')
    parser.add_argument('--query_size', type=int, default=4, help='Query size for the model')
    parser.add_argument('--ort_ratio', type=float, default=1, help='Orthogonal ratio')

    parser.add_argument('--pos_emd', action='store_true', help='Boolean flag for col positional embedding')
    parser.add_argument('--one_hot', action='store_true', help='Use one-hot encoding')
    parser.add_argument('--normal', action='store_true', help='Normalization flag')
    parser.add_argument('--last_layer', action='store_true', help='Use only the last layer')
    parser.add_argument('--grouped_parameters', action='store_true', help='Group parameters flag')
    parser.add_argument('--pe_pos', action='store_true', help='Positional encoding position flag')
    parser.add_argument('--gpt_aug', action='store_true', help='GPT agumnet flag')
    parser.add_argument('--use_valid', action='store_true', help='GPT agumnet flag')
    parser.add_argument('--orthogonal_loss', action='store_true', help='Use orthogonal loss')
    parser.add_argument('--natural', action='store_true', help='Use natural dataset')
    parser.add_argument('--default', action='store_true')

    args = parser.parse_args() #['--model_name_or_path', 'roberta-base']

    return args


def parse_em_args(args, data_name) -> APrompt4EMArgs:
    return APrompt4EMArgs(args, data_name)
