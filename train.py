import logging
import random
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LMNet
from pseudo_label import *
from args import APrompt4EMArgs
from data import PromptEMData, TypeDataset
from prompt import get_prompt_model, get_prompt_dataloader, read_prompt_dataset
from utils import evaluate, statistic_of_current_train_set, EL2N_score
from transformers import  get_cosine_schedule_with_warmup

def train_plm(args: APrompt4EMArgs, model, labeled_train_dataloader, optimizer, scaler):
    criterion = nn.CrossEntropyLoss()
    model.train()
    loss_total = []
    for batch in tqdm(labeled_train_dataloader):
        x, labels = batch
        x = torch.tensor(x).to(args.device)
        labels = torch.tensor(labels).to(args.device)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_total.append(loss.item())
    return np.array(loss_total).mean()


def eval_plm(args: APrompt4EMArgs, model, data_loader, return_acc=False):
    model.eval()
    y_truth = []
    y_pre = []
    for batch in tqdm(data_loader):
        x, labels = batch
        x = torch.tensor(x).to(args.device)
        y_truth.extend(labels)
        with torch.no_grad():
            logits = model(x)
            logits = torch.argmax(logits, dim=1)
            logits = logits.cpu().numpy().tolist()
            y_pre.extend(logits)
    return evaluate(np.array(y_truth), np.array(y_pre), return_acc=return_acc)


def train_prompt(args: APrompt4EMArgs, model, labeled_train_dataloader, optimizer, scaler,scheduler):
    model.train()
    loss_fn = CrossEntropyLoss()
    loss_total = []
    for _batch in tqdm(labeled_train_dataloader):
        labeled_batch = copy.deepcopy(_batch)
        labeled_batch = labeled_batch.to(args.device)
        y_truth = labeled_batch.label
        with autocast():
            logits,softemd = model(labeled_batch)
            loss = loss_fn(logits, y_truth)
            if softemd is not None and args.soft_token_num > 1 and args.orthogonal_loss:
                from orthogonal_loss import orthogonal_loss
                ort_loss =orthogonal_loss(softemd)
                #print('ort_loss',ort_loss)
                loss+=args.ort_ratio*ort_loss
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        loss_total.append(loss.item())

    return np.array(loss_total).mean()


def eval_prompt(args: APrompt4EMArgs, model, data_loader, return_acc=False):
    model.eval()
    y_truth_all = []
    y_pred_all = []
    for batch in data_loader:
        batch = batch.to(args.device)
        y_truth = batch.label
        with torch.no_grad():
            logits,_ = model(batch)
            y_pred = torch.argmax(logits, dim=-1)
            y_truth_all.extend(y_truth.cpu().numpy().tolist())
            y_pred_all.extend(y_pred.cpu().numpy().tolist())
    return evaluate(np.array(y_truth_all), np.array(y_pred_all), return_acc=return_acc)


def pruning_dataset(args: APrompt4EMArgs, data: PromptEMData, model, prompt=True) -> int:
    if prompt:
        labeled_dataset = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs, data.train_y)
        labeled_dataloader = get_prompt_dataloader(args, labeled_dataset, shuffle=False)
    else:
        labeled_dataset = TypeDataset(data, "train")
        labeled_dataloader = DataLoader(dataset=labeled_dataset, batch_size=args.batch_size, collate_fn=TypeDataset.pad)
    model.eval()
    all_pos_el2n = []
    all_neg_el2n = []
    for batch in tqdm(labeled_dataloader, desc="pruning..."):
        if hasattr(batch, "to"):
            batch = batch.to(args.device)
            y_truth = batch.label
        else:
            x, labels = batch
            x = torch.tensor(x).to(args.device)
            labels = torch.tensor(labels).to(args.device)
        with torch.no_grad():
            out_prob = []
            # mc-el2n
            for _ in range(args.mc_dropout_pass):
                if hasattr(batch, "to"):
                    _batch = copy.deepcopy(batch)
                    logits = model(_batch)
                else:
                    logits = model(x)
                logits = torch.softmax(logits, dim=-1)
                out_prob.append(logits.detach())
            out_prob = torch.stack(out_prob)
            out_prob = torch.mean(out_prob, dim=0)
            out_prob = out_prob.detach()
            if hasattr(batch, "to"):
                y_truth = y_truth.detach()
            else:
                y_truth = labels
            pos_el2n = EL2N_score(out_prob[y_truth == 1], y_truth[y_truth == 1])
            neg_el2n = EL2N_score(out_prob[y_truth == 0], y_truth[y_truth == 0])
            all_pos_el2n.extend(pos_el2n)
            all_neg_el2n.extend(neg_el2n)
    k = int(args.el2n_ratio * len(all_pos_el2n))
    values, indices = torch.topk(torch.tensor(all_pos_el2n), k=k)
    pos_ids = indices.numpy().tolist()
    k = int(args.el2n_ratio * len(all_neg_el2n))
    values, indices = torch.topk(torch.tensor(all_neg_el2n), k=k)
    neg_ids = indices.numpy().tolist()
    ids = pos_ids + neg_ids
    data.train_pairs = [x for (i, x) in enumerate(data.train_pairs) if i not in ids]
    data.train_y = [x for (i, x) in enumerate(data.train_y) if i not in ids]
    return len(ids)


class BestMetric:
    def __init__(self):
        self.valid_f1 = -1
        self.loss = 100
        self.test_metric = None
        self.state_dict = None


def inner_train(args: APrompt4EMArgs, model, optimizer, scaler, train_dataloader, valid_dataloader, test_dataloader,epoch,
                prompt=True,scheduler= None):

    loss = train_prompt(args, model, train_dataloader, optimizer, scaler,scheduler)
    test_p, test_r, test_f1 = eval_prompt(args, model, test_dataloader)
    if args.use_valid:
        valid_p, valid_r, valid_f1 =  eval_prompt(args, model, valid_dataloader)
    else:
        valid_p, valid_r, valid_f1 = test_p, test_r, test_f1

    logging.info(f"[Valid] Precision: {valid_p:.4f}, Recall: {valid_r:.4f}, F1: {valid_f1:.4f}")
    logging.info(f"[Test] Precision: {test_p:.4f}, Recall: {test_r:.4f}, F1: {test_f1:.4f}")
    return (valid_p, valid_r, valid_f1, test_p, test_r, test_f1,loss)


def update_best(model, metric, best: BestMetric ,epoch,args):
    valid_p, valid_r, valid_f1, test_p, test_r, test_f1,loss = metric
    if valid_f1 > best.valid_f1 or valid_f1 == best.valid_f1 and test_f1 > best.test_metric[2]:
        best.valid_f1 = valid_f1
        best.test_metric = (test_p, test_r, test_f1, epoch)
        best.state_dict = model.state_dict()
        torch.save({
            'model_state_dict': model.state_dict(),},
            args.model_file+"/best_model_torch",
        )
        import pickle
        with open(args.model_file+"/best_model","wb") as f:
            pickle.dump(model.cpu(),f)

        model.cuda()
    if loss < best.loss:
        best.loss = loss
        import pickle
        with open(args.model_file+"/bestloss_model","wb") as f:
            pickle.dump(model.cpu(),f)
        model.cuda()

def train_and_update_best(args: APrompt4EMArgs, model, optimizer, scaler, train_dataloader, valid_dataloader,
                          test_dataloader, best: BestMetric,epoch , prompt=True,scheduler=None):
    metric = inner_train(args, model, optimizer, scaler, train_dataloader, valid_dataloader, test_dataloader,epoch,
                             prompt,scheduler)
    update_best(model, metric, best,epoch,args)


def get_optimizer_grouped_parameters(
    model,
    plm_learning_rate, template_learning_rate, weight_decay,
    layerwise_learning_rate_decay
):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.template.named_parameters() if 'raw' not in n],
            "weight_decay": 0.0,
            "lr": template_learning_rate,
        },
    ]

    layers = [getattr(model, 'prompt_model').plm]
    no_decay = ["bias", "LayerNorm.weight"]
    layers.reverse()
    lr = plm_learning_rate
    for layer in layers:
        lr *= layerwise_learning_rate_decay
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    return optimizer_grouped_parameters


def self_training(args: APrompt4EMArgs, data: PromptEMData):
    args.soft_token_num = args.query_size if args.query_size else args.head_num
    train_set = read_prompt_dataset(data.left_entities, data.right_entities, data.train_pairs, data.train_y,data.leftpos,data.rightpos,data)
    test_set = read_prompt_dataset (data.left_entities, data.right_entities, data.test_pairs, data.test_y,data.leftpos,data.rightpos,data)
    train_loader = get_prompt_dataloader(args, train_set, shuffle=True)
    test_loader = get_prompt_dataloader(args, test_set, shuffle=False)
    if args.use_valid:
        valid_set = read_prompt_dataset(data.left_entities, data.right_entities, data.valid_pairs, data.valid_y,data.leftpos,data.rightpos)
        valid_loader = get_prompt_dataloader(args, valid_set, shuffle=False)
    else:
        valid_set = train_set
        valid_loader = train_loader

    best = BestMetric()
    for iter in range(1, args.num_iter + 1):
        # train the teacher model
        model, tokenizer, wrapperClass, template = get_prompt_model(args)
        model.to(args.device)
        #print('test', evaluate_prompt(model,test_loader))
        if args.grouped_parameters:
            parameters =  get_optimizer_grouped_parameters(model,args.learning_rate, args.template_learning_rate, args.weight_decay,args.layerwise_learning_rate_decay)
            optimizer = AdamW(params=parameters, lr=args.learning_rate)
        else:
            optimizer = AdamW(params=model.parameters(), lr=args.learning_rate)
        if args.warm_up_ratio !=-1:
            total_steps = len(train_loader)*args.teacher_epochs
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps= int(args.warm_up_ratio *total_steps),
                num_training_steps=total_steps
            )
        else:
            scheduler = None
        scaler = GradScaler()
        siz, pos, neg, per, acc = statistic_of_current_train_set(data)
        logging.info(f"[Current Train Set] Size: {siz} Pos: {pos} Neg: {neg} Per: {per:.2f} Acc: {acc:.4f}")
        for epoch in range(1, args.teacher_epochs + 1):
            logging.info(f"[Teacher] epoch#{epoch}")
            train_and_update_best(args, model, optimizer, scaler, train_loader, valid_loader, test_loader, best,epoch,scheduler=scheduler)
            p, r, f1,bestepoch = best.test_metric
            loss = best.loss
        logging.info(f"[Best Teacher in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f} bestepoch {bestepoch:3d}")

        with open(args.save_file,'a') as f:
            #print('args',args,file=f)
            print(f"[Best Teacher in iter#{iter}] Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f} bestepoch {bestepoch:3d} loss {loss:.4f}",file=f)






