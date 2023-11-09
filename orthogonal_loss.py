import torch
import math
def orthogonal_loss(input):
    '''
        # input is a batch of matrices - shape is [batch_size, num_rows, num_cols]
    identity = torch.eye(input.size(1)).unsqueeze(0).to(input.device)
    identity = identity.repeat(input.size(0), 1, 1)
    output = torch.bmm(input, input.transpose(1, 2)) - identity
    loss = output.norm(p='fro') / math.sqrt(input.size(0)) / input.size(2)# take average over batch size

    '''

    sample_losses = []
    for i in range(input.size(0)):
        identity = torch.eye(input.size(1)).to(input.device)
        output = torch.mm(input[i], input[i].t()) - identity
        sample_loss = output.norm(p='fro')
        sample_losses.append(sample_loss)

    # 计算平均loss
    loss = sum(sample_losses) / len(sample_losses) / input.size(2)
    return loss


# 示例
input = torch.randn(10, 5, 500)  # 10个5x5的矩阵
loss = orthogonal_loss(input)
print(loss)