import torch

def calc_loss_h(g_h, labels, centers, margin=0.1):
    code_length = g_h.size(1)
    y = torch.argmax(labels, dim=1)
    y_centers = centers[y] # batch_size * code_length
    dis = 0.5 - (g_h * y_centers).sum(1) / (2 * code_length)
    return torch.clamp(dis - margin, min=0).mean()


def calc_loss_l(g, beh_embedding, labels):
    code_length = g.size(1)
    y = torch.argmax(labels, dim=1)
    max_i = y.max()
    losses = []
    for i in range(max_i + 1):
        same_cls = torch.where(y == i)[0]
        if 0 <= same_cls.size(0) <= 2:
            continue

        embed_cls = beh_embedding[same_cls]
        embed_dist = torch.cdist(embed_cls, embed_cls, p=2)
        n = same_cls.size(0)
        embed_dist = embed_dist.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)  # delete diagonal

        g_cls = g[same_cls]
        g_dist = 0.5 - torch.matmul(g_cls, g_cls.t()) / (2 * code_length)
        g_dist = g_dist.flatten()[1:].view(n - 1, n + 1)[:, :-1].reshape(n, n - 1)  # delete diagonal
        for j in range(n):
            embed_dist_comb = torch.combinations(embed_dist[j], r=2)
            pos_inds = torch.where(embed_dist_comb[:, 0] < embed_dist_comb[:, 1])[0]
            neg_inds = torch.where(embed_dist_comb[:, 0] > embed_dist_comb[:, 1])[0]

            g_dist_comb = torch.combinations(g_dist[j], r=2)
            if pos_inds.size(0) == 0:
                loss1 = torch.tensor(0.0, device=g.device, dtype=torch.float)
            else:
                loss1 = torch.clamp(g_dist_comb[pos_inds, 0] - g_dist_comb[pos_inds, 1], min=0.0).sum()
            if neg_inds.size(0) == 0:
                loss2 = torch.tensor(0.0, device=g.device, dtype=torch.float)
            else:
                loss2 = torch.clamp(g_dist_comb[neg_inds, 1] - g_dist_comb[neg_inds, 0], min=0.0).sum()

            losses.append(loss1 + loss2)

    if len(losses) == 0:
        return torch.tensor(0.0, device=g.device, dtype=torch.float, requires_grad=True)

    return sum(losses) / len(losses)