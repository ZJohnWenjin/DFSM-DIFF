import torch

# return alpha
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

# generating()
def generalized_steps(x, seq, model, b, condition,**kwargs):
    # x 是初始输入图像0
    with torch.no_grad():
        # 获取批次大小 n
        n = x.size(0)
        # 表示下一个时间步的序列
        seq_next = [-1] + list(seq[:-1])
        # 用于保存每一步的去噪结果和
        x0_preds = []
        #  xs 用于保存每一步的输入图像
        xs = [x]
        # 对整个batch，每个时间步 i 和下一个时间步 j
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # 创建当前时间步 t 和下一个时间步 next_t 的张量
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            # b是β,t是time 返回at
            at = compute_alpha(b, t.long())
            # 计算当前和下一个时间步的 at 和 at_next
            at_next = compute_alpha(b, next_t.long())
            # 取当前全部batch步骤的输入图像 xt
            xt = xs[-1].to('cuda')
            # 计算出噪声
            et = model(xt, t,condition)
            # 从当前xt推断出x0
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            # 把当前时间段xt推断出的x0加入x0_preds
            x0_preds.append(x0_t.to('cpu'))
            # 加入随机噪声的强度 eta 和at,at+1计算得到
            c1 = (
                #                       ((1-at/at+1)*(1-at+1)/(1-at))^0.5
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            # (1-(at+1)-c1^2)^0.5
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            # 根据x0,at+1和噪音推出xt+1
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            # 用所有batch的xt 推断出xt+1
            xs.append(xt_next.to('cpu'))

    # 返回用所有batch的xt推断出xt+1 和 当前时间段xt推断出的x0加入返回
    return xs, x0_preds

# 去噪过程
def ddpm_steps(x, seq, model, b, condition, condition_idx):
    # 下面的at和at+1都是代表着连乘（α一把），真正的αt会用1-β表示
    with torch.no_grad():
        n = x.size(0)
        # 获取下各时间段
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        # 获取当前t和下个时间t+1
        for i, j in zip(reversed(seq), reversed(seq_next)):
            # 获取整个batch的t
            t = (torch.ones(n) * i).to(x.device)
            # 获取整个batch的t+1
            next_t = (torch.ones(n) * j).to(x.device)
            # 计算当前at
            at = compute_alpha(betas, t.long())
            # 计算at+1
            atm1 = compute_alpha(betas, next_t.long())
            # 计算beta t
            beta_t = 1 - at / atm1
            # 拿出当前x
            x = xs[-1].to('cuda')
            # 预测出噪声当前xt的噪声
            output = model(x, condition, condition_idx, t.float())
            e = output
            # 从at,噪声，预测出x0
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            # 加入x0_preds
            x0_preds.append(x0_from_e.to('cpu'))
            # 预测出噪声的均值，因为方差是固定的
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    # 返回 xs列表 和 x0_preds列表
    return xs, x0_preds
