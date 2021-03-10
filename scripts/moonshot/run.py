import graca
import torch
import random

def run(args):
    net = getattr(graca.models, args.model)()
    net = net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), 1e-4)
    rmse_tr_lr = [0 for _ in range(args.n_epochs)]
    rmse_tr_rl = [0 for _ in range(args.n_epochs)]
    rmse_te_lr = [0 for _ in range(args.n_epochs)]
    rmse_te_rl = [0 for _ in range(args.n_epochs)]
    ds_tr, ds_te = graca.data.collections.moonshot()

    for idx in range(300):
        random.shuffle(ds_tr)
        random.shuffle(ds_te)
        for gl, gr, yl, yr in ds_tr:
            optimizer.zero_grad()
            gl = gl.to("cuda")
            gr = gr.to("cuda")
            yl = torch.tensor(yl, device="cuda", dtype=torch.float32)
            yr = torch.tensor(yr, device="cuda", dtype=torch.float32)
            y_hat = net(gl, gr)
            y = yl - yr
            loss = torch.nn.MSELoss()(y, y_hat)
            if args.train_lr is True:
                loss.backward()
                optimizer.step()

            rmse_tr_lr[idx] += loss.item()


        for gr, gl, yr, yl in ds_tr:
            optimizer.zero_grad()
            gl = gl.to("cuda")
            gr = gr.to("cuda")
            yl = torch.tensor(yl, device="cuda", dtype=torch.float32)
            yr = torch.tensor(yr, device="cuda", dtype=torch.float32)
            y_hat = net(gl, gr)
            y = yl - yr
            loss = torch.nn.MSELoss()(y, y_hat)
            if args.train_rl is True:
                loss.backward()
                optimizer.step()
            rmse_tr_rl[idx] += loss.item()

        for gr, gl, yr, yl in ds_te:
            gl = gl.to("cuda")
            gr = gr.to("cuda")
            yl = torch.tensor(yl, device="cuda", dtype=torch.float32)
            yr = torch.tensor(yr, device="cuda", dtype=torch.float32)
            y_hat = net(gl, gr)
            y = yl - yr
            loss = torch.nn.MSELoss()(y, y_hat)
            rmse_te_rl[idx] += loss.item()

        for gl, gr, yl, yr in ds_te:
            gl = gl.to("cuda")
            gr = gr.to("cuda")
            yl = torch.tensor(yl, device="cuda", dtype=torch.float32)
            yr = torch.tensor(yr, device="cuda", dtype=torch.float32)
            y_hat = net(gl, gr)
            y = yl - yr
            loss = torch.nn.MSELoss()(y, y_hat)
            rmse_te_lr[idx] += loss.item()

    import numpy as np
    rmse_tr_lr = (rmse_tr_lr / len(ds_tr)) ** 0.5
    rmse_tr_rl = (rmse_tr_rl / len(ds_tr)) ** 0.5
    rmse_te_lr = (rmse_te_lr / len(ds_te)) ** 0.5
    rmse_te_rl = (rmse_te_rl / len(ds_te)) ** 0.5

    if args.out is None:
        args.out = args.model

    np.save(args.out + "/rmse_tr_lr.npy", rmse_tr_lr)
    np.save(args.out + "/rmse_tr_rl.npy", rmse_tr_lr)
    np.save(args.out + "/rmse_rl_lr.npy", rmse_te_lr)
    np.save(args.out + "/rmse_rl_rl.npy", rmse_te_lr)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=str, default=300)
    parser.add_argument("--model", type=str, default="GMN")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run(args)
