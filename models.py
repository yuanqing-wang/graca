import torch
import dgl
from dgl.nn.pytorch import GraphConv

class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, hl, hr):
        hl = hl / hl.norm(dim=-1, keepdim=True)
        hr = hr / hr.norm(dim=-1, keepdim=True)

        a = (hl[:, None, :] * hr[None, :, :]).sum(dim=-1)

        mu_lr = hr - a.softmax(dim=1).transpose(1,0 ) @ hl
        mu_rl = hl - a.softmax(dim=0) @ hr

        return mu_lr, mu_rl

class GMN(torch.nn.Module):
    def __init__(
        self,
        depth=3,
        in_features=74,
        hidden_features=128,
        out_features=1,
    ):
        super(GMN, self).__init__()
        _in_features = 2 * in_features
        _out_features = hidden_features
        for idx in range(depth):

            setattr(
                self,
                "ff%s" % idx,
                torch.nn.Sequential(
                    torch.nn.Linear(_in_features, _out_features),
                    torch.nn.ReLU(),
                )
            )

            _in_features = 2 * hidden_features


        self.depth = depth
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.gmn = GMN()

    def forward(self, gl, gr):
        gl = gl.local_var()
        gr = gr.local_var()
        xl = gl.ndata["feat"]
        xr = gr.ndata["feat"]
        for idx in range(self.depth):
            mu_lr, mu_rl = self.gmn(xl, xr)
            gl.ndata["x"] = xl
            gr.ndata["x"] = xr
            gl.update_all(dgl.function.copy_src(src='x', out='m'),
                             dgl.function.sum(msg='m', out='x'))

            xl = gl.ndata["x"]
            xr = gr.ndata["x"]

            xl = torch.cat([xl, mu_rl], dim=-1)
            xr = torch.cat([xr, mu_lr], dim=-1)

            xl = getattr(self, "ff%s"%idx)(xl)
            xr = getattr(self, "ff%s"%idx)(xr)

        gl.ndata["x"] = xl
        gr.ndata["x"] = xr

        xl = dgl.sum_nodes(gl, "x")
        xr = dgl.sum_nodes(gr, "x")
        x = self.ff(
            torch.cat(
                [xl, xr], dim=-1
            )
        )

class GMN(torch.nn.Module):
    def __init__(
        self,
        depth=3,
        in_features=74,
        hidden_features=128,
        out_features=1,
    ):
        super(GMN, self).__init__()
        _in_features = 2 * in_features
        _out_features = hidden_features
        for idx in range(depth):

            setattr(
                self,
                "ff%s" % idx,
                torch.nn.Sequential(
                    torch.nn.Linear(_in_features, _out_features),
                    torch.nn.ReLU(),
                )
            )

            _in_features = 2 * hidden_features


        self.depth = depth
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )
        self.gmn = GMN()

    def forward(self, gl, gr):
        gl = gl.local_var()
        gr = gr.local_var()
        xl = gl.ndata["feat"]
        xr = gr.ndata["feat"]
        for idx in range(self.depth):
            mu_lr, mu_rl = self.gmn(xl, xr)
            gl.ndata["x"] = xl
            gr.ndata["x"] = xr
            gl.update_all(dgl.function.copy_src(src='x', out='m'),
                             dgl.function.sum(msg='m', out='x'))

            xl = gl.ndata["x"]
            xr = gr.ndata["x"]

            xl = torch.cat([xl, mu_rl], dim=-1)
            xr = torch.cat([xr, mu_lr], dim=-1)

            xl = getattr(self, "ff%s"%idx)(xl)
            xr = getattr(self, "ff%s"%idx)(xr)

        gl.ndata["x"] = xl
        gr.ndata["x"] = xr

        xl = dgl.sum_nodes(gl, "x")
        xr = dgl.sum_nodes(gr, "x")
        x = self.ff(
            torch.cat(
                [xl, xr], dim=-1
            )
        )

        return x

class DeltaModel(torch.nn.Module):
    def __init__(
        self,
        depth=3,
        in_features=74,
        hidden_features=128,
        out_features=1,
        layer=GraphConv,
        activation=torch.nn.functional.relu,
    ):
        super(DeltaModel, self).__init__()
        _in_features = in_features
        _out_features = hidden_features
        for idx in range(depth):
            setattr(
                self,
                "gn%s" % idx,
                layer(
                    _in_features, _out_features,
                    activation=activation,
                )
            )
            _in_features = hidden_features

        self.depth = depth
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, out_features)
        )

    def forward(self, gl, gr):
        gl = gl.local_var()
        gr = gr.local_var()
        xl = gl.ndata["feat"]
        xr = gr.ndata["feat"]
        for idx in range(self.depth):

            xl = getattr(self, "gn%s" % idx)(gl, xl)
            xr = getattr(self, "gn%s" % idx)(gr, xr)

        gl.ndata["x"] = xl
        gr.ndata["x"] = xr

        xl = dgl.sum_nodes(gl, "x")
        xr = dgl.sum_nodes(gr, "x")
        x = self.ff(
            torch.cat(
                [xl, xr], dim=-1
            )
        )

        return x
