import torch.nn as nn
import torch
import math

class FCModule(nn.Module):
    def __init__(self,embed_dim=256):
        super(FCModule, self).__init__()
        self.active_relu = nn.ReLU()
        self.emb = nn.Embedding(100, embed_dim)
        self.ln1 = nn.Linear(embed_dim, 4*embed_dim)
        self.bn1 = nn.BatchNorm1d(4*embed_dim)
        self.ln2 = nn.Linear(4*embed_dim,embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        nn.init.xavier_uniform_(self.ln1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.ln2.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        x_emb = self.emb(x)
        x = self.bn1(self.active_relu(self.ln1(x_emb)))
        x = self.bn2(self.active_relu(x_emb + self.ln2(x)))
        return x



class sinEmbedModule(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout,
        dropout_rate,
        dropout_in_first_layer,
    ):
        super(sinEmbedModule, self).__init__()
        self.layers = nn.ModuleList()
        self.lambda_min = 10**-2.5
        self.lambda_max = 10**3.3
        self.x = torch.arange(0, embedding_dim, 2).to(torch.float64)
        self.x = (
            2
            * math.pi
            * (
                self.lambda_min
                * (self.lambda_max / self.lambda_min) ** (self.x / (embedding_dim - 2))
            )
            ** -1
        )
        dropout_starting_layer = 0 if dropout_in_first_layer else 1
        for i in range(2):
            self.layers.append(nn.Linear(embedding_dim, embedding_dim))
            if i==0:
                self.layers.append(nn.ReLU())
            # self.layers.append(nn.BatchNorm1d(embedding_dim))
            self.layers.append(nn.LayerNorm(embedding_dim))
            if dropout and i >= dropout_starting_layer:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, mz):
        self.x = self.x.to(mz.device)
        x = torch.einsum("bl,d->bld", mz, self.x)
        sinemb = torch.sin(x)
        cosemb = torch.cos(x)
        b, l, d = sinemb.shape
        x = torch.empty((b, l, 2 * d), dtype=mz.dtype, device=mz.device)
        x[:, :, ::2] = sinemb
        x[:, :, 1::2] = cosemb
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = layer(x)
        return x


class EmbedModule(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        dropout,
        dropout_rate,
        dropout_in_first_layer,
    ):
        super(EmbedModule, self).__init__()
        self.layers = nn.ModuleList()
        self.tkembd = nn.Embedding(20000, embedding_dim)
        self.sinemb = sinEmbedModule(
            embedding_dim, dropout, dropout_rate, dropout_in_first_layer
        )
        dropout_starting_layer = 0 if dropout_in_first_layer else 1
        for i in range(2):
            if i == 0:
                self.layers.append(nn.Linear(embedding_dim + 1, embedding_dim))
                self.layers.append(nn.ReLU())
            else:
                self.layers.append(nn.Linear(embedding_dim, embedding_dim))
            self.layers.append(nn.LayerNorm(embedding_dim))
            if dropout and i >= dropout_starting_layer:
                self.layers.append(nn.Dropout(dropout_rate))

    def forward(self, mz, intensity, precursor):
        mz = torch.cat([mz, precursor], dim=1)
        mzemb = self.sinemb(mz)
        intensity = torch.cat(
            [intensity, 2 * torch.ones((mzemb.shape[0], 1)).to(mz.device)], -1
        )
        x = torch.cat([mzemb, intensity.unsqueeze(-1)], dim=2)
        for layer in self.layers:
            if isinstance(layer, nn.BatchNorm1d):
                x = layer(x.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                x = layer(x)
        return x


class SinSiameseModel(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim: int = 512,
        dropout: bool = True,
        dropout_rate: float = 0.1,
        dropout_in_first_layer: bool = False,
        device='cuda',
        dtype = torch.float32,
        project_size = 200
    ):
        super(SinSiameseModel, self).__init__()
        self.device = device
        self.dtype = dtype
        self.input_dim = input_dim
        self.embd_model = EmbedModule(
            input_dim,
            embed_dim,
            dropout,
            dropout_rate,
            dropout_in_first_layer,
        )

        self.tranencoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 8, batch_first=True),
            6,
        )

        self.trandecoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, 8, batch_first=True), 
            6,
        )

        self.fc_ins = FCModule(embed_dim=embed_dim)
        self.fc_charge = FCModule(embed_dim=embed_dim)
        self.fc_adduct = FCModule(embed_dim=embed_dim)
        self.project = nn.Linear(embed_dim,project_size)
        self.active = nn.SELU()
        # self.head_model = HeadModule(self.embd_model, self.tranencoder)
    
    def inference(self, mz, intensity, precursor, cp, ins, charge, adduct):
        emb = self.embd_model(mz, intensity, precursor)
        B, L = mz.shape
        mask = torch.arange(L + 1).expand(B, L + 1).to(mz.device) < cp.unsqueeze(1)
        mask[:,-1] = True
        mask = torch.logical_not(mask)
        out = self.tranencoder(emb, src_key_padding_mask = mask)

        param_emb = torch.stack([self.fc_ins(ins), self.fc_adduct(adduct), self.fc_charge(charge)],dim=1)
        out = self.trandecoder(param_emb, out, memory_key_padding_mask = mask)

        out = torch.max(out,1)[0]

        return self.active(self.project(out))


    def forward(
        self, input
    ):
        mz = torch.tensor(input[0][:,0,:]).to(self.device).to(self.dtype)
        intensity = torch.tensor(input[0][:,1,:]).to(self.device).to(self.dtype)
        cp = (mz>0).sum(1)
        prmz = torch.tensor(input[3]).to(self.device).to(self.dtype)
        ins = torch.tensor(input[1]).to(self.device)
        charge = torch.tensor(input[2]).to(self.device)
        adduct = torch.tensor(input[4]).to(self.device)
        return self.inference(mz, intensity, prmz, cp, ins, charge, adduct)
