import torch
from torch import Tensor, softmax, sign
import torch.nn.functional as F
from torch.nn import (
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    Linear,
    Transformer,
    TransformerEncoder,
    TransformerDecoder,
    Module,
    Dropout,
    LayerNorm,
    Parameter,
    ModuleList,
    Linear,
    Sequential,
    ReLU,
    MultiheadAttention,
)

# from torch.nn.init import xavier_uniform_, constant_, kaiming_uniform_, kaiming_normal_, xavier_normal_


class GRN(Module):
    def __init__(
        self,
        d_model,
        d_grn=1024,
        dropout=0.1,
        g1_activation=F.sigmoid,
        g2_activation=F.elu,
        device="cuda",
    ):
        super().__init__()

        self.linear2 = Linear(d_model, d_grn, device=device)  # for g2
        self.linear1 = Linear(d_grn, d_model, device=device)  # for g1

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.g1_activation = g1_activation
        self.g2_activation = g2_activation

    def forward(self, x):
        
        g2 = self.dropout2(self.g2_activation(self.linear2(x)))

        g1 = self.linear1(g2)

        out = self.dropout1(self.g1_activation(g1) * g1)

        # swiglu_g1 = self.dropout1(F.silu(g1)*g1)

        # geglu_g1 = self.dropout1(F.gelu(g1)*g1)

        # glu_g1 = self.dropout1(F.sigmoid(g1)*g1)

        # transformer layers that we are subclassing already implement residual
        # connection and layernorm
        return out


# # # TODO change this
# class GRN(Module):
#     def __init__(self, d_model, d_grn=1024, dropout=0.1, device="cuda"):
#         super(GRN, self).__init__()

#         self.linear2 = Linear(d_model, d_grn, device=device)  # for g2
#         # * Project to 2xd_model as glu will then split in final dimension and gate
#         self.linear1 = Linear(d_grn, 2 * d_model, device=device)  # for g1

#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)

#     def forward(self, x):
#         # ? usually dropout after activations but could try both
#         # g2 = F.elu(self.dropout2(self.linear2(x)))
#         g2 = self.dropout2(F.elu(self.linear2(x)))

#         # ? Do we want dropout here?
#         # * HUGO: I would be tempted to replace the sigmoid in the glu with something like a gelu :shrug:
#         glu_g1 = self.dropout1(F.glu(self.linear1(g2), dim=-1))

#         # transformer layers that we are subclassing already implement residual
#         # connection and layernorm
#         return glu_g1


# can actually rewrite this whole thing using a single linear layer see below implementation
# class Time2Vec(Module):
#     def __init__(
#         self,
#         d_model,
#         activation_fn=torch.sin,
#         device="cuda",
#     ):
#         super(Time2Vec, self).__init__()
#         self.k = d_model - 1
#         self.linear = Linear(1, 1, device=device)
#         self.freqs = Parameter(torch.randn(self.k, device=device))
#         self.phases = Parameter(torch.randn(self.k, device=device))

#         self.activation_fn = activation_fn
#         self.device = device

#     def forward(self, t):
#         zero_component = self.linear(t)  # (bsz, T, 1)

#         k_components = self.activation_fn(t * self.freqs + self.phases)  # (bsz, T, k)

#         out = torch.cat([zero_component, k_components], dim=-1)  # (bsz, T, k+1=d_model)

#         return out


class Time2Vec(Module):
    def __init__(self, d_model, device="cuda"):
        super(Time2Vec, self).__init__()
        # Linear with bias equivalent to freqs and phase components of time2vec
        # just apply sin to components 1+
        self.linear = Linear(1, d_model, bias=True, device=device)

    def forward(self, t):
        # t dim is (bsz, T, 1)
        x = self.linear(t)  # (bsz, T, d_model)

        # The below two steps are needed to avoid errors in the backward pass when
        # modifying a tensor in place

        # Create a new tensor for the sine-transformed values
        sin_transformed = torch.sin(x[:, :, 1:])

        # Concatenate the first element (unchanged) with the sine-transformed values
        x = torch.cat([x[:, :, 0].unsqueeze(-1), sin_transformed], dim=-1)

        return x


class PortfolioTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        g1_activation=F.sigmoid,
        g2_activation=F.elu,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

        self.grn = GRN(
            d_model=d_model,
            d_grn=dim_feedforward,
            dropout=dropout,
            device=device,
            g1_activation=g1_activation,
            g2_activation=g2_activation,
        )

    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Hack to insert GRN. Overrides the feed forward block of the
        TransformerEncoderLayer with a GRN.
        """

        return self.grn(x)


class PortfolioTransformerDecoderLayer(TransformerDecoderLayer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 0.00001,
        batch_first: bool = False,
        norm_first: bool = False,
        device=None,
        dtype=None,
        g1_activation=F.sigmoid,
        g2_activation=F.elu,
    ) -> None:
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            batch_first=batch_first,
            norm_first=norm_first,
            device=device,
            dtype=dtype,
        )

        self.grn = GRN(
            d_model=d_model,
            d_grn=dim_feedforward,
            dropout=dropout,
            device=device,
            g1_activation=g1_activation,
            g2_activation=g2_activation,
        )

    def _ff_block(self, x: Tensor) -> Tensor:
        """
        Hack to insert GRN. Overrides the feed forward block of the
        TransformerEncoderLayer with a GRN.
        """
        return self.grn(x)


class PortfolioTransformer(Module):
    def __init__(
        self,
        n_assets,
        d_model,
        nhead,
        num_layers=4,
        tau=25,
        d_grn=2048,
        use_src_causal_mask=False,
        device="cuda",
        norm_first=False,
        batch_first=True,
        dropout=0.1,
        g1_activation=F.sigmoid,
        g2_activation=F.elu,
    ) -> None:
        super().__init__()

        # lets try sharing params for src and tgt
        # self.linear_in = ModuleList(
        #     [
        #         Linear(n_assets, d_model, device=device),
        #         Linear(n_assets, d_model, device=device),
        #     ]
        # )

        self.tau = tau

        self.linear_in = Linear(n_assets, d_model, device=device)

        # self.t2v = ModuleList(
        #     [
        #         Time2Vec(d_model=d_model, device=device),
        #         Time2Vec(d_model=d_model, device=device),
        #     ]
        # )

        self.t2v = Time2Vec(d_model=d_model, device=device)

        encoder_layer = PortfolioTransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            device=device,
            norm_first=norm_first,
            batch_first=batch_first,
            dropout=dropout,
            dim_feedforward=d_grn,
            g1_activation=g1_activation,
            g2_activation=g2_activation,
        )
        decoder_layer = PortfolioTransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            device=device,
            norm_first=norm_first,
            batch_first=batch_first,
            dropout=dropout,
            dim_feedforward=d_grn,
            g1_activation=g1_activation,
            g2_activation=g2_activation,
        )

        encoder = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        decoder = TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.transformer = Transformer(
            d_model=d_model,
            custom_encoder=encoder,
            custom_decoder=decoder,
            device=device,
            batch_first=batch_first,
            dropout=dropout,
            dim_feedforward=d_grn,
        )

        self.dropout_out = Dropout(dropout)

        self.output_layer = Linear(d_model, n_assets, device=device)

        # paper does not use masked self attention for the encoder inputs,
        # this makes sense in nlp use cases like translation
        # however for time series data I'm wondering if this can lead to data
        # leakage by allowing the encoder to learn features from future time steps?
        # maybe we should try both with and without this causal mask?
        self.use_src_causal_mask = use_src_causal_mask

        self.device = device

        # initialize all weights with xavier init
        # self._reset_parameters()

    # # why is this not working?
    # def _reset_parameters(self):
    #     """Initialize parameters in custom layers"""

    #     xavier_uniform_(self.t2v.linear.weight)
    #     kaiming_uniform_(self.linear_in.weight)
    #     kaiming_uniform_(self.output_layer.weight)
    #     constant_(self.t2v.linear.bias, 0.)
    #     constant_(self.linear_in.bias, 0.)
    #     constant_(self.output_layer.bias, 0.)

    def forward(
        self,
        input: Tensor,
    ) -> Tensor:

        # split asset returns into src and tgt sequences

        # src sequence, 0 -> tau
        src = input[:, : -self.tau, :].to(self.device)

        # tgt sequence for decoder input, asset returns for tau+1 -> t
        tgt = input[:, -self.tau : -1, :].to(self.device)

        mask = create_causal_mask(tgt.size(1))
        tgt_causal_mask = mask.to(self.device)

        src_causal_mask = None
        if self.use_src_causal_mask:
            mask = create_causal_mask(src.size(1))
            src_causal_mask = mask.to(self.device)

        embs = []
        for _, x in enumerate([src, tgt]):
            # input x has shape (bsz, T, N_assets)
            x = self.linear_in(x)  # (bsz, T, d_model)

            t = torch.arange(x.shape[1])  # (T)
            t = t.unsqueeze(0).unsqueeze(-1)  # (1,T,1)
            t = t.expand(x.shape[0], -1, -1).to(
                self.device, dtype=torch.float32
            )  # (bsz, T, 1)
            t2v = self.t2v(t)  # (bsz, T, d_model)

            x = x + t2v  # (bsz, T, d_model)

            embs.append(x)

        src, tgt = embs

        out = self.transformer(
            src=src,
            tgt=tgt,
            src_mask=src_causal_mask,
            tgt_mask=tgt_causal_mask,
        )  # (bsz, T, d_model)

        out = self.dropout_out(self.output_layer(out))  # (bsz, T, N_assets)

        # return out

        return sign(out) * softmax(out, dim=-1)


def create_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return mask
