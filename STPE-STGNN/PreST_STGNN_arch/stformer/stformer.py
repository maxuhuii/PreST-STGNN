import torch
from torch import nn
from timm.models.vision_transformer import trunc_normal_
import torch.nn.functional as F
from .patch import PatchEmbedding
from .mask import MaskGenerator
from .positional_encoding import PositionalEncoding
from .transformer_layers import TransformerLayers



def unshuffle(shuffled_tokens):
    dic = {}
    for k, v, in enumerate(shuffled_tokens):
        dic[v] = k
    unshuffle_index = []
    for i in range(len(shuffled_tokens)):
        unshuffle_index.append(dic[i])
    return unshuffle_index


class STFormer(nn.Module):
    """An efficient unsupervised pre-training model for Time Series based on transFormer blocks. (TSFormer)"""

    def __init__(self, patch_size, num_nodes, in_channel, embed_dim, num_heads, mlp_ratio, dropout, num_token, mask_ratio, encoder_depth, decoder_depth, mode="pre-train",supports=None):
        super().__init__()
        assert mode in ["pre-train", "forecasting"], "Error mode."

        self.patch_size = patch_size
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_token = num_token # 这里暂时先填N*T，但需要随着patch以及token策略而变化
        self.mask_ratio = mask_ratio
        self.encoder_depth = encoder_depth
        self.mode = mode
        self.mlp_ratio = mlp_ratio

        self.num_nodes = num_nodes

        # 用于进行邻接矩阵的生成
        self.supports = supports
        adjinit = supports[0]
        m, p, n = torch.svd(adjinit)
        initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
        initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
        self.nodevec1 = nn.Parameter(initemb1, requires_grad=True)
        self.nodevec2 = nn.Parameter(initemb2, requires_grad=True)
        '''
        self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10), requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(10,self.num_nodes ), requires_grad=True)
        '''
        self.adpA = None

        self.selected_feature = 0

        # norm layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)



        # encoder specifics
        '''
        注：这里暂时不用patch_embedding，只对输入的数据进行了dim_embedding
        
        '''
        # # patchify & embedding
        # self.patch_embedding = PatchEmbedding(patch_size, in_channel, embed_dim, norm_layer=None)
        self.dim_embedding = nn.Linear(in_channel, embed_dim, bias=True)
        # # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=0)

        # # masking
        self.mask = MaskGenerator(num_token, mask_ratio)
        # encoder
        self.encoder = TransformerLayers(embed_dim, encoder_depth, mlp_ratio, num_heads, dropout)

        # decoder specifics
        # transform layer
        self.enc_2_dec_emb = nn.Linear(embed_dim, embed_dim, bias=True)
        # # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # # decoder
        self.decoder = TransformerLayers(embed_dim, decoder_depth, mlp_ratio, num_heads, dropout)

        # # prediction (reconstruction) layer
        self.output_layer = nn.Linear(embed_dim, 1)
        self.initialize_weights()

    def initialize_weights(self):
        # positional encoding
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)
        # mask token
        trunc_normal_(self.mask_token, std=.02)

    def get_adp_graph(self, max_num_neigh = 40):
        adp_A = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
        threshold = 1/self.num_nodes
        tmp,_ = torch.kthvalue(-1*adp_A, max_num_neigh + 1,dim=1,keepdim=True)
        bin_mask = (torch.logical_and((adp_A > threshold), (adp_A > -tmp)).type_as(adp_A) - adp_A).detach() + adp_A
        adp_A = adp_A*bin_mask
        return adp_A

    def encoding(self, long_term_history,tod,dow, mask=True):
        """Encoding process of TSFormer: patchify, positional encoding, mask, Transformer layers.

        Args:
            long_term_history (torch.Tensor): Very long-term historical MTS with shape [B, N, 1, P * L],
                                                which is used in the TSFormer.
                                                P is the number of segments (patches).
            mask (bool): True in pre-training stage and False in forecasting stage.

        Returns:
            torch.Tensor: hidden states of unmasked tokens
            list: unmasked token index
            list: masked token index
        """
        # long_term_history:[32,207,1,2016]
        B, N, C, T = long_term_history.shape

        # STEP0. 对自适应邻接矩阵的学习
        adpA = self.get_adp_graph()
        # 将NXN矩阵进行扩张，成为NTXNT矩阵
        self.adpA = (adpA.repeat_interleave(T, dim=0)).repeat_interleave(T, dim=1)


        # STEP1. dim embedding
        # B,N,C,T->B,N,T,C
        long_term_history = long_term_history.permute(0,1,3,2)
        # B,N,T,C->B,N,T,C'
        hidden_states = self.dim_embedding(long_term_history)

        '''
        # patchify and embed input
        patches = self.patch_embedding(long_term_history)     # B, N, d, P
        patches = patches.transpose(-1, -2)         # B, N, P, d
        '''

        # STEP2. token化
        token = hidden_states.reshape(B,N*T,-1)
        tod = tod.reshape(B,N*T,-1)
        dow = dow.reshape(B,N*T,-1)

        # STEP3. positional embedding
        token = self.positional_encoding(token,tod,dow)
        A = self.adpA
        # STEP4. mask
        if mask:
            unmasked_token_index, masked_token_index = self.mask()
            encoder_input = token[:, unmasked_token_index, :]
            A = self.adpA[unmasked_token_index,:]
            A = A[:,unmasked_token_index]

        else:
            unmasked_token_index, masked_token_index = None, None
            encoder_input = token



        attention_mask = ~(A.bool())

        # STEP5. encoding
        hidden_states_unmasked = self.encoder(encoder_input,mask=attention_mask)
        hidden_states_unmasked = self.encoder_norm(hidden_states_unmasked)


        return hidden_states_unmasked, unmasked_token_index, masked_token_index

    def decoding(self, hidden_states_unmasked, masked_token_index,tod,dow):
        """Decoding process of TSFormer: encoder 2 decoder layer, add mask tokens, Transformer layers, predict.

        Args:
            hidden_states_unmasked (torch.Tensor): hidden states of masked tokens [B, N, P*(1-r), d].
            masked_token_index (list): masked token index

        Returns:
            torch.Tensor: reconstructed data
        """
        batch_size, NT, C = hidden_states_unmasked.shape
        tod = tod.reshape(batch_size,-1,1)
        dow = dow.reshape(batch_size,-1,1)

        # encoder 2 decoder layer
        hidden_states_unmasked = self.enc_2_dec_emb(hidden_states_unmasked)

        # add mask tokens
        hidden_states_masked = self.positional_encoding(
            self.mask_token.expand(batch_size, len(masked_token_index), hidden_states_unmasked.shape[-1]),
            tod,dow,
            index=masked_token_index
            )
        hidden_states_full = torch.cat([hidden_states_unmasked, hidden_states_masked], dim=-2)   # B, NT, C

        # decoding
        hidden_states_full = self.decoder(hidden_states_full)
        hidden_states_full = self.decoder_norm(hidden_states_full)
        # hidden_states_full:[8,207,168,96]

        # prediction (reconstruction)
        reconstruction_full = self.output_layer(hidden_states_full)

        return reconstruction_full

    def get_reconstructed_masked_tokens(self, reconstruction_full, real_value_full, unmasked_token_index, masked_token_index):
        """Get reconstructed masked tokens and corresponding ground-truth for subsequent loss computing.

        Args:
            reconstruction_full (torch.Tensor): reconstructed full tokens.
            real_value_full (torch.Tensor): ground truth full tokens.
            unmasked_token_index (list): unmasked token index.
            masked_token_index (list): masked token index.

        Returns:
            torch.Tensor: reconstructed masked tokens.
            torch.Tensor: ground truth masked tokens.
        """
        # get reconstructed masked tokens
        # batch_size, num_nodes, _, _ = reconstruction_full.shape
        # B , NT, C
        B, NT, C = reconstruction_full.shape
        reconstruction_masked_tokens = reconstruction_full[:, len(unmasked_token_index):, :]     # B, N, r*P, d
        # reconstruction_masked_tokens = reconstruction_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)     # B, r*P*d, N

        # real_value_full: [B,N,C,T]
        # B,N,C,T -> B,N,T,C
        real_value_full = real_value_full.permute(0,1,3,2)
        b,n,t,c = real_value_full.shape
        label_full = real_value_full[:,:,:,self.selected_feature]
        label_full = label_full.reshape(B,-1,1)
        # label_full = real_value_full.permute(0, 3, 1, 2).unfold(1, self.patch_size, self.patch_size)[:, :, :, self.selected_feature, :].transpose(1, 2)  # B, N, P, L
        label_masked_tokens = label_full[:, masked_token_index, :].contiguous() # B, N, r*P, d

        # label_masked_tokens = label_masked_tokens.view(batch_size, num_nodes, -1).transpose(1, 2)

        # 这里返回的都是 B, NT(1-mr), C的大小，这使得在计算误差的时候要重写一下
        return reconstruction_masked_tokens, label_masked_tokens

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor = None, batch_seen: int = None, epoch: int = None, **kwargs) -> torch.Tensor:
        """feed forward of the TSFormer.
            TSFormer has two modes: the pre-training mode and the forecasting mode,
                                    which are used in the pre-training stage and the forecasting stage, respectively.

        Args:
            history_data (torch.Tensor): very long-term historical time series with shape B, L * P, N, 1.

        Returns:
            pre-training:
                torch.Tensor: the reconstruction of the masked tokens. Shape [B, L * P * r, N, 1]
                torch.Tensor: the ground truth of the masked tokens. Shape [B, L * P * r, N, 1]
                dict: data for plotting.
            forecasting:
                torch.Tensor: the output of TSFormer of the encoder with shape [B, N, L, 1].
        """
        # reshape
        history_data = history_data.permute(0, 2, 3, 1)     # B, N, 1, L * P
        B, N, C, T = history_data.shape
        origin_data = history_data[:,:,:1,:]
        time_of_day = history_data[:,:,3:4,:]
        day_of_week = history_data[:,:,2:3,:]
        # feed forward
        if self.mode == "pre-train":
            # encoding
            hidden_states_unmasked, unmasked_token_index, masked_token_index = self.encoding(origin_data,time_of_day,day_of_week)
            # decoding
            reconstruction_full = self.decoding(hidden_states_unmasked, masked_token_index,time_of_day,day_of_week)
            # for subsequent loss computing

            # 在计算完这个误差后，这些tensor会变成unable ot get repr(代表这里些的有问题）
            reconstruction_masked_tokens, label_masked_tokens = self.get_reconstructed_masked_tokens(reconstruction_full, history_data, unmasked_token_index, masked_token_index)
            # 需要注意的是，由于get_reconstructed_masked_tokens的写法不一样了，在执行计算误差时也要相应的修改一下
            return reconstruction_masked_tokens, label_masked_tokens
        else:
            hidden_states_full, _, _ = self.encoding(origin_data,time_of_day,day_of_week, mask=False)
            hidden_states_full = hidden_states_full.view(B,N,T,-1)
            # 32,207,12,96
            return hidden_states_full
