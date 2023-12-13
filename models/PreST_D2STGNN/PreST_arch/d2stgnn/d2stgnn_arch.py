import torch
import torch.nn as nn
import torch.nn.functional as F

from .difusion_block import DifBlock
from .inherent_block import InhBlock
from .dynamic_graph_conv.dy_graph_conv import DynamicGraphConstructor
from .decouple.estimation_gate import EstimationGate


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, fk_dim=256, first=False, **model_args):
        super().__init__()
        self.spatial_gate = EstimationGate(
            model_args['node_hidden'], model_args['time_emb_dim'], 64, model_args['seq_length'])
        self.dif_layer = DifBlock(hidden_dim, fk_dim=fk_dim, **model_args)
        self.inh_layer = InhBlock(
            hidden_dim, fk_dim=fk_dim, first=first, **model_args)

    def forward(self, X: torch.Tensor, dynamic_graph: torch.Tensor, static_graph, E_u, E_d, T_D, D_W,learn_graph):
        """decouple layer

        Args:
            X (torch.Tensor): input data with shape (B, L, N, D)
            dynamic_graph (list of torch.Tensor): dynamic graph adjacency matrix with shape (B, N, k_t * N)
            static_graph (ist of torch.Tensor): the self-adaptive transition matrix with shape (N, N)
            E_u (torch.Parameter): node embedding E_u
            E_d (torch.Parameter): node embedding E_d
            T_D (torch.Parameter): time embedding T_D
            D_W (torch.Parameter): time embedding D_W

        Returns:
            torch.Tensor: the undecoupled signal in this layer, i.e., the X^{l+1}, which should be feeded to the next layer. shape [B, L', N, D].
            torch.Tensor: the output of the forecast branch of Diffusion Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
            torch.Tensor: the output of the forecast branch of Inherent Block with shape (B, L'', N, D), where L''=output_seq_len / model_args['gap'] to avoid error accumulation in auto-regression.
        """
        X_spa = self.spatial_gate(E_u, E_d, T_D, D_W, X)
        dif_backcast_seq_res, dif_forecast_hidden = self.dif_layer(
            X=X, X_spa=X_spa, dynamic_graph=dynamic_graph, static_graph=static_graph, learn_graph=learn_graph)
        inh_backcast_seq_res, inh_forecast_hidden = self.inh_layer(
            dif_backcast_seq_res)
        return inh_backcast_seq_res, dif_forecast_hidden, inh_forecast_hidden


class D2STGNN(nn.Module):
    """
    Paper: Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting
    Link: https://arxiv.org/abs/2206.09112
    Official Code: https://github.com/zezhishao/D2STGNN
    """
    def __init__(self, **model_args):
        super().__init__()
        # attributes
        self._in_feat = model_args['num_feat']
        self._hidden_dim = model_args['num_hidden']
        self._node_dim = model_args['node_hidden']
        self._forecast_dim = 256
        self._output_hidden = 512
        self._output_dim = model_args['seq_length']

        self._num_nodes = model_args['num_nodes']
        self._k_s = model_args['k_s']
        self._k_t = model_args['k_t']
        self._num_layers = 5
        self._time_in_day_size = model_args['time_in_day_size']
        self._day_in_week_size = model_args['day_in_week_size']

        # model_args['use_pre'] = False
        model_args['use_pre'] = True
        model_args['dy_graph'] = True
        model_args['sta_graph'] = True

        self._model_args = model_args

        # start embedding layer
        self.embedding = nn.Linear(self._in_feat, self._hidden_dim)
        self.st_embedding = nn.Linear(96,self._hidden_dim)
        self.tsformer_embedding = nn.Sequential(nn.Linear(96, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())
        self.relu = nn.ReLU()

        # time embedding
        self.T_i_D_emb = nn.Parameter(
            torch.empty(288, model_args['time_emb_dim']))
        self.D_i_W_emb = nn.Parameter(
            torch.empty(7, model_args['time_emb_dim']))

        # Decoupled Spatial Temporal Layer
        self.layers = nn.ModuleList([DecoupleLayer(
            self._hidden_dim, fk_dim=self._forecast_dim, first=True, **model_args)])
        for _ in range(self._num_layers - 1):
            self.layers.append(DecoupleLayer(
                self._hidden_dim, fk_dim=self._forecast_dim, **model_args))

        # dynamic and static hidden graph constructor
        if model_args['dy_graph']:
            self.dynamic_graph_constructor = DynamicGraphConstructor(
                **model_args)

        # node embeddings
        self.node_emb_u = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))
        self.node_emb_d = nn.Parameter(
            torch.empty(self._num_nodes, self._node_dim))

        # output layer
        self.out_fc_1 = nn.Linear(self._forecast_dim, self._output_hidden)
        self.out_fc_2 = nn.Linear(self._output_hidden, model_args['gap'])

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def _graph_constructor(self, **inputs):
        E_d = inputs['E_d']
        E_u = inputs['E_u']
        if self._model_args['sta_graph']:
            static_graph = [F.softmax(F.relu(torch.mm(E_d, E_u.T)), dim=1)]
        else:
            static_graph = []
        if self._model_args['dy_graph']:
            dynamic_graph = self.dynamic_graph_constructor(**inputs)
        else:
            dynamic_graph = []
        return static_graph, dynamic_graph

    def _calculate_random_walk_matrix(self, adj_mx):
        B, N, N = adj_mx.shape

        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[1])).unsqueeze(0).expand(B, N, N).to(adj_mx.device)
        d = torch.sum(adj_mx, 2)
        d_inv = 1. / d
        d_inv = torch.where(torch.isinf(d_inv), torch.zeros(d_inv.shape).to(adj_mx.device), d_inv)
        d_mat_inv = torch.diag_embed(d_inv)
        random_walk_mx = torch.bmm(d_mat_inv, adj_mx)
        return random_walk_mx

    def _prepare_inputs(self, X):
        num_feat = self._model_args['num_feat']
        # node embeddings
        node_emb_u = self.node_emb_u  # [N, d]
        node_emb_d = self.node_emb_d  # [N, d]
        # time slot embedding
        # [B, L, N, d]
        # In the datasets used in D2STGNN, the time_of_day feature is normalized to [0, 1]. We multiply it by 288 to get the index.
        # If you use other datasets, you may need to change this line.
        T_i_D = self.T_i_D_emb[(X[:, :, :, num_feat] * self._time_in_day_size).type(torch.LongTensor)]
        # [B, L, N, d]
        D_i_W = self.D_i_W_emb[(X[:, :, :, num_feat+1] * self._day_in_week_size).type(torch.LongTensor)]
        # traffic signals
        X = X[:, :, :, :num_feat]

        return X, node_emb_u, node_emb_d, T_i_D, D_i_W

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, hidden_states_st: torch.Tensor, hidden_states: torch.Tensor, sampled_adj: torch.Tensor, **kwargs) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L, N, C]
            
        Returns:
            torch.Tensor: outputs with shape [B, L, N, C]
        """
        # history_data:[32,12,207,3]
        X = history_data
        X[:,:,:,2] = X[:,:,:,2]/7
        # ==================== Prepare Input Data ==================== #
        X, E_u, E_d, T_D, D_W = self._prepare_inputs(X)

        # ========================= Construct Graphs ========================== #
        static_graph, dynamic_graph = self._graph_constructor(
            E_u=E_u, E_d=E_d, X=X, T_D=T_D, D_W=D_W)
        # static_graph[1,207,207] dynamic_graph:[4,32,207,621]
        learn_graph = [] + [self._calculate_random_walk_matrix(sampled_adj)]
        learn_graph = learn_graph + [self._calculate_random_walk_matrix(sampled_adj.transpose(-1, -2))]

        # Start embedding layer
        # X:[32,12,207,32]
        X = self.embedding(X)
        # hidden_states_st:[32,207,12,96]->[32,207,12,32]->[32,12,207,32]
        hidden_states_st = self.st_embedding(hidden_states_st).permute(0,2,1,3)
        X = self.relu(X+hidden_states_st)

        spa_forecast_hidden_list = []
        tem_forecast_hidden_list = []

        tem_backcast_seq_res = X
        for index, layer in enumerate(self.layers):
            tem_backcast_seq_res, spa_forecast_hidden, tem_forecast_hidden = layer(
                tem_backcast_seq_res, dynamic_graph, static_graph, E_u, E_d, T_D, D_W,learn_graph)
            spa_forecast_hidden_list.append(spa_forecast_hidden)
            tem_forecast_hidden_list.append(tem_forecast_hidden)

        # Output Layer
        # spa_forecast_hidden:[32,4,207,256]
        spa_forecast_hidden = sum(spa_forecast_hidden_list)
        # tem_forecast_hidden:[32,4,207,256]
        tem_forecast_hidden = sum(tem_forecast_hidden_list)
        forecast_hidden = spa_forecast_hidden + tem_forecast_hidden

        # hidden_states:[32,207,96]
        hidden_states = self.tsformer_embedding(hidden_states)
        hidden_states = hidden_states.unsqueeze(1)
        forecast_hidden += hidden_states


        # regression layer
        forecast = self.out_fc_2(
            F.relu(self.out_fc_1(F.relu(forecast_hidden))))
        forecast = forecast.transpose(1, 2).contiguous().view(
            forecast.shape[0], forecast.shape[2], -1)

        # reshape
        forecast = forecast.transpose(1, 2).unsqueeze(-1)
        return forecast
