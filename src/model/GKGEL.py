from .BaseModel import *

from torch_geometric.nn import GCN,GAT
def compute_svd_rotation(embeddings):
    # Compute the SVD of the embeddings
    U, S, V = torch.svd(embeddings,some=False)
    # Cseate rotation matrixs
    # rotation_matrix = torch.mm(U, V.t())
    return U, S, V
class GKGEL(BaseModel):
    def __init__(self, args, kg):
        super(GKGEL, self).__init__(args, kg)
        self.gcn = MAE(args, kg)
        self.mse_loss_func = nn.MSELoss(size_average=False)
        self.ent_weight, self.rel_weight, self.other_weight = None, None, None
        self.margin_loss_func = nn.MarginRankingLoss(float(self.args.margin), size_average=False).to(self.args.device)
        self.init_old_weight()
        self.fisher = dict()
        self.fisher_old = dict()
        for n, p in self.named_parameters():
            n = n.replace('.','_')
            self.fisher[n] = 0 * p.data
        self.inputs = {}
        self.register_hooks()
    def store_old_parameters(self):
        '''
        Store learned paramters and weights for regularization.
        '''
        self.args.snapshot -= 1
        param_weight = self.get_new_weight()
        self.args.snapshot += 1
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            value = param.data
            old_weight = getattr(self, 'old_weight_{}'.format(name))
            new_weight = param_weight[name]
            self.register_buffer('old_data_{}'.format(name), value)
            if '_embeddings' in name:
                if self.args.snapshot == 0:
                    old_weight = torch.zeros_like(new_weight)
                else:
                    old_weight = torch.cat([old_weight, torch.zeros(new_weight.size(0) - old_weight.size(0), 1).to(self.args.device)], dim=0)
            self.register_buffer('old_weight_{}'.format(name), old_weight + new_weight)

    def init_old_weight(self):
        '''
        Initialize the learned parameters for storage.
        '''
        for name, param in self.named_parameters():
            name_ = name.replace('.', '_')
            if 'ent_embeddings' in name_:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))
            elif 'rel_embeddings' in name_:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor([[]]))
                self.register_buffer('old_data_{}'.format(name_), torch.tensor([[]]))
            else:
                self.register_buffer('old_weight_{}'.format(name_), torch.tensor(0.0))
                self.register_buffer('old_data_{}'.format(name_), param.data)


    def switch_snapshot(self):
        '''
        Prepare for the training on next snapshot.
        '''
        '''store old parameters'''
        for n, param in self.named_parameters():
            n = n.replace('.', '_')
            if param.requires_grad:
                self.fisher[n] /= len(self.kg.snapshots[self.args.snapshot].train_new)
                self.fisher[n] = self.fisher[n].to(self.args.device)
                self.fisher_old[n] = self.fisher_old[n].to(self.args.device)
                self.fisher[n] = torch.cat([(self.fisher[n][:self.fisher_old[n].size(0)] + self.fisher_old[n] * self.args.snapshot+1) / (self.args.snapshot + 2),self.fisher[n][self.fisher_old[n].size(0):]],dim=0)

        self.store_old_parameters()
        '''expand embedding size for new entities and relations'''
        ent_embeddings, rel_embeddings = self.expand_embedding_size()
        new_ent_embeddings = ent_embeddings.weight.data
        new_rel_embeddings = rel_embeddings.weight.data
        '''inherit learned paramters'''
        new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = torch.nn.Parameter(self.ent_embeddings.weight.data)
        new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = torch.nn.Parameter(self.rel_embeddings.weight.data)
        self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
        self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)
        '''embedding transfer'''
        if self.args.using_embedding_transfer == 'True':
            # scale = self.args.emb_dim ** 0.5
            reconstruct_ent_embeddings, reconstruct_rel_embeddings = self.reconstruct()
            # new_ent_embeddings_attention,_ = self.calculate_entity_attention(reconstruct_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent],self.ent_embeddings.weight.data,self.ent_embeddings.weight.data,scale) #[10513,10513]
            # new_rel_embeddings_attention,_ = self.calculate_relation_attention(reconstruct_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel],self.rel_embeddings.weight.data,self.rel_embeddings.weight.data,scale) #[10513,10513]
            # new_ent_embeddings[:self.kg.snapshots[self.args.snapshot].num_ent] = new_ent_embeddings_attention
            # new_rel_embeddings[:self.kg.snapshots[self.args.snapshot].num_rel] = new_rel_embeddings_attention
            new_ent_embeddings[self.kg.snapshots[self.args.snapshot].num_ent:] = reconstruct_ent_embeddings[self.kg.snapshots[self.args.snapshot].num_ent:]
            new_rel_embeddings[self.kg.snapshots[self.args.snapshot].num_rel:] = reconstruct_rel_embeddings[self.kg.snapshots[self.args.snapshot].num_rel:]
            if self.args.embedding_transfer_norm == 'True':
                new_ent_embeddings = self.ent_layer_norm(new_ent_embeddings)
                new_rel_embeddings = self.ent_layer_norm(new_rel_embeddings)
            self.ent_embeddings.weight = torch.nn.Parameter(new_ent_embeddings)
            self.rel_embeddings.weight = torch.nn.Parameter(new_rel_embeddings)
        '''store the total number of facts containing each entity or relation'''
        new_ent_weight, new_rel_weight, new_other_weight = self.get_weight()
        self.register_buffer('new_weight_ent_embeddings_weight', new_ent_weight.clone().detach())
        self.register_buffer('new_weight_rel_embeddings_weight', new_rel_weight.clone().detach())
        '''get regularization weights'''
        self.new_weight_other_weight = new_other_weight
        '''4. store old fisher matrix '''
        if self.args.snapshot >= 0:
            self.fisher_old = {}
            for n, _ in self.named_parameters():
                n = n.replace('.','_')
                self.fisher_old[n] = self.fisher[n].clone()
    def reconstruct(self):
        '''
        Reconstruct the entity and relation embeddings.
        '''
        num_ent, num_rel = self.kg.snapshots[self.args.snapshot+1].num_ent, self.kg.snapshots[self.args.snapshot+1].num_rel
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot+1].edge_index, self.kg.snapshots[self.args.snapshot+1].edge_type
        try:
            old_entity_weight = self.old_weight_entity_embeddings
            old_relation_weight = self.old_weight_relation_embeddings
            old_x = self.old_data_entity_embeddings
            old_r = self.old_data_relation_embeddings
        except:
            old_entity_weight, old_relation_weight = None, None
            old_x, old_r = None, None
        new_embeddings, rel_embeddings = self.gcn(self.ent_embeddings.weight, self.rel_embeddings.weight, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return new_embeddings, rel_embeddings
    def register_hooks(self):
        def get_activation(name):
            def hook(module, input, output):
                self.inputs[name] = input[0].detach()
            return hook
        self.ent_embeddings.register_forward_hook(get_activation('ent_embeddings_weight'))
        # print(self.inputs['ent_embeddings_weight'])
        self.rel_embeddings.register_forward_hook(get_activation('rel_embeddings_weight'))
        if hasattr(self.gcn, 'conv_layers'):
            for idx, conv_layer in enumerate(self.gcn.conv_layers[0].GCN_model.convs):
                conv_layer.register_forward_hook(get_activation(f'gcn_conv_layers_0_GCN_model_convs_{idx}_lin_weight'))

    def epoch_post_processing(self, size=None):
        '''
        Process for next training iteration
        '''
        ''' update the fisher matrix for parameters '''

        for n, p in self.named_parameters():
            # print(f'n,p:',n,p.size())
            n = n.replace('.', '_')
            if p.grad is not None:
                if self.fisher[n].size(0) != p.grad.data.size(0):
                    self.fisher[n] = torch.cat([self.fisher[n].to(p.grad.data.device), torch.zeros(p.grad.data.size(0)-self.fisher[n].size(0), self.fisher[n].size(1)).to(p.grad.data.device)], dim=0).to(p.grad.data.device)
                self.fisher[n] = self.fisher[n].to(p.grad.data.device)
                if p.data.dim() > 1:
                    mat_input_cov = torch.mm(self.inputs[n].t(), self.inputs[n]) / self.inputs[n].size(0)
                    L_output_conv = torch.mm(p.grad.t(), p.grad) / p.grad.size(0)
                    u1,s1,v1= compute_svd_rotation(mat_input_cov)
                    u2,s2,v2= compute_svd_rotation(L_output_conv)

                    Q1 = torch.mm(torch.inverse(torch.mm(u1.t(), u1)), u1.t())
                    Q2 = torch.mm(u2.t(), torch.inverse(torch.mm(u2, u2.t())))
                    rotated_matrix =torch.mm(torch.mm(p.grad.data,Q1), Q2)
                    self.fisher[n] += size * rotated_matrix.pow(2)
                else:
                    self.fisher[n] += size * p.grad.data.pow(2)
                # Apply scaling factor
                self.fisher[n] *= self.args.scaling_factor
                if self.args.max_fisher is not None:
                    self.fisher[n] = torch.clamp(self.fisher[n], max=self.args.max_fisher)
    def pre_snapshot(self):
        if self.args.snapshot >= 0:
            self.fisher_old = {}
            for n, p in self.named_parameters():
                n = n.replace('.','_')
                self.fisher_old[n] = self.fisher[n].clone()
                if 'embeddings' in n:
                    self.inputs[n] = p.data

    def get_new_weight(self):
        '''
        Calculate the regularization weights for entities and relations.
        :return: weights for entities and relations.
        '''
        ent_weight, rel_weight, other_weight = self.get_weight()
        weight = dict()
        for name, param in self.named_parameters():
            name_ = name.replace('.','_')
            if 'ent_embeddings' in name_:
                weight[name_] = ent_weight
            elif 'rel_embeddings' in name_:
                weight[name_] = rel_weight
            else:
                weight[name_] = other_weight
        return weight

    def new_loss(self, head, rel, tail=None, label=None):
        return self.margin_loss(head, rel, tail, label).mean()

    def lkge_regular_loss(self):
        '''
        Calculate regularization loss to avoid catastrophic forgetting.
        :return: regularization loss.
        '''
        if self.args.snapshot == 0:
            return torch.tensor(0.0).to(self.args.device)
        losses = []
        '''get samples number of entities and relations'''
        new_ent_weight, new_rel_weight, new_other_weight = self.new_weight_ent_embeddings_weight, self.new_weight_rel_embeddings_weight, self.new_weight_other_weight
        '''calculate regularization loss'''
        for name, param in self.named_parameters():
            name = name.replace('.', '_')
            if 'ent_embeddings' in name:
                new_weight = new_ent_weight
            elif 'rel_embeddings' in name:
                new_weight = new_rel_weight
            else:
                new_weight = new_other_weight
            new_data = param
            old_weight = getattr(self, 'old_weight_{}'.format(name))
            old_data = getattr(self, 'old_data_{}'.format(name))
            if type(new_weight) != int:
                new_weight = new_weight[:old_weight.size(0)]
                new_data = new_data[:old_data.size(0)]
            data_loss = (((new_data - old_data) * old_weight / (new_weight+old_weight+ 1e-8)) ** 2)
            fisher = self.fisher[name][:old_data.size(0)].to(self.args.device)
            # magnitude =torch.norm(fisher)
            # normalized_importance = fisher / (magnitude + 1e-8)
            loss = torch.sum(fisher*float(self.args.online_EWC_weight)*data_loss)
            # else:
            #     data_loss = ((new_data - old_data) ** 2)
            #     loss = torch.sum(float(self.args.online_EWC_weight)*data_loss)
           
            losses.append(loss)

        return sum(losses)


class TransE(GKGEL):
    def __init__(self, args, kg):
        super(TransE, self).__init__(args, kg)


    def MAE_loss(self):
        '''
        Calculate the MAE loss by masking and reconstructing embeddings.
        :return: MAE loss
        '''
        num_ent = self.kg.snapshots[self.args.snapshot].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot].num_rel
        '''get subgraph(edge indexs and relation types of all facts in the training facts)'''
        edge_index = self.kg.snapshots[self.args.snapshot].edge_index
        edge_type = self.kg.snapshots[self.args.snapshot].edge_type

        '''reconstruct'''
        ent_embeddings, rel_embeddings = self.embedding('Train')
        try:
            old_entity_weight = self.old_weight_entity_embeddings
            old_relation_weight = self.old_weight_relation_embeddings
            old_x = self.old_data_entity_embeddings
            old_r = self.old_data_relation_embeddings
        except:
            old_entity_weight, old_relation_weight = None, None
            old_x, old_r = None, None
        ent_embeddings_reconstruct, rel_embeddings_reconstruct = self.gcn(ent_embeddings, rel_embeddings, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return(self.mse_loss_func(ent_embeddings_reconstruct, ent_embeddings[:num_ent]) / num_ent + self.mse_loss_func(
            rel_embeddings_reconstruct, rel_embeddings[:num_rel]) / num_rel)

    def loss(self, head, rel, tail=None, label=None):
        '''
        :param head: subject entity
        :param rel: relation
        :param tail: object entity
        :param label: positive or negative facts
        :return: new facts loss + MAE loss + regularization loss
        '''
        loss = 0.0
        new_loss = self.new_loss(head, rel, tail, label)/head.size(0)
        loss += new_loss
        if self.args.using_reconstruct_loss:
            MAE_loss = self.MAE_loss()
            loss += float(self.args.reconstruct_weight)*MAE_loss
        if self.args.using_regular_loss:
            regular_loss = self.lkge_regular_loss()
            loss += float(self.args.regular_weight)*regular_loss
        return loss,new_loss,MAE_loss,regular_loss

    def get_weight(self):
        '''get the total number of samples containing each entity or relation'''
        num_ent = self.kg.snapshots[self.args.snapshot+1].num_ent
        num_rel = self.kg.snapshots[self.args.snapshot+1].num_rel
        ent_weight, rel_weight, other_weight = self.gcn.get_weight(num_ent, num_rel)
        return ent_weight, rel_weight, other_weight


class MAE(nn.Module):
    def __init__(self, args, kg):
        super(MAE, self).__init__()
        self.args = args
        self.kg = kg
        '''masked KG auto encoder'''
        self.conv_layers = nn.ModuleList()
        for i in range(args.num_layer):
            self.conv_layers.append(ConvLayer(args, kg))

    def forward(self, ent_embeddings, rel_embeddings, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r):
        '''
        Reconstruct embeddings for all entities and relations
        :param x: input entity embeddings
        :param r: input relation embeddings
        :param edge_index: (s, o)
        :param edge_type: (r)
        :param num_ent: entity number
        :param num_rel: relation number
        :return: reconstructed embeddings
        '''
        x, r = ent_embeddings, rel_embeddings
        for i in range(self.args.num_layer):
            x, r = self.conv_layers[i](x, r, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r)
        return x, r

    def get_weight(self, num_ent, num_rel):
        '''get the total number of samples containing each entity or relation'''
        edge_index, edge_type = self.kg.snapshots[self.args.snapshot+1].edge_index, self.kg.snapshots[self.args.snapshot+1].edge_type
        other_weight = edge_index.size(1)
        ent_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_index[0], dim_size=num_ent)
        rel_weight = scatter_add(src=torch.ones_like(edge_index[0]).unsqueeze(1), dim=0, index=edge_type, dim_size=num_rel)
        return ent_weight + 1, rel_weight + 1, other_weight

class ConvLayer(nn.Module):
    def __init__(self, args, kg):
        super(ConvLayer, self).__init__()
        self.args = args
        self.kg = kg
        self.GCN_model = GCN(in_channels=self.args.emb_dim, hidden_channels=200, \
                            out_channels=self.args.emb_dim, num_layers=3)
    def forward(self, x, r, edge_index, edge_type, num_ent, num_rel, old_entity_weight, old_relation_weight, old_x, old_r):
        '''
        Reconstruct embeddings for all entities and relations
        :param x: input entity embeddings
        :param r: input relation embeddings
        :param edge_index: (s, o)
        :param edge_type: (r)
        :param num_ent: entity number
        :param num_rel: relation number
        :return: reconstructed embeddings
        '''
        '''avoid the reliance for learned facts'''
        if old_entity_weight == None:  # for embedding transfer
            edge_index, edge_type = self.add_loop_edge(edge_index, edge_type, num_ent, num_rel)
            r = torch.cat([r, torch.zeros(1, r.size(1)).to(self.args.device)], dim=0)
            neigh_t = torch.index_select(x, 0, edge_index[1])
            neigh_r = torch.index_select(r, 0, edge_type)
            neigh_h = torch.index_select(x, 0, edge_index[0])
            ent_embed = scatter_mean(src=neigh_h + neigh_r, dim=0, index=edge_index[1], dim_size=num_ent)
            rel_embed = scatter_mean(src=neigh_t - neigh_h, dim=0, index=edge_type, dim_size=num_rel + 1)
            ent_embed = self.GCN_model(ent_embed.float(), edge_index,edge_attr = edge_type,batch_size =self.args.batch_size)

            ent_embed = torch.relu(ent_embed.double())
            # print(ent_embed.dtype,rel_embed.dtype)
            return ent_embed, rel_embed[:-1]
        else:
            print(f'MAE is working')
            '''prepare old parameter and the number of |N(x)|'''
            if x.size(0) > old_entity_weight.size(0):
                old_entity_weight = torch.cat((old_entity_weight, torch.zeros(x.size(0)-old_entity_weight.size(0))), dim=0)
                old_x = torch.cat((old_x, torch.zeros(x.size(0)-old_entity_weight.size(0), x.size(1))), dim=0)
            if r.size(0) > old_relation_weight.size(0):
                old_relation_weight = torch.cat((old_relation_weight, torch.zeros(x.size(0) - old_relation_weight.size(0))),dim=0)
                old_r = torch.cat((old_r, torch.zeros(r.size(0) - old_relation_weight.size(0), r.size(1))), dim=0)

            '''add self-loop edges'''
            edge_index, edge_type = self.add_loop_edge(edge_index, edge_type, num_ent, num_rel)
            r = torch.cat([r, torch.zeros(1, r.size(1)).to(self.args.device)], dim=0)

            '''get neighbor embeddings'''
            neigh_t = torch.index_select(x, 0, edge_index[1])
            neigh_r = torch.index_select(r, 0, edge_type)
            neigh_h = torch.index_select(x, 0, edge_index[0])

            '''calculate entity embeddings'''
            ent_embed_new = scatter_add(src=neigh_h + neigh_r, dim=0, index=edge_index[1], dim_size=num_ent)
            ent_embed_old = old_entity_weight.unsqueeze(1) * old_x
            ent_embed = ent_embed_old + ent_embed_new
            ent_involving_num = old_entity_weight + scatter_add(src=torch.ones(edge_index.size(1)), index=edge_index[1], dim_size = num_ent)
            ent_embed = ent_embed/ent_involving_num
            ent_embed = self.GCN_model(ent_embed.float(), edge_index,edge_attr = edge_type,batch_size =self.args.batch_size)
            ent_embed = torch.relu(ent_embed)

            '''calculate relation embeddings'''
            rel_embed_new = scatter_add(src=neigh_t + neigh_h, dim=0, index=edge_index[1], dim_size=num_rel)
            rel_embed_old = old_relation_weight.unsqueeze(1) * old_r
            rel_embed = rel_embed_old + rel_embed_new
            rel_involving_num = old_relation_weight + scatter_add(src=torch.ones(edge_type.size(0)), index=edge_type,
                                                                dim_size=num_rel)
            rel_embed = rel_embed / rel_involving_num

            return ent_embed, rel_embed[:-1]

    def add_loop_edge(self, edge_index, edge_type, num_ent, num_rel):
        '''add self-loop edge for entities'''
        u, v = torch.arange(0, num_ent).unsqueeze(0).to(self.args.device), torch.arange(0, num_ent).unsqueeze(0).to(self.args.device)
        r = torch.zeros(num_ent).to(self.args.device).long()
        loop_edge = torch.cat([u, v], dim=0)
        edge_index = torch.cat([edge_index, loop_edge], dim=-1)
        edge_type = torch.cat([edge_type, r+num_rel], dim=-1)
        return edge_index, edge_type





