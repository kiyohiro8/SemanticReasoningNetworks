
import numpy as np
import torch
import torch.nn as nn



class SRNModel(nn.Module):
    def __init__(self, params):
        super(SRNModel, self).__init__()
        resnet_name = params["resnet_name"]
        self.feature_extraction = ResNetFPN(resnet_name)
        self.seq_modeling = EncodeTransformer()
        self.decoder = Decoder

    def forward(self, image, is_train=True):
        cnn_feature = self.feature_extraction(image)
        b, c, h, w = cnn_feature.shape

        cnn_feature = cnn_feature.permute(0, 1, 3, 2)
        cnn_feature = cnn_feature.contiguous().view(b, c, -1)
        cnn_feature = cnn_feature.permute(0, 2, 1) # batch, seq, feature

        contextual_feature = self.seq_modeling(cnn_feature)[0]

        prediction = self.decoder(contextual_feature)

        return prediction


class Decoder(nn.Module):
    def __init__(self, n_dim, n_class, max_character=25, n_position=256, n_GSRM_layer=4):
        self.pvam = PVAModule(max_character=max_character, n_position=n_position)
        self.w_e = nn.Linear(n_dim, n_class)

        self.gsrm = GSRModule(
            n_class=n_class,
            pad=n_class-1, 
            n_dim=n_dim,
            n_position=max_character, 
            n_layers=n_GSRM_layer
            )
        self.w_s = nn.Linear(n_dim, n_class)
        self.w_f = nn.Linear(n_dim, n_class)

    def forward(self, cnn_feature):
        pvam_out = self.w_e(self.pvam(cnn_feature))

        s = self.gsrm(pvam_out)[0]
        gsrm_out = self.w_s(s)

        f = pvam_out + s
        f_out = self.w_f(f)

        return pvam_out, gsrm_out, f_out


class PVAModule(nn.Module):
    """
    Parallel Visual Attention
    """
    def __init__(self, n_dim, max_character, n_position):
        super(PVAModule, self).__init__()
        self.char_length = max_character
        self.embed = nn.Embedding(max_character, n_dim)

        self.w0 = nn.Linear(max_character, n_position)
        self.wv = nn.Linear(n_dim, n_dim)
        self.we = nn.Linear(n_dim, max_character)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        reading_order = torch.arange(self.char_length, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)
        reading_order_embed = self.embed(reading_order)

        t = self.w0(reading_order_embed.permute(0, 2, 1))
        t = self.active(t.permute(0, 2, 1) + self.mv(enc_output))
        attention = self.softmax(self.we(t).permute(0, 2, 1))

        output = torch.bmm(attention, enc_output)

        return output


class GSRModule(nn.Module):
    def __init__(self, n_dim, n_class, pad, n_layer, n_position):
        super(GSRModule, self).__init__()

        self.pad = pad
        self.embed = nn.Embedding(n_class, n_dim)
        self.transformer = EncodeTransformer(n_layers=n_layer, n_position=n_position)

    def forward(self, enc_output):
        enc_argmax = enc_output.argmax(dim=-1)
        e = self.embed(enc_argmax)

        s = self.transformer(e, None)

        return s
     



class EncodeTransformer(nn.Module):
    def __init__(
        self, 
        d_word_vec=512, 
        n_layers=2, 
        n_head=8, 
        d_k=64, 
        d_v=64,
        d_model=512, 
        d_inner=1024, 
        dropout=0.1, 
        n_position=256
        ):
        super(EncodeTransformer, self).__init__()
        self.position_enc = PositionalEncoder(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layer_stack.append(EncodeLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout))
        self.layer_norm = nn.LayerNorm(d_model, epis=1e-6)

    def forward(self, cnn_feature, src_mask, return_attention=False):
        enc_attention_list = []

        enc_output = self.dropout(self.position_enc(cnn_feature))

        for enc_layer in self.layers:
            enc_output, enc_attention = enc_layer(enc_output, mask=src_mask)
            if return_attention:
                enc_attention_list.extend(enc_attention)
        enc_output = self.layer_norm(enc_output)

        if return_attention:
            return enc_output, enc_attention_list
        else:
            return enc_output


class EncodeLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncodeLayer, self).__init__()
        self.self_attension = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_feedforward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, mask=None):
        enc_output, enc_attention = self.self_attension(enc_input, enc_input, enc_input, mask=mask)
        enc_output = self.pos_feedforward(enc_output)
        return enc_output, enc_attention


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)     # 4*21*512 ---- 4*21*8*64
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1) if mask is not None else None # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn        


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(self.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attention_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class PositionalEncoder(nn.Module):
    def __init__(self, d_hid, n_position):
        super(PositionalEncoder, self).__init__()

        self.resister_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))
    
    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        sinusoid_table = np.array([self._get_position_angle_vac(pos, d_hid) for pos in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def _get_position_angle_vac(self, pos, d_hid):
        return [pos / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class ResNetFPN(nn.Module):
    resnet_blocks_dict = {
        "resnet50": {
            "num": [3, 4, 6, 3],
            "block": BottleneckBlock
        }
    }
    resnet_name_list = ["resnet50"]
    def __init__(self, resnet_name="resnet50"):
        assert resnet_name in ResNetFPN.resnet_name_list
        base_features = 64
        self.conv1 = nn.Conv2d(3, base_features, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_features)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        block_name = ResNetFPN.resnet_blocks_dict[resnet_name]["block"]
        nums_layers = ResNetFPN.resnet_blocks_dict[resnet_name]["num"]

        self.layer1 = ResLayer(
            block_name, 
            base_features, 
            base_features * 2, 
            nums_layers[0], 
            stride=1
            )
        self.layer2 = ResLayer(
            block_name, 
            base_features * 2,
            base_features * 4, 
            nums_layers[1], 
            stride=2
            )
        self.layer3 = ResLayer(
            block_name, 
            base_features * 4, 
            base_features * 8, 
            nums_layers[2], 
            stride=2
        )
        self.layer4 = ResLayer(
            block_name, 
            base_features * 8, 
            base_features * 16, 
            nums_layers[3], 
            stride=2
        )

        self.fpn = FeaturePyramidNetwork()
        self._initialize_weight()

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        visual_feature = self.fpn(x2, x3, x4)
        return visual_feature

    def _initialize_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class FeaturePyramidNetwork(nn.Module):
    def __init__(self, C3_feature, C4_feature, C5_feature, feature_size=256):
        super(FeaturePyramidNetwork, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_feature, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_feature, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_feature, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

    def forward(self, C3, C4, C5):
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return P3_x


class ResLayer(nn.Module):
    def __init__(self, block, in_features, out_features, num_blocks, stride=1):
        super(ResLayer, self).__init__()
        downsample = stride != 1 or in_features != out_features * block.expansion
        layers = [block(in_features, out_features, stride, downsample)]
        for _ in range(1, num_blocks):
            layers.append(block(out_features, out_features))
        
        return nn.Sequential(*layers)


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_features, out_features, stride=1, downsample=False):
        super(BottleneckBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_features)
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_features)
        self.conv3 = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample:
            self.downsample_layer = nn.Sequential(
                nn.Conv2d(in_features, out_features,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_features),
            )
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample:
            residual = self.downsample_layer(x)

        out += residual
        out = self.relu(out)

        return out


        