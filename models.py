import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from torch.cuda.amp import autocast
from einops.layers.torch import Rearrange
from einops import rearrange
from linear_attention_transformer import LinearAttentionTransformer

mask_value = -1e9

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, face_encoder, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        if face_encoder is not None: self.face_encoder = face_encoder
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src=None, tgt=None, src_mask=None, tgt_mask=None, return_encoder_out=False, 
                                no_grad_face=False, just_extra=False):
        "Take in and process masked src and target sequences."
        # print(src.size())
        # print(tgt.size())
        # print(src_mask.size())
        # print(tgt_mask.size())
        # input()
        encoder_out, src_mask = self.encode(src, src_mask, no_grad_face)
        
        if just_extra:
            return self.extra_generator(encoder_out)

        decoder_out = self.decode(encoder_out, src_mask,
                            tgt, tgt_mask)

        if return_encoder_out:
            return decoder_out, encoder_out

        return decoder_out

    def encode(self, src, src_mask,  no_grad_face=False):
        if hasattr(self, 'face_encoder'):
            if no_grad_face:
                with torch.no_grad():
                    src = self.face_encoder(src)
            else:
                src = self.face_encoder(src)

        if isinstance(src, tuple):
            src, src_mask = src

            if len(src_mask.size()) == 4:
                if self.src_embed is not None: src_embeddings = self.src_embed(src)
                else: src_embeddings = src
                B, T, num_parts, C = src_embeddings.size()
                src_embeddings = src_embeddings.reshape(B, T * num_parts, C)
                src_mask = src_mask.reshape(B, 1, T * num_parts)
            else:
                if self.src_embed is not None: src_embeddings = self.src_embed(src)
                else: src_embeddings = src

        else:
            if self.src_embed is not None: src_embeddings = self.src_embed(src)
            else: src_embeddings = src
        #import ipdb;ipdb.set_trace()
        encoder_out = self.encoder(src_embeddings, src_mask)
        # print(encoder_out.size(), encoder_out.max(), encoder_out.min(), encoder_out.mean())
        return encoder_out, src_mask

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # print('tgt', tgt.size())
        # print(tgt.reshape(-1).max())
        # input()
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, x, reshape=True):
        scores = self.proj(x)
        if reshape: scores = scores.view(-1, scores.size(-1))
        return scores

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N, face_encoder=None, pos=None, in_proj=None,
                final_norm=True):
        super(Encoder, self).__init__()
        if face_encoder is not None: self.face_encoder = face_encoder
        if in_proj is not None:
            self.in_proj = in_proj
        if pos is not None: self.pos = pos
        self.layers = clones(layer, N)

        if final_norm: self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        if mask is None:
            if len(x.size()) > 3:
                mask = x.sum((1, 3, 4)).bool().unsqueeze(1)
            else:
                mask = x.sum(1, keepdim=True)

        if hasattr(self, 'face_encoder'):
            x = self.face_encoder(x)

        if x.size(1) == self.layers[0].size: x = x.transpose(1, 2)

        if hasattr(self, 'in_proj'):
            if x.size(1) == 512: x = x.transpose(1, 2)
            x = self.in_proj(x)

        if hasattr(self, 'pos'):
            x = self.pos(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = (self.norm(x) if hasattr(self, 'norm') else x)

        return x

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    with autocast(enabled=False):
        if mask is not None:
            scores = scores.float()
            # print(scores)
            scores = scores.masked_fill(mask == 0, mask_value)
            # input(scores)
        # print(scores.min(), scores.max())
        # input()
        p_attn = F.softmax(scores, dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        if self_attn is not None: self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        if hasattr(self, 'self_attn'):
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, gelu=False):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        if gelu:
            self.act = F.gelu
        else:
            self.act = F.relu

    def forward(self, x):
        return self.w_2(self.dropout(self.act(self.w_1(x))))

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


def build_model(visual_dim=512, num_layers=12, d_model=768, dropout=0.1, multilingual=True, 
        vocab=None, **kwargs):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy

    d_ff = 4*d_model
    h = d_model//64

    attn = MultiHeadedAttention(h, d_model, dropout=dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    if vocab is None:
        vocab = 51864 + int(multilingual)

    model = EncoderDecoder(
        None,
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), num_layers),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), num_layers),

        nn.Sequential(nn.Linear(visual_dim, d_model), c(position)) \
                    if visual_dim != d_model else nn.Sequential(c(position)),
        nn.Sequential(Embeddings(d_model, vocab), c(position)),
        Generator(d_model, vocab))
    
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#### visual parts

class Conv3d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, bias=True, residual=False, 
                    *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv3d(cin, cout, kernel_size, stride, padding, bias=bias),
                            nn.BatchNorm3d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class CNN_3d(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.encoder = nn.Sequential(
            Conv3d(3, 64, kernel_size=5, stride=(1, 2, 2), padding=2),  # 48, 48

            Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 24, 24
            Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 12, 12
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 6, 6
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),

            Conv3d(512, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 3, 3
            Conv3d(512, d_model, kernel_size=(1, 3, 3), stride=1, padding=(0, 0, 0)),)

    @autocast()
    def forward(self, faces, mask):
        assert faces.size(3) == 96
        assert faces.size(4) == 96
        face_embeddings = self.encoder(faces) # (B, C, T, 1, 1)
            
        return face_embeddings.squeeze(3).squeeze(3).transpose(1, 2) # (B, T, C)

### VTP modules:

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats):
        super().__init__()
        self.row_embed = nn.Embedding(64, num_pos_feats)
        self.col_embed = nn.Embedding(64, num_pos_feats)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return x + pos

class CNN_3d_featextractor(nn.Module):
    def __init__(self, d_model, till):
        super().__init__()
        layers = [Conv3d(3, 64, kernel_size=5, stride=(1, 2, 2), padding=2)]  # 48, 48

        if till <= 24:
            layers.extend([Conv3d(64, 128, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 24, 24
                Conv3d(128, 128, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),])
        
        if till <= 12:
            layers.extend([Conv3d(128, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 12, 12
                    Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
                    Conv3d(256, 256, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),])

        if till == 6:
            layers.extend([Conv3d(256, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)), # 6, 6
                        Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),
                        Conv3d(512, 512, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), residual=True),])

        self.encoder = nn.Sequential(*layers)

    @autocast()
    def forward(self, faces):
        assert faces.size(3) == 96
        assert faces.size(-1) == 96

        face_embeddings = self.encoder(faces) # (B, C, T, H, W)

        return face_embeddings

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                            act_layer=nn.ReLU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VTP_wrapper(nn.Module):
    def __init__(self, feat_extrator, encoder, in_dim, out_dim):
        super().__init__()
        self.feat_extrator = feat_extrator

        self.hwposition = PositionEmbeddingLearned(in_dim//2)

        self.pooler = nn.Linear(out_dim, 1)

        self.encoder = encoder

    @autocast()
    def forward(self, faces, mask=None):
        faces = self.feat_extrator(faces)
        faces = faces.transpose(1, 2) # (B, T, C, H, W)
        if mask is not None:
            lens = mask.long().sum((1, 2)) # (B,)
        else:
            lens = torch.ones(faces.size(0)).long() * faces.size(1)
        indiv_faces = []
        for l, fs in zip(lens, faces):
            indiv_faces.append(fs[:l])

        face_tokens = torch.cat(indiv_faces, dim=0) # (B*, C, H, W)

        face_tokens = self.hwposition(face_tokens)

        face_embeddings = self.encoder(face_tokens) # (B, hw, C)

        pooling_weights = F.softmax(self.pooler(face_embeddings), dim=1) # (B, hw, 1)
        self.face_attn_weights = pooling_weights

        face_embeddings = (face_embeddings * pooling_weights).sum(1) # (B, C)

        video_embeddings = []
        max_len = faces.size(1)
        start = 0

        for l in lens:
            cur_face_embeddings = face_embeddings[start : start + l]
            if l != max_len:
                cur_face_embeddings = torch.cat([cur_face_embeddings, torch.zeros((max_len - l, 
                                    cur_face_embeddings.size(1)), 
                                    device=cur_face_embeddings.device)], dim=0)
            start += l
            video_embeddings.append(cur_face_embeddings)
            
        video_embeddings = torch.stack(video_embeddings, dim=0)
        return video_embeddings # (B, T, C)

class VTP(nn.Module):   
    def __init__(self, num_layers, dims, heads, patch_sizes, initial_resolution=48, 
                initial_dim=64, trans_block=LinearAttentionTransformer):
        '''
            Num layers per block, dim and #heads for the layers of each block, 
                patch sizes for downsampling
        ''' 
        super().__init__()

        self.transformer_blocks = nn.ModuleList([])
        self.patch_projectors = nn.ModuleList([]) # converts spatial maps to patches

        cur_dim = initial_dim
        cur_res = initial_resolution
        for l, dim, h, p in zip(num_layers, dims, heads, patch_sizes):
            input_dim = (cur_dim * p * p)
            if input_dim == dim:
                self.patch_projectors.append(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                        p1 = p, p2 = p))
            else:
                self.patch_projectors.append(nn.Sequential(
                                                 Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', 
                                                    p1 = p, p2 = p),
                                                 Mlp(input_dim, dim, dim)))
            cur_res /= p
            self.transformer_blocks.append(
                                    trans_block(dim=dim, 
                                    max_seq_len=(cur_res * cur_res),
                                    heads=h, depth=l, ff_dropout = 0.1, 
                                    attn_layer_dropout = 0.1, attn_dropout = 0.1))
            cur_dim = dim

    @autocast()
    def forward(self, face_tokens, cls_token=None):
        x = face_tokens
        for i, (patch_maker, transformer) in enumerate(zip(self.patch_projectors, 
                                                            self.transformer_blocks)):
            x = patch_maker(x)
            if i == len(self.patch_projectors) - 1 and cls_token is not None:
                x = torch.cat([cls_token, x], dim=1)

            x = transformer(x)

            if i != len(self.patch_projectors) - 1:
                r = int(x.size(1) ** 0.5)
                feature_map_projector = Rearrange('b (h w) c -> b c h w', 
                                                    h=r, w=r)
                x = feature_map_projector(x)

        return x

def build_visual_encoder(d_model=512):
    linear_pooler = VTP(num_layers=[3, 3], dims=[256, 512], heads=[8, 8], 
                                patch_sizes=[1, 2], initial_resolution=24, initial_dim=128)

    model = VTP_wrapper(CNN_3d_featextractor(d_model, till=24), linear_pooler, in_dim=128, 
                                out_dim=512)

    return model

if __name__ == "__main__":
    model = build_visual_encoder()
    print(model)