import torch
import torch.nn as nn
import torch.nn.functional as F

# [B, 720, 96] ,[B, 720, 96],[B, 720, 96], [B, 720, 96] -->[B, 720, 96]
# signal_temporal_concat_feature,signal_spatial_concat_feature,depth_temporal_feature,depth_spatial_feature
class CrossAttentionLayer(nn.Module):
    """交叉注意力层"""

    def __init__(self, dim_x, dim_y, dim, num_heads=8, dropout=0.1, use_pre_norm=True):
        super().__init__()
        self.dim_x = dim_x  # x的特征维度(最后一维)
        self.dim_y = dim_y  # y的特征维度(最后一维)
        self.dim = dim  # 注意力计算的中间维度
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"

        # 可选的预归一化
        self.use_pre_norm = use_pre_norm
        if use_pre_norm:
            self.norm_q = nn.LayerNorm(dim_x)
            self.norm_k = nn.LayerNorm(dim_y)
            self.norm_v = nn.LayerNorm(dim_y)
        else:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity() 
            self.norm_v = nn.Identity()

        # 线性投影:Q来自x,K/V来自y
        self.q_proj = nn.Linear(dim_x, dim)  # x→Q:[B,T1,dim_x]→[B,T1,dim]
        self.k_proj = nn.Linear(dim_y, dim)  # y→K:[B,T2,dim_y]→[B,T2,dim]
        self.v_proj = nn.Linear(dim_y, dim)  # y→V:[B,T2,dim_y]→[B,T2,dim]

        # 输出投影:将注意力结果投影回x的特征维度
        self.out_proj = nn.Linear(dim, dim_x)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # 注意力缩放因子

    def forward(self, x, y, mask=None):
        """
        输入:
            x: [B, T1, dim_x]  # 查询端
            y: [B, T2, dim_y]  # 键值端
            mask: [B, T1, T2]  # 可选掩码(需匹配[T1,T2]维度,避免广播错误)
        输出:
            output: [B, T1, dim_x]  # 输出序列长度=查询端长度T1
        """
        # 检查批量大小一致性
        B, T1, _ = x.shape
        B_y, T2, _ = y.shape
        assert B == B_y, f"x和y的批量大小必须一致(x: {B}, y: {B_y})"

        # 生成Q、K、V并拆分多头(维度适配T1和T2)
        # Q: [B, T1, dim] → [B, T1, H, hd] → [B, H, T1, hd]
        q = self.q_proj(self.norm_q(x)).reshape(B, T1, self.num_heads, self.head_dim).transpose(1, 2)
        # K: [B, T2, dim] → [B, T2, H, hd] → [B, H, T2, hd]
        k = self.k_proj(self.norm_k(y)).reshape(B, T2, self.num_heads, self.head_dim).transpose(1, 2)
        # V: [B, T2, dim] → [B, T2, H, hd] → [B, H, T2, hd]
        v = self.v_proj(self.norm_v(y)).reshape(B, T2, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数(维度:[B, H, T1, T2])
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # k.transpose(-2,-1)→[B,H,hd,T2]

        # 应用掩码(若有):注意掩码需为[T1,T2]维度
        if mask is not None:
            assert mask.shape == (B, T1, T2), f"掩码维度需为[B,T1,T2](当前:{mask.shape})"
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重并应用到V(输出长度=T1)
        attn_weights = F.softmax(attn_scores, dim=-1)  # 在T2维度softmax(键端长度)
        attn_weights = self.dropout(attn_weights)
        output = attn_weights @ v  # [B,H,T1,T2] @ [B,H,T2,hd] → [B,H,T1,hd]

        # 重塑输出(合并多头,投影回dim_x)
        output = output.transpose(1, 2).contiguous()  # [B, T1, H, hd]
        output = output.reshape(B, T1, self.dim)  # [B, T1, dim]
        output = self.out_proj(output)  # [B, T1, dim_x]

        return output

# [B, 720, 96] ,[B, 720, 96],[B, 720, 96], [B, 720, 96] -->[B, 720, 96]
# signal_temporal_concat_feature,signal_spatial_concat_feature,depth_temporal_feature,depth_spatial_feature
# [B, 720, 96] ,[B, 96, 720],[B, 720, 96], [B, 96, 720] -->[B, 720, 96]
class MultimodalFusion(nn.Module):
    def __init__(self, dim=96, num_heads=8, dropout=0.1):
        super().__init__()

        self.gate_norm_1 = nn.LayerNorm(normalized_shape=(720, 96))  
        self.gate_norm_2 = nn.LayerNorm(normalized_shape=(720, 96))  
        self.gate_norm_3 = nn.LayerNorm(normalized_shape=(720, 96))  
        self.gate_norm_4 = nn.LayerNorm(normalized_shape=(720, 96))  
        # 定义四个交叉注意力层,分别用于四种融合
        self.signal_temporal_spatial_attn = CrossAttentionLayer(96, 720, dim, num_heads, dropout, use_pre_norm=True)
        self.image_temporal_spatial_attn = CrossAttentionLayer(96, 720, dim, num_heads, dropout, use_pre_norm=True)
        self.cross_modal_temporal_attn = CrossAttentionLayer(96, 96, dim, num_heads, dropout, use_pre_norm=True)
        self.cross_modal_spatial_attn = CrossAttentionLayer(720, 720, dim, num_heads, dropout, use_pre_norm=True)

        # 反向注意力
        self.signal_spatial_temporal_attn = CrossAttentionLayer(720, 96, dim, num_heads, dropout, use_pre_norm=True)
        self.image_spatial_temporal_attn = CrossAttentionLayer(720, 96, dim, num_heads, dropout, use_pre_norm=True)
        self.cross_modal_temporal_attn_rev = CrossAttentionLayer(96, 96, dim, num_heads, dropout, use_pre_norm=True)
        self.cross_modal_spatial_attn_rev = CrossAttentionLayer(720, 720, dim, num_heads, dropout, use_pre_norm=True)

        self.gate_1 = nn.Sequential(
            nn.Linear(dim ,dim),  
            nn.Sigmoid()  # 权重归一化到[0,1]
        )
        self.gate_2 = nn.Sequential(
            nn.Linear(dim ,dim),  
            nn.Sigmoid()  # 权重归一化到[0,1]
        )
        self.gate_3 = nn.Sequential(
            nn.Linear(dim ,dim),  
            nn.Sigmoid()  # 权重归一化到[0,1]
        )
        self.gate_4 = nn.Sequential(
            nn.Linear(dim ,dim),  
            nn.Sigmoid()  # 权重归一化到[0,1]
        )

# [B, 720, 96] ,[B, 720, 96],[B, 720, 96], [B, 720, 96] -->[B, 720, 96]
# signal_temporal_concat_feature,signal_spatial_concat_feature,depth_temporal_feature,depth_spatial_feature
    def forward(self, signal_temporal, signal_spatial, image_temporal, image_spatial):
        # 空间特征维度调整（适配注意力层的B,L,C）
        signal_spatial = signal_spatial.permute(0, 2, 1)  #[B, 96, 720]
        image_spatial = image_spatial.permute(0, 2, 1)  #[B, 96, 720]
        # [B, 720, 96] ,[B, 96, 720],[B, 720, 96], [B, 96, 720] -->[B, 720, 96]

        # 1. 模态内时空交叉注意力融合
        signal_s2t = self.signal_spatial_temporal_attn(signal_spatial, signal_temporal)  #[B, 96, 720]
        signal_t2s = self.signal_temporal_spatial_attn(signal_temporal, signal_spatial)  #[B, 720, 96]
        signal_temporal_spatial = signal_spatial.permute(0, 2, 1) + signal_t2s + signal_s2t.permute(0, 2, 1)  # [B, 720, 96]

        image_t2s = self.image_temporal_spatial_attn(image_temporal, image_spatial)  # [B, 720, 96]
        image_s2t = self.image_spatial_temporal_attn(image_spatial, image_temporal)  #[B, 96, 720]
        image_temporal_spatial = image_temporal + image_t2s + image_s2t.permute(0, 2, 1)  # [B, 720, 96]

        # 2. 跨模态交叉注意力融合
        sig2img_temp = self.cross_modal_temporal_attn(signal_temporal, image_temporal)  # [B, 720, 96]
        img2sig_temp = self.cross_modal_temporal_attn_rev(image_temporal, signal_temporal)  # [B, 720, 96]
        cross_temporal = signal_temporal + sig2img_temp + img2sig_temp  # [B, 720, 96]

        img2sig_spat = self.cross_modal_spatial_attn(image_spatial, signal_spatial)  #[B, 96, 720]
        sig2img_spat = self.cross_modal_spatial_attn_rev(signal_spatial, image_spatial)  #[B, 96, 720]
        cross_spatial = image_spatial.permute(0, 2, 1) + sig2img_spat.permute(0, 2, 1) + img2sig_spat.permute(0, 2, 1)  # [B, 720, 96]

        gate_input_1 = self.gate_norm_1(signal_temporal_spatial)
        gate_input_2 = self.gate_norm_2(image_temporal_spatial)
        gate_input_3 = self.gate_norm_3(cross_temporal)
        gate_input_4 = self.gate_norm_4(cross_spatial)

        w1 = self.gate_1(gate_input_1)  
        w2 = self.gate_2(gate_input_2)
        w3 = self.gate_3(gate_input_3)
        w4 = self.gate_4(gate_input_4)

        fused = (
            w1 * signal_temporal_spatial +  # 特征1 × 权重1
            w2 * image_temporal_spatial +   # 特征2 × 权重2
            w3 * cross_temporal +           # 特征3 × 权重3
            w4 * cross_spatial              # 特征4 × 权重4
        )  # [B, 720, 96]

        return fused