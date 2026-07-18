import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from easydict import EasyDict as edict
import copy
import os  # 新增：用于处理目录创建

def conv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=2, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

def linear_block(in_c, out_c):
    return nn.Sequential(
        nn.Linear(in_c, out_c),
        nn.BatchNorm1d(out_c),
        nn.ReLU(),
    )

def pixel_bias(outViewN, outW, outH, renderDepth):
    initTile = torch.cat([
        torch.ones([outViewN, outH, outW]).float() * renderDepth, 
        torch.zeros([outViewN, outH, outW]).float(),
    ], dim=0) 

    return initTile.unsqueeze_(dim=0) 

# 修改：ConvModule现在返回各个分支的卷积输出，用于保存
class ConvModule(nn.Module):
    def __init__(self, options=None, in_ch=9, out_ch=32):
        super(ConvModule, self).__init__()
        self.options = options
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.actvn = nn.ReLU()

        self.conv3 = nn.Conv1d(self.in_ch, self.out_ch, 3, padding=1)
        self.conv5 = nn.Conv1d(self.in_ch, self.out_ch, 5, padding=2)
        self.conv7 = nn.Conv1d(self.in_ch, self.out_ch, 7, padding=3)
        self.conv9 = nn.Conv1d(self.in_ch, self.out_ch, 9, padding=4)

    def forward(self, sig):
        # 分别计算并保存每个卷积核的输出
        conv3_out = self.conv3(sig)
        conv5_out = self.conv5(sig)
        conv7_out = self.conv7(sig)
        conv9_out = self.conv9(sig)
        
        out = self.actvn(conv3_out + conv5_out + conv7_out + conv9_out)
        
        # 修改：返回中间结果供保存
        return conv3_out, conv5_out, conv7_out, conv9_out, out
  
class PoolModule(nn.Module):
    def __init__(self, options=None, poolFunc='Max', kernel_size=2, pad=0):
        super(PoolModule, self).__init__()
        self.options = options
        if poolFunc=='Max':
            self.pool = nn.MaxPool1d(kernel_size=kernel_size, padding=pad)
        else:
            self.pool = nn.AvgPool1d(kernel_size=kernel_size, padding=pad)

    def forward(self, x):
        out = self.pool(x)
        return out
  
class LinearModule(nn.Module):
    def __init__(self, in_ch=640, out_ch=320):
        super(LinearModule, self).__init__()
        self.LinearM = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.LinearM(x)
        return out

# 修改：ComplexEncoder现在收集并返回所有需要的中间特征
class ComplexEncoder(nn.Module):
    def __init__(self, options=None):
        self.options = options
        super(ComplexEncoder, self).__init__()
        self.actvn = nn.ReLU()

        # Conv module
        self.conv1Real = ConvModule(in_ch=9, out_ch=32)
        self.conv1Imag = ConvModule(in_ch=9, out_ch=32)

        self.conv2 = ConvModule(in_ch=32, out_ch=64)
        self.conv3 = ConvModule(in_ch=64, out_ch=128)
        self.conv4 = ConvModule(in_ch=128, out_ch=256)
        self.conv5 = ConvModule(in_ch=256, out_ch=512)

        # Pooling
        self.pool1 = PoolModule(poolFunc='Max', kernel_size=4)
        self.pool2 = PoolModule(poolFunc='Max', kernel_size=4)
        self.pool3 = PoolModule(poolFunc='Max', kernel_size=4, pad=1)
        self.pool4 = PoolModule(poolFunc='Max', kernel_size=2)
        self.pool5 = PoolModule(poolFunc='Max', kernel_size=2)

        # FC
        self.fc_1 = LinearModule(5632, 1024)
        self.fc_2 = LinearModule(1024, 512)
        self.fc_3 = nn.Linear(512, 128)

    def forward(self, real, imag):
        # 用于存储该Encoder所有需要保存的中间特征
        saved_tensors = {}

        # --- Layer 1 ---
        # 获取实部和虚部分支的中间卷积输出
        r3, r5, r7, r9, out_1_real = self.conv1Real(real)
        i3, i5, i7, i9, out_1_imag = self.conv1Imag(imag)
        
        # 保存：conv1Real 和 conv1Imag 的多尺度特征
        saved_tensors['conv1Real_branches'] = {'conv3': r3, 'conv5': r5, 'conv7': r7, 'conv9': r9}
        saved_tensors['conv1Imag_branches'] = {'conv3': i3, 'conv5': i5, 'conv7': i7, 'conv9': i9}

        out_1 = out_1_real + out_1_imag
        out_1 = self.pool1(out_1)
        saved_tensors['out_1'] = out_1 # 保存：池化后的 out_1

        # --- Layer 2 ---
        _, _, _, _, out_2_conv = self.conv2(out_1)
        out_2 = self.pool2(out_2_conv)
        saved_tensors['out_2'] = out_2 # 保存：池化后的 out_2

        # --- Layer 3 ---
        _, _, _, _, out_3_conv = self.conv3(out_2)
        out_3 = self.pool3(out_3_conv)
        saved_tensors['out_3'] = out_3 # 保存：池化后的 out_3

        # --- Layer 4 ---
        _, _, _, _, out_4_conv = self.conv4(out_3)
        out_4 = self.pool4(out_4_conv)
        saved_tensors['out_4'] = out_4 # 保存：池化后的 out_4

        # --- Layer 5 ---
        _, _, _, _, out_5_conv = self.conv5(out_4)
        out_5 = self.pool5(out_5_conv)
        saved_tensors['out_5'] = out_5 # 保存：池化后的 out_5
        
        # FC Layers
        out = out_5.view([-1, 5632])
        out = self.fc_1(out)
        out = self.fc_2(out)
        out = self.fc_3(out)

        # 修改：返回最终输出和中间特征字典
        return out, saved_tensors

def deconv2d_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )

# 修改：Decoder 现在收集并返回反卷积和上采样的中间特征
class Decoder(nn.Module):
    """Build Decoder"""
    def __init__(self, outViewN, outW, outH, renderDepth):
        super(Decoder, self).__init__()
        self.outViewN = outViewN

        self.relu = nn.ReLU()
        self.fc1 = linear_block(512, 1024)
        self.fc2 = linear_block(1024, 2048)
        self.fc3 = linear_block(2048, 4096)
        self.deconv1 = deconv2d_block(256, 192)
        self.deconv2 = deconv2d_block(192, 128)
        self.deconv3 = deconv2d_block(128, 96)
        self.deconv4 = deconv2d_block(96, 64)
        self.deconv5 = deconv2d_block(64, 48)
        self.pixel_conv = nn.Conv2d(48, outViewN*2, 1, stride=1, bias=False)
        self.pixel_bias = pixel_bias(outViewN, outW, outH, renderDepth)

    def forward(self, x):
        # 用于存储Decoder的中间特征
        saved_tensors = {}
        
        x = self.relu(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = x.view([-1, 256, 4, 4])
        
        # --- Decoder Layer 1 ---
        x_interp = F.interpolate(x, scale_factor=2)
        saved_tensors['decoder_interp_1'] = x_interp # 保存：Interpolate 之后
        x = self.deconv1(x_interp)
        saved_tensors['decoder_deconv_1'] = x       # 保存：Deconv 之后

        # --- Decoder Layer 2 ---
        x_interp = F.interpolate(x, scale_factor=2)
        saved_tensors['decoder_interp_2'] = x_interp
        x = self.deconv2(x_interp)
        saved_tensors['decoder_deconv_2'] = x

        # --- Decoder Layer 3 ---
        x_interp = F.interpolate(x, scale_factor=2)
        saved_tensors['decoder_interp_3'] = x_interp
        x = self.deconv3(x_interp)
        saved_tensors['decoder_deconv_3'] = x

        # --- Decoder Layer 4 ---
        x_interp = F.interpolate(x, scale_factor=2)
        saved_tensors['decoder_interp_4'] = x_interp
        x = self.deconv4(x_interp)
        saved_tensors['decoder_deconv_4'] = x

        # --- Decoder Layer 5 ---
        x_interp = F.interpolate(x, scale_factor=2)
        saved_tensors['decoder_interp_5'] = x_interp
        x = self.deconv5(x_interp)
        saved_tensors['decoder_deconv_5'] = x

        # Final Layer
        x = self.pixel_conv(x) + self.pixel_bias.to(x.device)
        XYZ, maskLogit = torch.split(
            x, [self.outViewN, self.outViewN], dim=1)

        return XYZ, maskLogit, saved_tensors

# 修改：主模型类，负责整合所有保存逻辑并写入文件
class Signal2Pixel_4views_Model(nn.Module):
    def __init__(self, options=None):
        super(Signal2Pixel_4views_Model, self).__init__()
        self.options = options
        if options.encoder_shared:
            self.encoder = ComplexEncoder(options=options)
        else:
            self.encoder0 = ComplexEncoder(options=options)
            self.encoder1 = ComplexEncoder(options=options)
            self.encoder2 = ComplexEncoder(options=options)
            self.encoder3 = ComplexEncoder(options=options)
        self.decoder = Decoder(outViewN=options.outViewN, outW=128, outH=128, renderDepth=options.renderDepth)
   
    def forward(self, signals, save_path=None):
        '''
        signals: list of 4 views signal dicts
        save_path: 如果提供字符串路径，则将特征保存到该目录
        '''
        outs = []
        all_encoder_saves = [] # 存储4个Encoder的中间特征
        
        if self.options.encoder_shared:
            # View 0
            out0, saves0 = self.encoder(signals[0]['signal_real'], signals[0]['signal_imag'])
            outs.append(out0)
            all_encoder_saves.append(saves0)
            
            # View 1
            out1, saves1 = self.encoder(signals[1]['signal_real'], signals[1]['signal_imag'])
            outs.append(out1)
            all_encoder_saves.append(saves1)
            
            # View 2
            out2, saves2 = self.encoder(signals[2]['signal_real'], signals[2]['signal_imag'])
            outs.append(out2)
            all_encoder_saves.append(saves2)
            
            # View 3
            out3, saves3 = self.encoder(signals[3]['signal_real'], signals[3]['signal_imag'])
            outs.append(out3)
            all_encoder_saves.append(saves3)
            
            outs = torch.concat(outs, dim=1)
        else:
            # View 0
            out0, saves0 = self.encoder0(signals[0]['signal_real'], signals[0]['signal_imag'])
            outs.append(out0)
            all_encoder_saves.append(saves0)
            
            # View 1
            out1, saves1 = self.encoder1(signals[1]['signal_real'], signals[1]['signal_imag'])
            outs.append(out1)
            all_encoder_saves.append(saves1)
            
            # View 2
            out2, saves2 = self.encoder2(signals[2]['signal_real'], signals[2]['signal_imag'])
            outs.append(out2)
            all_encoder_saves.append(saves2)
            
            # View 3
            out3, saves3 = self.encoder3(signals[3]['signal_real'], signals[3]['signal_imag'])
            outs.append(out3)
            all_encoder_saves.append(saves3)
            
            outs = torch.concat(outs, dim=1)
        
        # 保存：进入Decoder之前的特征
        decoder_input_feature = outs
        
        # Decoder Forward
        XYZ, maskLogit, decoder_saves = self.decoder(outs)

        # --- 保存逻辑 ---
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            
            # 构建保存字典
            checkpoint = {
                'encoder_features': all_encoder_saves, # List[Dict], 索引0-3对应View 0-3
                'decoder_input': decoder_input_feature, # 512维特征向量
                'decoder_features': decoder_saves        # Decoder内部特征
            }
            
            # 保存文件
            file_path = os.path.join(save_path, 'intermediate_features.pth')
            torch.save(checkpoint, file_path)
            print(f"所有中间特征已保存至: {file_path}")

        return XYZ, maskLogit

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 配置参数
    options = edict()
    options.encoder_shared = True  # 假设共享权重
    options.outViewN = 4           # 输出视角数量
    options.renderDepth = 5.0      # 渲染深度初始值
    
    # 初始化模型
    model = Signal2Pixel_4views_Model(options).to(device)
    
    # 构造模拟输入数据 (List of 4 views)
    # 符合forward要求的list of dicts
    signals = []
    for _ in range(4):
        s_real = torch.randn(8, 9, 2800).to(device)
        s_imag = torch.randn(8, 9, 2800).to(device)
        signals.append({'signal_real': s_real, 'signal_imag': s_imag})
    
    # 前向传播并指定保存目录
    # 特征将保存在当前目录下的 'saved_tensors' 文件夹中
    XYZ, maskLogit = model(signals, save_path='./saved_tensors')
    
    print("Output shape check:")
    print("XYZ shape:", XYZ.shape)
    print("maskLogit shape:", maskLogit.shape)
    
    # --- 如何加载并检查保存的特征 (示例) ---
    print("\n正在加载保存的特征进行验证...")
    loaded_data = torch.load(os.path.join('./saved_tensors', 'intermediate_features.pth'), map_location=device)
    
    print("\n特征结构预览:")
    print(f"- 包含 {len(loaded_data['encoder_features'])} 个View的Encoder特征")
    print(f"  - View 0 'out_5' shape: {loaded_data['encoder_features'][0]['out_5'].shape}")
    print(f"  - View 0 'conv1Real_conv3' shape: {loaded_data['encoder_features'][0]['conv1Real_branches']['conv3'].shape}")
    print(f"- Decoder Input shape: {loaded_data['decoder_input'].shape}")
    print(f"- Decoder 'deconv_5' shape: {loaded_data['decoder_features']['decoder_deconv_5'].shape}")