import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision.transforms.functional as TF

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(  # Transform g (decoder feature)
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(  # Transform x (encoder feature)
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(  # Compute attention map
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
    
    def forward(self, g, x):
        g1 = self.W_g(g)  # [B, F_int, H, W]
        x1 = self.W_x(x)  # [B, F_int, H, W]
        psi = torch.relu(g1 + x1)  # Combine
        psi = self.psi(psi)  # [B, 1, H, W] (attention map)
        return x * psi  # Attended feature

class DoubleConv(nn.Module):
    def __init__(self, inchannels , outchannels):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(inchannels,outchannels, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels,outchannels, kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)
    
class Attention_UNET(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, features=[64,128,256,512]):
        super(Attention_UNET,self).__init__()
        self.ups=nn.ModuleList()
        self.downs=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2, stride=2)
        
        #Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels=feature
        
        #Up part of UNET
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2,stride=2))
            self.ups.append(DoubleConv(feature*2,feature))
            #Adding attention
            self.ups.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature//2))
        
        #Middle part and final part
        self.bottleneck=DoubleConv(features[-1],features[-1]*2)
        self.final_conv=nn.Conv2d(features[0],out_channels,kernel_size=1)
        
    def forward(self,x):
        skip_connections=[]
        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x=self.pool(x)

        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1]#reverse the list
        
        for idx in range(0,len(self.ups),3):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//3]
            
            if x.shape!=skip_connection.shape:
                x=TF.resize(x,size=skip_connection.shape[2:])
            
            #Applying attention
            attn_skip = self.ups[idx+2](x, skip_connection)
            concat_skip=torch.concat((attn_skip,x),dim=1)
            x=self.ups[idx+1](concat_skip)
                
        return self.final_conv(x)
    
def test():
    x=torch.randn((3,1,161,161))
    model=Attention_UNET(in_channels=1,out_channels=1)
    preds=model(x)
    assert preds.shape==x.shape
    
if __name__=="__main__":
    test()