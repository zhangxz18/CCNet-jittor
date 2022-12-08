import jittor as jt
from jittor import nn
from jittor import Module

def INF(B,H,W):  # debug .cuda() .repeat()
     return -jt.diag(jt.array(float("inf")).repeat(H), 0).unsqueeze(0).repeat(B*W,1,1)

class CC_module(Module):
    def __init__(self,in_dim):
        super(CC_module, self).__init__()
        self.query_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1) #[C', H, W] C'=C/8
        self.key_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(jt.zeros(1))

    def execute(self, x):  # TODO debug .view() .permute()
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).view(m_batchsize*height,-1,width).permute(0, 2, 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).view(m_batchsize*height,-1,width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).view(m_batchsize*height,-1,width)
        energy_H = (nn.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = nn.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(jt.concat([energy_H, energy_W], 3))

        att_H = concate[:,:,:,0:height].permute(0,2,1,3).view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].view(m_batchsize*height,width,width)
        out_H = nn.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)  # batch matmul
        out_W = nn.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return self.gamma*(out_H + out_W) + x


if __name__ == '__main__':
    jt.flags.use_cuda = 0

    model = CC_module(64)
    x = jt.randn(2, 64, 5, 6)
    out = model(x)
    print(out.shape)