import torch
from torch import nn
from torch import Tensor
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    import torch.nn.init as init
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)



class CrossVector(nn.Module):

    def create_Embedding_layer(self,in_channel,mid_channel,out_channel):
        return nn.Sequential(
            nn.Linear(in_channel, mid_channel, bias=True),
            nn.BatchNorm1d(mid_channel),
            self.activate,
            nn.Linear(mid_channel, out_channel, bias=True),
            nn.BatchNorm1d(out_channel),
            self.activate,
        )

    def __init__(self,ImageDim=4096, TexDim=100,CatDim=300, ImageEmDim=10, TexEmDim=10,CatEmDim=30,HidDim=150, FinalDim=1):
        super(CrossVector, self).__init__()
        self.ImageDim = ImageDim
        self.TexDim = TexDim
        self.CatDim=CatDim
        self.ImageEmDim = ImageEmDim
        self.TexEmDim = TexEmDim
        self.CatEmDim=CatEmDim
        self.HidDim = HidDim
        self.FinalDim = FinalDim
        self.useRelu = True
        self.activate = nn.ReLU() if self.useRelu else nn.Tanh()
        self.ImageMidDim = 1000
        self.TextMidDim = 100
        self.CatMidDim=100


        self.ImageSeq_t = self.create_Embedding_layer(self.ImageDim,self.ImageMidDim,self.ImageEmDim)
        self.ImageSeq_c = self.create_Embedding_layer(self.ImageDim, self.ImageMidDim, self.ImageEmDim)
        self.TextSeq_t = self.create_Embedding_layer(self.TexDim,self.TextMidDim,self.TextMidDim)
        self.TextSeq_c = self.create_Embedding_layer(self.TexDim,self.TextMidDim,self.TextMidDim)
        self.CatSeq_t = self.create_Embedding_layer(self.CatDim,self.CatMidDim,self.CatEmDim)
        self.CatSeq_c = self.create_Embedding_layer(self.CatDim, self.CatMidDim, self.CatEmDim)
        self.Sigmoid=nn.Sigmoid()
        self.Loss = nn.BCELoss(reduction='sum')

        self.init_network()


    def init_network(self):
        self.apply(weight_init)

    def forward(self, imageFeature_1,textFeature_1,categoryFeature_1,imageFeature_2,textFeature_2,categoryFeature_2,score_2):
        '''
        **_1 means main item
        **_2 means complementary item
        '''
        ImageFeature_t = self.ImageSeq_t(imageFeature_1)
        ImageFeature_c = self.ImageSeq_t(imageFeature_2)
        TextFeature_t=self.TextSeq_t(textFeature_1)
        TextFeature_c=self.TextSeq_t(textFeature_2)
        CatFeature_t=self.CatSeq_t(categoryFeature_1)
        CatFeature_c=self.CatSeq_t(categoryFeature_2)
        # B*Len
        target_vector=torch.cat((ImageFeature_t,TextFeature_t,CatFeature_t),dim=1)
        context_vector = torch.cat((ImageFeature_c, TextFeature_c, CatFeature_c), dim=1)
        res=torch.sum(target_vector*context_vector,dim=1)
        prop = self.Sigmoid(res)

        return prop


    def predict(self,textFeature_1,imageFeature_1,categoryFeature_1,textFeature_2,imageFeature_2,categoryFeature_2,score_2,train=False,y=None):
        prop=self.forward(textFeature_1,imageFeature_1,categoryFeature_1,textFeature_2,imageFeature_2,categoryFeature_2,score_2)
        if train:
            loss=self.Loss(prop,y)
            return prop,loss
        res=torch.ge(prop.cpu().detach(),0.5)
        return res