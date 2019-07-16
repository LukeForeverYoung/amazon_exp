import torch
from torch import nn
from torch import Tensor

def train_step(item,optimizer,net):
    visual_1 = item[0].cuda()
    text_1 = item[1].cuda()
    visual_2 = item[2].cuda()
    text_2 = item[3].cuda()
    rating_2 = item[4].cuda()
    label = item[5].float().cuda()  # 只有训练的时候需要转化为float
    _,loss = net.predict(visual_1, text_1, visual_2, text_2, rating_2,train=True,y=label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def test_step(item,net):
    visual_1 = item[0].cuda()
    text_1 = item[1].cuda()
    visual_2 = item[2].cuda()
    text_2 = item[3].cuda()
    rating_2 = item[4].cuda()
    label = item[5]
    res = net.predict(visual_1, text_1, visual_2, text_2, rating_2)

    return [res[i]==label[i]for i in range(len(res))]

def predict(item,net):
    pass

class Net(nn.Module):
    def __init__(self,ImageDim=4096, TexDim=100, ImageEmDim=10, TexEmDim=10, HidDim=100, FinalDim=10):
        super(Net, self).__init__()
        self.ImageDim = ImageDim
        self.TexDim = TexDim
        self.ImageEmDim = ImageEmDim
        self.TexEmDim = TexEmDim
        self.HidDim = HidDim
        self.FinalDim = FinalDim

        self.ImageLinear=nn.Linear(self.ImageDim,self.ImageEmDim,bias=False)
        self.TextLinear=nn.Linear(self.TexDim,self.TexEmDim,bias=False)
        self.MLP=nn.Sequential(
            nn.Linear(self.ImageEmDim+self.TexEmDim+1,self.HidDim), #need bias
            nn.Tanh(),
            nn.Linear(self.HidDim,self.FinalDim,bias=False)
        )
        self.Margin=nn.Parameter(torch.zeros(1,requires_grad=True))
        self.Sigmoid=nn.Sigmoid()
        self.Loss=nn.BCELoss(reduction='sum')

        self.init_weight()

    def init_weight(self):
        torch.nn.init.normal_(self.ImageLinear.weight, std=0.01)
        torch.nn.init.normal_(self.TextLinear.weight, std=0.1)
        torch.nn.init.normal_(self.MLP[0].weight, std=0.1)
        torch.nn.init.normal_(self.MLP[2].weight, std=0.1)
        torch.nn.init.zeros_(self.MLP[0].bias)


    def forward(self, imageFeature_1,textFeature_1,imageFeature_2,textFeature_2,score_2):
        '''
        **_1 means main item
        **_2 means complementary item
        '''
        textDifference = self.TextLinear(textFeature_1) - self.TextLinear(textFeature_2)
        imageDifference = self.ImageLinear(imageFeature_1) - self.ImageLinear(imageFeature_2)
        concatVector = torch.cat((textDifference,imageDifference,score_2),dim=1)
        concatVector = self.MLP(concatVector)
        distance = torch.sum(concatVector**2,dim=1)
        prop = self.Sigmoid(distance-self.Margin)

        return prop

    def predict(self,textFeature_1,imageFeature_1,textFeature_2,imageFeature_2,score_2,train=False,y=None):
        prop=self.forward(textFeature_1,imageFeature_1,textFeature_2,imageFeature_2,score_2)
        if train:
            loss=self.Loss(prop,y)
            return prop,loss
        res=torch.ge(prop.cpu().detach(),0.5)
        return res


