import torch
from torch import nn
from torch import Tensor

def train_step(item,optimizer,net,add_category=False):
    if add_category:
        visual_1 = item[0].cuda()
        text_1 = item[1].cuda()
        category_1=item[2].cuda()
        visual_2 = item[3].cuda()
        text_2 = item[4].cuda()
        category_2=item[5].cuda()
        rating_2 = item[6].cuda()
        label = item[7].float().cuda()  # 只有训练的时候需要转化为float
        _, loss = net.predict(visual_1, text_1,category_1, visual_2, text_2,category_2, rating_2, train=True, y=label)
    else:
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
    #print('mid loss:',loss.item())
    return loss.item()

def test_step(item,net,add_category=False):
    if add_category:
        visual_1 = item[0].cuda()
        text_1 = item[1].cuda()
        category_1 = item[2].cuda()
        visual_2 = item[3].cuda()
        text_2 = item[4].cuda()
        category_2 = item[5].cuda()
        rating_2 = item[6].cuda()
        label = item[7]
        res = net.predict(visual_1, text_1,category_1, visual_2, text_2,category_2, rating_2)
    else:
        visual_1 = item[0].cuda()
        text_1 = item[1].cuda()
        visual_2 = item[2].cuda()
        text_2 = item[3].cuda()
        rating_2 = item[4].cuda()
        label = item[5]
        res = net.predict(visual_1, text_1, visual_2, text_2, rating_2)
    pre_positive=0
    positive=0
    for i in range(len(res)):
        if label[i]==1:
            positive+=1
            if res[i]==1:
                pre_positive+=1
    return [res[i]==label[i]for i in range(len(res))],pre_positive,positive



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
            #nn.BatchNorm1d(self.HidDim),
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
        torch.nn.init.zeros_(self.Margin.data)

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

class Net_category(nn.Module):
    def __init__(self,ImageDim=4096, TexDim=100,CatDim=300, ImageEmDim=10, TexEmDim=10,CatEmDim=30,HidDim=150, FinalDim=10):
        super(Net_category, self).__init__()
        self.ImageDim = ImageDim
        self.TexDim = TexDim
        self.CatDim=CatDim
        self.ImageEmDim = ImageEmDim
        self.TexEmDim = TexEmDim
        self.CatEmDim=CatEmDim
        self.HidDim = HidDim
        self.FinalDim = FinalDim

        self.ImageLinear=nn.Linear(self.ImageDim,self.ImageEmDim,bias=False)
        self.TextLinear=nn.Linear(self.TexDim,self.TexEmDim,bias=False)
        self.CategoryLinear = nn.Linear(self.CatDim, self.CatEmDim, bias=False)
        self.MLP=nn.Sequential(
            nn.Linear(self.ImageEmDim+self.TexEmDim+self.CatEmDim+1,self.HidDim), #need bias
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
        torch.nn.init.normal_(self.CategoryLinear.weight, std=0.1)
        torch.nn.init.normal_(self.MLP[0].weight, std=0.1)
        torch.nn.init.normal_(self.MLP[2].weight, std=0.1)
        torch.nn.init.zeros_(self.MLP[0].bias)


    def forward(self, imageFeature_1,textFeature_1,categoryFeature_1,imageFeature_2,textFeature_2,categoryFeature_2,score_2):
        '''
        **_1 means main item
        **_2 means complementary item
        '''
        textDifference = self.TextLinear(textFeature_1) - self.TextLinear(textFeature_2)
        imageDifference = self.ImageLinear(imageFeature_1) - self.ImageLinear(imageFeature_2)
        categoryDifference= self.CategoryLinear(categoryFeature_1) - self.CategoryLinear(categoryFeature_2)
        concatVector = torch.cat((textDifference,imageDifference,categoryDifference,score_2),dim=1)
        concatVector = self.MLP(concatVector)
        distance = torch.sum(concatVector**2,dim=1)
        prop = self.Sigmoid(distance-self.Margin)

        return prop

    def predict(self,textFeature_1,imageFeature_1,categoryFeature_1,textFeature_2,imageFeature_2,categoryFeature_2,score_2,train=False,y=None):
        prop=self.forward(textFeature_1,imageFeature_1,categoryFeature_1,textFeature_2,imageFeature_2,categoryFeature_2,score_2)
        if train:
            loss=self.Loss(prop,y)
            return prop,loss
        res=torch.ge(prop.cpu().detach(),0.5)
        return res


class Net_bayes(nn.Module):
    def __init__(self,ImageDim=4096, TexDim=100, ImageEmDim=10, TexEmDim=10, HidDim=100, FinalDim=1):
        super(Net_bayes, self).__init__()
        self.ImageDim = ImageDim
        self.TexDim = TexDim
        self.ImageEmDim = ImageEmDim
        self.TexEmDim = TexEmDim
        self.HidDim = HidDim
        self.FinalDim = FinalDim

        self.useRelu=True
        self.activate=nn.ReLU() if self.useRelu else nn.Tanh()
        '''
        self.ImageLinear=nn.Linear(self.ImageDim,self.ImageEmDim,bias=False)
        self.TextLinear=nn.Linear(self.TexDim,self.TexEmDim,bias=False)
        self.MLP_1=nn.Sequential(
            nn.Linear(self.ImageEmDim+self.TexEmDim+1,self.HidDim), #need bias
            nn.Tanh(),
        )
        self.MLP_2 = nn.Sequential(
            nn.Linear(self.ImageEmDim + self.TexEmDim + 1, self.HidDim),  # need bias
            nn.Tanh(),
        )
        self.FL=nn.Linear(self.HidDim,self.FinalDim,bias=True)
        self.Sigmoid=nn.Sigmoid()
        self.Loss=nn.BCELoss(reduction='sum')
        '''
        self.ImageMidDim=1000
        self.TextMidDim=100


        self.ImageSeq=nn.Sequential(
            nn.Linear(self.ImageDim * 2, self.ImageMidDim, bias=True),
            nn.BatchNorm1d(self.ImageMidDim),
            self.activate,
            nn.Linear(self.ImageMidDim, self.ImageEmDim, bias=True),
            nn.BatchNorm1d(self.ImageEmDim),
            self.activate,
        )
        self.TextSeq = nn.Sequential(
            nn.Linear(self.TexDim * 2, self.TextMidDim, bias=True),
            nn.BatchNorm1d(self.TextMidDim),
            self.activate,
            nn.Linear(self.TextMidDim, self.TexEmDim, bias=True),
            nn.BatchNorm1d(self.TexEmDim),
            self.activate,
        )
        self.MLP = nn.Sequential(
            nn.Linear(self.ImageEmDim + self.TexEmDim + 1, self.HidDim),  # need bias
            self.activate,
            nn.Linear(self.HidDim, self.FinalDim, bias=True)
        )
        self.Sigmoid=nn.Sigmoid()
        self.Loss = nn.BCELoss(reduction='sum')
        self.init_weight()

    def init_weight(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
        self.apply(init_weights)

    def forward(self, imageFeature_1,textFeature_1,imageFeature_2,textFeature_2,score_2):
        '''
        **_1 means main item
        **_2 means complementary item
        '''
        ImageFeature=torch.cat((imageFeature_1,imageFeature_2),dim=1)
        TextFeature=torch.cat((textFeature_1,textFeature_2),dim=1)
        ImageFeature=self.ImageSeq(ImageFeature)
        TextFeature=self.TextSeq(TextFeature)
        concatVector = torch.cat((ImageFeature,TextFeature,score_2),dim=1)
        concatVector = self.MLP(concatVector)
        prop = self.Sigmoid(concatVector)

        return prop

    def predict(self,textFeature_1,imageFeature_1,textFeature_2,imageFeature_2,score_2,train=False,y=None):
        prop=self.forward(textFeature_1,imageFeature_1,textFeature_2,imageFeature_2,score_2)
        if train:
            loss=self.Loss(prop,y)
            return prop,loss
        res=torch.ge(prop.cpu().detach(),0.5)
        return res

class Net_BN_category(nn.Module):
    def __init__(self,ImageDim=4096, TexDim=100,CatDim=300, ImageEmDim=10, TexEmDim=10,CatEmDim=30,HidDim=150, FinalDim=1):
        super(Net_BN_category, self).__init__()
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


        self.ImageSeq = nn.Sequential(
            nn.Linear(self.ImageDim * 2, self.ImageMidDim, bias=True),
            nn.BatchNorm1d(self.ImageMidDim),
            self.activate,
            nn.Linear(self.ImageMidDim, self.ImageEmDim, bias=True),
            nn.BatchNorm1d(self.ImageEmDim),
            self.activate,
        )
        self.TextSeq = nn.Sequential(
            nn.Linear(self.TexDim * 2, self.TextMidDim, bias=True),
            nn.BatchNorm1d(self.TextMidDim),
            self.activate,
            nn.Linear(self.TextMidDim, self.TexEmDim, bias=True),
            nn.BatchNorm1d(self.TexEmDim),
            self.activate,
        )
        self.CatSeq = nn.Sequential(
            nn.Linear(self.CatDim * 2, self.CatMidDim, bias=True),
            nn.BatchNorm1d(self.CatMidDim),
            self.activate,
            nn.Linear(self.CatMidDim, self.CatEmDim, bias=True),
            nn.BatchNorm1d(self.CatEmDim),
            self.activate,
        )
        self.MLP = nn.Sequential(
            nn.Linear(self.ImageEmDim + self.TexEmDim+self.CatEmDim + 1, self.HidDim),  # need bias
            self.activate,
            nn.Linear(self.HidDim, self.FinalDim, bias=True)
        )
        self.Sigmoid = nn.Sigmoid()
        self.Loss = nn.BCELoss(reduction='sum')

        self.init_weight()


    def init_weight(self):
        def init_weights(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)

        self.apply(init_weights)


    def forward(self, imageFeature_1,textFeature_1,categoryFeature_1,imageFeature_2,textFeature_2,categoryFeature_2,score_2):
        '''
        **_1 means main item
        **_2 means complementary item
        '''
        ImageFeature = torch.cat((imageFeature_1, imageFeature_2), dim=1)
        TextFeature = torch.cat((textFeature_1, textFeature_2), dim=1)
        CatFeature=torch.cat((categoryFeature_1,categoryFeature_2),dim=1)
        ImageFeature = self.ImageSeq(ImageFeature)
        TextFeature = self.TextSeq(TextFeature)
        CatFeature=self.CatSeq(CatFeature)
        concatVector = torch.cat((ImageFeature, TextFeature,CatFeature, score_2), dim=1)
        concatVector = self.MLP(concatVector)
        prop = self.Sigmoid(concatVector)

        return prop


    def predict(self,textFeature_1,imageFeature_1,categoryFeature_1,textFeature_2,imageFeature_2,categoryFeature_2,score_2,train=False,y=None):
        prop=self.forward(textFeature_1,imageFeature_1,categoryFeature_1,textFeature_2,imageFeature_2,categoryFeature_2,score_2)
        if train:
            loss=self.Loss(prop,y)
            return prop,loss
        res=torch.ge(prop.cpu().detach(),0.5)
        return res

