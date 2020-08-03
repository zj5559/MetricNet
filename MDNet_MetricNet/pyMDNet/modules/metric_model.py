import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from collections import OrderedDict
from torch.autograd import Variable

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in') # For old pytorch, you may use kaiming_normal.
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, droprate=0.0, relu=True, bnorm=True, num_bottleneck=512, linear=True, return_f = False):
        super(ClassBlock, self).__init__()
        self.return_f = return_f
        add_block = []
        if linear:
            add_block += [nn.Linear(input_dim, num_bottleneck)]
        else:
            num_bottleneck = input_dim
        if bnorm:
            add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.LeakyReLU(0.1)]
        if droprate>0:
            add_block += [nn.Dropout(p=droprate)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.add_block = add_block
        self.classifier = classifier
    def forward(self, x):
        x = self.add_block(x)
        if self.return_f:
            f = x
            x = self.classifier(x)
            return x,f
        else:
            x = self.classifier(x)
            return x

class DistanceBlock(nn.Module):
    def __init__(self, input_dim):
        super(DistanceBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, 1024)]
        add_block += [nn.BatchNorm1d(1024)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        distance = []
        distance += [nn.Linear(1024, 1024)]
        distance = nn.Sequential(*distance)
        distance.apply(weights_init_classifier)

        self.add_block = add_block
        self.distance = distance
    def forward(self, x):
        x = self.add_block(x)
        x = self.distance(x)
        return x
class DistanceBlock2(nn.Module):
    def __init__(self, input_dim):
        super(DistanceBlock2, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, 1024)]
        add_block += [nn.BatchNorm1d(1024)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)

        distance = []
        distance += [nn.Linear(1024, 512)]
        distance = nn.Sequential(*distance)
        distance.apply(weights_init_classifier)

        self.add_block = add_block
        self.distance = distance
    def forward(self, x):
        x = self.add_block(x)
        x = self.distance(x)
        return x

class ft_net(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        # self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        # x = self.classifier(x)
        return d
class classifier_ft(nn.Module):

    def __init__(self, droprate=0.5, stride=2):
        super(classifier_ft, self).__init__()
        self.model = nn.Sequential(nn.Linear(1024,512),
                                   nn.ReLU(),
                                   nn.Linear(512,2))
        # for m in self.model.modules():
        #     if isinstance(m, nn.Linear):
        #         init.normal_(m.weight.data, std=0.001)
        #         init.constant_(m.bias.data, 0.0)
    def set_all_params_no_learnable(self):
        for k, p in self.named_parameters():
            p.requires_grad=False
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params
    def forward(self, x):
        x = self.model(x)
        return x
class ft_net_no_class(nn.Module):

    def __init__(self, droprate=0.5, stride=2):
        super(ft_net_no_class, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.distance = DistanceBlock2(2048)
        self.classifier_2=nn.Sequential(nn.LeakyReLU(0.1),
                                            nn.Linear(512,2))#change distance block from 1024 to 512#nn.BatchNorm1d(512),
        for m in self.classifier_2.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, std=0.001)
                init.constant_(m.bias.data, 0.0)
    def forward(self, x,flag='all'):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        return d
    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params
class ft_net_classifier(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_classifier, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1, 1)
            model_ft.layer4[0].conv2.stride = (1, 1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock2(2048)
        self.classifier_2 = nn.Sequential(nn.LeakyReLU(0.1),
                                      nn.Linear(512, 2))  # change distance block from 1024 to 512#nn.BatchNorm1d(512),
        for m in self.classifier_2.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight.data, std=0.001)
                init.constant_(m.bias.data, 0.0)
    def set_learnable_params(self, layers):
        for k, p in self.named_parameters():
            if any([k.startswith(l) for l in layers]):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def get_learnable_params(self):
        params = OrderedDict()
        for k, p in self.named_parameters():
            if p.requires_grad:
                params[k] = p
        return params

    def set_all_params_no_learnable(self):
        for k, p in self.named_parameters():
            p.requires_grad = False

    def forward(self, x, flag='all'):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        return d
class LOFBlock_fc(nn.Module):
    def __init__(self):
        super(LOFBlock_fc, self).__init__()
        add_block=[]
        add_block+=[nn.BatchNorm1d(1024)]
        add_block+=[nn.LeakyReLU(0.1)]
        add_block+=[nn.Linear(1024,256)]
        add_block+=[nn.BatchNorm1d(256)]
        add_block+=[nn.LeakyReLU(0.1)]
        add_block+=[nn.Linear(256,128)]
        add_block=nn.Sequential(*add_block)
        add_block.apply(weights_init_classifier)
        self.lof=add_block
    def forward(self,x):
        x=self.lof(x)
        return x
class LOFBlock_fc_bottleneck(nn.Module):
    def __init__(self):
        super(LOFBlock_fc_bottleneck, self).__init__()
        add_block=[]
        add_block+=[nn.BatchNorm1d(1024)]
        add_block+=[nn.LeakyReLU(0.1)]
        add_block+=[nn.Linear(1024,256)]
        add_block+=[nn.BatchNorm1d(256)]
        add_block+=[nn.LeakyReLU(0.1)]
        add_block+=[nn.Linear(256,64)]
        add_block += [nn.BatchNorm1d(64)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Linear(64, 128)]
        add_block=nn.Sequential(*add_block)
        add_block.apply(weights_init_classifier)
        self.lof=add_block
    def forward(self,x):
        x=self.lof(x)
        return x
class LOFBlock_fc_bottleneck_128_2(nn.Module):
    def __init__(self):
        super(LOFBlock_fc_bottleneck_128_2, self).__init__()
        add_block=[]
        add_block += [nn.BatchNorm1d(1024)]
        add_block+=[nn.Linear(1024,256)]
        add_block += [nn.BatchNorm1d(256)]
        # add_block += [nn.LeakyReLU(0.1)]
        add_block+=[nn.Linear(256,64)]
        add_block += [nn.BatchNorm1d(64)]
        # add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Linear(64, 128)]
        add_block=nn.Sequential(*add_block)
        add_block.apply(weights_init_classifier)
        self.lof=add_block
    def forward(self,x):
        x=self.lof(x)
        return x
class ft_net_lof_fc_bottleneck_128(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):#stride=2
        super(ft_net_lof_fc_bottleneck_128, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
        self.lof = LOFBlock_fc_bottleneck_128_2()
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        lof=self.lof(d)
        # x = self.classifier(x)
        return d,lof
class LOFBlock_fc_bottleneck_512_2(nn.Module):
    def __init__(self):
        super(LOFBlock_fc_bottleneck_512_2, self).__init__()
        add_block=[]
        add_block+=[nn.Linear(1024,256)]
        add_block += [nn.BatchNorm1d(256)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block+=[nn.Linear(256,512)]
        add_block=nn.Sequential(*add_block)
        add_block.apply(weights_init_classifier)
        self.lof=add_block
    def forward(self,x):
        x=self.lof(x)
        return x
class ft_net_lof_fc_bottleneck_512(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):#stride=2
        super(ft_net_lof_fc_bottleneck_512, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
        self.lof = LOFBlock_fc_bottleneck_512_2()
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        lof=self.lof(d)
        # x = self.classifier(x)
        return d,lof
class ft_net_lof_fc(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):#stride=2
        super(ft_net_lof_fc, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
        self.lof = LOFBlock_fc()
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        lof=self.lof(d)
        # x = self.classifier(x)
        return d,lof
class ft_net_lof_fc_bottleneck(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):#stride=2
        super(ft_net_lof_fc_bottleneck, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
        self.lof = LOFBlock_fc_bottleneck()
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        lof=self.lof(d)
        # x = self.classifier(x)
        return d,lof
class ft_net_lof_fc_test(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):#stride=2
        super(ft_net_lof_fc_test, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
        self.lof = LOFBlock_fc()
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        # lof=self.lof(d)
        # x = self.classifier(x)
        return d,d

class LOFBlock_cnn(nn.Module):
    ############to do
    def __init__(self):
        super(LOFBlock_cnn, self).__init__()
        add_block=[]
        add_block+=[nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1,bias=False)]
        add_block+=[nn.BatchNorm2d(32)]
        add_block+=[nn.LeakyReLU(0.1)]
        add_block+=[nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        add_block+=[nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,bias=False)]
        add_block += [nn.BatchNorm2d(64)]
        add_block+=[nn.LeakyReLU(0.1)]
        add_block += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        add_block+=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1,bias=False)]
        add_block += [nn.BatchNorm2d(128)]
        add_block+=[nn.LeakyReLU(0.1)]
        add_block += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]

        add_block+=[nn.AdaptiveAvgPool2d((1,1))]
        add_block=nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.lof=add_block
    def forward(self,x):
        x=self.lof(x)
        return x

class ft_net_lof_cnn(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_lof_cnn, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
        self.lof=LOFBlock_cnn()
    def forward(self, x,flag='all'):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        d_cnn=d.reshape(d.shape[0],1,32,32)
        lof=self.lof(d_cnn)
        lof=lof.squeeze()
        return d,lof
class LOFBlock(nn.Module):
    def __init__(self, input_dim):
        super(LOFBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, 1024)]
        add_block += [nn.BatchNorm1d(1024)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Linear(1024, 512)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_classifier)
        self.lof = add_block
    def forward(self, x):
        x = self.lof(x)
        return x

class ft_net_lof_fc_3part(nn.Module):

    def __init__(self, class_num, droprate=0.5, stride=2):
        super(ft_net_lof_fc_3part, self).__init__()
        model_ft = models.resnet50(pretrained=True)
        # avg pooling to global pooling
        if stride == 1:
            model_ft.layer4[0].downsample[0].stride = (1,1)
            model_ft.layer4[0].conv2.stride = (1,1)
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.model = model_ft
        self.classifier = ClassBlock(2048, class_num)
        self.distance = DistanceBlock(2048)
        self.lof=LOFBlock(2048)
    def forward(self, x,flag='all'):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = x.view(x.size(0), x.size(1))
        d = self.distance(x)
        lof=self.lof(x)
        return d,lof