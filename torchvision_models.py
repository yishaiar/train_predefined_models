# from typing import Any
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torchvision
import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

def get_model_torchVision(model_name = 'resnet18', num_classes=2,weights=None,modelSummary = False ):#pretrained=True
    # model = getattr(torchvision.models, model_name)(pretrained=False)
    model = getattr(torchvision.models, model_name)(weights = weights, num_classes=num_classes, progress=True)
    if modelSummary:
        # print(model)
        name,param = list(model.named_parameters())[-1]
        print(f'num_classes: {num_classes}, last layer: {name}, last layer params: {param.shape}')
        print('')
    return model

class get_model_lightning(pl.LightningModule):
    # def __init__(self, *args: Any, **kwargs: Any) -> None:
    def __init__(self, model_name = 'resnet18', num_classes=2,optimizer = 'Adam',lr = 0.001, weights=None,criterion ='CrossEntropyLoss',scheduler_step = 7 ):
        super(get_model_lightning,self).__init__()
        self.back_bone = get_model_torchVision(model_name = model_name,num_classes=num_classes,weights = weights)

        self.save_hyperparameters()
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler_step = scheduler_step
        # ----------------------
        # 1 INITIALIZATION
        self.criterion_ = getattr(torch.nn, criterion)()# CrossEntropyLoss
        self.train_acc = Accuracy(task = 'multiclass',num_classes = num_classes)
        self.val_acc = Accuracy(task = 'multiclass',num_classes = num_classes)
        self.test_acc = Accuracy(task = 'multiclass',num_classes = num_classes)
    
    #needed y.long() only for criterion - calculate only here (not in forward) thats why we need to use self.criterion_ instead of self.criterion
    def criterion(self, logits, y): 
          return self.criterion_(logits, y.long())

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.optimizer)(self.parameters(), lr=self.lr)
        if self.scheduler_step is not None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.scheduler_step, gamma=0.1)#gamma=0.1 - multiply lr by 0.1 every step_size epochs (reduce by magnitude)
            return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}
        return optimizer
    
    def forward(self, x):
        return self.back_bone(x.float())
        # possible to add here more layers..

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)#outputs of last fully connected layer before softmax
        
        loss = self.criterion(logits, y)#CrossEntropyLoss
        preds = torch.argmax(logits, dim=1)#get the index of the max log-probability; equivalent to _,preds = torch.max(logits, dim=1)

        self.train_acc(preds, y)#Accuracy
        self.log('train_loss', loss, prog_bar=True)#on_step=True, on_epoch=True,logger=True
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], prog_bar=True)#actual lr inside optimizer
        return loss
    def on_train_epoch_end(self):# -> None
        # train acc logged after each epoch and not after each batch (such as in val/test)
        # otherwise it would be noisy (weights dont change after each batch)
        self.log('train_acc', self.train_acc.compute())
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)#outputs of last fully connected layer before softmax
        loss = self.criterion(logits, y)#CrossEntropyLoss
        preds = torch.argmax(logits, dim=1)#get the index of the max log-probability; equivalent to _,preds = torch.max(logits, dim=1)
        
        self.log('val_loss', loss, prog_bar=True)
        self.val_acc.update(preds, y)
        self.log('val_loss', self.val_acc.compute(), prog_bar=True)
        
        return loss
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)#outputs of last fully connected layer before softmax
        loss = self.criterion(logits, y)#CrossEntropyLoss
        preds = torch.argmax(logits, dim=1)#get the index of the max log-probability; equivalent to _,preds = torch.max(logits, dim=1)
        
        self.log('test_loss', loss, prog_bar=True)
        self.test_acc.update(preds, y)
        self.log('test_loss', self.test_acc.compute(), prog_bar=True)
        
        return loss
    def predict_step(self, batch, batch_idx):
        x, y = batch#y is y_true, prefd is y_pred
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        return {'preds': preds, 'y': y}
    
    if __name__ =='__main__':
        get_model_lightning(model_name = model_name,num_classes=num_classes,weights = weights)
        model = get_model_lightning()
        print(model)

       
            
