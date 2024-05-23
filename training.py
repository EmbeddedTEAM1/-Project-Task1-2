import torch
import torchvision
from cnn.center_dataset import CenterDataset
# import ipywidgets
import torch.nn.functional as f

def get_model():
    model = torchvision.models.alexnet(num_classes=2, dropout=0.0)
    return model

device = torch.device('cuda')
model = get_model()
#model.load_state_dict(torch.load('road_model_new_left.pth'))
model = model.to(device)


batch_size = 32

dataset = CenterDataset('dataset', random_hflip=False)
train_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=0,
    batch_size=batch_size,
    shuffle=True,
)

epoch = 100
#learning_rate = 2e-3
learning_rate = 2e-4

'''
epoch_slider = ipywidgets.IntSlider(description='Epochs', value=epoch, min=1, max=200, step=1)
lr_slider = ipywidgets.FloatSlider(description='lr', value=learning_rate, min=1e-4, max=1e-2, step=1e-4, readout_format='.4f')
train_button = ipywidgets.Button(description='Train', icon='tasks')
loss_text = ipywidgets.Textarea(description='Progress', value='', rows=15, layout=ipywidgets.Layout(width="50%", height="auto"))
layout = ipywidgets.VBox([ipywidgets.HBox([epoch_slider, lr_slider, train_button]), loss_text])
'''

def train_model():
    global epoch
    for e in range(epoch):
        #loss_text.value += "<<<<< Epoch {:d} >>>>>\n".format(epoch)
        print("epoch:", e)
        train_step()                


def train_step():
    global model, device

    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.SGD(model.parameters(), lr=lr_slider.value, momentum=0.9)

        #train_button.disabled = True                
        model = model.train()        

        num_iters = len(train_loader)
        for ii, (images, labels) in enumerate(train_loader):
            # send data to device
            images = images.to(device)
            labels = labels.to(device)
            
            # zero gradients of parameters
            optimizer.zero_grad()

            # execute model to get outputs
            outputs = model(images)

            # compute MSE loss over x coordinates            
            loss = f.mse_loss(outputs, labels, reduction='sum')

            # run backpropogation to accumulate gradients
            loss.backward()

            # step optimizer to adjust parameters
            optimizer.step()

            if ii % 10 == 0:
                xlbl, ylbl = labels[0].cpu()
                xlbl = ( xlbl.item() / 2 + 0.5 ) * 640
                ylbl = ( ylbl.item() / 2 + 0.5 ) * 360
                xpre, ypre = outputs[0].cpu()
                xpre = ( xpre.item() / 2 + 0.5 ) * 640
                ypre = ( ypre.item() / 2 + 0.5 ) * 360
                '''
                msg = "[{:04d} / {:04d}] loss: {:.4f} | labels: ({:.2f}, {:.2f}), outpus: ({:.2f}, {:.2f})\n".format(ii, num_iters, loss.item(), xlbl, ylbl, xpre, ypre)
                loss_text.value += msg                
                '''
                
                print(ii,'/',num_iters,'    loss:',loss.item())
                
    except Exception as e:
        print(e)
        pass
        
    model = model.eval()
    torch.save(model.state_dict(), 'road_model_new_right.pth')
    
    # train_button.disabled = False
    
# train_button.on_click(train_model)    
train_model()
# display(layout)

model = get_model()
#model.load_state_dict(torch.load('road_model_new_left.pth'))
model = model.to(device)

# from torch2trt import TRTModule
# model = TRTModule()
# model.load_state_dict(torch.load('road_following_model.pth'))
