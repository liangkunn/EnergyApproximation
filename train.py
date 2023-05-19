import time
import torch


# training function
def train(model, train_loader, val_loader, criterion, optimizer, epochs, scheduler=None):
    # put records to the log
    log_dict= {'train_loss': [],
               'val_loss': []
              }

    model.train()
    start_time = time.time()
    for epoch in range(epochs):
        
        train_loss = 0
        for i, (data, labels) in enumerate(train_loader):

            data = data.float()
            labels = labels.float()
            
            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()   

            predictions = model(data)

            loss = criterion(predictions.squeeze(), labels)
            train_loss += loss.item()

            # Backward pass
            loss.backward()
            optimizer.step()

        log_dict['train_loss'].append(train_loss/len(train_loader))
        
        val_loss = 0
        for i, (data, labels) in enumerate(val_loader):

            data = data.float()
            labels = labels.float()
                
            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()
                    
            predictions = model(data)

            loss = criterion(predictions.squeeze(), labels)
            val_loss += loss.item()

        log_dict['val_loss'].append(val_loss/len(val_loader))

        if scheduler is not None:
            # update the learning rate
            scheduler.step()

        if (epoch + 1) % 2 == 0:
            print("{}/{} Epochs | Train Loss={:.4f} | Val_loss={:.4f}".format(epoch+1, epochs, train_loss/len(train_loader), val_loss/len(val_loader)))
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60)) 
    return log_dict
