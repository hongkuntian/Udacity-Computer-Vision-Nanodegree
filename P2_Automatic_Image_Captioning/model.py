import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM cell
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        # Embed the captions
        embeded_captions = self.embedding(captions[:, :-1])
        
        # Concatenate feature and embeded captions into single tensor
        inputs = torch.cat([features.unsqueeze(1), embeded_captions], 1)
        
        # Pass inputs to LSTM
        output, _ = self.lstm(inputs)
        
        output = self.fc(output)
        
        return output
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        sentence = []
        
        for i in range(max_len):
            output, states = self.lstm(inputs, states)
            output = self.fc(output)
            output = output.squeeze(1)
            _, predicted_index = torch.max(output, dim=1)
            word_ind = int(predicted_index.cpu().numpy()[0])
            sentence.append(word_ind)
            
            # Break from loop once we reach <end> token
            if predicted_index == 1:
                break
                
            inputs = self.embedding(predicted_index)
            inputs = inputs.unsqueeze(1)
        
        return sentence
        