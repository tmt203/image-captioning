import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # pretrained model resnet50
        resnet = models.resnet50(pretrained=True)
        
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # replace the classifier with a fully connected embedding layer
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(self.embed_size, 
                            self.hidden_size, 
                            self.num_layers, 
                            batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.vocab_size)
#         self.hidden = (torch.zeros(1, 1, self.hidden_size), torch.zeros(1, 1, self.hidden_size))
        
    
    def forward(self, features, captions):
        
        caption_embed = self.word_embedding(captions[:, :-1])
        caption_embed = torch.cat((features.unsqueeze(dim=1), caption_embed),1)
        output, self.hidden = self.lstm(caption_embed)
#         print(output.shape)
        output = self.fc(output)
#         print(self.embed_size, self.vocab_size)
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        """Samples captions for given image features."""
        output = []
        (h, c) = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device), torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device))
        for i in range(max_len):
            x, (h, c) = self.lstm(inputs, (h, c))
            x = self.fc(x)
            x = x.squeeze(1)
            predict = x.argmax(dim=1)
            if predict.item() == 1:
                break
            output.append(predict.item())
            inputs = self.word_embedding(predict.unsqueeze(0))
        return output
    
#-------- Attention Mechanism ---------#
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim, dropout):
        """
        decoder_dim is hidden_size for lstm cell
        """
        super(DecoderWithAttention, self).__init__()  # Correct superclass name
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution
        
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, device):  # Added device argument
        """
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param device: device to be used (cuda or cpu)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """
        encoder_out = encoder_out.to(device)
        encoded_captions = encoded_captions.to(device)
        
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        
        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        
        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        
        decode_length = encoded_captions.size(1)-1
        
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, decode_length, vocab_size).to(device)
        alphas = torch.zeros(batch_size, decode_length, num_pixels).to(device)
        
        for t in range(decode_length):
            
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
                        
            h, c = self.decode_step(torch.cat([embeddings[:, t, :], attention_weighted_encoding], dim=1), (h, c))  #(batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:, t, :] = preds
            alphas[:, t, :] = alpha

        return predictions, alphas
    
    def sample(self, encoder_out, data_loader, device, max_len=20):
        # Initialize the hidden state and cell state
        h, c = self.init_hidden_state(encoder_out)

        # Initialize the input token as <start>
        start_token = torch.tensor(data_loader.dataset.vocab.word2idx['<start>']).unsqueeze(0).to(device)

        # Initialize the input token list
        sampled_ids = [start_token.item()]

        # Loop until reaching the maximum length or the <end> token
        for i in range(max_len):
            # Embed the input token
            inputs = self.embedding(start_token)

            # Generate the attention-weighted encoding
            attention_weighted_encoding, _ = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding

            # Add a singleton dimension to inputs and attention_weighted_encoding tensors            
            inputs_expanded = inputs.unsqueeze(1)
            
            if inputs_expanded.dim() == 4:
                inputs_expanded = inputs_expanded.squeeze(1)           

            # Check if attention_weighted_encoding needs to be squeezed
            if attention_weighted_encoding.dim() == 4:
                attention_weighted_encoding_expanded = attention_weighted_encoding.unsqueeze(1).squeeze(2)
            else:
                attention_weighted_encoding_expanded = attention_weighted_encoding.unsqueeze(1)

            # Concatenate the input with attention-weighted encoding
            concatenated_input = torch.cat([inputs_expanded, attention_weighted_encoding_expanded], dim=2)

            # Pass the concatenated input through the LSTM decoder
            h, c = self.decode_step(concatenated_input.squeeze(1), (h, c))

            # Generate the output logits
            outputs = self.fc(h)

            # Choose the token with the highest probability as the next input token
            predicted_token = outputs.argmax(1)

            # Append the predicted token to the sampled token list
            sampled_ids.append(predicted_token.item())

            # If the predicted token is the <end> token, break the loop
            if predicted_token == data_loader.dataset.vocab.word2idx['<end>']:
                break

            # Update the input token for the next iteration
            start_token = predicted_token.unsqueeze(0)

        return sampled_ids

