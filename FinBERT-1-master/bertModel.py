from __future__ import print_function, division

#importing torch library
import torch
#importing torch.nn library. Essential Library for building and training a neural network using torch
import torch.nn as nn
#importing torch.optim library. Essential library used for optimization of neural networks
import torch.optim as optim
#importing transformers
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
#Creating a class BertClassification. In this class we define our BertModel for training for sentiment analysis. this cass inherits from nn.module
class BertClassification(nn.Module):
   #This function is used to initialize our BertModel. It has 3 parameters. Number of labels, vocab used and url path where weights are present
    def __init__(self, weight_path, num_labels=2, vocab="base-cased"):
        #This line calls the contructor for the nn.Module class
        super(BertClassification, self).__init__()
        #this line initializes num_labels
        self.num_labels = num_labels
        #this line initializes vocab
        self.vocab = vocab 
        #this line defines the model and configures it according to the type of vocabulary used
        if self.vocab == "base-cased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=28996, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        elif self.vocab == "base-uncased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
            
        elif self.vocab == "finance-cased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=28573, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

        elif self.vocab =="finance-uncased":
            self.bert = BertModel.from_pretrained(weight_path)
            self.config = BertConfig(vocab_size_or_config_json_file=30873, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
        #the next 2 upcoming lines create new layers to the model.
        #this is the first layer added. This is the Dropout Layer. this is used to reduce the amount of overfitting present. numberOfInputs=noOfHiddenElements in the nn. numberOfOutputs=numberOfInputs. Dropout Layers don't Change the dimentions
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        #this line here is used to create the next layer. the next layer is Classification layer. It is a linear Classifier. 
        #this is a linear classifier. The number of inputs to the classifier are hidden_size and the number of outputs are num_labels
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        #this is a funtion used to initialize the weights of classification layers
        nn.init.xavier_normal(self.classifier.weight)
    #this function is used to forward propagate in the neural network. it returns the logits(last layer output of 3 size) which must then be given to softmax. 
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, graphEmbeddings=None):
        #this line returns the sequence output and the pooled output from the model after forward propagation. 
        #pooled output contains the information regarding the classification output
        
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
#         print(pooled_output.tolist())
#         print(_)
#         print(pooled_output.dtype)
#         print(self.bert)
        #now in this line we pass the pooled_outputs to the dropout layers. 
        pooled_output = self.dropout(outputs[1])
        #from here we get logits by giving the pooled_output from dropout layer to the classifier layer
        logits = self.classifier(pooled_output)
#         logits=0
        #in this line we return the logits
        return logits

class dense_opt():
    def __init__(self, model):
        super(dense_opt, self).__init__()
        #self.lrlast = .001: This line initializes an attribute lrlast of the dense_opt instance with the value 0.001. This attribute seems to represent the learning rate for the last (or final) layers of the model during optimization.
        self.lrlast = .001
        #self.lrmain = .00001: This line initializes another attribute lrmain of the dense_opt instance with the value 0.00001. This attribute seems to represent the learning rate for the main (or earlier) layers of the model during optimization.
        self.lrmain = .00001
        self.optim = optim.Adam(
        [
                #{"params": model.bert.parameters(), "lr": self.lrmain}: This defines the first parameter group. It specifies that the parameters of the model.bert component should be optimized with a learning rate of self.lrmain.
            {"params":model.bert.parameters(),"lr": self.lrmain},
            #{"params": model.classifier.parameters(), "lr": self.lrlast}: This defines the second parameter group. It specifies that the parameters of the model.classifier component should be optimized with a learning rate of self.lrlast.
          {"params":model.classifier.parameters(), "lr": self.lrlast},
       ])
    
    def get_optim(self):
        #return self.optim
        return self.optim