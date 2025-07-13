import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#Data loading and preprocessing
#It uses session_state to prevent reloading on every interaction.
if "data_loaded_dl" not in st.session_state:
    st.session_state.data_loaded_dl = True
    try:
        df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("ERROR: 'data.csv' not found. Please make sure the file is in the same directory as the script.")
        st.stop()

    df = df.drop(["id","Unnamed: 32"], axis=1)

    le = LabelEncoder()
    y = le.fit_transform(df["diagnosis"])
    st.session_state.target_names = le.classes_

    x = df.drop(columns=["diagnosis"])
    scaler = StandardScaler()
    x = pd.DataFrame(scaler.fit_transform(x), columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Save to session state.
    st.session_state.x_train_dl = x_train
    st.session_state.x_test_dl = x_test
    st.session_state.y_train_dl = y_train
    st.session_state.y_test_dl = y_test
    st.session_state.x_columns_dl = x.columns

#Get data from session state.
x_train = st.session_state.x_train_dl
x_test = st.session_state.x_test_dl
y_train = st.session_state.y_train_dl
y_test = st.session_state.y_test_dl
target_names = st.session_state.target_names

# Pytorch custom dataset
class MyDataset(Dataset):
    # Convert pandas df to float tensors.
    def __init__(self,x,y):
        self.x=torch.tensor(x.values,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32).view(-1,1)

    def __len__(self):
        # Return the total number of samples.
        return len(self.x)

    def __getitem__(self,idx):
        # Retrieve a single sample at a given index.
        return self.x[idx],self.y[idx]

training_data=MyDataset(x_train,y_train)
testing_data=MyDataset(x_test,y_test)

#Streamlit UI for dynamic model creation
st.header("Make your own neural network")
st.subheader(":green[Insert a hidden layer num]")

# Get the desired number of hidden layers from the user.
hidden_num = st.number_input("", min_value=1, step=1, value=2, placeholder="hidden num")

# Dictionary to map user's string selection to actual PyTorch activation functions.
activation_fn_dict = {
        "Tanh": nn.Tanh,
        "ReLU": nn.ReLU,
        "Leaky ReLU": nn.LeakyReLU
    }
nöronlarliste=[]
aktivasyonlar=[]

# Loop to get the number of neurons and activation function for each hidden layer.
for i in range(hidden_num):
    st.markdown(f":green[Select {i + 1}. hidden layer number ]")
    nöron_num = st.number_input("Please write neuron number", step=1,value=1,key=f"n_{i}")
    nöronlarliste.append(nöron_num)
    activation_selected = st.selectbox("Please choose activation function", ["Tanh", "ReLU", "Leaky ReLU"],key=f"a_{i}")
    act_fn = activation_fn_dict[activation_selected]
    aktivasyonlar.append(act_fn)

# Option for the user to add a Dropout layer for regularization.
use_dropout = st.toggle("Use Dropout", value=False)
dropout_rate = st.slider("Dropout rate", 0.0, 0.9, 0.5, step=0.05) if use_dropout else None

#Dynamic Pytorch NN Model
class MyModel(nn.Module):
    def __init__(self, input_dim, nöron_size, activations, use_dropout=False, dropout_rate=0.5):
        super(MyModel, self).__init__()
        # Use ModuleList to correctly register a variable number of layers.
        self.layers = nn.ModuleList()
        prev_size = input_dim

        for n, act_fn in zip(nöron_size, activations):
            self.layers.append(nn.Linear(prev_size, n))

            self.layers.append(act_fn())

            if use_dropout:
                self.layers.append(nn.Dropout(dropout_rate))

            prev_size = n

        # Final output layer with 1 neuron for binary classification.
        self.output_layer = nn.Linear(prev_size, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        # Defines the forward pass of the network.
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        x = self.activation(x)
        return x


loss_fn=nn.BCELoss()
#Choose optimizer algorithm
optimizer_selected=st.selectbox("Optimizer algorithm: ",["ADAM","SGD","AdaGrad","RMS-Prop"],
             index=0,
             placeholder="Choose an optimizer")
optimizer_dict = {
        "ADAM": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "AdaGrad": torch.optim.Adagrad,
        "RMS-Prop": torch.optim.RMSprop,
    }

lr_options = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1]

lr_rate = st.select_slider(
    "Learning Rate",
    options=lr_options,
    value=1e-3
)
#Choose batch number
batch=st.number_input("Insert a batch number", value=32, placeholder="Batch num", max_value=len(training_data),step=1)
train_dataloader=DataLoader(training_data, batch_size=batch,shuffle=True)
test_dataloader=DataLoader(testing_data, batch_size=batch, shuffle=True)

#Model Train
def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    for batch, (x_batch, y_batch) in enumerate(dataloader):
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        # Standard PyTorch training steps: zero gradients, backpropagate, update weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_loop(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    # Disable gradient calculations for efficiency during inference.
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            pred = model(x_batch)
            test_loss += loss_fn(pred, y_batch).item()
            predicted = (pred > 0.5).float()
            correct += (predicted == y_batch).sum().item()

    test_loss /= len(dataloader)
    accuracy = correct / len(dataloader.dataset)

    return test_loss, accuracy

epoch=st.number_input("Choose an epoch number",step=1,value=10)

# This block runs when the user clicks the 'Train model' button.
if st.button("Train model"):
    model = MyModel(x_train.shape[1], nöronlarliste, aktivasyonlar,use_dropout=use_dropout,dropout_rate=dropout_rate)

    optimizer = optimizer_dict[optimizer_selected](model.parameters(), lr=lr_rate)
    # Display the architecture of the generated model.
    st.subheader("Generated Model Architecture")
    st.code(model)
    status_text = st.empty()
    progress_bar = st.progress(0)

    # The main training loop over the specified number of epochs.
    for t in range(epoch):
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loss, accuracy = test_loop(test_dataloader, model,
                                               loss_fn)
        status_text.text(f"Epoch {t + 1}/{epoch} - Accuracy: {accuracy * 100:.2f}%, Avg loss: {test_loss:.4f}")
        progress_bar.progress((t + 1) / epoch)

    y_pred = model(torch.tensor(x_test.values, dtype=torch.float32))
    y_pred_cls = (y_pred.detach().numpy() > 0.5).astype(int)
    st.header("Final Model Performance")
    st.subheader("Overall Test Accuracy")
    final_accuracy = accuracy_score(y_test, y_pred_cls)
    st.metric("Final Test Accuracy", f"{final_accuracy * 100:.2f}%")

    #Display the accuracy with a pie chart
    accuracy_df = pd.DataFrame({
        'Category': ['Correct Predictions', 'Incorrect Predictions'],
        'Ratio': [final_accuracy, 1 - final_accuracy]
    })
    fig_acc = px.pie(accuracy_df, values='Ratio', names='Category',
                     title='Correct vs. Incorrect Prediction Rates',
                     color_discrete_sequence=['#013220', '#77DD77'])
    st.plotly_chart(fig_acc, use_container_width=True)

    st.subheader("Classification Metrics (Precision, Recall, F1-Score)")

    #Get the report as a dictionary to visualize it
    report_dict = classification_report(y_test, y_pred_cls, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    plot_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])
    plot_df = plot_df.drop(columns=['support'])

    #Create a grouped bar chart with Plotly
    fig_report = px.bar(plot_df, barmode='group',
                        title="Performance Metrics by Class",
                        labels={'value': 'Score', 'index': 'Class', 'variable': 'Metric'},
                        height=500,
                        text_auto='.2f')
    fig_report.update_yaxes(range=[0, 1.05])
    st.plotly_chart(fig_report, use_container_width=True)

    #Display the detailed numerical report in a text box
    st.text("Detailed Numerical Report:")
    st.code(classification_report(y_test, y_pred_cls, target_names=target_names))