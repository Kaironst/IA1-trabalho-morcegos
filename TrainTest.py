import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from SpeciesDataset import SpeciesDataset
from NeuralNetwork import NeuralNetwork


def trainingLoop(dataloader, model, lossFunction, optimizer):
    model.train()
    totalLoss = 0

    for features, label, names in dataloader:

        features = features.to(device).float()
        # saída é 2d [batch_size,1], então precisa adicionar dimensão extra [[0/1]] com unsqueeze
        label = label.to(device).float().unsqueeze(1)

        # passagem pela rede
        prediction = model(features)
        # calculando loss
        loss = lossFunction(prediction, label)

        # backpropagation para calcular gradiente e otimizador

        # reseta gradiente
        optimizer.zero_grad()
        # calcula gradiente novo com backpropagation
        loss.backward()
        # atualiza pesos
        optimizer.step()

        totalLoss = loss.item()
        avgLoss = totalLoss / len(dataloader)
        print(f"Training Loss: {avgLoss:.4f}")


def testLoop(dataloader, model):
    # modo de avaliação
    model.eval()
    errors = 0
    wrongGuesses = []

    # sem gradientes para rodar mais rapido (inutil fora de treinamento)
    with torch.no_grad():
        for features, labels, names in dataloader:
            features = features.to(device).float()
            logits = model(features)
            # transforma logits em 0-1
            probabilities = torch.sigmoid(
                logits
            )
            # arredonda para 0 ou 1
            predictions = (probabilities >= 0.5).int()

            for name, probability, prediction, label in zip(names, probabilities, predictions, labels):
                print(
                    f"Species: {name.strip()} | Probability: {
                        probability.item():.4f} |"
                    f" Prediction: {prediction.item()}"
                )
                if label.item() != prediction:
                    errors += 1
                    wrongGuesses.append(name)
        print(f"errors: {errors}")
        print(wrongGuesses)


device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"
print(f"using {device} device")

model = NeuralNetwork().to(device)

lossFunction = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

trainingDataset = SpeciesDataset(
    data_path="./data/wpc50Training.dat",
    name_path="./data/NamesTraining.txt",
    is_training=True
)
testDataset = SpeciesDataset(
    data_path="./data/wpc50.dat",
    name_path="./data/NamesWithClass.txt",
    is_training=True
)

trainingLoader = DataLoader(trainingDataset, batch_size=8, shuffle=True)
testLoader = DataLoader(testDataset, batch_size=8, shuffle=True)

epochs = 40
for epoch in range(epochs):
    trainingLoop(trainingLoader, model, lossFunction, optimizer)

testLoop(testLoader, model)
