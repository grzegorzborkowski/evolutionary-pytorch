import Models
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

class ModelEvaluator():

    # TODO: refactor is so that is has fewer params
    @staticmethod
    def evalOneMax(individual, number_of_epochs, learning_rate, input_layer_size, output_layer_size, hidden_size,
                   train_X, test_X, train_y, test_y):
        print("Evaluatio individual", individual)
        model_factory = Models.ModelFactory(input_layer_size, output_layer_size, hidden_size)
        model = model_factory.get_model(individual)
        loss_fn = nn.CrossEntropyLoss()

        for t in range(number_of_epochs):
            y_pred = model(train_X)
            loss = loss_fn(y_pred, train_y)

            if t % 25 == 0:
                pass
                # print (t, loss.item())

            model.zero_grad()
            loss.backward()

            with torch.no_grad():
                for param in model.parameters():
                    param.data -= learning_rate * param.grad

        predict_out = model(test_X)
        _, predict_y = torch.max(predict_out, 1)
        accuracy = accuracy_score(test_y, predict_y)
        print(accuracy)
        return accuracy,