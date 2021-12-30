import numpy as np

class Generation_Func(object):
    def __init__(self, n=100):
        self.n = n
        self.inputs_linera = None
        self.labels_linera = None
        self.inputs_xor = None
        self.labels_xor = None

    
    def generation_linear(self):
        pts = np.random.uniform(0, 1, (self.n, 2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0] - pt[1])/1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                 labels.append(1)

        return [np.array(inputs), np.array(labels).reshape(self.n, 1)]

    def generation_XOR_easy(self):
        inputs = []
        labels = []

        for i in range(11):
            inputs.append([0.1*i, 0.1*i])
            labels.append(0)

            if 0.1*i == 0.5:
                continue
            inputs.append([0.1*i, 1 - 0.1*i])
            labels.append(1)

        return [np.array(inputs), np.array(labels).reshape(len(labels), 1)]







