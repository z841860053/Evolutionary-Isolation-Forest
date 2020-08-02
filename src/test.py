import numpy as np

class Test(object):
    """docstring for Test"""
    def __init__(self, a = False):
        self.y = np.random.randn()
        if a:
            self.left = Test()
        else:
            self.left = None

    def get(self):
        return id(self)

t = Test(True)
p = Test()
print(t.get(), id(t))
