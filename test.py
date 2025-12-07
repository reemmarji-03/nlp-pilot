import numpy as np


x = [True, True, False]
x=np.array(x)
print(x.all())


y = np.arange(10)
print(np.any([y>5, y<2], axis=0))