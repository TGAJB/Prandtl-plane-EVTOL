class test_class():
    def __init__(self, func):
        self.func = func
    def run_func(self):
        return self.func()

global_var = 0
def PLS():
    return global_var

HMM = test_class(PLS)

print(HMM.run_func())

global_var = 9

print(HMM.run_func())
