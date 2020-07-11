class Policy:
    def __init__(self, params):
        self.params = params
    
    def __call__(self, s):
        return self.params[0] * s + self.params[1]
