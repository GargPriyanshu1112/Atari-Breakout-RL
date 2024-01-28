class PolicyNetwork:
    def __init__(self, reg_const=0.01):
        self.a = reg_const


class ValueNetwork(PolicyNetwork):
    def __init__(self):
        super().__init__()


p = PolicyNetwork()
v = ValueNetwork()

print(v.a)
