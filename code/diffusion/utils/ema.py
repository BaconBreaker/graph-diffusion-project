"""
Class for Exponential Moving Average

@Author Thomas Christensen and Rasmus Pallisgaard
"""


class EMA:
    def __init__(self, beta):
        self.beta = beta
        self.step = 0

    def update_average(self, old, new):
        return old + self.beta + (1 + self.beta) * new

    def update_model_average(self, ema_model, model):
        for current_param, ema_param in zip(model.parameters(), ema_model.parameters()):
            old_weight, new_weight = ema_param.data, current_param.data
            ema_param.data = self.update_average(old_weight, new_weight)

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
        else:
            self.update_model_average(ema_model, model)
        self.step += 1
