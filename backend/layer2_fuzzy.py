import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class FuzzyEngagementSystem:
    # camada 2 — inferência fuzzy Mamdani
    # entradas: probabilidade positiva (saída do NB) e nota do aluno (1-5)
    # saída: score de engajamento de 0 a 10

    def __init__(self):
        self._build()

    def _build(self):
        # universos de discurso de cada variável
        u_sent = np.arange(0, 1.01, 0.01)
        u_rate = np.arange(1, 5.01, 0.1)
        u_eng  = np.arange(0, 10.01, 0.1)

        sentiment  = ctrl.Antecedent(u_sent, 'sentiment')
        rating     = ctrl.Antecedent(u_rate, 'rating')
        engagement = ctrl.Consequent(u_eng, 'engagement')

        # funções de pertinência do sentimento
        sentiment['negative'] = fuzz.trapmf(u_sent, [0.00, 0.00, 0.30, 0.50])
        sentiment['neutral']  = fuzz.trimf (u_sent, [0.30, 0.50, 0.70])
        sentiment['positive'] = fuzz.trapmf(u_sent, [0.50, 0.70, 1.00, 1.00])

        # funções de pertinência da nota
        rating['low']    = fuzz.trapmf(u_rate, [1.0, 1.0, 2.0, 2.5])
        rating['medium'] = fuzz.trimf (u_rate, [2.0, 3.0, 4.0])
        rating['high']   = fuzz.trapmf(u_rate, [3.5, 4.0, 5.0, 5.0])

        # funções de pertinência do engajamento
        engagement['low']    = fuzz.trapmf(u_eng, [0.0, 0.0, 2.0, 4.0])
        engagement['medium'] = fuzz.trimf (u_eng, [3.0, 5.0, 7.0])
        engagement['high']   = fuzz.trapmf(u_eng, [6.0, 8.0, 10.0, 10.0])

        # base de regras com 9 regras (cobre todo o espaço de entrada)
        rules = [
            ctrl.Rule(sentiment['positive'] & rating['high'],   engagement['high']),
            ctrl.Rule(sentiment['positive'] & rating['medium'], engagement['high']),
            ctrl.Rule(sentiment['positive'] & rating['low'],    engagement['medium']),
            ctrl.Rule(sentiment['neutral']  & rating['high'],   engagement['medium']),
            ctrl.Rule(sentiment['neutral']  & rating['medium'], engagement['medium']),
            ctrl.Rule(sentiment['neutral']  & rating['low'],    engagement['low']),
            ctrl.Rule(sentiment['negative'] & rating['high'],   engagement['medium']),
            ctrl.Rule(sentiment['negative'] & rating['medium'], engagement['low']),
            ctrl.Rule(sentiment['negative'] & rating['low'],    engagement['low']),
        ]

        system = ctrl.ControlSystem(rules)
        self.sim = ctrl.ControlSystemSimulation(system)

        self.universes = {
            'sentiment':  (u_sent, sentiment),
            'rating':     (u_rate, rating),
            'engagement': (u_eng,  engagement),
        }

    def compute(self, positive_prob: float, rating: float) -> dict:
        # clipa os valores para não estourar o universo do skfuzzy
        sp = float(np.clip(positive_prob, 0.01, 0.99))
        rt = float(np.clip(rating, 1.0, 5.0))

        self.sim.input['sentiment'] = sp
        self.sim.input['rating']    = rt
        self.sim.compute()

        score = float(self.sim.output['engagement'])

        if score >= 6.5:
            level = 'Alto'
        elif score >= 4.0:
            level = 'Médio'
        else:
            level = 'Baixo'

        return {
            'engagement_score': round(score, 2),
            'engagement_level': level,
            'inputs': {'positive_prob': round(sp, 3), 'rating': round(rt, 2)},
        }
