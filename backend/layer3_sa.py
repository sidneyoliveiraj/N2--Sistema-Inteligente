import math
import random


class StudySchedulerSA:
    """
    Camada III: Simulated Annealing para otimização do plano de estudos.

    Problema: dado N tópicos com seus scores de engajamento, encontrar
    a sequência que minimize o custo total do plano de estudos.

    Função de custo:
      - Penaliza aumentos bruscos de dificuldade (quedas no engajamento).
      - Penaliza início com tópico de baixo engajamento.
      - Incentiva término com o tópico mais desafiador.

    Vizinhança: troca de dois tópicos aleatórios (swap).
    """

    def __init__(
        self,
        initial_temp: float = 500.0,
        cooling_rate: float = 0.97,
        min_temp: float = 0.5,
        iters_per_temp: int = 60,
    ):
        self.initial_temp  = initial_temp
        self.cooling_rate  = cooling_rate
        self.min_temp      = min_temp
        self.iters_per_temp = iters_per_temp

    # ------------------------------------------------------------------
    # Função objetivo
    # ------------------------------------------------------------------

    def _cost(self, sequence: list[int], scores: list[float]) -> float:
        n = len(sequence)
        total = 0.0

        for i in range(n - 1):
            curr = scores[sequence[i]]
            nxt  = scores[sequence[i + 1]]
            diff = curr - nxt

            if diff > 0:
                # Queda gradual de engajamento → progressão normal de dificuldade
                total += diff * 0.5
            else:
                # Salto para cima depois de queda → sequência desconexa, penalidade maior
                total += abs(diff) * 2.0

        # Premia início com tópico de alto engajamento (motiva o aluno)
        total += (10.0 - scores[sequence[0]]) * 2.0
        # Penaliza levemente encerrar com tópico fácil (o mais desafiador deve ser último)
        total += scores[sequence[-1]] * 0.5

        return total

    # ------------------------------------------------------------------
    # Otimização SA
    # ------------------------------------------------------------------

    def optimize(self, topics: list[dict]) -> dict:
        n = len(topics)
        if n == 0:
            return {'sequence': [], 'initial_cost': 0, 'final_cost': 0, 'convergence': []}
        if n == 1:
            t = topics[0]
            return {
                'sequence': [{'name': t['name'], 'engagement_score': t['engagement_score'], 'position': 1}],
                'initial_cost': 0,
                'final_cost': 0,
                'convergence': [],
            }

        names  = [t['name']             for t in topics]
        scores = [t['engagement_score'] for t in topics]

        # Solução inicial: ordem decrescente de engajamento (greedy)
        current    = sorted(range(n), key=lambda i: scores[i], reverse=True)
        curr_cost  = self._cost(current, scores)
        best       = current[:]
        best_cost  = curr_cost
        initial_cost = curr_cost

        temp = self.initial_temp
        convergence: list[float] = []

        while temp > self.min_temp:
            for _ in range(self.iters_per_temp):
                i, j = random.sample(range(n), 2)
                neighbor = current[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                nb_cost = self._cost(neighbor, scores)
                delta   = nb_cost - curr_cost

                # Aceita solução melhor ou pior (com probabilidade de Boltzmann)
                if delta < 0 or random.random() < math.exp(-delta / temp):
                    current   = neighbor
                    curr_cost = nb_cost

                if curr_cost < best_cost:
                    best      = current[:]
                    best_cost = curr_cost

            convergence.append(round(curr_cost, 2))
            temp *= self.cooling_rate

        result_sequence = [
            {
                'name':             names[idx],
                'engagement_score': round(scores[idx], 2),
                'position':         pos + 1,
            }
            for pos, idx in enumerate(best)
        ]

        # Reduz a curva de convergência para ~20 pontos (visualização)
        step = max(1, len(convergence) // 20)
        convergence_sampled = convergence[::step]

        return {
            'sequence':     result_sequence,
            'initial_cost': round(initial_cost, 2),
            'final_cost':   round(best_cost, 2),
            'convergence':  convergence_sampled,
        }
