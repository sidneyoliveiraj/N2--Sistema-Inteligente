import math
import random


class StudySchedulerSA:
    # camada 3 — Simulated Annealing para ordenar os tópicos na melhor sequência de estudo
    # a ideia é minimizar o custo total, penalizando sequências que variam muito de engajamento

    def __init__(
        self,
        initial_temp: float = 500.0,
        cooling_rate: float = 0.97,
        min_temp: float = 0.5,
        iters_per_temp: int = 60,
    ):
        self.initial_temp   = initial_temp
        self.cooling_rate   = cooling_rate
        self.min_temp       = min_temp
        self.iters_per_temp = iters_per_temp

    def _cost(self, sequence: list[int], scores: list[float]) -> float:
        n = len(sequence)
        total = 0.0

        for i in range(n - 1):
            curr = scores[sequence[i]]
            nxt  = scores[sequence[i + 1]]
            diff = curr - nxt

            if diff > 0:
                # queda gradual de engajamento é ok
                total += diff * 0.5
            else:
                # subida depois de queda é ruim — sequência desconexa
                total += abs(diff) * 2.0

        # penaliza começar com tópico de baixo engajamento
        total += (10.0 - scores[sequence[0]]) * 2.0
        # penaliza terminar com tópico muito fácil
        total += scores[sequence[-1]] * 0.5

        return total

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

        # começa com a sequência ordenada por engajamento decrescente
        current      = sorted(range(n), key=lambda i: scores[i], reverse=True)
        curr_cost    = self._cost(current, scores)
        best         = current[:]
        best_cost    = curr_cost
        initial_cost = curr_cost

        temp = self.initial_temp
        convergence: list[float] = []

        while temp > self.min_temp:
            for _ in range(self.iters_per_temp):
                # gera vizinho trocando dois tópicos de posição
                i, j = random.sample(range(n), 2)
                neighbor = current[:]
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

                nb_cost = self._cost(neighbor, scores)
                delta   = nb_cost - curr_cost

                # aceita o vizinho se for melhor ou pela probabilidade de Boltzmann
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

        # reduz para ~20 pontos para facilitar a visualização no frontend
        step = max(1, len(convergence) // 20)
        convergence_sampled = convergence[::step]

        return {
            'sequence':     result_sequence,
            'initial_cost': round(initial_cost, 2),
            'final_cost':   round(best_cost, 2),
            'convergence':  convergence_sampled,
        }
