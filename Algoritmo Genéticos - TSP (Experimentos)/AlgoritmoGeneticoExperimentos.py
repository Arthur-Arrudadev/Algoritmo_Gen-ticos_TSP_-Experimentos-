import random
import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Classe TSP13 (mantida igual)
class TSP13:
    def __init__(self, matriz_distancias: List[List[int]]):
        self.matriz_distancias = matriz_distancias
        self.num_cidades = len(matriz_distancias)
        self.nomes_cidades = [
            "New York", "Los Angeles", "Chicago", "Minneapolis", "Denver",
            "Dallas", "Seattle", "Boston", "San Francisco", "St. Louis",
            "Houston", "Phoenix", "Salt Lake City"
        ]
        
    def calcular_distancia(self, rota: List[int]) -> float:
        distancia_total = 0
        for i in range(len(rota) - 1):
            cidade_atual = rota[i]
            proxima_cidade = rota[i + 1]
            distancia_total += self.matriz_distancias[cidade_atual][proxima_cidade]
        distancia_total += self.matriz_distancias[rota[-1]][rota[0]]
        return distancia_total
    
    def calcular_fitness(self, rota: List[int]) -> float:
        distancia = self.calcular_distancia(rota)
        if distancia == 0:
            return float('inf')
        return 1.0 / distancia
    
    def validar_rota(self, rota: List[int]) -> bool:
        if len(rota) != self.num_cidades:
            return False
        cidades_visitadas = set(rota)
        if len(cidades_visitadas) != self.num_cidades:
            return False
        if min(rota) < 0 or max(rota) >= self.num_cidades:
            return False
        if len(rota) != len(set(rota)):
            return False
        return True
    
    def gerar_rota_aleatoria(self) -> List[int]:
        todas_cidades = list(range(self.num_cidades))
        cidades_embaralhadas = todas_cidades.copy()
        random.shuffle(cidades_embaralhadas)
        return cidades_embaralhadas

# Matriz de distancias
USA13 = [
    [0, 2451, 713, 1018, 1631, 1374, 2408, 213, 2571, 875, 1420, 2145, 1972],
    [2451, 0, 1745, 1524, 831, 1240, 959, 2596, 403, 1589, 1374, 357, 579],
    [713, 1745, 0, 355, 920, 803, 1737, 851, 1858, 262, 940, 1453, 1260],
    [1018, 1524, 355, 0, 700, 862, 1395, 1123, 1584, 466, 1056, 1280, 987],
    [1631, 831, 920, 700, 0, 663, 1021, 1769, 949, 796, 879, 586, 371],
    [1374, 1240, 803, 862, 663, 0, 1681, 1551, 1765, 547, 225, 887, 999],
    [2408, 959, 1737, 1395, 1021, 1681, 0, 2493, 678, 1724, 1891, 1114, 701],
    [213, 2596, 851, 1123, 1769, 1551, 2493, 0, 2699, 1038, 1605, 2300, 2099],
    [2571, 403, 1858, 1584, 949, 1765, 678, 2699, 0, 1744, 1645, 653, 600],
    [875, 1589, 262, 466, 796, 547, 1724, 1038, 1744, 0, 679, 1272, 1162],
    [1420, 1374, 940, 1056, 879, 225, 1891, 1605, 1645, 679, 0, 1017, 1200],
    [2145, 357, 1453, 1280, 586, 887, 1114, 2300, 653, 1272, 1017, 0, 504],
    [1972, 579, 1260, 987, 371, 999, 701, 2099, 600, 1162, 1200, 504, 0],
]

# Classe AlgoritmoGeneticoTSP (mantida igual)
class AlgoritmoGeneticoTSP:
    def __init__(self, tsp: TSP13, tamanho_populacao: int = 50, 
                 geracoes: int = 400, taxa_crossover: float = 0.9,
                 taxa_mutacao: float = 0.05, elitismo: int = 5,
                 tamanho_torneio: int = 3):
        self.tsp = tsp
        self.tamanho_populacao = tamanho_populacao
        self.geracoes = geracoes
        self.taxa_crossover = taxa_crossover
        self.taxa_mutacao = taxa_mutacao
        self.elitismo = elitismo
        self.tamanho_torneio = tamanho_torneio
    
    def criar_populacao(self) -> List[List[int]]:
        return [self.tsp.gerar_rota_aleatoria() for _ in range(self.tamanho_populacao)]
    
    def calcular_fitness_populacao(self, populacao: List[List[int]]) -> List[float]:
        return [self.tsp.calcular_fitness(rota) for rota in populacao]
    
    def selecao_torneio(self, populacao: List[List[int]], fitness: List[float], k: int = None) -> List[int]:
        if k is None:
            k = self.tamanho_torneio
        indices = random.sample(range(len(populacao)), k)
        melhor_idx = max(indices, key=lambda i: fitness[i])
        return populacao[melhor_idx]
    
    def crossover_OX(self, pai1: List[int], pai2: List[int]) -> Tuple[List[int], List[int]]:
        size = len(pai1)
        ponto1 = random.randint(0, size - 2)
        ponto2 = random.randint(ponto1 + 1, size - 1)
        
        filho1 = [-1] * size
        filho2 = [-1] * size
        
        filho1[ponto1:ponto2] = pai1[ponto1:ponto2]
        filho2[ponto1:ponto2] = pai2[ponto1:ponto2]
        
        self._preencher_filho_OX(filho1, pai2, ponto1, ponto2)
        self._preencher_filho_OX(filho2, pai1, ponto1, ponto2)
        
        return filho1, filho2
    
    def _preencher_filho_OX(self, filho: List[int], pai: List[int], ponto1: int, ponto2: int):
        size = len(filho)
        current_pos = ponto2
        
        for i in range(ponto2, size):
            cidade = pai[i]
            if cidade not in filho:
                filho[current_pos % size] = cidade
                current_pos += 1
        
        for i in range(0, ponto2):
            cidade = pai[i]
            if cidade not in filho:
                filho[current_pos % size] = cidade
                current_pos += 1
    
    def mutacao_troca(self, rota: List[int]) -> List[int]:
        mutado = rota.copy()
        if random.random() < self.taxa_mutacao:
            i, j = random.sample(range(len(rota)), 2)
            mutado[i], mutado[j] = mutado[j], mutado[i]
        return mutado
    
    def contar_individuos_unicos(self, populacao: List[List[int]]) -> int:
        rotas_unicas = set(tuple(rota) for rota in populacao)
        return len(rotas_unicas)
    
    def executar_ag(self, monitorar_diversidade: bool = False) -> Tuple[List[int], float, List[float], List[float]]:
        populacao = self.criar_populacao()
        fitness = self.calcular_fitness_populacao(populacao)
        
        historico_melhor_fitness = []
        historico_diversidade = []
        melhor_fitness_global = max(fitness)
        melhor_rota_global = populacao[fitness.index(melhor_fitness_global)]
        
        for geracao in range(self.geracoes):
            nova_populacao = []
            
            if self.elitismo > 0:
                indices_ordenados = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
                for i in range(min(self.elitismo, len(indices_ordenados))):
                    nova_populacao.append(populacao[indices_ordenados[i]])
            
            while len(nova_populacao) < self.tamanho_populacao:
                pai1 = self.selecao_torneio(populacao, fitness)
                pai2 = self.selecao_torneio(populacao, fitness)
                
                if random.random() < self.taxa_crossover:
                    filho1, filho2 = self.crossover_OX(pai1, pai2)
                else:
                    filho1, filho2 = pai1.copy(), pai2.copy()
                
                filho1 = self.mutacao_troca(filho1)
                filho2 = self.mutacao_troca(filho2)
                
                nova_populacao.extend([filho1, filho2])
            
            populacao = nova_populacao[:self.tamanho_populacao]
            fitness = self.calcular_fitness_populacao(populacao)
            
            melhor_fitness_atual = max(fitness)
            if melhor_fitness_atual > melhor_fitness_global:
                melhor_fitness_global = melhor_fitness_atual
                melhor_rota_global = populacao[fitness.index(melhor_fitness_atual)]
            
            historico_melhor_fitness.append(melhor_fitness_global)
            
            if monitorar_diversidade:
                diversidade = self.contar_individuos_unicos(populacao)
                historico_diversidade.append(diversidade)
        
        return melhor_rota_global, melhor_fitness_global, historico_melhor_fitness, historico_diversidade

# Funcao executar_experimentos (mantida igual)
def executar_experimentos():
    tsp = TSP13(USA13)
    
    config_base = {
        'tamanho_populacao': 50,
        'geracoes': 400,
        'taxa_crossover': 0.9,
        'taxa_mutacao': 0.05,
        'elitismo': 5,
        'tamanho_torneio': 3
    }
    
    resultados_experimentos = {}
    
    # EXPERIMENTO 1: Tamanho da Populacao
    print("EXPERIMENTO 1: Tamanho da Populacao")
    tamanhos_populacao = [20, 50, 100]
    resultados_experimento1 = {}
    
    for tamanho in tamanhos_populacao:
        print(f"  Testando tamanho de populacao: {tamanho}")
        fitness_finais = []
        tempos_execucao = []
        historicos_convergencia = []
        
        for execucao in range(10):  # Reduzido para 10 execucoes para ser mais rapido
            inicio = time.time()
            ag = AlgoritmoGeneticoTSP(tsp, tamanho_populacao=tamanho, **{k: v for k, v in config_base.items() if k != 'tamanho_populacao'})
            melhor_rota, melhor_fitness, historico, _ = ag.executar_ag()
            fim = time.time()
            
            fitness_finais.append(melhor_fitness)
            tempos_execucao.append(fim - inicio)
            historicos_convergencia.append(historico)
        
        resultados_experimento1[tamanho] = {
            'fitness_finais': fitness_finais,
            'tempos_execucao': tempos_execucao,
            'historicos_convergencia': historicos_convergencia
        }
    
    resultados_experimentos['populacao'] = resultados_experimento1
    
    # EXPERIMENTO 2: Taxa de Mutacao
    print("\nEXPERIMENTO 2: Taxa de Mutacao")
    taxas_mutacao = [0.01, 0.05, 0.1, 0.2]
    resultados_experimento2 = {}
    
    for taxa in taxas_mutacao:
        print(f"  Testando taxa de mutacao: {taxa*100}%")
        fitness_finais = []
        historicos_convergencia = []
        
        for execucao in range(10):
            ag = AlgoritmoGeneticoTSP(tsp, taxa_mutacao=taxa, **{k: v for k, v in config_base.items() if k != 'taxa_mutacao'})
            melhor_rota, melhor_fitness, historico, _ = ag.executar_ag()
            
            fitness_finais.append(melhor_fitness)
            historicos_convergencia.append(historico)
        
        resultados_experimento2[taxa] = {
            'fitness_finais': fitness_finais,
            'historicos_convergencia': historicos_convergencia
        }
    
    resultados_experimentos['mutacao'] = resultados_experimento2
    
    # EXPERIMENTO 3: Tamanho do Torneio
    print("\nEXPERIMENTO 3: Tamanho do Torneio")
    tamanhos_torneio = [2, 3, 5, 7]
    resultados_experimento3 = {}
    
    for tamanho in tamanhos_torneio:
        print(f"  Testando tamanho de torneio: {tamanho}")
        fitness_finais = []
        historicos_convergencia = []
        historicos_diversidade = []
        
        for execucao in range(10):
            ag = AlgoritmoGeneticoTSP(tsp, tamanho_torneio=tamanho, **{k: v for k, v in config_base.items() if k != 'tamanho_torneio'})
            melhor_rota, melhor_fitness, historico, diversidade = ag.executar_ag(monitorar_diversidade=True)
            
            fitness_finais.append(melhor_fitness)
            historicos_convergencia.append(historico)
            historicos_diversidade.append(diversidade)
        
        resultados_experimento3[tamanho] = {
            'fitness_finais': fitness_finais,
            'historicos_convergencia': historicos_convergencia,
            'historicos_diversidade': historicos_diversidade
        }
    
    resultados_experimentos['torneio'] = resultados_experimento3
    
    # EXPERIMENTO 4: Elitismo
    print("\nEXPERIMENTO 4: Elitismo")
    valores_elitismo = [0, 1, 5, 10]
    resultados_experimento4 = {}
    
    for elitismo_pct in valores_elitismo:
        elitismo = max(1, int(config_base['tamanho_populacao'] * elitismo_pct / 100))
        print(f"  Testando elitismo: {elitismo_pct}% ({elitismo} individuos)")
        fitness_finais = []
        historicos_convergencia = []
        
        for execucao in range(10):
            ag = AlgoritmoGeneticoTSP(tsp, elitismo=elitismo, **{k: v for k, v in config_base.items() if k != 'elitismo'})
            melhor_rota, melhor_fitness, historico, _ = ag.executar_ag()
            
            fitness_finais.append(melhor_fitness)
            historicos_convergencia.append(historico)
        
        resultados_experimento4[elitismo_pct] = {
            'fitness_finais': fitness_finais,
            'historicos_convergencia': historicos_convergencia
        }
    
    resultados_experimentos['elitismo'] = resultados_experimento4
    
    return resultados_experimentos

# FUNCAO SIMPLIFICADA APENAS PARA GRAFICOS
def gerar_graficos_experimentos(resultados_experimentos):
    # Configuracao de estilo
    plt.style.use('default')
    
    # Paleta de cores
    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # ===== EXPERIMENTO 1: TAMANHO POPULACAO =====
    plt.figure(figsize=(15, 5))
    resultados_pop = resultados_experimentos['populacao']
    
    # Grafico 1: Convergencia
    plt.subplot(1, 3, 1)
    for i, (tamanho, dados) in enumerate(resultados_pop.items()):
        historico_medio = np.mean(dados['historicos_convergencia'], axis=0)
        plt.plot(historico_medio, label=f'Pop={tamanho}', color=cores[i], linewidth=2)
    plt.title('Convergencia - Tamanho Populacao')
    plt.xlabel('Geracao')
    plt.ylabel('Melhor Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Grafico 2: Boxplot
    plt.subplot(1, 3, 2)
    dados_boxplot = [dados['fitness_finais'] for dados in resultados_pop.values()]
    box = plt.boxplot(dados_boxplot, patch_artist=True, widths=0.6)
    for patch, color in zip(box['boxes'], cores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xticks([1, 2, 3], [f'Pop={t}' for t in resultados_pop.keys()])
    plt.title('Fitness Final - Tamanho Populacao')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    
    # Grafico 3: Tempo
    plt.subplot(1, 3, 3)
    tempos_medios = [np.mean(dados['tempos_execucao']) for dados in resultados_pop.values()]
    plt.bar([f'Pop={t}' for t in resultados_pop.keys()], tempos_medios, 
            color=cores[:len(tempos_medios)], alpha=0.7)
    plt.title('Tempo Medio de Execucao')
    plt.ylabel('Tempo (segundos)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ===== EXPERIMENTO 2: TAXA MUTACAO =====
    plt.figure(figsize=(10, 5))
    resultados_mut = resultados_experimentos['mutacao']
    
    plt.subplot(1, 2, 1)
    for i, (taxa, dados) in enumerate(resultados_mut.items()):
        historico_medio = np.mean(dados['historicos_convergencia'], axis=0)
        plt.plot(historico_medio, label=f'Mut={taxa*100}%', color=cores[i], linewidth=2)
    plt.title('Convergencia - Taxa Mutacao')
    plt.xlabel('Geracao')
    plt.ylabel('Melhor Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    dados_boxplot = [dados['fitness_finais'] for dados in resultados_mut.values()]
    box = plt.boxplot(dados_boxplot, patch_artist=True, widths=0.6)
    for patch, color in zip(box['boxes'], cores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xticks([1, 2, 3, 4], [f'Mut={t*100}%' for t in resultados_mut.keys()])
    plt.title('Fitness Final - Taxa Mutacao')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ===== EXPERIMENTO 3: TAMANHO TORNEIO =====
    plt.figure(figsize=(15, 5))
    resultados_torn = resultados_experimentos['torneio']
    
    plt.subplot(1, 3, 1)
    for i, (tamanho, dados) in enumerate(resultados_torn.items()):
        historico_medio = np.mean(dados['historicos_convergencia'], axis=0)
        plt.plot(historico_medio, label=f'Torn={tamanho}', color=cores[i], linewidth=2)
    plt.title('Convergencia - Tamanho Torneio')
    plt.xlabel('Geracao')
    plt.ylabel('Melhor Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    dados_boxplot = [dados['fitness_finais'] for dados in resultados_torn.values()]
    box = plt.boxplot(dados_boxplot, patch_artist=True, widths=0.6)
    for patch, color in zip(box['boxes'], cores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xticks([1, 2, 3, 4], [f'Torn={t}' for t in resultados_torn.keys()])
    plt.title('Fitness Final - Tamanho Torneio')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    for i, (tamanho, dados) in enumerate(resultados_torn.items()):
        diversidade_media = np.mean(dados['historicos_diversidade'], axis=0)
        plt.plot(diversidade_media, label=f'Torn={tamanho}', color=cores[i], linewidth=2)
    plt.title('Diversidade - Tamanho Torneio')
    plt.xlabel('Geracao')
    plt.ylabel('Individuos Unicos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # ===== EXPERIMENTO 4: ELITISMO =====
    plt.figure(figsize=(12, 5))
    resultados_elit = resultados_experimentos['elitismo']
    
    plt.subplot(1, 2, 1)
    for i, (pct, dados) in enumerate(resultados_elit.items()):
        historico_medio = np.mean(dados['historicos_convergencia'], axis=0)
        plt.plot(historico_medio, label=f'Elit={pct}%', color=cores[i], linewidth=2)
    plt.title('Convergencia - Elitismo')
    plt.xlabel('Geracao')
    plt.ylabel('Melhor Fitness')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    dados_boxplot = [dados['fitness_finais'] for dados in resultados_elit.values()]
    box = plt.boxplot(dados_boxplot, patch_artist=True, widths=0.6)
    for patch, color in zip(box['boxes'], cores):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    plt.xticks([1, 2, 3, 4], [f'Elit={e}%' for e in resultados_elit.keys()])
    plt.title('Fitness Final - Elitismo')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Execucao principal simplificada
if __name__ == "__main__":
    print("INICIANDO ANALISE DE PARAMETROS DO ALGORITMO GENETICO")
    print("="*60)
    
    # Executa experimentos
    resultados = executar_experimentos()
    
    # Gera apenas os graficos
    gerar_graficos_experimentos(resultados)
    
    print("\n" + "="*80)
    print("GRAFICOS GERADOS COM SUCESSO")
    print("="*80)