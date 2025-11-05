# Importacao das bibliotecas necessarias
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Classe principal para resolver o Problema do Caixeiro Viajante (TSP)
class TSP13:
    
    # Metodo construtor da classe TSP13
    def __init__(self, matriz_distancias: List[List[int]]):
        # Armazena a matriz de distancias entre as cidades
        self.matriz_distancias = matriz_distancias
        # Define o numero total de cidades no problema
        self.num_cidades = len(matriz_distancias)
        # Lista com os nomes das 13 cidades americanas
        self.nomes_cidades = [
            "New York", "Los Angeles", "Chicago", "Minneapolis", "Denver",
            "Dallas", "Seattle", "Boston", "San Francisco", "St. Louis",
            "Houston", "Phoenix", "Salt Lake City"
        ]
        
    # Funcao para calcular a distancia total de uma rota
    def calcular_distancia(self, rota: List[int]) -> float:
        # Inicializa a distancia total como zero
        distancia_total = 0
        
        # Loop para calcular distancias entre cidades consecutivas na rota
        for i in range(len(rota) - 1):
            # Obtem o indice da cidade atual
            cidade_atual = rota[i]
            # Obtem o indice da proxima cidade
            proxima_cidade = rota[i + 1]
            # Adiciona a distancia entre cidade atual e proxima cidade
            distancia_total += self.matriz_distancias[cidade_atual][proxima_cidade]
        
        # Adiciona a distancia de retorno para a cidade inicial (fecha o ciclo)
        distancia_total += self.matriz_distancias[rota[-1]][rota[0]]
        
        # Retorna a distancia total calculada
        return distancia_total
    
    # Funcao para calcular o fitness de uma rota
    def calcular_fitness(self, rota: List[int]) -> float:
        # Calcula a distancia total da rota
        distancia = self.calcular_distancia(rota)
        
        # Verifica se a distancia e zero para evitar divisao por zero
        if distancia == 0:
            # Retorna infinito se distancia for zero
            return float('inf')
        # Retorna o inverso da distancia (quanto menor a distancia, maior o fitness)
        return 1.0 / distancia
    
    # Funcao para validar se uma rota e valida
    def validar_rota(self, rota: List[int]) -> bool:
        # Verifica se a rota tem o numero correto de cidades
        if len(rota) != self.num_cidades:
            return False
        
        # Cria um conjunto com as cidades visitadas na rota
        cidades_visitadas = set(rota)
        
        # Verifica se todas as cidades foram visitadas
        if len(cidades_visitadas) != self.num_cidades:
            return False
        
        # Verifica se os indices das cidades estao no intervalo valido
        if min(rota) < 0 or max(rota) >= self.num_cidades:
            return False
        
        # Verifica se nao ha cidades duplicadas na rota
        if len(rota) != len(set(rota)):
            return False
        
        # Se passou por todas as verificacoes, a rota e valida
        return True
    
    # Funcao para gerar uma rota aleatoria valida
    def gerar_rota_aleatoria(self) -> List[int]:
        # Cria uma lista com todas as cidades (indices de 0 a 12)
        todas_cidades = list(range(self.num_cidades))
        
        # Faz uma copia da lista de cidades
        cidades_embaralhadas = todas_cidades.copy()
        # Embaralha aleatoriamente a ordem das cidades
        random.shuffle(cidades_embaralhadas)
        
        # Retorna a rota aleatoria gerada
        return cidades_embaralhadas

# Matriz de distancias entre as 13 cidades americanas em quilometros
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

# Classe que implementa o Algoritmo Genetico para resolver o TSP
class AlgoritmoGeneticoTSP:
    
    # Metodo construtor do algoritmo genetico
    def __init__(self, tsp: TSP13, tamanho_populacao: int = 50, 
                 geracoes: int = 400, taxa_crossover: float = 0.9,
                 taxa_mutacao: float = 0.05, elitismo: int = 5):
        # Armazena a instancia do problema TSP
        self.tsp = tsp
        # Define o tamanho da populacao
        self.tamanho_populacao = tamanho_populacao
        # Define o numero de geracoes a serem executadas
        self.geracoes = geracoes
        # Define a probabilidade de crossover
        self.taxa_crossover = taxa_crossover
        # Define a probabilidade de mutacao
        self.taxa_mutacao = taxa_mutacao
        # Define o numero de individuos elitistas
        self.elitismo = elitismo
    
    # Funcao para criar a populacao inicial
    def criar_populacao(self) -> List[List[int]]:
        # Cria uma lista com rotas aleatorias
        return [self.tsp.gerar_rota_aleatoria() for _ in range(self.tamanho_populacao)]
    
    # Funcao para calcular o fitness de toda a populacao
    def calcular_fitness_populacao(self, populacao: List[List[int]]) -> List[float]:
        # Calcula o fitness para cada rota na populacao
        return [self.tsp.calcular_fitness(rota) for rota in populacao]
    
    # Funcao de selecao por torneio
    def selecao_torneio(self, populacao: List[List[int]], fitness: List[float], k: int = 3) -> List[int]:
        # Seleciona k indices aleatorios da populacao
        indices = random.sample(range(len(populacao)), k)
        # Encontra o indice com maior fitness entre os selecionados
        melhor_idx = max(indices, key=lambda i: fitness[i])
        # Retorna o individuo correspondente ao melhor fitness
        return populacao[melhor_idx]
    
    # Funcao de crossover Order Crossover (OX) para TSP
    def crossover_OX(self, pai1: List[int], pai2: List[int]) -> Tuple[List[int], List[int]]:
        # Obtem o tamanho dos pais
        size = len(pai1)
        
        # Escolhe aleatoriamente o primeiro ponto de corte
        ponto1 = random.randint(0, size - 2)
        # Escolhe aleatoriamente o segundo ponto de corte (apos o primeiro)
        ponto2 = random.randint(ponto1 + 1, size - 1)
        
        # Inicializa os filhos com valores vazios (-1)
        filho1 = [-1] * size
        filho2 = [-1] * size
        
        # Copia o segmento entre os pontos de corte do pai1 para o filho1
        filho1[ponto1:ponto2] = pai1[ponto1:ponto2]
        # Copia o segmento entre os pontos de corte do pai2 para o filho2
        filho2[ponto1:ponto2] = pai2[ponto1:ponto2]
        
        # Preenche o restante do filho1 com elementos do pai2
        self._preencher_filho_OX(filho1, pai2, ponto1, ponto2)
        # Preenche o restante do filho2 com elementos do pai1
        self._preencher_filho_OX(filho2, pai1, ponto1, ponto2)
        
        # Retorna os dois filhos gerados
        return filho1, filho2
    
    # Funcao auxiliar para preencher os filhos no crossover OX
    def _preencher_filho_OX(self, filho: List[int], pai: List[int], ponto1: int, ponto2: int):
        # Obtem o tamanho do filho
        size = len(filho)
        # Inicia a posicao de preenchimento a partir do ponto2
        current_pos = ponto2
        
        # Percorre o pai do ponto2 ate o final
        for i in range(ponto2, size):
            cidade = pai[i]
            # Se a cidade nao esta no filho, adiciona na posicao atual
            if cidade not in filho:
                filho[current_pos % size] = cidade
                current_pos += 1
        
        # Percorre o pai do inicio ate o ponto2
        for i in range(0, ponto2):
            cidade = pai[i]
            # Se a cidade nao esta no filho, adiciona na posicao atual
            if cidade not in filho:
                filho[current_pos % size] = cidade
                current_pos += 1
    
    # Funcao de mutacao por troca de duas cidades
    def mutacao_troca(self, rota: List[int]) -> List[int]:
        # Cria uma copia da rota para nao modificar a original
        mutado = rota.copy()
        
        # Verifica se a mutacao deve ser aplicada baseado na taxa de mutacao
        if random.random() < self.taxa_mutacao:
            # Seleciona duas posicoes aleatorias diferentes
            i, j = random.sample(range(len(rota)), 2)
            # Troca as cidades nas posicoes selecionadas

            mutado[i], mutado[j] = mutado[j], mutado[i]
        
        # Retorna a rota mutada (ou original se nao houve mutacao)
        return mutado
    
    # Funcao principal que executa o algoritmo genetico
    def executar_ag(self) -> Tuple[List[int], float, List[float]]:
        # Cria a populacao inicial
        populacao = self.criar_populacao()
        # Calcula o fitness da populacao inicial
        fitness = self.calcular_fitness_populacao(populacao)
        
        # Inicializa o historico do melhor fitness
        historico_melhor_fitness = []
        # Encontra o melhor fitness inicial
        melhor_fitness_global = max(fitness)
        # Encontra a melhor rota inicial
        melhor_rota_global = populacao[fitness.index(melhor_fitness_global)]
        
        # Loop principal sobre as geracoes
        for geracao in range(self.geracoes):
            # Inicializa a nova populacao
            nova_populacao = []
            
            # Aplica elitismo: mantem os melhores individuos
            # Ordena os indices pelo fitness em ordem decrescente
            indices_ordenados = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
            # Adiciona os melhores individuos a nova populacao
            for i in range(self.elitismo):
                nova_populacao.append(populacao[indices_ordenados[i]])
            
            # Preenche o restante da nova populacao

            while len(nova_populacao) < self.tamanho_populacao:
                # Seleciona dois pais usando torneio
                pai1 = self.selecao_torneio(populacao, fitness)
                pai2 = self.selecao_torneio(populacao, fitness)
                
                # Aplica crossover com probabilidade definida
                if random.random() < self.taxa_crossover:
                    # Gera dois filhos usando crossover OX
                    filho1, filho2 = self.crossover_OX(pai1, pai2)
                else:
                    # Se nao houve crossover, os filhos sao copias dos pais
                    filho1, filho2 = pai1.copy(), pai2.copy()
                
                # Aplica mutacao por troca nos filhos
                filho1 = self.mutacao_troca(filho1)
                filho2 = self.mutacao_troca(filho2)
                
                # Adiciona os filhos a nova populacao
                nova_populacao.extend([filho1, filho2])
            
            # Mantem o tamanho correto da populacao
            populacao = nova_populacao[:self.tamanho_populacao]
            # Recalcula o fitness da nova populacao

            fitness = self.calcular_fitness_populacao(populacao)
            
            # Encontra o melhor fitness da geracao atual
            melhor_fitness_atual = max(fitness)
            # Atualiza o melhor global se necessario
            if melhor_fitness_atual > melhor_fitness_global:
                melhor_fitness_global = melhor_fitness_atual
                melhor_rota_global = populacao[fitness.index(melhor_fitness_atual)]
            
            # Armazena o melhor fitness no historico
            historico_melhor_fitness.append(melhor_fitness_global)
        
        # Retorna a melhor rota, melhor fitness e historico
        return melhor_rota_global, melhor_fitness_global, historico_melhor_fitness

# Bloco principal de execucao do programa
if __name__ == "__main__":
    # Cabecalho do programa
    print("=" * 80)
    print("IMPLEMENTACAO AG PARA TSP - CONFIGURACAO PADRAO")
    print("=" * 80)
    
    # Configuracao padrao conforme especificado
    print("CONFIGURACAO PADRAO:")
    print("- Populacao: 50 individuos")
    print("- Geracoes: 400")
    print("- Selecao: Torneio (tamanho 3)")
    print("- Crossover: Order Crossover (OX) - taxa 90%")
    print("- Mutacao: Swap - taxa 5%")
    print("- Elitismo: manter os 5 melhores")
    print()
    
    # Cria uma instancia do problema TSP com a matriz de distancias
    tsp = TSP13(USA13)
    
    # Executa 30 vezes o algoritmo genetico para analise estatistica
    print("EXECUTANDO 30 EXECUCOES DO ALGORITMO GENETICO...")
    print("-" * 50)
    
    # Listas para armazenar os resultados das execucoes
    resultados_fitness = []
    resultados_distancia = []
    historicos_convergencia = []
    
    # Executa o algoritmo genetico 30 vezes
    for execucao in range(30):
        # Cria nova instancia com configuracao padrao
        ag_tsp = AlgoritmoGeneticoTSP(
            tsp=tsp,
            tamanho_populacao=50,
            geracoes=400,
            taxa_crossover=0.9,
            taxa_mutacao=0.05,
            elitismo=5
        )
        
        # Executa o algoritmo
        melhor_rota, melhor_fitness, historico = ag_tsp.executar_ag()
        # Calcula a distancia da melhor rota
        melhor_distancia = 1.0 / melhor_fitness
        
        # Armazena os resultados
        resultados_fitness.append(melhor_fitness)
        resultados_distancia.append(melhor_distancia)
        historicos_convergencia.append(historico)
        
        # Exibe o resultado da execucao atual
        print(f"Execucao {execucao + 1:2d}: Fitness = {melhor_fitness:.6f}, Distancia = {melhor_distancia:.0f} km")
    
    # 1. Calcula a media e o desvio padrao da qualidade (fitness)
    media_fitness = np.mean(resultados_fitness)
    desvio_padrao_fitness = np.std(resultados_fitness)
    media_distancia = np.mean(resultados_distancia)
    desvio_padrao_distancia = np.std(resultados_distancia)
    
    print("\n" + "=" * 80)
    print("RESULTADOS ESTATISTICOS - 30 EXECUCOES")
    print("=" * 80)
    
    print(f"FITNESS (qualidade da solucao):")
    print(f"Media: {media_fitness:.6f}")
    print(f"Desvio Padrao: {desvio_padrao_fitness:.6f}")
    print(f"Melhor Fitness: {max(resultados_fitness):.6f}")
    print(f"Pior Fitness: {min(resultados_fitness):.6f}")
    
    print(f"\nDISTANCIA (km):")
    print(f"Media: {media_distancia:.0f} km")
    print(f"Desvio Padrao: {desvio_padrao_distancia:.0f} km")
    print(f"Melhor Distancia: {min(resultados_distancia):.0f} km")
    print(f"Pior Distancia: {max(resultados_distancia):.0f} km")
    
    # 2. Grafico de Convergencia do AG (melhor fitness por iteracao)
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Grafico de convergencia (todas as execucoes)
    plt.subplot(1, 3, 1)
    for i, historico in enumerate(historicos_convergencia):
        plt.plot(historico, alpha=0.3, linewidth=0.5)
    
    # Plota a media da convergencia
    historico_medio = np.mean(historicos_convergencia, axis=0)
    plt.plot(historico_medio, 'r-', linewidth=2, label='Media das execucoes')
    plt.title('Convergencia do Algoritmo Genetico\n(Melhor Fitness por Geracao)', fontsize=12, fontweight='bold')
    plt.xlabel('Geracao')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Subplot 2: Boxplot dos resultados finais de fitness
    plt.subplot(1, 3, 2)
    plt.boxplot(resultados_fitness)
    plt.title('Distribuicao dos Fitness Finais', fontsize=12, fontweight='bold')
    plt.ylabel('Fitness')
    plt.grid(True, alpha=0.3)
    
    # Subplot 3: Boxplot das distancias finais
    plt.subplot(1, 3, 3)
    plt.boxplot(resultados_distancia)
    plt.title('Distribuicao das Distancias Finais (km)', fontsize=12, fontweight='bold')
    plt.ylabel('Distancia (km)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Grafico adicional: Evolucao da distancia media
    plt.figure(figsize=(10, 6))
    
    # Converte fitness para distancia nos historicos
    historicos_distancia = []
    for historico in historicos_convergencia:
        historico_distancia = [1.0 / fit for fit in historico]
        historicos_distancia.append(historico_distancia)
    
    # Calcula a media das distancias por geracao
    historico_distancia_medio = np.mean(historicos_distancia, axis=0)
    
    plt.plot(historico_distancia_medio, 'b-', linewidth=2)
    plt.title('Evolucao da Distancia Media\n(30 Execucoes)', fontsize=14, fontweight='bold')
    plt.xlabel('Geracao')
    plt.ylabel('Distancia Media (km)')
    plt.grid(True, alpha=0.3)
    
    # Adiciona linha horizontal para a melhor distancia encontrada
    melhor_distancia_global = min(resultados_distancia)
    plt.axhline(y=melhor_distancia_global, color='r', linestyle='--', 
                label=f'Melhor distancia: {melhor_distancia_global:.0f} km')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Exibe a melhor solucao geral encontrada
    melhor_execucao_idx = np.argmax(resultados_fitness)
    melhor_fitness_global = resultados_fitness[melhor_execucao_idx]
    melhor_distancia_global = resultados_distancia[melhor_execucao_idx]
    
    print("\n" + "=" * 80)
    print("MELHOR SOLUCAO ENCONTRADA (entre as 30 execucoes)")
    print("=" * 80)
    print(f"Execucao: {melhor_execucao_idx + 1}")
    print(f"Fitness: {melhor_fitness_global:.6f}")
    print(f"Distancia: {melhor_distancia_global:.0f} km")
    
    # Executa novamente para obter a melhor rota
    ag_tsp = AlgoritmoGeneticoTSP(
        tsp=tsp,
        tamanho_populacao=50,
        geracoes=400,
        taxa_crossover=0.9,
        taxa_mutacao=0.05,
        elitismo=5
    )
    melhor_rota, melhor_fitness, _ = ag_tsp.executar_ag()
    
    print(f"\nMelhor rota valida: {tsp.validar_rota(melhor_rota)}")