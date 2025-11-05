# Importacao das bibliotecas necessarias
import random  # Para geracao de numeros aleatorios e embaralhamento
import numpy as np  # Para calculos numericos e operacoes com arrays
import matplotlib.pyplot as plt  # Para criacao de graficos e visualizacoes
from typing import List, Tuple  # Para definicao de tipos de dados

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
    
    # Funcao para converter uma rota de indices para nomes de cidades
    def converter_rota_para_nomes(self, rota: List[int]) -> List[str]:
        # Para cada indice na rota, obtem o nome correspondente da cidade
        return [self.nomes_cidades[i] for i in rota]
    
    # Funcao para plotar a rota em um grafico
    def plotar_rota(self, rota: List[int], titulo: str = "Rota TSP"):
        # Coordenadas aproximadas para visualizacao das cidades
        coordenadas = {
            0: (75, 45),   # New York
            1: (10, 32),   # Los Angeles
            2: (50, 42),   # Chicago
            3: (45, 50),   # Minneapolis
            4: (25, 40),   # Denver
            5: (40, 30),   # Dallas
            6: (15, 55),   # Seattle
            7: (80, 45),   # Boston
            8: (5, 35),    # San Francisco
            9: (48, 38),   # St. Louis
            10: (42, 25),  # Houston
            11: (20, 35),  # Phoenix
            12: (22, 43)   # Salt Lake City
        }
        
        # Cria uma nova figura para o grafico
        plt.figure(figsize=(12, 8))
        
        # Plota as cidades no grafico
        for cidade, (x, y) in coordenadas.items():
            plt.plot(x, y, 'o', markersize=8, label=self.nomes_cidades[cidade] if cidade == 0 else "")
            plt.text(x + 1, y + 1, self.nomes_cidades[cidade], fontsize=9)
        
        # Cria a rota completa incluindo o retorno ao inicio
        rota_completa = rota + [rota[0]]
        
        # Plota as conexoes entre as cidades
        for i in range(len(rota_completa) - 1):
            cidade_atual = rota_completa[i]
            proxima_cidade = rota_completa[i + 1]
            
            x1, y1 = coordenadas[cidade_atual]
            x2, y2 = coordenadas[proxima_cidade]
            
            # Desenha a linha entre as cidades
            plt.plot([x1, x2], [y1, y2], 'b-', alpha=0.6, linewidth=1)
            # Adiciona seta para indicar direcao
            plt.arrow((x1 + x2) / 2, (y1 + y2) / 2, 
                     (x2 - x1) * 0.1, (y2 - y1) * 0.1,
                     head_width=1, head_length=1, fc='red', ec='red')
        
        # Calcula a distancia total para exibir no titulo
        distancia = self.calcular_distancia(rota)
        plt.title(f"{titulo}\nDistancia Total: {distancia:.0f} km", fontsize=14, fontweight='bold')
        plt.xlabel("Coordenada X (Aproximada)")
        plt.ylabel("Coordenada Y (Aproximada)")
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

# Matriz de distancias entre as 13 cidades americanas em quilometros
# Cada linha representa uma cidade de origem
# Cada coluna representa uma cidade de destino
# O valor na posicao [i][j] e a distancia da cidade i para a cidade j
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
                 geracoes: int = 500, taxa_crossover: float = 0.8,
                 taxa_mutacao: float = 0.02, elitismo: int = 2):
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
    
    # Funcao de mutacao por inversao de um segmento
    def mutacao_inversao(self, rota: List[int]) -> List[int]:
        # Cria uma copia da rota para nao modificar a original
        mutado = rota.copy()
        
        # Verifica se a mutacao deve ser aplicada baseado na taxa de mutacao
        if random.random() < self.taxa_mutacao:
            # Seleciona aleatoriamente o inicio do segmento
            i = random.randint(0, len(rota) - 2)
            # Seleciona aleatoriamente o fim do segmento (apos o inicio)
            j = random.randint(i + 1, len(rota) - 1)
            # Inverte a ordem do segmento selecionado
            mutado[i:j+1] = reversed(mutado[i:j+1])
        
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
                
                # Aplica mutacao adicional por inversao com probabilidade 30%
                if random.random() < 0.3:
                    filho1 = self.mutacao_inversao(filho1)
                    filho2 = self.mutacao_inversao(filho2)
                
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
            
            # Exibe informacoes de progresso a cada 50 geracoes
            if geracao % 50 == 0:
                # Converte fitness para distancia para exibicao
                distancia = 1.0 / melhor_fitness_global
                print(f"Geracao {geracao}: Melhor distancia = {distancia:.0f} km")
        
        # Retorna a melhor rota, melhor fitness e historico
        return melhor_rota_global, melhor_fitness_global, historico_melhor_fitness

# Bloco principal de execucao do programa
if __name__ == "__main__":
    # Cabecalho do programa
    print("=" * 80)
    print("ATIVIDADE 5 - PROBLEMA DO CAIXEIRO VIAJANTE (TSP) - 13 CIDADES")
    print("=" * 80)
    
    # Cria uma instancia do problema TSP com a matriz de distancias
    tsp = TSP13(USA13)
    
    # Exibe a lista de cidades do problema
    print("CIDADES DO PROBLEMA:")
    for i, cidade in enumerate(tsp.nomes_cidades):
        print(f"{i:2d}. {cidade}")
    print()
    
    # Teste das funcoes basicas do TSP
    print("TESTE DAS FUNCOES BASICAS:")
    print("-" * 40)
    
    # Gera uma rota aleatoria para teste
    rota_teste = tsp.gerar_rota_aleatoria()
    # Calcula a distancia da rota teste
    distancia_teste = tsp.calcular_distancia(rota_teste)
    # Calcula o fitness da rota teste
    fitness_teste = tsp.calcular_fitness(rota_teste)
    # Valida se a rota teste e valida
    valida_teste = tsp.validar_rota(rota_teste)
    
    # Exibe os resultados do teste
    print(f"Rota teste: {rota_teste}")
    print(f"Rota em nomes: {tsp.converter_rota_para_nomes(rota_teste)}")
    print(f"Distancia total: {distancia_teste:.0f} km")
    print(f"Fitness: {fitness_teste:.6f}")
    print(f"Rota valida: {valida_teste}")
    print()
    
    # Execucao do algoritmo genetico
    print("EXECUTANDO ALGORITMO GENETICO...")
    print("-" * 40)
    
    # Cria uma instancia do algoritmo genetico com parametros definidos
    ag_tsp = AlgoritmoGeneticoTSP(
        tsp=tsp,
        tamanho_populacao=50,
        geracoes=500,
        taxa_crossover=0.8,
        taxa_mutacao=0.02,
        elitismo=2
    )
    
    # Executa o algoritmo genetico e obtem os resultados
    melhor_rota, melhor_fitness, historico = ag_tsp.executar_ag()
    # Converte o fitness para distancia
    melhor_distancia = 1.0 / melhor_fitness
    
    # Exibe a melhor solucao encontrada
    print("\n" + "=" * 80)
    print("MELHOR SOLUCAO ENCONTRADA:")
    print("=" * 80)
    
    print(f"Melhor distancia: {melhor_distancia:.0f} km")
    print(f"Melhor fitness: {melhor_fitness:.6f}")
    print(f"Rota valida: {tsp.validar_rota(melhor_rota)}")
    print()
    
    # Exibe a melhor rota em formato legivel
    print("MELHOR ROTA ENCONTRADA:")
    # Converte a rota de indices para nomes de cidades
    rota_nomes = tsp.converter_rota_para_nomes(melhor_rota)
    # Exibe cada cidade na ordem da rota
    for i, cidade in enumerate(rota_nomes):
        print(f"{i+1:2d}. {cidade}")
    # Exibe o retorno a cidade inicial
    print(f"{14:2d}. {rota_nomes[0]} (retorno)")
    
    # Plotar evolucao do fitness
    plt.figure(figsize=(12, 5))
    
    # Grafico da evolucao da distancia
    plt.subplot(1, 2, 1)
    distancias_historicas = [1.0 / fit for fit in historico]
    plt.plot(distancias_historicas, 'b-', linewidth=2)
    plt.title('Evolucao da Melhor Distancia', fontsize=12, fontweight='bold')
    plt.xlabel('Geracao')
    plt.ylabel('Distancia (km)')
    plt.grid(True, alpha=0.3)
    
    # Grafico da evolucao do fitness
    plt.subplot(1, 2, 2)
    plt.plot(historico, 'g-', linewidth=2)
    plt.title('Evolucao do Melhor Fitness', fontsize=12, fontweight='bold')
    plt.xlabel('Geracao')
    plt.ylabel('Fitness (1/distancia)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Plotar a melhor rota encontrada
    tsp.plotar_rota(melhor_rota, "Melhor Rota Encontrada - Algoritmo Genetico")
    
    # Analise estatistica com multiplas execucoes
    print("\n" + "=" * 80)
    print("ANALISE ESTATISTICA (10 EXECUCOES)")
    print("=" * 80)
    
    # Lista para armazenar os resultados das execucoes
    resultados_distancia = []
    # Executa o algoritmo genetico 10 vezes
    for execucao in range(10):
        # Cria nova instancia com menos geracoes para teste rapido
        ag_tsp = AlgoritmoGeneticoTSP(tsp, geracoes=200)
        # Executa o algoritmo
        melhor_rota, melhor_fitness, _ = ag_tsp.executar_ag()
        # Calcula a distancia da melhor rota
        melhor_distancia = 1.0 / melhor_fitness
        # Armazena o resultado
        resultados_distancia.append(melhor_distancia)
        # Exibe o resultado da execucao atual
        print(f"Execucao {execucao + 1}: {melhor_distancia:.0f} km")
    
    # Exibe o resumo estatistico dos resultados
    print(f"\nRESUMO ESTATISTICO:")
    print(f"Media: {np.mean(resultados_distancia):.0f} km")
    print(f"Desvio Padrao: {np.std(resultados_distancia):.0f} km")
    print(f"Melhor: {min(resultados_distancia):.0f} km")
    print(f"Pior: {max(resultados_distancia):.0f} km")