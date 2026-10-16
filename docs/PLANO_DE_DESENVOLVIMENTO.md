# Plano de desenvolvimento do TCC 2

## 1 Escopo do projeto

O projeto realizará a reconstrução tridimensional e a análise temporal de três conjuntos utilizados pelo FluidNexus:

1. ScalarReal, amostra do ScalarFlow;
2. FluidNexus-Smoke;
3. FluidNexus-Ball.

Para cada conjunto, o resultado final deverá apresentar a evolução visual da reconstrução em frames representativos, incluindo inicialmente os frames 0, 10, 20 e 40. O projeto também produzirá métricas quantitativas de qualidade visual e coerência física.

O reconhecimento, a classificação e a segmentação de padrões não fazem parte deste escopo.

## 2 Etapas de desenvolvimento

### Etapa 1 Preparação do ambiente

Objetivo: criar um ambiente reproduzível para executar o código oficial e as rotinas próprias do projeto.

Atividades:

- registrar sistema operacional, GPU, memória de vídeo, memória RAM e espaço disponível;
- verificar compatibilidade com CUDA e PyTorch;
- documentar as versões das dependências;
- configurar variáveis de caminho sem incluir dados locais no Git;
- executar um teste simples de GPU e importação das bibliotecas principais.

Entregáveis:

- arquivo de ambiente reproduzível;
- instruções de instalação;
- relatório curto de compatibilidade do computador.

Critério de conclusão: o ambiente deve importar as dependências necessárias e reconhecer corretamente o dispositivo de execução.

### Etapa 2 Aquisição e inventário dos dados

Objetivo: obter primeiro as amostras menores dos três conjuntos e registrar sua estrutura real.

Atividades:

- baixar ScalarReal, FluidNexus-Smoke e FluidNexus-Ball;
- validar a integridade dos arquivos baixados;
- identificar cenas, câmeras, quantidade de frames, resolução e calibração disponível;
- escolher uma cena inicial de cada conjunto;
- criar uma divisão clara entre dados brutos, processados e resultados.

Entregáveis:

- inventário dos datasets;
- manifesto das cenas selecionadas;
- instruções de download sem versionar os arquivos grandes.

Critério de conclusão: uma cena de cada conjunto deve estar acessível e descrita com quantidade de câmeras e frames confirmada.

### Etapa 3 Preparação dos vídeos e câmeras

Objetivo: converter os dados para o formato esperado pelo pipeline de reconstrução.

Atividades:

- extrair e organizar os frames;
- conferir sincronização entre as vistas;
- validar os parâmetros das câmeras;
- padronizar nomes, resolução e sequência temporal;
- selecionar a câmera central como entrada e reservar as demais vistas para avaliação.

Entregáveis:

- rotina de preparação dos dados;
- estrutura padronizada por dataset e cena;
- relatório de validação das sequências.

Critério de conclusão: os frames de entrada e as vistas de referência devem estar sincronizados e carregáveis pelo pipeline.

### Etapa 4 Execução inicial da reconstrução

Objetivo: reproduzir uma reconstrução completa com a configuração oficial antes de modificar o método.

Atividades:

- executar a síntese de novas vistas;
- executar o refinamento temporal dos vídeos;
- otimizar a representação de partículas físicas;
- otimizar as partículas visuais;
- registrar tempo de execução, memória utilizada e arquivos gerados.

Entregáveis:

- primeira reconstrução funcional;
- logs de execução;
- registro dos parâmetros utilizados.

Critério de conclusão: uma cena deve gerar reconstruções renderizáveis em todos os frames selecionados.

### Etapa 5 Reconstrução dos três conjuntos

Objetivo: aplicar o mesmo protocolo experimental aos três casos.

Atividades:

- executar o pipeline em ScalarReal;
- executar o pipeline em FluidNexus-Smoke;
- executar o pipeline em FluidNexus-Ball;
- manter os parâmetros comuns sempre que tecnicamente possível;
- documentar todas as diferenças necessárias entre os conjuntos.

Entregáveis:

- reconstruções completas dos três conjuntos;
- configurações versionadas;
- tabela com parâmetros, duração e consumo de recursos.

Critério de conclusão: os três experimentos devem terminar com resultados válidos e reproduzíveis.

### Etapa 6 Resultados visuais por frame

Objetivo: mostrar de forma clara a evolução temporal de cada fluido reconstruído.

Atividades:

- renderizar inicialmente os frames 0, 10, 20 e 40;
- manter câmera, enquadramento, resolução e escala visual consistentes;
- criar um painel separado para cada conjunto;
- incluir identificação do dataset, cena, frame e ponto de vista;
- verificar se os frames escolhidos representam mudanças perceptíveis no fluido.

Entregáveis:

- painel visual do ScalarReal;
- painel visual do FluidNexus-Smoke;
- painel visual do FluidNexus-Ball.

Critério de conclusão: cada painel deve permitir comparar claramente o estado inicial, a evolução e o estado final selecionado.

### Etapa 7 Avaliação quantitativa

Objetivo: medir a qualidade visual e física das reconstruções.

Métricas principais:

- PSNR, em que valores maiores indicam maior fidelidade;
- SSIM, em que valores maiores indicam maior semelhança estrutural;
- LPIPS, em que valores menores indicam maior semelhança perceptual;
- divergência do campo de velocidade, em que valores menores indicam maior compatibilidade com a incompressibilidade.

Atividades:

- calcular as métricas por frame e por vista;
- consolidar média, mediana e dispersão por cena;
- registrar falhas e frames inválidos;
- comparar os três conjuntos sem tratá-los como observações equivalentes;
- incluir tempo de processamento e uso máximo de memória como métricas de custo.

Entregáveis:

- tabelas quantitativas por conjunto;
- tabela comparativa consolidada;
- arquivos de resultados em formato aberto, como CSV ou JSON.

Critério de conclusão: todos os valores apresentados no TCC devem ser rastreáveis aos arquivos de resultados produzidos pelo código.

### Etapa 8 Validação e reprodutibilidade

Objetivo: garantir que os resultados possam ser repetidos e auditados.

Atividades:

- fixar sementes aleatórias quando aplicável;
- registrar versões de código, dependências, modelos e datasets;
- criar comandos únicos para preparação, reconstrução, avaliação e geração das figuras;
- testar o pipeline a partir de um ambiente limpo;
- verificar se dados brutos e modelos pesados continuam fora do Git.

Entregáveis:

- instruções completas de reprodução;
- configurações finais dos experimentos;
- checklist de validação preenchido.

Critério de conclusão: outra execução com a mesma configuração deve reproduzir os resultados dentro das variações esperadas.

### Etapa 9 Análise e redação dos resultados

Objetivo: transformar os experimentos em evidências para o TCC 2.

Atividades:

- descrever o comportamento visual de cada reconstrução;
- relacionar métricas e observações qualitativas;
- analisar limitações, artefatos e perda de detalhes finos;
- discutir diferenças entre fumaça livre, fumaça com interação sólida e ScalarFlow;
- incorporar figuras e tabelas aos capítulos de metodologia e resultados.

Entregáveis:

- texto da metodologia implementada;
- capítulo de resultados;
- discussão das limitações;
- figuras e tabelas finais.

Critério de conclusão: cada conclusão deve estar apoiada por uma figura, uma métrica ou uma observação experimental documentada.

## 3 Ordem prática de execução

A implementação seguirá esta ordem:

1. ScalarReal, por ser a amostra menor e mais adequada para validar o ambiente;
2. FluidNexus-Smoke, para validar o pipeline no dataset principal;
3. FluidNexus-Ball, para analisar a interação entre fumaça e objeto sólido;
4. geração dos painéis dos frames;
5. cálculo e consolidação das métricas;
6. redação dos resultados e conclusões.

## 4 Definição de pronto do projeto

O desenvolvimento será considerado concluído quando:

- os três conjuntos tiverem pelo menos uma cena reconstruída;
- os frames selecionados tiverem sido renderizados com apresentação consistente;
- PSNR, SSIM, LPIPS e divergência tiverem sido calculados quando os dados necessários estiverem disponíveis;
- tempo de execução e consumo de memória estiverem registrados;
- os resultados puderem ser reproduzidos por comandos documentados;
- as figuras, tabelas e limitações estiverem incorporadas ao TCC 2.
