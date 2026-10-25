# Plano de desenvolvimento

Este documento descreve as etapas do projeto na ordem de execução, com entradas, saídas e critério de conclusão de cada uma. O código de todas as etapas está em [`notebooks/tcc2_pipeline.ipynb`](../notebooks/tcc2_pipeline.ipynb) e [`src/tcc_seg.py`](../src/tcc_seg.py).

## 1. Escopo

1. **Reconstrução 3D:** reconstruir ScalarReal, FluidNexus-Smoke e FluidNexus-Ball com o FluidNexus em duas configurações (leve e alta) e comparar a qualidade visual.
2. **Segmentação 3D sem ground truth:** treinar 3D U-Net, V-Net, PointNet++ e Point Transformer com pseudo-rótulos no ScalarFlow e avaliar no ScalarFlow e nas reconstruções do FluidNexus com métricas que não dependem de rótulo manual.

## 2. Requisitos

| Item | Requisito |
|---|---|
| Execução | Google Colab com GPU (L4 recomendada, T4 funciona) e RAM alta |
| Armazenamento | Google Drive com espaço para os datasets, os modelos e os resultados |
| Rede | acesso ao Hugging Face (FluidNexus), ao GitHub (código oficial) e ao rsync do mediaTUM (ScalarFlow) |

## 3. Organização no Google Drive

Todas as etapas leem e gravam em `MyDrive/TCC2_FluidNexus/`:

| Pasta | Conteúdo |
|---|---|
| `codigo/FluidNexus/` | código oficial do FluidNexus |
| `dados/compactados/`, `dados/extraidos/` | arquivos .zip e imagens do FluidNexus |
| `dados/processados/` | dados no formato do reconstrutor |
| `dados/scalarflow/sim_XXXXXX/` | densidade e velocidade do ScalarFlow (.npz) |
| `configuracoes/` | configurações JSON de cada execução |
| `modelos/<Fluido>_20_160[_alta]/` | reconstruções (fundo, físico, visual) |
| `segmentacao/` | dados padronizados, modelos e resultados da segmentação |
| `resultados/`, `figuras/`, `videos/`, `logs/` | tabelas, figuras, vídeos e registros |

## 4. Etapas

### Etapa 1. Dados do FluidNexus

- **Notebook:** seção 1.
- **Ações:** criar a estrutura de pastas, baixar ScalarReal, FluidNexus-Smoke e FluidNexus-Ball do Hugging Face e extrair as imagens.
- **Saída:** `dados/extraidos/` com as câmeras e frames de cada conjunto.
- **Concluído quando:** os três conjuntos estão extraídos e a contagem de imagens por câmera confere.

### Etapa 2. Ambiente do FluidNexus

- **Notebook:** seção 2.
- **Ações:** criar o ambiente Python 3.12, instalar PyTorch e as dependências, compilar as extensões CUDA para a GPU detectada.
- **Saída:** ambiente em `/content/fluidnexus_py312`.
- **Concluído quando:** o FluidNexus importa sem erro e as extensões CUDA carregam.
- **Atenção:** o ambiente fica no disco local do Colab. Rode esta etapa de novo sempre que abrir uma sessão nova.

### Etapa 3. Reconstrução 3D (leve e alta)

- **Notebook:** seção 3.
- **Configuração comum:** frames 20 a 160; treino nas câmeras 0, 1, 3 e 4; câmera 2 reservada para teste.
- **Leve:** resolução reduzida e menos iterações. Serve de base de comparação.
- **Alta:** configuração oficial dos autores (resolução cheia, iterações oficiais).

| Fluido | Etapas | Passo entre frames |
|---|---|---|
| ScalarReal | físico + visual | 1 |
| Smoke | fundo + físico + visual | 2 |
| Ball | fundo + físico + visual | 2 |

- **Saída:** `modelos/<Fluido>_20_160/` (leve) e `modelos/<Fluido>_20_160_alta/` (alta), com renders e nuvens de pontos por frame.
- **Concluído quando:** existem renders e nuvens de pontos para todos os frames nas duas versões.
- **Retomada:** as células pulam as etapas já concluídas e limpam pastas incompletas. Se a sessão cair, basta rodar de novo.

### Etapa 4. Avaliação da reconstrução

- **Notebook:** seção 4.
- **Ações:** comparar cada versão com a foto real da câmera de teste usando PSNR, SSIM e LPIPS em três regiões (imagem inteira, só fumaça, só fundo). Gerar figuras dos frames 40, 80, 120 e 160 e vídeos da reconstrução.
- **Saída:** Tabela 1 (`resultados/comparacao_leve_x_alta/`), figuras e vídeos.
- **Concluído quando:** a Tabela 1 tem os três fluidos nas duas versões.

### Etapa 5. Dados do ScalarFlow

- **Notebook:** seção 5.
- **Ações:** baixar 10 sequências (`sim_000000` a `sim_000009`) por rsync, extrair densidade e velocidade e salvar compactado. Cada `.tar` tem cerca de 4,7 GB e é apagado depois da extração.
- **Saída:** `dados/scalarflow/sim_XXXXXX/densidade.npz` (151 frames, 100 x 178 x 100) e `velocidade_meia_res.npz` (150 frames, 50 x 89 x 50 x 3).
- **Concluído quando:** as 10 sequências estão no Drive e listadas em `resumo_sequencias.csv`.

### Etapa 6. Preparação para a segmentação

- **Notebook:** seção 6, células 1/5.
- **Ações:**
  1. Reamostrar todas as densidades para a grade 64 x 112 x 64 e normalizar pelo percentil 99,9.
  2. Voxelizar as nuvens de pontos do FluidNexus (versão alta) com filtro gaussiano (sigma 1) e compressão `log1p`.
  3. Gerar pseudo-rótulos por limiar de Otsu por sequência (voxels acima de 0,02; limiar entre 0,05 e 0,6), removendo componentes com menos de 20 voxels.
- **Divisão:**

| Conjunto | Sequências |
|---|---|
| Treino | ScalarFlow `sim_000000` a `sim_000005` |
| Validação | ScalarFlow `sim_000006` e `sim_000007` |
| Teste | ScalarFlow `sim_000008` e `sim_000009`, FluidNexus ScalarReal, Smoke e Ball |

- **Saída:** `segmentacao/dados/<sequencia>.npz` e `.json`.

### Etapa 7. Treino dos modelos

- **Notebook:** seção 6, células 2/5 e 3/5.
- **Modelos:** 3D U-Net e V-Net (MONAI), PointNet++ e Point Transformer (PyTorch).
- **Treino:** perda BCE + Dice, validação a cada época, limite de 20 min por modelo. As redes de pontos usam amostras de pontos dos voxels candidatos.
- **Saída:** `segmentacao/modelos/<modelo>.pt` e `.json` (parâmetros, tempo e memória).

### Etapa 8. Avaliação sem ground truth

- **Notebook:** seção 6, célula 4/5.
- **Métricas por sequência:**

| Métrica | O que mede | Melhor |
|---|---|---|
| IoU temporal | sobreposição entre máscaras de frames seguidos | maior |
| Variação de volume | mudança relativa do volume entre frames | menor |
| Deslocamento do centroide | movimento do centro da máscara entre frames | menor |
| Borda HD95 | estabilidade do contorno entre frames | menor |
| Flicker | voxels que acendem e apagam entre frames | menor |
| Consistência física | IoU entre a máscara do frame seguinte e a máscara atual transportada pela velocidade (só ScalarFlow) | maior |
| Estrutura fina | quanto das partes finas do pseudo-rótulo é mantido | maior |
| Custo | parâmetros, tempo de treino, memória e tempo de inferência | menor |

- **Saída:** `segmentacao/resultados/` com `metricas_por_sequencia.csv`, Tabela 2 (média e desvio por domínio), Tabela 3 (custo) e figuras.
- **Concluído quando:** todas as combinações de modelo e sequência de teste estão na tabela.

## 5. Cuidados de execução

- **Salvar no Drive antes de desligar:** ao final de uma sessão, rode `drive.flush_and_unmount()` antes de desconectar o ambiente. Arquivos grandes podem não ter sido enviados ao Drive ainda.
- **Sessões longas:** as etapas pulam o que já foi feito. Em caso de queda, reconecte, rode a Etapa 2 (se for usar o FluidNexus) e repita a célula interrompida.
- **Reprodutibilidade:** mantenha a mesma divisão de câmeras, frames e sequências. Os valores principais estão nas constantes do início de `src/tcc_seg.py` (`GRADE`, `SPLIT`, `LIMIAR_CAND`, `MINUTOS`).

## 6. Entregáveis

| Entregável | Local no repositório | Status |
|---|---|---|
| Tabela 1: reconstrução leve x alta | `results/tabelas/tabela1_leve_x_alta.csv` | concluído |
| Tabela 2: métricas sem ground truth | `results/tabelas/tabela2_metricas_sem_gt.csv` | concluído |
| Tabela 3: custo computacional | `results/tabelas/tabela3_custo_computacional.csv` | concluído |
| Figuras da reconstrução e da segmentação | `results/figuras/` | concluído |
| Texto do TCC (metodologia, resultados, discussão) | `docs/` | em andamento |

## 7. Próximos passos

1. Aumentar o número de sequências de teste do ScalarFlow para reforçar a estatística.
2. Escrever os capítulos de metodologia, resultados e discussão.
