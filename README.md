# Assistente clínico institucional

Este repositório apresenta um protótipo local para organizar informações clínicas sintéticas antes da revisão de um profissional. Ele combina quatro elementos: ajuste fino de um modelo de linguagem, consulta a dados estruturados, recuperação de protocolos e um fluxo de decisão que aplica limites antes de permitir uma resposta.

O protótipo não diagnostica, não prescreve, não altera prontuários e não envia alertas externos. Todas as informações clínicas presentes no projeto são sintéticas.

## Como o projeto funciona

1. O corpus reúne perguntas frequentes, protocolos, exemplos de laudo, procedimentos, um modelo de receita sem conteúdo prescritivo e situações de segurança.
2. A preparação verifica campos obrigatórios, remove formatos diretos de identificação reconhecidos, identifica duplicidades e divide os exemplos entre treino, validação e teste.
3. O treinamento carrega o modelo FLAN-T5-small e aplica LoRA. Essa técnica mantém os pesos originais do modelo e treina um adaptador menor, adequado ao ambiente local.
4. Na consulta, a aplicação lê o paciente no SQLite em modo somente leitura e recupera os protocolos mais relacionados à pergunta por comparação lexical.
5. O LangChain monta o prompt com pergunta, dados do paciente e protocolos. O LangGraph decide se deve emitir um alerta, recusar a solicitação, continuar para geração ou reter a saída.
6. A saída somente é disponibilizada quando passa pelas verificações de contexto e segurança. A resposta inclui as fontes recuperadas e o aviso de revisão profissional.

## Estrutura

| Local | Finalidade |
|---|---|
| `data/raw` | dados sintéticos de entrada: exemplos, protocolos, pacientes e exames |
| `src/clinical_assistant` | módulos da aplicação, do treinamento, da recuperação e do fluxo |
| `scripts` | preparação, banco SQLite, avaliação, execução de notebooks e geração do PNG do fluxo |
| `notebooks` | explicação executável de cada etapa |
| `docs` | relatório, roteiro de apresentação e diagrama do fluxo |
| `tests` | verificações automatizadas de dados, segurança, recuperação e grafo |

Os diretórios `data/processed`, `models`, `logs`, `artifacts` e `docs/results` são produzidos localmente durante a execução. Eles estão no `.gitignore` e não devem ser incluídos no repositório.

## Instalação

Use Python 3.11 ou 3.12 de 64 bits. Na raiz do projeto, crie o ambiente e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Não é necessário ativar o ambiente virtual. Os exemplos abaixo usam diretamente o executável instalado nele.

## Preparação dos dados e do banco

Execute os comandos a seguir antes do treinamento ou da aplicação:

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/init_database.py
```

O primeiro comando cria as partições em `data/processed`. São 15 exemplos de treino, cinco de validação e cinco de teste. A validação escolhe a época do treinamento; o teste é reservado para a comparação final. O segundo comando cria o arquivo SQLite somente quando ele ainda não existe.

## Treinamento da LLM

```powershell
$env:HF_HOME = "$PWD\.hf-cache"
$env:HF_HUB_DISABLE_XET = "1"
.\.venv\Scripts\python.exe -m clinical_assistant.training --epochs 6
```

Na primeira execução, o modelo base é obtido e armazenado no cache local. O treinamento salva o adaptador LoRA em `models/clinical-t5-lora` e registra métricas, comparações de respostas e a curva de perdas em `docs/results`. Esses arquivos são evidências locais de execução e não são versionados.

## Execução da aplicação

O modo `demo` é um gerador determinístico usado somente para testes da orquestração. Para carregar a LLM ajustada, utilize o modo `t5` depois de executar o treinamento:

```powershell
$env:ASSISTANT_BACKEND = "t5"
$env:ADAPTER_PATH = "models/clinical-t5-lora"
.\.venv\Scripts\python.exe -m clinical_assistant.cli --patient-id PAC-0001 --question "Exames de acompanhamento de diabetes"
```

O retorno pode ser retido quando a resposta gerada não encontra suporte literal no contexto recuperado. Essa retenção é intencional: ela reduz saídas não sustentadas, mas não substitui avaliação clínica.

## Fluxo de decisão

O diagrama está disponível em [docs/langgraph_flow.svg](docs/langgraph_flow.svg). Para gerar uma imagem PNG localmente, execute:

```powershell
.\.venv\Scripts\python.exe scripts/render_langgraph_flow.py
```

O PNG é salvo em `artifacts/langgraph_flow.png`. O fluxo inicia pela anonimização e pela triagem textual. Sinais críticos encerram o processo em um alerta local; solicitações de prescrição ou diagnóstico são recusadas; somente perguntas com paciente e fontes recuperadas seguem para a LLM. A saída gerada passa por uma verificação final antes de ser apresentada para revisão.

## Notebooks, avaliação e testes

Os notebooks devem ser executados nesta ordem:

1. `01_preparacao_dados.ipynb` — curadoria, anonimização e partições;
2. `02_treinamento_lora.ipynb` — ajuste fino e leitura dos resultados locais;
3. `03_assistente_langchain.ipynb` — consulta estruturada, recuperação e prompt;
4. `04_fluxo_langgraph.ipynb` — decisões do grafo, auditoria e diagrama;
5. `05_avaliacao.ipynb` — avaliação das respostas e testes automatizados.

Para executar os notebooks com um kernel novo e depois rodar a suíte de testes:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
.\.venv\Scripts\python.exe -m pytest -q
```

As saídas dos notebooks não devem ser salvas no repositório. Para limpar saídas existentes antes de versionar, use o comando de limpeza descrito na seção seguinte.

## Limpeza antes de versionar

```powershell
.\.venv\Scripts\python.exe scripts/clear_notebook_outputs.py
```

Confira também `git status` antes de criar um commit. Apenas código-fonte, dados sintéticos de entrada, notebooks sem saídas e documentação devem aparecer como arquivos versionáveis.

## Limitações

O corpus é pequeno e sintético. A anonimização usa padrões conhecidos e não garante remoção irreversível de toda informação identificável. A recuperação lexical depende de palavras em comum e pode não reconhecer sinônimos. O modelo ajustado pode repetir instruções ou produzir texto inadequado; por isso o fluxo aplica retenção e não autoriza condutas. O protótipo não foi validado por especialistas nem integrado a sistemas hospitalares.

O relatório detalha as escolhas e a avaliação em [docs/relatorio_tecnico.md](docs/relatorio_tecnico.md). O roteiro da demonstração está em [docs/roteiro_video.md](docs/roteiro_video.md).
