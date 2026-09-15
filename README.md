# Assistente clínico institucional

Experimento acadêmico com corpus sintético, ajuste fino LoRA, consulta SQLite, recuperação lexical de protocolos, LangChain e LangGraph. O modelo organiza texto para revisão profissional. Não há execução de condutas ou envio externo de alertas.

O treinamento local usa **FLAN-T5-small com LoRA em CPU**. A escolha do modelo considerou o computador disponível: cerca de 6 GB de RAM, vídeo integrado AMD e ausência de CUDA.

## Estado do experimento

- Ajuste fino local realizado: 15 exemplos de treino, cinco de validação e cinco de teste.
- Adaptador local: `models/clinical-t5-lora`.
- Evidências versionáveis: [resultados de treinamento](docs/results/training.json), [curva](docs/results/loss.png) e [execução integrada](docs/results/system.json).
- A perda de teste caiu de 3,3887 para 3,1454, mas a qualidade gerativa continua insuficiente: há cópia de instruções, mistura de idiomas e repetição.
- O sistema retém respostas que não correspondam literalmente às evidências. O caso clínico integrado foi retido.
- O vídeo ainda precisa ser gravado, conforme [roteiro](docs/roteiro_video.md).

## Estrutura

| Pasta | Conteúdo |
|---|---|
| `data/raw` | 25 exemplos e quatro protocolos sintéticos; CSV de pacientes e exames |
| `data/processed` | partições e SQLite reconstruíveis |
| `src/clinical_assistant` | preparação do prompt, modelo, recuperação, grafo, auditoria e avaliação |
| `scripts` | preparação, avaliação integrada e execução de notebooks |
| `notebooks` | percurso explicado em cinco cadernos com saídas salvas |
| `docs` | relatório com arquitetura e descrição do modelo, roteiro e resultados |
| `tests` | regressões com bancos temporários |
| `models`, `.hf-cache`, `logs` | pesos e artefatos locais, excluídos do Git |

## Instalação em Windows

O ambiente reproduzido usa Windows e Python 3.12.10 de 64 bits. Instale essa versão de Python antes de criar o ambiente virtual. Não é necessário alterar o PATH se você usar o caminho completo do executável.

Em uma cópia nova do projeto, com Python instalado, execute na raiz:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Para usar o Python local já instalado neste diretório:

```powershell
.\.tools\python\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Não é necessário ativar o ambiente. Isso evita alterações na política de execução do PowerShell. A lista de versões foi verificada no ambiente Windows/Python 3.12; outros sistemas precisam de validação própria.

O arquivo `requirements.txt` reúne as versões completas das dependências usadas no treinamento, na execução e nos testes. O `pyproject.toml` define os metadados e as dependências diretas do pacote; para reproduzir o ambiente, use o comando acima.

## Preparação

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py
.\.venv\Scripts\python.exe scripts/init_database.py
```

A preparação valida os exemplos e escreve três partições fixas, sem sobreposição de identificadores. REC-007, o modelo de receita, permanece no treino. O inicializador do SQLite preserva uma base já existente. Para reconstruir, faça antes uma cópia e mova a base antiga para outro nome.

Todos os registros são sintéticos e estão incluídos em `data/raw`. Não é necessário baixar uma base médica externa. Os dados disponíveis são suficientes para um experimento exploratório, não para validar desempenho clínico.

## Treinamento local

```powershell
$env:HF_HOME = "$PWD\.hf-cache"
$env:HF_HUB_DISABLE_XET = "1"
.\.venv\Scripts\python.exe -m clinical_assistant.training --epochs 6
```

O primeiro uso baixa `google/flan-t5-small` do Hugging Face, sem token, e exige internet e espaço livre para bibliotecas, cache e pesos. Reserve alguns GB. Em Windows, o aviso sobre ausência de symlinks não impede o download.

A execução salva o melhor adaptador por perda de validação, resultados e curva. Uma nova execução substitui os resultados e o adaptador desse experimento: copie-os antes caso deseje comparar execuções. O teste não participa da seleção da época.

## Execução do assistente

Modo de teste de software, com gerador determinístico:

```powershell
$env:ASSISTANT_BACKEND = "demo"
.\.venv\Scripts\python.exe -m clinical_assistant.cli --patient-id PAC-0001 --question "Exames de acompanhamento de diabetes"
```

Modelo efetivamente ajustado:

```powershell
$env:ASSISTANT_BACKEND = "t5"
$env:ADAPTER_PATH = "models/clinical-t5-lora"
.\.venv\Scripts\python.exe -m clinical_assistant.cli --patient-id PAC-0001 --question "Exames de acompanhamento de diabetes"
.\.venv\Scripts\python.exe scripts/evaluate_system.py
```

O retorno pode ser retido por falta de correspondência literal com o contexto. Esse bloqueio é um resultado esperado quando o modelo produz uma saída inadequada, não evidência de que respondeu corretamente. A avaliação salva também o rascunho sintético para análise.

As configurações podem ser copiadas de `.env.example` para `.env`. Variáveis de ambiente já definidas têm precedência. Tokens nunca devem ser versionados. Os registros locais de auditoria ficam em `logs/audit.jsonl`.

## Notebooks e testes

Selecione `.venv/Scripts/python.exe` como interpretador no editor de notebooks. Execute os cadernos em ordem:

1. [Preparação dos dados](notebooks/01_preparacao_dados.ipynb): privacidade e partições;
2. [Treinamento LoRA](notebooks/02_treinamento_lora.ipynb): configuração e análise da curva;
3. [Integração LangChain](notebooks/03_assistente_langchain.ipynb): consulta e inferência com o adaptador;
4. [Fluxo LangGraph](notebooks/04_fluxo_langgraph.ipynb): bloqueios e auditoria;
5. [Avaliação](notebooks/05_avaliacao.ipynb): comparação de modelos e testes.

Para executar tudo em kernels novos e salvar saídas:

```powershell
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
.\.venv\Scripts\python.exe -m pytest -q
```

O notebook 02 lê a evidência existente por padrão. Defina `RUN_TRAINING=True` nele para repetir o treinamento. Em um clone novo, os pesos não estão no Git: execute o treinamento antes dos notebooks 03–05.

Os próprios arquivos `.ipynb` são a fonte dos cadernos e devem ser editados diretamente. Eles compartilham módulos de `src/clinical_assistant` e artefatos em disco, sem depender da memória de outro notebook.

## Limites

As regras textuais não cobrem toda a linguagem clínica. Correspondência literal não garante pertinência ou validade médica. Não há autenticação de profissionais, aprovação clínica eletrônica ou conexão com hospital. Alertas são exibidos localmente e não enviados a uma equipe. Os hashes dos logs não garantem anonimização irreversível.

O [relatório técnico](docs/relatorio_tecnico.md) explica objetivos, conceitos, metodologia, arquitetura, resultados e limitações do experimento.
