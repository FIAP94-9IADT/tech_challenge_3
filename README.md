# Assistente Clínico Institucional

Protótipo acadêmico de um assistente para apoio à consulta de protocolos internos e à organização de informações clínicas. A aplicação combina ajuste fino eficiente de um modelo de linguagem, recuperação de contexto, consultas estruturadas e um fluxo controlado por grafo.

O sistema trabalha exclusivamente com dados sintéticos. As respostas são apoio informacional: não substituem avaliação profissional, não emitem diagnóstico e não prescrevem. Toda conduta permanece sujeita à validação de um profissional habilitado.

## Funcionalidades

- preparação, anonimização e curadoria de exemplos clínicos sintéticos;
- fine-tuning supervisionado de um modelo LLaMA com quantização em 4 bits e adaptadores LoRA;
- recuperação de trechos de protocolos com indicação de fonte;
- consulta somente leitura a prontuários e exames em SQLite;
- identificação de exames pendentes e sinais de alerta;
- pipeline LangChain que recebe o modelo ajustado e contexto institucional;
- fluxo LangGraph com estado tipado, nós especializados e rotas condicionais;
- validação de saída, aviso de revisão humana e log JSONL para auditoria;
- avaliação de anonimização, recuperação, segurança e qualidade das respostas.

## Estrutura

```text
.
├── data/
│   ├── raw/                  # exemplos, protocolos e registros sintéticos
│   └── processed/            # artefatos gerados localmente
├── docs/                     # relatório, diagramas e roteiro de demonstração
├── notebooks/                # percurso completo, da preparação à avaliação
├── scripts/                  # preparação dos dados e criação do SQLite
├── src/clinical_assistant/   # implementação modular
└── tests/                    # testes unitários e de integração
```

## Instalação

Requer Python 3.11 ou 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

As dependências pesadas de treinamento são opcionais, pois o ajuste fino é indicado para ambiente com GPU:

```bash
python -m pip install -e ".[training]"
```

Copie `.env.example` para `.env`. Tokens nunca devem ser versionados.

## Preparação dos dados

```bash
python scripts/prepare_data.py
python scripts/init_database.py
```

O primeiro comando anonimiza e valida os exemplos, separa treino e teste e produz um relatório de curadoria. O segundo cria `data/processed/hospital.db` com pacientes e exames sintéticos. Ambos são determinísticos.

## Execução local

O modo demonstrativo não baixa modelos e permite verificar todo o fluxo:

```bash
python -m clinical_assistant.cli --patient-id PAC-0001 --question "Quais exames estão pendentes e quais protocolos devem ser consultados?"
```

Para usar o adaptador treinado, defina no `.env`:

```dotenv
ASSISTANT_BACKEND=huggingface
BASE_MODEL_ID=meta-llama/Llama-2-7b-hf
ADAPTER_PATH=models/clinical-lora
```

O acesso ao modelo base no Hugging Face depende da aceitação de sua licença e de um token com permissão. Consulte o notebook `02_fine_tuning_lora.ipynb` para o treinamento em GPU.

## Notebooks

Execute na ordem:

1. `01_preparacao_dados.ipynb` — inspeção, anonimização, curadoria e divisão;
2. `02_fine_tuning_lora.ipynb` — tokenização, quantização, LoRA, treinamento e avaliação;
3. `03_assistente_langchain.ipynb` — modelo customizado, prontuário e recuperação;
4. `04_fluxo_langgraph.ipynb` — execução do grafo, alertas, validação e logs;
5. `05_avaliacao.ipynb` — métricas e análise dos resultados.

## Testes

```bash
pytest -q
```

Os testes não exigem GPU nem chamada externa. Eles verificam anonimização, consultas parametrizadas, recuperação de fontes, limites de atuação, roteamento do grafo e registro de auditoria.

## Segurança e governança

- Todos os dados incluídos no repositório são fictícios e identificados como sintéticos.
- A camada de acesso aceita apenas identificadores no formato institucional e consultas SQL previamente definidas.
- Entradas passam por remoção de identificadores pessoais antes de qualquer processamento pelo modelo.
- Solicitações de prescrição, alteração autônoma de tratamento ou substituição de avaliação profissional são bloqueadas.
- Sinais críticos produzem alerta de priorização, sem determinar diagnóstico ou conduta.
- Cada resposta informa as fontes recuperadas, as etapas percorridas e a necessidade de validação humana.
- O log registra hashes e metadados operacionais; não armazena a pergunta clínica em texto aberto.

Detalhes de arquitetura, decisões metodológicas, métricas e limitações estão em `docs/relatorio_tecnico.md`.
