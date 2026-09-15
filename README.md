# Assistente clínico institucional

Protótipo local que organiza informações clínicas sintéticas para revisão profissional. Ele combina ajuste fino de uma LLM, consulta a dados estruturados, recuperação de protocolos e um fluxo de segurança com LangChain e LangGraph.

O sistema não diagnostica, não prescreve, não altera prontuários e não envia alertas externos.

## Instalação

Use Python 3.11 ou 3.12 de 64 bits. Na raiz do projeto, crie o ambiente e instale as dependências:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Selecione `.venv\Scripts\python.exe` como interpretador no editor de notebooks.

## Execução pelos notebooks

Execute os notebooks na ordem indicada. Cada um explica o que está sendo feito, por que a etapa é necessária e quais limitações devem ser consideradas.

1. `01_preparacao_dados.ipynb` prepara, valida e separa os dados sintéticos.
2. `02_treinamento_lora.ipynb` realiza o ajuste fino quando ainda não existem resultados locais.
3. `03_assistente_langchain.ipynb` consulta o banco local, recupera protocolos e monta o contexto da LLM.
4. `04_fluxo_langgraph.ipynb` apresenta o diagrama, as decisões de segurança e a auditoria.
5. `05_avaliacao.ipynb` compara respostas, executa os casos integrados e roda os testes automatizados.

Na primeira execução do notebook de treinamento, o modelo base será obtido e o adaptador LoRA será criado localmente. Em seguida, os notebooks posteriores usarão esse adaptador. O treinamento pode levar alguns minutos em CPU.

## Arquivos do projeto

| Local | Conteúdo |
|---|---|
| `data/raw` | exemplos, protocolos, pacientes e exames sintéticos |
| `notebooks` | percurso de execução e explicação do projeto |
| `src/clinical_assistant` | módulos da aplicação |
| `docs/relatorio_tecnico.pdf` | relatório técnico final |
| `docs/langgraph_flow.svg` | diagrama do fluxo LangGraph |
| `tests` | testes automatizados |

Os diretórios `data/processed`, `models`, `logs`, `artifacts` e `docs/results` são gerados localmente e não devem ser versionados. Os notebooks versionados permanecem sem saídas de execução.

## Limitações

Os dados são sintéticos e o modelo não foi validado para uso clínico. A recuperação depende de termos presentes nos protocolos, e a resposta gerada pode ser retida quando não houver suporte suficiente no contexto. Toda saída requer revisão profissional.
