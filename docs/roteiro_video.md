# Roteiro de demonstração — duração aproximada de 12 minutos

## 0:00–1:00 — problema e limites

Apresentar a finalidade de apoio informacional, os dados sintéticos e a regra de revisão humana. Destacar que não há diagnóstico ou prescrição autônoma.

## 1:00–3:00 — dados e preparação

Abrir `01_preparacao_dados.ipynb`. Mostrar as categorias, um exemplo antes e depois da anonimização, a validação estrutural e a separação de treino e teste.

## 3:00–5:30 — modelo customizado

Abrir `02_fine_tuning_lora.ipynb`. Explicar modelo base, tokenização, quantização em 4 bits, PEFT e adaptadores LoRA. Mostrar os hiperparâmetros, a contagem de parâmetros treináveis e a curva de perda obtida na execução em GPU. Comparar uma resposta do modelo base com a do adaptador, sem omitir casos de erro.

## 5:30–7:30 — LangChain e contexto

Abrir `03_assistente_langchain.ipynb`. Consultar `PAC-0001`, mostrar exames pendentes e recuperar o protocolo de dor torácica. Exibir o prompt composto com prontuário e fontes.

## 7:30–10:00 — fluxo automatizado

Abrir `04_fluxo_langgraph.ipynb`. Executar a pergunta: “Há dor torácica e falta de ar. Quais exames estão pendentes?”. Mostrar alerta, resposta, fontes e lista de nós executados. Em seguida, solicitar uma prescrição direta e mostrar a recusa.

## 10:00–11:00 — logs e validação

Abrir o log JSONL. Evidenciar hash, fontes, rota e etapas, confirmando que a pergunta não foi gravada em texto aberto.

## 11:00–12:00 — avaliação e limitações

Abrir `05_avaliacao.ipynb`. Apresentar métricas, testes automatizados, limitações do corpus sintético e necessidade de validação por especialistas antes de qualquer uso real.
