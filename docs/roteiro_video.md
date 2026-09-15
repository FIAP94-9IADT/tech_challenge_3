# Roteiro de apresentação

O vídeo deve mostrar a execução local, sem apresentar o sistema como ferramenta de atendimento. A duração prevista é de até quinze minutos.

1. Apresente a finalidade do protótipo e a composição do corpus sintético.
2. Abra o notebook `01_preparacao_dados.ipynb`, execute a preparação e explique as verificações de estrutura, anonimização e separação entre treino, validação e teste.
3. Abra o notebook `02_treinamento_lora.ipynb`, mostre a configuração do ajuste LoRA e explique que o adaptador altera apenas uma pequena parte dos parâmetros do modelo base.
4. No notebook `03_assistente_langchain.ipynb`, execute uma consulta estruturada, a recuperação de protocolos e a montagem do contexto enviado ao modelo.
5. No notebook `04_fluxo_langgraph.ipynb`, apresente o diagrama e execute um alerta, uma recusa e uma pergunta com contexto disponível. Mostre o registro de auditoria sem expor texto livre.
6. No notebook `05_avaliacao.ipynb`, apresente a comparação entre o modelo base e o adaptador, os testes automatizados e as limitações observadas.
7. Encerre destacando que qualquer conteúdo disponibilizado pelo sistema depende de revisão profissional e que o protótipo não prescreve, diagnostica nem envia alertas externos.
