# Ficha do adaptador clínico

## Finalidade

Adaptador LoRA destinado à organização de respostas de apoio baseadas em contexto institucional. Não é um dispositivo médico e não deve ser utilizado para diagnóstico, prescrição ou decisão autônoma.

## Modelo base

`meta-llama/Llama-2-7b-hf`, condicionado à licença e ao acesso concedido pelo mantenedor.

## Dados

Exemplos inteiramente sintéticos de perguntas frequentes, protocolos, registros e recusas de segurança. O relatório gerado por `scripts/prepare_data.py` informa distribuição, fontes e resultado da verificação de identificadores.

## Técnica

Fine-tuning supervisionado com quantização de 4 bits e adaptadores LoRA. Somente os parâmetros dos adaptadores são atualizados.

## Usos inadequados

- atendimento direto ao paciente sem supervisão;
- definição ou confirmação diagnóstica;
- escolha, dose, início ou suspensão de tratamento;
- uso com dados pessoais sem base legal, controles e governança;
- extrapolação para especialidades ou populações não avaliadas.

## Avaliação necessária

Perda de validação, comparação com modelo base, rastreabilidade de fontes, taxa de segurança, testes adversariais, análise de vieses e revisão clínica independente.
