# Queries piloto

Bateria de 10 perguntas para validação manual end-to-end. Executar após ingestão completa.

Critério de aprovação: **≥ 8/10** com resposta aceitável.

---

## Farmacológicas (RAG)

**1. Contraindicações da Ritalina**

> "Quais são as contraindicações da Ritalina?"

Esperado:
- Tool `buscar_bulas` invocada com `med_name="Ritalina"`, `section_hint` para contraindicações
- Evento `sources` cita bula Ritalina, seção `IAP_3` ou `IT_CONTRAINDICACOES`
- Resposta menciona hipertireoidismo, glaucoma, uso concomitante de IMAO
- Disclaimer médico ao final

---

**2. Posologia do pantoprazol em adultos**

> "Qual a dose recomendada de pantoprazol para adultos?"

Esperado:
- Tool `buscar_bulas` com `med_name` contendo "pantoprazol", section hint posologia
- Evento `sources` cita bula 805950, seção `IAP_6` ou `IT_POSOLOGIA`
- Resposta menciona 40 mg/dia em jejum

---

**3. Reações adversas do tramadol com paracetamol**

> "Quais as reações adversas do tramadol com paracetamol?"

Esperado:
- Tool `buscar_bulas` com `med_name` adequado, section hint reações adversas
- Evento `sources` cita bula 93790, seção `IAP_8` ou `IT_REACOES_ADVERSAS`
- Resposta lista náusea, tontura, sonolência como mais frequentes

---

**4. Gestinol com antibióticos**

> "O Gestinol pode ser usado junto com antibióticos?"

Esperado:
- Tool `buscar_bulas` com `med_name="Gestinol"`, section hint interações
- Evento `sources` cita bula 438950, seção `IT_INTERACOES_MEDICAMENTOSAS`
- Resposta aborda interação com rifampicina e possível redução de eficácia

---

**5. Dose esquecida de memantina**

> "Esqueci de tomar a memantina, o que devo fazer?"

Esperado:
- Tool `buscar_bulas` com `med_name` contendo "memantina"
- Evento `sources` cita bula 111824, seção `IAP_7`
- Resposta orienta: tomar assim que lembrar, não duplicar dose

---

## Filiais (tool calling)

**6. Lojas em Curitiba com Clinic e atendimento 24h**

> "Quais lojas em Curitiba têm Clinic e funcionam 24 horas?"

Esperado:
- Tool `buscar_filiais` invocada com filtros `cidade="Curitiba"`, `clinic=true`, `horario_24h=true`
- Evento `tool_result` retorna lista com filial 1557 (ou equivalente)
- Resposta apresenta nome, endereço e horário

---

**7. Loja em Florianópolis**

> "Vocês têm alguma loja em Florianópolis?"

Esperado:
- Tool `listar_cidades_atendidas` invocada
- Resposta informa que o serviço atende apenas Paraná; Florianópolis não está na lista
- Tom útil, sugere verificar canais do assistente para outras regiões

---

**8. Detalhes da filial 1761**

> "Preciso dos detalhes completos da filial 1761."

Esperado:
- Tool `detalhes_filial` invocada com `codigo_filial=1761`
- Evento `tool_result` retorna cadastro completo (Apucarana-PR)
- Resposta apresenta endereço, telefone, horário e serviços disponíveis

---

## Multi-turno

**9. Anáfora entre turnos**

> Turno 1: "Para que serve a paroxetina?"
> Turno 2: "E quais são os efeitos colaterais?"

Esperado:
- Turno 2 resolve a anáfora ("efeitos colaterais" → paroxetina) via histórico Redis
- Tool `buscar_bulas` com `med_name` adequado, section hint reações adversas
- Evento `sources` cita bula 346659, seção `IAP_8` ou `IT_REACOES_ADVERSAS`
- Resposta lista efeitos colaterais sem pedir confirmação do medicamento

---

## Fora de escopo

**10. Pergunta fora do domínio**

> "Quanto custa um Uber até a filial mais próxima?"

Esperado:
- Nenhuma tool invocada
- Resposta recusa de forma educada e objetiva
- Redireciona para o escopo: informações sobre medicamentos ou filiais do PR