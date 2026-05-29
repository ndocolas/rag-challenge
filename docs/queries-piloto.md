# Pilot Queries

Battery of 10 questions for manual end-to-end validation. Run after full ingestion.

Passing criterion: **≥ 8/10** with acceptable response.

---

## Pharmacological (RAG)

**1. Ritalina contraindications**

> "Quais são as contraindicações da Ritalina?"

Expected:
- Tool `buscar_bulas` invoked with `med_name="Ritalina"`, `section_hint` for contraindications
- `sources` event cites Ritalina leaflet, section `IAP_3` or `IT_CONTRAINDICACOES`
- Response mentions hyperthyroidism, glaucoma, concomitant MAOI use
- Medical disclaimer at the end

---

**2. Pantoprazole dosage for adults**

> "Qual a dose recomendada de pantoprazol para adultos?"

Expected:
- Tool `buscar_bulas` with `med_name` containing "pantoprazol", section hint posology
- `sources` event cites leaflet 805950, section `IAP_6` or `IT_POSOLOGIA`
- Response mentions 40 mg/day on an empty stomach

---

**3. Adverse reactions of tramadol with paracetamol**

> "Quais as reações adversas do tramadol com paracetamol?"

Expected:
- Tool `buscar_bulas` with appropriate `med_name`, section hint adverse reactions
- `sources` event cites leaflet 93790, section `IAP_8` or `IT_REACOES_ADVERSAS`
- Response lists nausea, dizziness, drowsiness as most frequent

---

**4. Gestinol with antibiotics**

> "O Gestinol pode ser usado junto com antibióticos?"

Expected:
- Tool `buscar_bulas` with `med_name="Gestinol"`, section hint interactions
- `sources` event cites leaflet 438950, section `IT_INTERACOES_MEDICAMENTOSAS`
- Response covers interaction with rifampicin and possible efficacy reduction

---

**5. Missed memantine dose**

> "Esqueci de tomar a memantina, o que devo fazer?"

Expected:
- Tool `buscar_bulas` with `med_name` containing "memantina"
- `sources` event cites leaflet 111824, section `IAP_7`
- Response advises: take as soon as remembered, do not double dose

---

## Branches (tool calling)

**6. Stores in Curitiba with Clinic and 24h service**

> "Quais lojas em Curitiba têm Clinic e funcionam 24 horas?"

Expected:
- Tool `buscar_filiais` invoked with filters `cidade="Curitiba"`, `clinic=true`, `horario_24h=true`
- `tool_result` event returns list with branch 1557 (or equivalent)
- Response presents name, address and hours

---

**7. Store in Florianópolis**

> "Vocês têm alguma loja em Florianópolis?"

Expected:
- Tool `listar_cidades_atendidas` invoked
- Response informs that the service covers Paraná only; Florianópolis is not on the list
- Helpful tone, suggests checking assistant channels for other regions

---

**8. Details of branch 1761**

> "Preciso dos detalhes completos da filial 1761."

Expected:
- Tool `detalhes_filial` invoked with `codigo_filial=1761`
- `tool_result` event returns full record (Apucarana-PR)
- Response presents address, phone, hours and available services

---

## Multi-turn

**9. Anaphora across turns**

> Turn 1: "Para que serve a paroxetina?"
> Turn 2: "E quais são os efeitos colaterais?"

Expected:
- Turn 2 resolves the anaphora ("efeitos colaterais" → paroxetina) via Redis history
- Tool `buscar_bulas` with appropriate `med_name`, section hint adverse reactions
- `sources` event cites leaflet 346659, section `IAP_8` or `IT_REACOES_ADVERSAS`
- Response lists side effects without asking for medication confirmation

---

## Out of scope

**10. Out-of-domain question**

> "Quanto custa um Uber até a filial mais próxima?"

Expected:
- No tool invoked
- Response declines politely and objectively
- Redirects to scope: medication information or branches in PR
