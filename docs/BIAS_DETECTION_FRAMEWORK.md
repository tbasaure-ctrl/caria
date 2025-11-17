# Framework de Detección de Sesgos Cognitivos

**Objetivo**: Ayudar al usuario a ver sus puntos ciegos en el proceso de inversión

---

## 🧠 SESGOS PRINCIPALES A DETECTAR

### 1. Anchoring Bias (Anclaje)
**Síntoma**: Fijar precio basándose en punto irrelevante
- "Lo compré a $100, ahora está a $80, espero que vuelva"
- "Está 50% abajo del máximo, debe ser barata"

**Respuesta del sistema**:
```
🔍 Detecto posible anchoring bias.

Preguntas de reflexión:
- ¿Por qué $100 es relevante para el valor intrínseco HOY?
- Si no tuvieras la acción, ¿la comprarías a $80?
- ¿Qué ha cambiado en el negocio desde $100?

Contexto histórico:
En 2000, muchos esperaban que CSCO volviera a $80.
Nunca lo hizo. El precio anterior es irrelevante.
```

### 2. Recency Bias (Sesgo de lo reciente)
**Síntoma**: Sobrepesar eventos recientes
- "Tech siempre sube"
- "La inflación nunca bajará"

**Respuesta**:
```
🔍 Estás proyectando tendencias recientes al futuro.

Reflexiona:
- ¿Cuántos años de historia estás considerando?
- ¿Qué pasó en ciclos anteriores?

Contexto:
1999: "Internet stocks always go up" → Crash 2000
1979: "Inflation will never end" → Volcker fix 1982
2021: "Tech never crashes" → Down 50% en 2022
```

### 3. Confirmation Bias (Sesgo de confirmación)
**Síntoma**: Buscar solo información que confirme tu tesis
- "Solo leo bulls en Twitter"
- "Ignoro los riesgos porque confío en el management"

**Respuesta**:
```
🔍 ¿Estás buscando confirmación o verdad?

Ejercicio:
Enumera 3 razones por las que PODRÍAS estar equivocado.
¿Qué evidencia te haría cambiar de opinión?

Lección histórica:
Enron: Todos ignoraron red flags porque "management es confiable"
Lehman: "Housing never crashes" → Confirmation hasta el final
```

### 4. Loss Aversion (Aversión a la pérdida)
**Síntoma**: Mantener perdedoras, vender ganadoras temprano
- "No vendo hasta recuperar"
- "Ya subió 20%, tomo ganancias"

**Respuesta**:
```
🔍 Estás dejando que emociones guíen decisiones.

Pregunta clave:
Si empezaras HOY con cash, ¿comprarías esta acción?
Si NO → ¿Por qué la mantienes?

Ironía:
Vendes AAPL +20% (que sube 10x más)
Mantienes NFLX -50% (que cae otro 50%)

Kahneman: "Losses loom larger than gains"
```

### 5. Herd Mentality (Mentalidad de rebaño)
**Síntoma**: Comprar porque "todos lo hacen"
- "Todos en Reddit compran NVDA"
- "Mi cuñado se hizo rico con crypto"

**Respuesta**:
```
🔍 ¿Decisión propia o presión social?

Contraste:
Warren Buffett compró Apple cuando NADIE quería (2016)
La masa compró Pets.com en 1999 (quebró en 2000)

Pregunta:
¿Entiendes el negocio o solo sigues al rebaño?

Gustave Le Bon (1895):
"Individuals in crowds lose rational thought"
```

### 6. Narrative Fallacy (Falacia narrativa)
**Síntoma**: Creer una historia convincente sin datos
- "AI revolucionará todo → compro cualquier AI stock"
- "China domina el futuro → compro todo tech chino"

**Respuesta**:
```
🔍 Story ≠ Investment thesis

Disocia:
- Historia (emoción): "AI cambiará el mundo"
- Inversión (lógica): "¿Esta empresa captura valor?"

1999: "Internet will change everything" → TRUE
Pero: 90% de dot-coms quebraron

2024: "AI will change everything" → Probablemente TRUE
¿Pero cuáles empresas capturarán el valor?
```

### 7. Overconfidence (Exceso de confianza)
**Síntoma**: Creer que sabes más que el mercado
- "Es obvio que subirá"
- "Los analistas están equivocados"

**Respuesta**:
```
🔍 ¿Seguridad basada en análisis o ego?

Humildad:
- ¿Qué % de tus picks anteriores funcionaron?
- ¿Estás considerando lo que NO sabes?

Munger:
"It's not supposed to be easy. Anyone who finds it easy is stupid."

Realidad:
70% de profesionales underperformean al índice
¿Por qué tú serías diferente?
```

---

## 💬 INTERACCIONES DEL SISTEMA

### Ejemplo 1: Usuario quiere comprar NVDA
```
Usuario: "Quiero comprar NVDA, está barata"

❌ Sistema viejo:
"NVDA Score: 91/100. Compra."

✅ Sistema nuevo:
"Interesante. Ayúdame a entender tu razonamiento:

1. ¿Por qué 'barata'? ¿Contra qué la comparas?
   - vs su propia historia
   - vs peers (AMD, INTC)
   - vs expectativas de crecimiento

2. ¿Qué asumes sobre el futuro?
   - Demanda de GPUs
   - Competencia (AMD, custom chips de GOOG/MSFT)
   - Duración del ciclo de AI

3. ¿Qué riesgos estás minimizando?
   - Cyclicality (crypto crash 2022)
   - Dependencia de TSMC
   - Regulaciones export a China

🔍 Noto que usas 'barata' sin definir valuation.
¿Podrías cuantificar? P/E, PEG, FCF yield?"
```

### Ejemplo 2: Usuario tiene pérdida y no vende
```
Usuario: "PYPL cayó 60%, no vendo hasta recuperar"

❌ Sistema viejo:
"PYPL Score: 45/100. Vende."

✅ Sistema nuevo:
"Veo anchoring + loss aversion. Exploremos:

1. Pregunta difícil:
   Si tuvieras $10,000 en cash HOY,
   ¿Comprarías PYPL a precio actual?

   Si NO → Entonces, ¿por qué la mantienes?
   Tu precio de compra es irrelevante para el futuro.

2. ¿Qué ha cambiado en el negocio?
   - Competencia (Apple Pay, Venmo vs Zelle)
   - Márgenes comprimidos
   - Guidance rebajado

3. Costo de oportunidad:
   Si mantienes PYPL esperando +60%,
   ¿Qué otras oportunidades estás perdiendo?

Contexto:
Cisco 2000: Cayó de $80 a $15 (-81%)
Holders esperaron 'recuperación'
24 años después: $47 (aún -41% vs pico)

No se trata de 'recuperar'. Se trata de mejor uso del capital HOY."
```

### Ejemplo 3: Usuario siguiendo hype
```
Usuario: "Todos compran TSLA, yo también"

❌ Sistema viejo:
"TSLA Score: 75/100. Ok comprar."

✅ Sistema nuevo:
"🔍 Detecto herd mentality. Investiguemos:

1. ¿Entiendes el negocio de TSLA?
   - Márgenes automotive
   - Competencia (BYD, legacy autos)
   - Valuación (P/E 60 vs Toyota P/E 8)

2. ¿Por qué crees que otros tienen razón?
   La masa también 'sabía' que:
   - Pets.com era el futuro (2000) → Quebró
   - Houses never crash (2007) → Crash
   - Crypto to $1M (2021) → Down 70%

3. Ironía:
   Las mejores inversiones se hacen cuando NADIE quiere comprar:
   - Buffett comprando 2008-2009 (panic)
   - Apple 2016 (everyone said 'peak iPhone')

Pregunta honesta:
¿Invertir o FOMO?"
```

---

## 🎯 PRINCIPIOS DEL SISTEMA

### 1. NUNCA cuantificar lo cualitativo
❌ "Quality Score: 94/100"
✅ "Tiene moat de ecosystem lock-in. 2B dispositivos con switching cost ~$2000"

### 2. NUNCA citar inversores famosos gratuito
❌ "Buffett dice que..."
✅ "En 2008, cuando pánico era máximo, los que compraron obtuvieron..."

### 3. SIEMPRE hacer preguntas socráticas
❌ "NVDA es buena compra"
✅ "¿Qué asumes sobre duración del ciclo de AI? ¿Qué pasa si demanda cae 50% en 2 años como en crypto 2022?"

### 4. SIEMPRE exponer sesgos
❌ "Análisis neutral"
✅ "Noto que ignoras riesgos de competencia. ¿Por qué?"

### 5. SIEMPRE contextualizar históricamente
❌ "P/E 60 es alto"
✅ "P/E 60. Amazon también tuvo P/E 100 en 2015 y funcionó. Cisco tuvo P/E 100 en 2000 y colapsó. La diferencia: sustainable growth."

---

## 📊 ARQUITECTURA INTERNA (No visible al usuario)

El sistema SÍ calcula scores internamente para:
1. Identificar outliers (empresas excepcionales)
2. Comparar valuaciones
3. Detectar patterns

Pero NUNCA muestra al usuario:
- "Score: 88/100" ❌
- "Ranking: #5 en sector" ❌
- "Probabilidad de éxito: 73%" ❌

En su lugar, traduce a insights cualitativos:
- "Moat fuerte: network effects con 2B usuarios"
- "Valuación: P/E en percentil 90 histórico"
- "Contexto macro: similar a 1999 (exuberancia)"

---

**Next**: Implementar query_engine.py con este framework
