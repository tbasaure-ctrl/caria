import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

S0 = 100_000.0      
mu = 0.10          
sigma = 0.25         
anios = 5            
pasos_por_anio = 12  
rutas = 10000       
aporte_por_paso = 0.0
fee_anual = 0.00     
semilla = 42         

np.random.seed(semilla)
dt = 1.0 / pasos_por_anio
n = int(anios * pasos_por_anio)
mu_eff = mu - fee_anual
drift = (mu_eff - 0.5 * sigma**2) * dt
dif = sigma * sqrt(dt)

Z = np.random.standard_normal((rutas, n)) 
paths = np.empty((rutas, n+1), float)
paths[:, 0] = S0
for t in range(n):
    paths[:, t+1] = paths[:, t] * np.exp(drift + dif * Z[:, t]) + aporte_por_paso

finales = paths[:, -1]
invertido = S0 + aporte_por_paso * n
moic = finales / max(invertido, 1e-9)


def pctl(x, q): return float(np.percentile(x, q))
var5 = pctl(finales - invertido, 5)
cvar5 = float((finales - invertido)[(finales - invertido) <= var5].mean())

resumen = pd.DataFrame([{
    "Horizonte_anios": anios,
    "Inicial": S0,
    "Invertido": invertido,
    "Rutas": rutas,
    "Pasos_por_anio": pasos_por_anio,
    "Mu_anual": mu,
    "Sigma_anual": sigma,
    "Fee_anual": fee_anual,
    "Aporte_por_paso": aporte_por_paso,
    "Final_media": float(np.mean(finales)),
    "Final_mediana": float(np.median(finales)),
    "Final_P5": pctl(finales, 5),
    "Final_P50": pctl(finales, 50),
    "Final_P95": pctl(finales, 95),
    "Prob_final_menor_invertido": float(np.mean(finales < invertido)),
    "MOIC_mediana": float(np.median(moic)),
    "VaR_5pct_$": var5,
    "CVaR_5pct_$": cvar5
}])

resumen.to_csv("resumen.csv", index=False)
pd.DataFrame(paths[:50, :]).to_csv("paths_muestra.csv", index=False)

plt.figure(); plt.hist(finales, bins=60)
plt.xlabel("Valor final"); plt.ylabel("Frecuencia")
plt.title("Distribución de valores finales"); plt.savefig("histograma.png", bbox_inches="tight"); plt.close()

plt.figure()
t = np.arange(n+1) / pasos_por_anio
for i in range(min(50, rutas)):
    plt.plot(t, paths[i, :])
plt.xlabel("Años"); plt.ylabel("Valor")
plt.title("Caminos simulados (muestra)")
plt.savefig("caminos.png", bbox_inches="tight"); plt.close()

print("Listo. Archivos: resumen.csv, paths_muestra.csv, histograma.png, caminos.png")
