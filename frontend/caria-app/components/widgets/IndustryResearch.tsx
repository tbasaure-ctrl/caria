import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';

// Interfaces
interface StockPick {
    ticker: string;
    name: string;
    thesis: string;
    type: 'Value' | 'Growth' | 'Turnaround' | 'Defensive' | 'Speculative';
}

interface IndustryReport {
    id: string;
    title: string;
    subtitle: string;
    icon: string;
    readTime: string;
    tags: string[];
    isFeatured?: boolean;
    content: {
        overview: string; // Short summary for card
        fullText: string; // Full text for modal
        trends: { title: string; description: string }[];
        picks: StockPick[];
        conclusion?: string;
    };
}

// DATA: Contenido COMPLETO del informe proporcionado
const REPORT_DATA: IndustryReport[] = [
    {
        id: 'staples-nov-2025',
        title: 'Consumo Básico (Consumer Staples)',
        subtitle: 'INDUSTRIA DEL MES: Refugio Táctico y Valor',
        icon: '🛒',
        readTime: '8 min read',
        tags: ['Industry of the Month', 'Defensive', 'High Conviction'],
        isFeatured: true,
        content: {
            overview: `Designado como la industria focal para Noviembre 2025. Históricamente, este sector actúa como un "proxy de bonos" con la ventaja del crecimiento del dividendo. Ante la incertidumbre económica, los inversores buscan la seguridad de la demanda inelástica.`,
            fullText: `1. Industria del Mes: Consumo Básico (Consumer Staples)

1.1 Tesis de Inversión y Racionalidad de la Selección
La designación del sector de Consumo Básico como la industria focal para noviembre de 2025 responde a una confluencia de factores técnicos, fundamentales y estacionales que rara vez se alinean con tanta precisión. A menudo malinterpretado como un refugio aburrido para inversores conservadores, el sector está experimentando una transformación interna y una dispersión de valoraciones que ofrece oportunidades de generación de alfa significativas para el inversor activo.

1.1.1 La Rotación Defensiva en un Entorno de Volatilidad
Durante gran parte de 2024 y el inicio de 2025, el capital fluyó desproporcionadamente hacia la tecnología y los servicios de comunicación, impulsado por la promesa de la inteligencia artificial. Sin embargo, a medida que las valoraciones en esos sectores se estiraron y los rendimientos de los bonos comenzaron a estabilizarse, se ha observado una rotación clásica hacia la defensa.
El mecanismo detrás de este movimiento es doble. Primero, la compresión de los rendimientos de los bonos del Tesoro hace que los dividendos de las empresas de consumo básico sean comparativamente más atractivos. Históricamente, este sector actúa como un "proxy de bonos" con la ventaja añadida del crecimiento del dividendo. Segundo, ante la incertidumbre de si la economía estadounidense puede mantener su ritmo de crecimiento sin reavivar la inflación, los inversores buscan la seguridad de la demanda inelástica: la gente sigue comprando pasta de dientes, alimentos y productos de limpieza independientemente del PIB.

1.1.2 Estacionalidad Histórica: El "Efecto Noviembre"
El análisis cuantitativo de los patrones de mercado revela que noviembre es, estadísticamente, un mes excepcionalmente fuerte para el sector de consumo básico. Al examinar el comportamiento del ETF Consumer Staples Select Sector SPDR Fund (XLP) durante los últimos 25 años, se identifican tendencias claras de estacionalidad positiva.
Noviembre muestra una continuación robusta del impulso iniciado en octubre y ofrece la mayor probabilidad histórica de retornos positivos en el Q4 (75%). Este patrón se atribuye a menudo al posicionamiento de los gestores de fondos antes del cierre del año fiscal y al aumento tangible en el consumo de productos básicos durante la temporada festiva.

1.2 Análisis Fundamental Profundo: Tendencias y Divergencias
El sector de consumo básico en noviembre de 2025 no es un bloque monolítico. Existe una divergencia crítica en las valoraciones y el desempeño operativo entre los grandes minoristas y los fabricantes de productos empaquetados.

1.2.1 La Bifurcación de Valoraciones: Minoristas vs. Fabricantes
Una de las anomalías más notables del mercado actual es la extrema dispersión en los múltiplos de valoración dentro del mismo sector GICS.
El segmento sobrevalorado (Retailers): Empresas como Costco (COST) y Walmart (WMT) cotizan a múltiplos de precio/ganancias (P/E) que rivalizan con las acciones de crecimiento tecnológico de alto vuelo (>40x-50x). Estas valoraciones descuentan un escenario de ejecución perfecta y crecimiento perpetuo difícil de justificar.
El segmento infravalorado (Packaged Food): En contraste agudo, si excluimos a estos gigantes minoristas, el resto del sector cotiza con un descuento atractivo (~11% bajo valor razonable). Empresas sólidas como Kraft Heinz (KHC) y General Mills (GIS) han sido penalizadas excesivamente por temores exagerados.

1.2.2 El Impacto de los Agonistas GLP-1: Realidad vs. Histeria
Durante 2023 y 2024, una sombra se cernió sobre el sector debido a los medicamentos GLP-1. Hacia finales de 2025, esta visión se ha matizado. Las grandes empresas están pivotando (Nestlé, General Mills) lanzando productos altos en proteína diseñados para usuarios de GLP-1, y los volúmenes de ventas en categorías clave se han mantenido estables.

1.2.3 Compresión de Márgenes y la Batalla de la Marca Privada
La inflación acumulada ha llevado a una "bajada de categoría" (trade-down). Las empresas con fuerte "pricing power" han logrado mantener márgenes brutos mediante eficiencias operativas, superando estimaciones de EPS a pesar de un crecimiento de ingresos modesto.`,
            trends: [
                {
                    title: "El Efecto Noviembre",
                    description: "Estadísticamente, noviembre es excepcionalmente fuerte para el sector (75% de frecuencia de ganancias)."
                },
                {
                    title: "Adaptación a GLP-1",
                    description: "Lanzamiento de productos altos en proteína para acompañar a usuarios de Ozempic/Wegovy."
                }
            ],
            picks: [
                { ticker: 'KHC', name: 'Kraft Heinz', type: 'Value', thesis: 'Infravaloración extrema. Reestructuración de deuda exitosa y mejora de márgenes ignorada por el mercado.' },
                { ticker: 'GIS', name: 'General Mills', type: 'Defensive', thesis: 'Jugador defensivo clásico. Adaptación superior a tendencias de salud (Blue Buffalo).' },
                { ticker: 'SFM', name: 'Sprouts Farmers Market', type: 'Growth', thesis: 'Beneficiario del auge de alimentación saludable/GLP-1. Expansión de márgenes con productos frescos.' },
                { ticker: 'OLLI', name: "Ollie's Bargain Outlet", type: 'Growth', thesis: 'Modelo "caza del tesoro" ideal para consumidor sensible al precio. Adquisición de exceso de inventario.' },
                { ticker: 'EL', name: 'Estée Lauder', type: 'Turnaround', thesis: 'Valoración deprimida por Asia. Potencial rebote violento si estabiliza inventarios dada su marca.' }
            ]
        }
    },
    {
        id: 'macro-nov-2025',
        title: 'Estrategia Macro Global',
        subtitle: 'Panorama Económico & Asset Allocation',
        icon: '🌍',
        readTime: '5 min read',
        tags: ['Macro', 'Strategy'],
        content: {
            overview: `El mercado entra en una fase de rotación táctica. La Fed ajusta tasas al rango 3.75%-4.00%. Rotación desde "growth at any price" hacia calidad y balance.`,
            fullText: `Panorama Macroeconómico y Estrategia de Asignación de Activos - Noviembre 2025

El penúltimo mes de 2025 se despliega en un contexto económico global que desafía las categorizaciones simplistas de "aterrizaje suave" o "recesión inminente". Los mercados financieros, tras un año marcado por la euforia tecnológica y la recalibración de las expectativas de política monetaria, han entrado en una fase de rotación táctica distintiva. A medida que la Reserva Federal y otros bancos centrales importantes ajustan sus tasas de interés—recortando recientemente al rango de 3.75%-4.00%—los inversores se encuentran reevaluando la prima de riesgo en sus carteras.

La narrativa predominante ha girado desde la búsqueda desenfrenada de crecimiento ("growth at any price") hacia una apreciación renovada por la calidad del balance, la previsibilidad de los flujos de caja y la resiliencia operativa. Este cambio de sentimiento no es un accidente, sino una respuesta racional a un entorno donde, si bien la inflación se ha enfriado considerablemente desde los picos de años anteriores, los costos de endeudamiento permanecen en niveles que penalizan a las empresas con apalancamiento excesivo o modelos de negocio no probados.

En este escenario, noviembre de 2025 emerge como un punto de inflexión crítico. Históricamente, este mes ha servido como un barómetro para el posicionamiento de fin de año, y los datos actuales sugieren una bifurcación clara: mientras los sectores cíclicos enfrentan vientos en contra por la desaceleración económica secuencial prevista para 2026, los sectores defensivos y de innovación sanitaria están capturando la atención del capital institucional.`,
            trends: [
                { title: "Rotación a Calidad", description: "Preferencia por flujos de caja predecibles y balances sólidos." },
                { title: "Bifurcación", description: "Sectores defensivos y salud capturan capital institucional vs cíclicos." }
            ],
            picks: [],
            conclusion: "Recomendación Final: Construir una cartera 'barbell' (pesa): un núcleo defensivo robusto en consumo básico y seguros de nicho, equilibrado con apuestas satélite de alto crecimiento en robótica médica y biotecnología con catalizadores cercanos."
        }
    },
    {
        id: 'pharma-nov-2025',
        title: 'Salud y Farmacéutica',
        subtitle: 'Innovación y Boom de M&A',
        icon: '🧬',
        readTime: '4 min read',
        tags: ['Biotech', 'M&A'],
        content: {
            overview: `Ecosistema en tensión por "patent cliff" y regulación, catalizando innovación y M&A. Big Pharma compra crecimiento (Oncología, Neurociencia, Obesidad).`,
            fullText: `2. Sector Salud y Farmacéutica: Innovación bajo Presión Regulatoria

2.1 Estado de la Industria: Un Ecosistema en Tensión
El sector de Salud y Farmacéutica presenta una dicotomía fascinante en noviembre de 2025. Por un lado, se enfrenta a vientos en contra regulatorios y de mercado significativos: la expiración de patentes clave (el "patent cliff"), la presión sobre los precios de los medicamentos en EE.UU. debido a las negociaciones de Medicare, y un entorno de financiación difícil para las pequeñas biotecnológicas. Por otro lado, esta presión está actuando como un catalizador para una innovación desenfrenada y una consolidación agresiva.

2.2 Tendencias Dominantes
2.2.1 El Renacimiento de las Fusiones y Adquisiciones (M&A)
Ante la inminente pérdida de exclusividad de sus medicamentos más vendidos, las grandes farmacéuticas ("Big Pharma") están desplegando sus balances para comprar crecimiento. 2025 ha sido testigo de una oleada de acuerdos estratégicos (Merck, Sanofi, Novartis, Lilly). Esta tendencia valida la tesis de que la innovación más valiosa está ocurriendo fuera de los laboratorios internos de las grandes corporaciones, en el ecosistema biotecnológico de mediana y pequeña capitalización.

2.2.2 Áreas Terapéuticas de Alto Valor
La inversión se concentra en verticales donde la ciencia está rompiendo barreras históricas:
- Oncología: ADCs y T-cell engagers.
- Neurociencia: Renacimiento en tratamientos para Alzheimer y Esquizofrenia.
- Obesidad y Metabolismo: Próxima generación de tratamientos metabólicos (mejor tolerabilidad/preservación muscular).

2.3 Perspectivas y Oportunidades de Inversión
La perspectiva para finales de 2025 y principios de 2026 es de volatilidad continua pero con oportunidades asimétricas en biotecnología. Investigar "más allá de las clásicas" implica mirar empresas con catalizadores binarios (lecturas de datos clínicos).`,
            trends: [
                { title: "M&A Renacimiento", description: "Big Pharma desplegando capital para comprar innovación externa." },
                { title: "Áreas Hot", description: "Oncología, Neurociencia y Metabolismo (Next-gen Obesity)." }
            ],
            picks: [
                { ticker: 'KALA', name: 'Kala Bio', type: 'Speculative', thesis: 'Catalizador binario a fin de 2025 (Fase 2b CHASE). Enfermedad ocular rara sin cura.' },
                { ticker: 'KAPA', name: 'Kairos Pharma', type: 'Speculative', thesis: 'Datos interinos Fase 2 cáncer próstata. Área oncológica lucrativa.' }
            ]
        }
    },
    {
        id: 'medtech-nov-2025',
        title: 'Dispositivos Médicos',
        subtitle: 'Revolución: Robótica e IA',
        icon: '🦾',
        readTime: '4 min read',
        tags: ['Growth', 'Tech'],
        content: {
            overview: `Crecimiento estructural predecible (CAGR 6%). IA operativa en diagnósticos y auge de robótica quirúrgica y dispositivos desechables.`,
            fullText: `3. Sector de Dispositivos Médicos: La Revolución Silenciosa de la Tecnología Sanitaria

3.1 Caracterización: Crecimiento Estructural y Resiliencia
A diferencia de la biotecnología, que a menudo depende de resultados binarios de ensayos clínicos, el sector de dispositivos médicos ofrece una trayectoria de crecimiento más predecible, impulsada por la demografía (envejecimiento global) y la necesidad de eficiencia hospitalaria. Se proyecta que el mercado global alcance los $678.8 mil millones en 2025.

3.2 Tendencias Tecnológicas y de Mercado
3.2.1 Inteligencia Artificial y Robótica Quirúrgica
La IA ha pasado de ser una promesa a una realidad operativa (ej: patología). En el quirófano, la robótica está permitiendo procedimientos mínimamente invasivos que reducen la estancia hospitalaria, crítico para la eficiencia.

3.2.2 El Auge de los Dispositivos Desechables (Single-Use)
Tendencia masiva hacia el reemplazo de instrumentos reutilizables por dispositivos de un solo uso para eliminar contaminación cruzada y reducir costos de esterilización. Mercado proyectado a crecer significativamente.

3.3 Oportunidades de Inversión de Alto Crecimiento
Buscamos empresas que redefinen el estándar de cuidado ("Standard of Care").`,
            trends: [
                { title: "Robótica Quirúrgica", description: "Procedimientos mínimamente invasivos, reducen estancia hospitalaria." },
                { title: "Dispositivos Desechables", description: "Eliminación de contaminación y costos de esterilización." }
            ],
            picks: [
                { ticker: 'TMDX', name: 'TransMedics Group', type: 'Growth', thesis: 'Sistema OCS mantiene órganos vivos. Creando su propio mercado (logística de trasplantes).' },
                { ticker: 'PRCT', name: 'PROCEPT BioRobotics', type: 'Growth', thesis: 'Robótica en Urología (Aquablation). Crecimiento ingresos 43% YoY.' },
                { ticker: 'DCTH', name: 'Delcath Systems', type: 'Speculative', thesis: 'Oncología intervencionista (hígado). Enfoque tecnológico único.' }
            ]
        }
    },
    {
        id: 'insurance-nov-2025',
        title: 'Seguros & Insurtech',
        subtitle: 'Eficiencia, IA y Nichos',
        icon: '🛡️',
        readTime: '3 min read',
        tags: ['Financials', 'AI'],
        content: {
            overview: `Modernización forzada por costos. Clave: especialistas de nicho (E&S) e Insurtech 2.0. IA reduce tiempos de reclamos un 80%.`,
            fullText: `4. Sector de Seguros y Managed Care: Eficiencia, IA y Nichos Rentables

4.1 Panorama del Sector: Modernización Forzada
El sector de seguros está atravesando una revolución silenciosa impulsada por la necesidad. Los costos crecientes de las reclamaciones (inflación social, clima) obligan a modernizarse. 2025 es mixto: generalistas luchan, especialistas en nichos y "Insurtech 2.0" prosperan.

4.2 Tendencias Transformadoras
4.2.1 IA Generativa en el Procesamiento de Reclamaciones
La implementación operativa de la IA es la mayor tendencia. Automatización completa del manejo de reclamaciones (reducción de tiempo 80%, costos 30%) y detección de fraude en tiempo real. Adopción de Modelos de Lenguaje Pequeños (SLMs).

4.2.2 El Auge del Mercado E&S (Excess & Surplus)
A medida que los riesgos climáticos hacen ciertas regiones "inasegurables" para aseguradoras estándar, el mercado E&S explota. Tienen libertad de precios para asumir riesgos complejos rentablemente.

4.3 Oportunidades de Inversión: Nichos y Eficiencia
Evitar aseguradoras expuestas a catástrofes sin poder de precios. Buscar especialistas.`,
            trends: [
                { title: "IA Operativa", description: "Reducción drástica de tiempos de reclamo y detección de fraude." },
                { title: "Mercado E&S", description: "Crecimiento en seguros de líneas excedentes por riesgos complejos." }
            ],
            picks: [
                { ticker: 'SKWD', name: 'Skyward Specialty', type: 'Growth', thesis: 'Rey del Nicho E&S. Crecimiento primas 26% anual.' },
                { ticker: 'PRI', name: 'Primerica', type: 'Defensive', thesis: 'Modelo distribución eficiente. ROE 27.2% líder. Máquina de flujo de caja.' },
                { ticker: 'CB', name: 'Chubb', type: 'Value', thesis: 'Estándar de Oro. Disciplina de suscripción y balance global.' }
            ]
        }
    }
];

// Componente de Detalle (Modal de Lectura)
const ReportModal: React.FC<{ report: IndustryReport; onClose: () => void }> = ({ report, onClose }) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-fade-in">
            <div 
                className="w-full max-w-4xl max-h-[90vh] overflow-y-auto bg-[#050A14] border border-accent-gold/30 rounded-xl shadow-2xl custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header del Informe */}
                <div className="sticky top-0 z-10 bg-[#050A14]/95 backdrop-blur border-b border-white/10 px-8 py-6 flex justify-between items-start">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <span className="text-3xl">{report.icon}</span>
                            <h2 className="text-2xl md:text-3xl font-display text-white tracking-wide">
                                {report.title}
                            </h2>
                        </div>
                        <p className="text-accent-gold font-medium text-sm uppercase tracking-widest">
                            {report.subtitle}
                        </p>
                    </div>
                    <button 
                        onClick={onClose}
                        className="p-2 rounded-full hover:bg-white/10 text-text-muted hover:text-white transition-colors"
                    >
                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {/* Contenido del Informe */}
                <div className="p-8 space-y-10">
                    {/* Texto Completo */}
                    <div className="prose prose-invert max-w-none">
                        <p className="text-text-secondary text-lg leading-relaxed whitespace-pre-line font-serif">
                            {report.content.fullText}
                        </p>
                    </div>

                    {/* Tendencias Clave */}
                    {report.content.trends.length > 0 && (
                        <div className="grid md:grid-cols-2 gap-6">
                            {report.content.trends.map((trend, idx) => (
                                <div key={idx} className="bg-bg-tertiary/50 p-6 rounded-lg border border-white/5 hover:border-accent-cyan/30 transition-colors">
                                    <h4 className="text-accent-cyan font-bold text-xs uppercase tracking-wider mb-3 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-accent-cyan"></span>
                                        Tendencia {idx + 1}
                                    </h4>
                                    <h3 className="text-white font-display text-xl mb-2">{trend.title}</h3>
                                    <p className="text-sm text-text-muted leading-relaxed">{trend.description}</p>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Selección de Acciones (Picks) */}
                    {report.content.picks.length > 0 && (
                        <div className="bg-white/5 rounded-xl p-8 border border-white/10">
                            <div className="flex items-center gap-4 mb-8">
                                <div className="h-px flex-1 bg-white/10"></div>
                                <span className="text-accent-gold font-display text-2xl">Top Picks & Thesis</span>
                                <div className="h-px flex-1 bg-white/10"></div>
                            </div>

                            <div className="space-y-6">
                                {report.content.picks.map((pick) => (
                                    <div 
                                        key={pick.ticker} 
                                        className="group relative overflow-hidden rounded-lg bg-[#0B1221] border border-white/10 hover:border-accent-gold/50 transition-all duration-300 p-6"
                                    >
                                        <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-4">
                                            <div className="flex items-center gap-4">
                                                <span className="text-3xl font-display font-bold text-white group-hover:text-accent-gold transition-colors tracking-tight">
                                                    {pick.ticker}
                                                </span>
                                                <div className="flex flex-col">
                                                    <span className="text-sm font-medium text-text-primary">{pick.name}</span>
                                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider w-fit mt-1 ${
                                                        pick.type === 'Value' ? 'bg-blue-500/20 text-blue-400' :
                                                        pick.type === 'Growth' ? 'bg-green-500/20 text-green-400' :
                                                        pick.type === 'Turnaround' ? 'bg-orange-500/20 text-orange-400' :
                                                        pick.type === 'Speculative' ? 'bg-purple-500/20 text-purple-400' :
                                                        'bg-gray-500/20 text-gray-400'
                                                    }`}>
                                                        {pick.type}
                                                    </span>
                                                </div>
                                            </div>
                                        </div>
                                        <p className="text-sm text-text-secondary leading-relaxed border-l-2 border-white/10 pl-4">
                                            {pick.thesis}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Conclusión */}
                    {report.content.conclusion && (
                        <div className="bg-accent-gold/5 border border-accent-gold/20 rounded-lg p-8 text-center">
                            <p className="text-accent-gold font-medium italic font-display text-xl leading-relaxed">
                                "{report.content.conclusion}"
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export const IndustryResearch: React.FC = () => {
    const [selectedReport, setSelectedReport] = useState<IndustryReport | null>(null);

    const featuredReport = REPORT_DATA.find(r => r.isFeatured);
    const otherReports = REPORT_DATA.filter(r => !r.isFeatured);

    return (
        <WidgetCard 
            title="Industry Research" 
            tooltip="Deep dive analysis into sectors with high alpha potential. Updated monthly."
        >
            <div className="space-y-6">
                <div className="flex justify-between items-end">
                    <h4 className="text-xs text-text-muted uppercase tracking-widest">November 2025 Edition</h4>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-accent-primary/10 text-accent-primary font-medium">
                        Strategy Report
                    </span>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 h-full">
                    {/* FEATURED CARD (Left/Top - Large) */}
                    {featuredReport && (
                        <div 
                            onClick={() => setSelectedReport(featuredReport)}
                            className="lg:col-span-2 group cursor-pointer rounded-xl p-6 bg-gradient-to-br from-bg-tertiary to-[#0F1623] border border-white/10 hover:border-accent-gold/40 transition-all duration-300 relative overflow-hidden min-h-[200px] flex flex-col justify-between"
                        >
                            <div className="absolute top-0 right-0 p-3 opacity-10 group-hover:opacity-20 transition-opacity">
                                <span className="text-8xl">{featuredReport.icon}</span>
                            </div>
                            <div>
                                <div className="flex items-center gap-2 mb-3">
                                    <span className="text-xs font-bold bg-accent-gold/20 text-accent-gold px-2 py-1 rounded uppercase tracking-wider">
                                        Industry of the Month
                                    </span>
                                    <span className="text-[10px] text-text-muted flex items-center gap-1">
                                        <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                        {featuredReport.readTime}
                                    </span>
                                </div>
                                <h3 className="text-2xl font-display font-bold text-white mb-2 group-hover:text-accent-gold transition-colors">
                                    {featuredReport.title}
                                </h3>
                                <p className="text-sm text-text-secondary leading-relaxed max-w-md">
                                    {featuredReport.content.overview}
                                </p>
                            </div>
                            <div className="mt-6 flex items-center text-xs font-bold text-accent-gold uppercase tracking-wider">
                                Read Full Analysis <span className="ml-2 group-hover:translate-x-1 transition-transform">→</span>
                            </div>
                        </div>
                    )}

                    {/* OTHER REPORTS GRID */}
                    {otherReports.map((report) => (
                        <div 
                            key={report.id}
                            onClick={() => setSelectedReport(report)}
                            className="group cursor-pointer rounded-lg p-5 bg-bg-tertiary border border-white/5 hover:border-accent-cyan/30 hover:bg-white/5 transition-all duration-300 flex flex-col h-full"
                        >
                            <div className="flex justify-between items-start mb-3">
                                <div className="w-10 h-10 rounded-full bg-bg-primary flex items-center justify-center text-xl border border-white/10 group-hover:border-accent-cyan/50 transition-colors">
                                    {report.icon}
                                </div>
                                <span className="text-[10px] text-text-subtle">{report.readTime}</span>
                            </div>
                            
                            <h3 className="text-sm font-bold text-white group-hover:text-accent-cyan transition-colors font-display tracking-wide mb-1">
                                {report.title}
                            </h3>
                            <p className="text-xs text-text-muted line-clamp-2 mb-3 flex-grow">
                                {report.subtitle}
                            </p>
                            
                            <div className="flex gap-1 mt-auto">
                                {report.tags.slice(0, 1).map(tag => (
                                    <span key={tag} className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-text-muted border border-white/5">
                                        {tag}
                                    </span>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {selectedReport && (
                <ReportModal 
                    report={selectedReport} 
                    onClose={() => setSelectedReport(null)} 
                />
            )}
        </WidgetCard>
    );
};
