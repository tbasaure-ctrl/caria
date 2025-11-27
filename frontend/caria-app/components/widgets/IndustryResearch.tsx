import React, { useState } from 'react';
import { WidgetCard } from './WidgetCard';

// Interfaces para estructurar el reporte
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
    icon: string; // Emoji o path
    readTime: string;
    tags: string[];
    content: {
        overview: string;
        trends: { title: string; description: string }[];
        picks: StockPick[];
        conclusion?: string;
    };
}

// DATA: Contenido extraído y estructurado del informe proporcionado
const REPORT_DATA: IndustryReport[] = [
    {
        id: 'macro-nov-2025',
        title: 'Estrategia Global: Noviembre 2025',
        subtitle: 'Panorama Macroeconómico y Asignación de Activos',
        icon: '🌍',
        readTime: '3 min read',
        tags: ['Macro', 'Strategy', 'Rates'],
        content: {
            overview: `El penúltimo mes de 2025 se despliega en un contexto económico que desafía las categorizaciones simplistas. Tras un año de euforia tecnológica, los mercados han entrado en una fase de rotación táctica distintiva. Con la Fed ajustando tasas al rango 3.75%-4.00%, los inversores reevalúan la prima de riesgo.
            
            La narrativa ha girado desde el "crecimiento a cualquier precio" hacia una apreciación por la calidad del balance y flujos de caja predecibles. Noviembre emerge como un punto de inflexión crítico: los sectores cíclicos enfrentan vientos en contra por la desaceleración prevista para 2026, mientras que los sectores defensivos y de innovación sanitaria capturan el capital institucional.`,
            trends: [
                {
                    title: "Rotación hacia Calidad",
                    description: "Los costos de endeudamiento penalizan el apalancamiento excesivo. El mercado premia ahora la resiliencia operativa sobre el crecimiento especulativo."
                },
                {
                    title: "Bifurcación Sectorial",
                    description: "Sectores defensivos (Consumo Básico) y Salud toman el relevo mientras la tecnología busca consolidar valoraciones."
                }
            ],
            picks: [], // Es macro, no tiene picks específicos
            conclusion: "Recomendación Final: Construir una cartera 'barbell' (pesa): un núcleo defensivo robusto en consumo básico y seguros, equilibrado con apuestas satélite de alto crecimiento en robótica médica y biotecnología."
        }
    },
    {
        id: 'staples-nov-2025',
        title: 'Consumo Básico (Consumer Staples)',
        subtitle: 'INDUSTRIA DEL MES: Refugio Táctico y Valor',
        icon: '🛒',
        readTime: '5 min read',
        tags: ['Defensive', 'High Conviction', 'Dividends'],
        content: {
            overview: `Designado como la industria focal para Noviembre 2025. Históricamente, este sector actúa como un "proxy de bonos" con la ventaja del crecimiento del dividendo. Ante la incertidumbre económica, los inversores buscan la seguridad de la demanda inelástica.
            
            Existe una "Bifurcación de Valoraciones": Los minoristas masivos (Costco, Walmart) están sobrevalorados (P/E >40x), mientras que los fabricantes de alimentos envasados cotizan con descuentos atractivos (~11% bajo valor razonable) debido a temores exagerados sobre los fármacos GLP-1.`,
            trends: [
                {
                    title: "El Efecto Noviembre",
                    description: "Estadísticamente, noviembre es excepcionalmente fuerte para el sector (75% de frecuencia de ganancias en los últimos 25 años)."
                },
                {
                    title: "Adaptación a GLP-1",
                    description: "Empresas como Nestlé y General Mills están lanzando productos altos en proteína para acompañar a usuarios de Ozempic/Wegovy, mitigando el impacto en volumen."
                }
            ],
            picks: [
                {
                    ticker: 'KHC',
                    name: 'Kraft Heinz',
                    type: 'Value',
                    thesis: 'Calificada con 5 estrellas por Morningstar. Infravaloración extrema que ignora su reestructuración de deuda y mejora de márgenes.'
                },
                {
                    ticker: 'GIS',
                    name: 'General Mills',
                    type: 'Defensive',
                    thesis: 'Jugador defensivo clásico que ha demostrado capacidad superior para adaptarse a tendencias de salud (Blue Buffalo).'
                },
                {
                    ticker: 'SFM',
                    name: 'Sprouts Farmers Market',
                    type: 'Growth',
                    thesis: 'Se beneficia del auge de alimentación saludable y "limpia" impulsado por la tendencia GLP-1. Expansión de márgenes con productos frescos.'
                },
                {
                    ticker: 'OLLI',
                    name: "Ollie's Bargain Outlet",
                    type: 'Growth',
                    thesis: 'Modelo de "caza del tesoro" ideal para un consumidor sensible al precio. Adquiere exceso de inventario a precios de ganga.'
                },
                {
                    ticker: 'EL',
                    name: 'Estée Lauder',
                    type: 'Turnaround',
                    thesis: 'Valoración deprimida por debilidad en Asia. Posee marcas de prestigio valiosas; potencial rebote violento si estabiliza inventarios.'
                }
            ]
        }
    },
    {
        id: 'medtech-nov-2025',
        title: 'Dispositivos Médicos',
        subtitle: 'La Revolución Silenciosa: Robótica e IA',
        icon: '🦾',
        readTime: '4 min read',
        tags: ['Growth', 'Tech', 'Innovation'],
        content: {
            overview: `A diferencia de la biotecnología binaria, MedTech ofrece crecimiento predecible impulsado por el envejecimiento global y la eficiencia hospitalaria. Mercado proyectado a $678.8B en 2025.
            
            La IA ha pasado a ser una realidad operativa en diagnósticos, y la robótica permite procedimientos mínimamente invasivos que reducen la estancia hospitalaria.`,
            trends: [
                {
                    title: "Robótica Quirúrgica",
                    description: "Permite procedimientos ultra-precisos, reduciendo costos hospitalarios a largo plazo."
                },
                {
                    title: "Dispositivos Desechables",
                    description: "Tendencia masiva hacia instrumentos de un solo uso para eliminar contaminación y costos de esterilización."
                }
            ],
            picks: [
                {
                    ticker: 'TMDX',
                    name: 'TransMedics Group',
                    type: 'Growth',
                    thesis: 'La "Logística de la Vida". Su sistema OCS mantiene órganos donados vivos fuera del cuerpo. Está creando su propio mercado (20.9% share).'
                },
                {
                    ticker: 'PRCT',
                    name: 'PROCEPT BioRobotics',
                    type: 'Growth',
                    thesis: 'Robótica en Urología (Aquablation). Crecimiento de ingresos del 43% YoY. Adopción exponencial por cirujanos.'
                },
                {
                    ticker: 'DCTH',
                    name: 'Delcath Systems',
                    type: 'Speculative',
                    thesis: 'Oncología intervencionista (hígado). Small-cap validada con subida del 202% interanual.'
                }
            ]
        }
    },
    {
        id: 'pharma-nov-2025',
        title: 'Salud y Farmacéutica',
        subtitle: 'Innovación bajo Presión y Boom de M&A',
        icon: '🧬',
        readTime: '4 min read',
        tags: ['Biotech', 'M&A', 'High Risk'],
        content: {
            overview: `Un ecosistema en tensión por el "patent cliff" y presión regulatoria de precios. Esto actúa como catalizador para una innovación desenfrenada y consolidación agresiva.
            
            Las "Big Pharma" (Merck, Sanofi, Lilly) están desplegando balances masivos para comprar crecimiento externo, validando que la innovación real ocurre en las mid-caps.`,
            trends: [
                {
                    title: "Renacimiento de M&A",
                    description: "Oleada de adquisiciones multimillonarias en oncología de precisión y enfermedades raras."
                },
                {
                    title: "Áreas Hot",
                    description: "Oncología (ADCs, T-cell engagers), Neurociencia (Alzheimer) y Metabolismo (Next-gen Obesity)."
                }
            ],
            picks: [
                {
                    ticker: 'KALA',
                    name: 'Kala Bio',
                    type: 'Speculative',
                    thesis: 'Catalizador binario a fin de 2025 (Fase 2b CHASE) para enfermedad ocular rara sin cura. Potencial de revalorización múltiple.'
                },
                {
                    ticker: 'KAPA',
                    name: 'Kairos Pharma',
                    type: 'Speculative',
                    thesis: 'Datos interinos de Fase 2 en cáncer de próstata. Aborda una de las áreas oncológicas más lucrativas.'
                }
            ]
        }
    },
    {
        id: 'insurance-nov-2025',
        title: 'Seguros & Insurtech',
        subtitle: 'Eficiencia, IA y Nichos Rentables',
        icon: '🛡️',
        readTime: '3 min read',
        tags: ['Financials', 'AI', 'Niche'],
        content: {
            overview: `El sector atraviesa una modernización forzada por costos climáticos e inflación social. La clave en 2025 es evitar aseguradoras generalistas expuestas a catástrofes y buscar especialistas de nicho (E&S) e Insurtech 2.0.
            
            La IA Generativa está reduciendo tiempos de reclamos en un 80% y detectando fraudes que antes pasaban desapercibidos.`,
            trends: [
                {
                    title: "Auge del Mercado E&S",
                    description: "Las aseguradoras de 'Exceso y Superávit' tienen libertad de precios para asumir riesgos complejos que las estándar no tocan."
                }
            ],
            picks: [
                {
                    ticker: 'SKWD',
                    name: 'Skyward Specialty',
                    type: 'Growth',
                    thesis: 'El Rey del Nicho E&S. Crecimiento de primas del 26% anual. Poder de fijación de precios superior.'
                },
                {
                    ticker: 'PRI',
                    name: 'Primerica',
                    type: 'Defensive',
                    thesis: 'Modelo de distribución masiva extremadamente eficiente. ROE del 27.2% (líder). Máquina de flujo de caja.'
                },
                {
                    ticker: 'CB',
                    name: 'Chubb',
                    type: 'Value',
                    thesis: 'El Estándar de Oro. Disciplina de suscripción legendaria y balance global para navegar volatilidad.'
                }
            ]
        }
    }
];

// Componente de Detalle (Modal de Lectura)
const ReportModal: React.FC<{ report: IndustryReport; onClose: () => void }> = ({ report, onClose }) => {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/90 backdrop-blur-sm animate-fade-in">
            <div 
                className="w-full max-w-3xl max-h-[90vh] overflow-y-auto bg-[#050A14] border border-accent-gold/30 rounded-xl shadow-2xl custom-scrollbar"
                onClick={(e) => e.stopPropagation()}
            >
                {/* Header del Informe */}
                <div className="sticky top-0 z-10 bg-[#050A14]/95 backdrop-blur border-b border-white/10 px-8 py-6 flex justify-between items-start">
                    <div>
                        <div className="flex items-center gap-3 mb-2">
                            <span className="text-2xl">{report.icon}</span>
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
                <div className="p-8 space-y-8">
                    {/* Overview */}
                    <div className="prose prose-invert max-w-none">
                        <p className="text-text-secondary text-lg leading-relaxed whitespace-pre-line">
                            {report.content.overview}
                        </p>
                    </div>

                    {/* Tendencias Clave */}
                    {report.content.trends.length > 0 && (
                        <div className="grid md:grid-cols-2 gap-4">
                            {report.content.trends.map((trend, idx) => (
                                <div key={idx} className="bg-bg-tertiary p-5 rounded-lg border border-white/5">
                                    <h4 className="text-accent-cyan font-bold text-xs uppercase tracking-wider mb-2">
                                        Tendencia {idx + 1}
                                    </h4>
                                    <h3 className="text-white font-display text-lg mb-2">{trend.title}</h3>
                                    <p className="text-sm text-text-muted leading-relaxed">{trend.description}</p>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* Selección de Acciones (Picks) */}
                    {report.content.picks.length > 0 && (
                        <div>
                            <div className="flex items-center gap-4 mb-6">
                                <div className="h-px flex-1 bg-white/10"></div>
                                <span className="text-accent-gold font-display text-xl">Top Picks & Thesis</span>
                                <div className="h-px flex-1 bg-white/10"></div>
                            </div>

                            <div className="space-y-4">
                                {report.content.picks.map((pick) => (
                                    <div 
                                        key={pick.ticker} 
                                        className="group relative overflow-hidden rounded-lg bg-white/5 border border-white/10 hover:border-accent-gold/50 transition-all duration-300"
                                    >
                                        <div className="absolute top-0 left-0 w-1 h-full bg-accent-gold opacity-50 group-hover:opacity-100 transition-opacity" />
                                        <div className="p-5 pl-7">
                                            <div className="flex justify-between items-start mb-2">
                                                <div className="flex items-center gap-3">
                                                    <span className="text-2xl font-display font-bold text-white group-hover:text-accent-gold transition-colors">
                                                        {pick.ticker}
                                                    </span>
                                                    <span className="text-sm text-text-muted">{pick.name}</span>
                                                </div>
                                                <span className={`text-[10px] font-bold px-2 py-1 rounded uppercase tracking-wider ${
                                                    pick.type === 'Value' ? 'bg-blue-500/20 text-blue-400' :
                                                    pick.type === 'Growth' ? 'bg-green-500/20 text-green-400' :
                                                    pick.type === 'Turnaround' ? 'bg-orange-500/20 text-orange-400' :
                                                    pick.type === 'Speculative' ? 'bg-purple-500/20 text-purple-400' :
                                                    'bg-gray-500/20 text-gray-400'
                                                }`}>
                                                    {pick.type}
                                                </span>
                                            </div>
                                            <p className="text-sm text-text-secondary leading-relaxed">
                                                {pick.thesis}
                                            </p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Conclusión */}
                    {report.content.conclusion && (
                        <div className="bg-accent-gold/10 border border-accent-gold/20 rounded-lg p-6 text-center">
                            <p className="text-accent-gold font-medium italic font-display text-lg">
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

    return (
        <WidgetCard 
            title="Industry Research" 
            tooltip="Deep dive analysis into sectors with high alpha potential. Updated monthly."
        >
            <div className="space-y-4">
                <div className="flex justify-between items-end mb-2">
                    <h4 className="text-xs text-text-muted uppercase tracking-widest">November 2025 Edition</h4>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-accent-primary/10 text-accent-primary font-medium">
                        New Report
                    </span>
                </div>

                <div className="space-y-3">
                    {REPORT_DATA.map((report) => (
                        <div 
                            key={report.id}
                            onClick={() => setSelectedReport(report)}
                            className="group cursor-pointer rounded-lg p-4 bg-bg-tertiary border border-white/5 hover:border-accent-cyan/30 hover:bg-white/5 transition-all duration-300"
                        >
                            <div className="flex items-start gap-4">
                                <div className="w-10 h-10 rounded-full bg-bg-primary flex items-center justify-center text-xl group-hover:scale-110 transition-transform duration-300 border border-white/10 group-hover:border-accent-cyan/50">
                                    {report.icon}
                                </div>
                                <div className="flex-1 min-w-0">
                                    <h3 className="text-sm font-bold text-white group-hover:text-accent-cyan transition-colors truncate font-display tracking-wide">
                                        {report.title}
                                    </h3>
                                    <p className="text-xs text-text-muted mt-1 truncate group-hover:text-text-secondary transition-colors">
                                        {report.subtitle}
                                    </p>
                                    <div className="flex items-center gap-3 mt-2.5">
                                        <span className="text-[10px] text-text-subtle flex items-center gap-1">
                                            <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
                                            {report.readTime}
                                        </span>
                                        <div className="flex gap-1">
                                            {report.tags.slice(0, 2).map(tag => (
                                                <span key={tag} className="text-[9px] px-1.5 py-0.5 rounded bg-white/5 text-text-muted border border-white/5">
                                                    {tag}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                                <div className="self-center opacity-0 group-hover:opacity-100 transform translate-x-2 group-hover:translate-x-0 transition-all duration-300">
                                    <svg className="w-5 h-5 text-accent-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                                    </svg>
                                </div>
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
