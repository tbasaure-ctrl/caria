import React, { useState, useEffect } from 'react';
import { WidgetCard } from './WidgetCard';
import { fetchBusinessCycle, fetchCurrencies, fetchHeatmap, BusinessCyclePoint, CurrencyRate, HeatmapCell } from '../../services/apiService';
import { BusinessCycleClock } from './GlobalEconomicMonitor/BusinessCycleClock';
import { MacroHeatmap } from './GlobalEconomicMonitor/MacroHeatmap';
import { CurrencyMonitor } from './GlobalEconomicMonitor/CurrencyMonitor';
import { CountryDetails } from './GlobalEconomicMonitor/CountryDetails';

type TabType = 'business-cycle' | 'heatmap' | 'currencies' | 'country-details';

export const GlobalEconomicMonitor: React.FC<{ id?: string }> = ({ id }) => {
    const [activeTab, setActiveTab] = useState<TabType>('business-cycle');
    const [businessCycleData, setBusinessCycleData] = useState<BusinessCyclePoint[]>([]);
    const [currencyData, setCurrencyData] = useState<CurrencyRate[]>([]);
    const [heatmapData, setHeatmapData] = useState<HeatmapCell[]>([]);
    const [selectedCountry, setSelectedCountry] = useState<string | null>(null);
    const [loading, setLoading] = useState<Record<string, boolean>>({
        businessCycle: false,
        currencies: false,
        heatmap: false,
    });
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        loadData();
    }, []);

    const loadData = async () => {
        try {
            // Load business cycle data
            setLoading(prev => ({ ...prev, businessCycle: true }));
            try {
                const bcData = await fetchBusinessCycle();
                setBusinessCycleData(bcData.points);
            } catch (err) {
                console.error('Error loading business cycle:', err);
            } finally {
                setLoading(prev => ({ ...prev, businessCycle: false }));
            }

            // Load currency data
            setLoading(prev => ({ ...prev, currencies: true }));
            try {
                const currData = await fetchCurrencies();
                setCurrencyData(currData.rates);
            } catch (err) {
                console.error('Error loading currencies:', err);
            } finally {
                setLoading(prev => ({ ...prev, currencies: false }));
            }

            // Load heatmap data
            setLoading(prev => ({ ...prev, heatmap: true }));
            try {
                const hmData = await fetchHeatmap();
                setHeatmapData(hmData.cells);
            } catch (err) {
                console.error('Error loading heatmap:', err);
            } finally {
                setLoading(prev => ({ ...prev, heatmap: false }));
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load economic data');
        }
    };

    const tabs = [
        { id: 'business-cycle' as TabType, label: 'Business Cycle', icon: '📊' },
        { id: 'heatmap' as TabType, label: 'Heatmap', icon: '🗺️' },
        { id: 'currencies' as TabType, label: 'Currencies', icon: '💱' },
        { id: 'country-details' as TabType, label: 'Country Details', icon: '🌍' },
    ];

    return (
        <WidgetCard
            id={id}
            title="GLOBAL ECONOMIC MONITOR"
            tooltip="Comprehensive view of global economic health using leading, coincident, and lagging indicators"
        >
            <div className="space-y-4">
                {/* Tab Navigation */}
                <div className="flex flex-wrap gap-2 border-b border-white/10 pb-3">
                    {tabs.map(tab => (
                        <button
                            key={tab.id}
                            onClick={() => {
                                setActiveTab(tab.id);
                                if (tab.id === 'country-details' && !selectedCountry && businessCycleData.length > 0) {
                                    setSelectedCountry(businessCycleData[0].country_code);
                                }
                            }}
                            className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${
                                activeTab === tab.id
                                    ? 'bg-accent-cyan/20 text-accent-cyan border border-accent-cyan/30'
                                    : 'text-text-muted hover:text-text-secondary hover:bg-white/5'
                            }`}
                        >
                            <span className="mr-1.5">{tab.icon}</span>
                            {tab.label}
                        </button>
                    ))}
                </div>

                {/* Error Message */}
                {error && (
                    <div className="bg-negative/10 border border-negative/30 text-negative text-xs p-3 rounded">
                        {error}
                    </div>
                )}

                {/* Tab Content */}
                <div className="min-h-[400px]">
                    {activeTab === 'business-cycle' && (
                        <BusinessCycleClock
                            data={businessCycleData}
                            loading={loading.businessCycle}
                            onCountrySelect={setSelectedCountry}
                        />
                    )}

                    {activeTab === 'heatmap' && (
                        <MacroHeatmap
                            data={heatmapData}
                            loading={loading.heatmap}
                            onCountrySelect={setSelectedCountry}
                        />
                    )}

                    {activeTab === 'currencies' && (
                        <CurrencyMonitor
                            data={currencyData}
                            loading={loading.currencies}
                        />
                    )}

                    {activeTab === 'country-details' && (
                        <CountryDetails
                            countryCode={selectedCountry}
                            onCountryChange={setSelectedCountry}
                            availableCountries={businessCycleData.map(p => ({
                                code: p.country_code,
                                name: p.country_name,
                            }))}
                        />
                    )}
                </div>
            </div>
        </WidgetCard>
    );
};

