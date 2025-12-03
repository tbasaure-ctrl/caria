import React, { useState, useEffect } from 'react';
import { fetchCountryDetails } from '../../../services/apiService';

interface CountryDetailsProps {
    countryCode: string | null;
    onCountryChange: (code: string) => void;
    availableCountries: Array<{ code: string; name: string }>;
}

export const CountryDetails: React.FC<CountryDetailsProps> = ({
    countryCode,
    onCountryChange,
    availableCountries,
}) => {
    const [details, setDetails] = useState<any>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        if (countryCode) {
            loadDetails(countryCode);
        }
    }, [countryCode]);

    const loadDetails = async (code: string) => {
        setLoading(true);
        setError(null);
        try {
            const data = await fetchCountryDetails(code);
            setDetails(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load country details');
        } finally {
            setLoading(false);
        }
    };

    if (!countryCode) {
        return (
            <div className="flex flex-col items-center justify-center h-[400px] space-y-4">
                <div className="text-text-muted text-sm">Select a country to view details</div>
                <select
                    onChange={(e) => onCountryChange(e.target.value)}
                    className="bg-bg-primary border border-white/10 rounded px-3 py-2 text-white text-sm"
                >
                    <option value="">Choose a country...</option>
                    {availableCountries.map(country => (
                        <option key={country.code} value={country.code}>
                            {country.name}
                        </option>
                    ))}
                </select>
            </div>
        );
    }

    if (loading) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">Loading country details...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-negative text-sm">{error}</div>
            </div>
        );
    }

    if (!details) {
        return (
            <div className="flex items-center justify-center h-[400px]">
                <div className="text-text-muted text-sm">No details available</div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Country Selector */}
            <div>
                <label className="block text-xs text-text-muted mb-2">Select Country</label>
                <select
                    value={countryCode}
                    onChange={(e) => onCountryChange(e.target.value)}
                    className="w-full bg-bg-primary border border-white/10 rounded px-3 py-2 text-white text-sm"
                >
                    {availableCountries.map(country => (
                        <option key={country.code} value={country.code}>
                            {country.name}
                        </option>
                    ))}
                </select>
            </div>

            {/* Country Info */}
            <div className="space-y-4">
                <div>
                    <h3 className="text-lg font-semibold text-white mb-2">{details.country_name}</h3>
                    <p className="text-xs text-text-muted">Region: {details.region}</p>
                </div>

                {/* Business Cycle Point */}
                {details.business_cycle_point && (
                    <div className="bg-white/5 border border-white/10 rounded p-4">
                        <h4 className="text-sm font-medium text-white mb-3">Business Cycle Position</h4>
                        <div className="grid grid-cols-2 gap-4 text-xs">
                            <div>
                                <span className="text-text-muted">Phase:</span>
                                <span className="ml-2 text-white capitalize">{details.business_cycle_point.phase}</span>
                            </div>
                            <div>
                                <span className="text-text-muted">Momentum:</span>
                                <span className="ml-2 text-white">{details.business_cycle_point?.x.toFixed(3)}</span>
                            </div>
                            <div>
                                <span className="text-text-muted">Deviation:</span>
                                <span className="ml-2 text-white">{details.business_cycle_point.y.toFixed(3)}</span>
                            </div>
                        </div>
                    </div>
                )}

                {/* Currency Info */}
                {details.currency && (
                    <div className="bg-white/5 border border-white/10 rounded p-4">
                        <h4 className="text-sm font-medium text-white mb-3">Currency</h4>
                        <div className="text-xs space-y-2">
                            <div>
                                <span className="text-text-muted">Pair:</span>
                                <span className="ml-2 text-white font-mono">{details.currency.currency_pair}</span>
                            </div>
                            <div>
                                <span className="text-text-muted">Rate:</span>
                                <span className="ml-2 text-white font-mono">{details.currency.rate.toFixed(4)}</span>
                            </div>
                            {details.currency.change_1d !== undefined && (
                                <div>
                                    <span className="text-text-muted">1D Change:</span>
                                    <span className={`ml-2 font-mono ${
                                        details.currency.change_1d >= 0 ? 'text-positive' : 'text-negative'
                                    }`}>
                                        {details.currency.change_1d >= 0 ? '+' : ''}
                                        {details.currency.change_pct_1d?.toFixed(2)}%
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* Indicators */}
                {details.indicators && details.indicators.length > 0 && (
                    <div className="bg-white/5 border border-white/10 rounded p-4">
                        <h4 className="text-sm font-medium text-white mb-3">Economic Indicators</h4>
                        <div className="space-y-2">
                            {details.indicators.map((indicator: any, index: number) => (
                                <div key={index} className="flex justify-between items-center text-xs py-2 border-b border-white/5 last:border-0">
                                    <div>
                                        <span className="text-white">{indicator.name}</span>
                                        <span className="ml-2 text-text-muted text-[10px] capitalize">
                                            ({indicator.temporal_class})
                                        </span>
                                    </div>
                                    <div className="text-right">
                                        <span className="text-white font-mono">{indicator.value.toFixed(2)}</span>
                                        {indicator.unit && (
                                            <span className="ml-1 text-text-muted text-[10px]">{indicator.unit}</span>
                                        )}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {(!details.indicators || details.indicators.length === 0) && (
                    <div className="text-text-muted text-xs text-center py-8">
                        Indicator data will be available once data sources are fully integrated
                    </div>
                )}
            </div>
        </div>
    );
};

