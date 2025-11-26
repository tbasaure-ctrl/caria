/**
 * Configurable industry list for Industry Research feature
 * Easily add/remove industries by modifying this array
 */
export interface IndustryConfig {
    name: string;
    status: 'Emerging' | 'Mature' | 'Overheated' | 'Under Pressure';
    representative_tickers: string[];
}

export const BASE_INDUSTRIES: IndustryConfig[] = [
    {
        name: 'Consumer Staples',
        status: 'Mature',
        representative_tickers: ['PG', 'KO', 'WMT']
    },
    {
        name: 'Healthcare & Pharma',
        status: 'Mature',
        representative_tickers: ['JNJ', 'PFE', 'UNH']
    },
    {
        name: 'Medical Devices',
        status: 'Emerging',
        representative_tickers: ['ABT', 'TMO', 'ISRG']
    },
    {
        name: 'Insurance & Managed Care',
        status: 'Mature',
        representative_tickers: ['UNH', 'ANTM', 'CI']
    },
    {
        name: 'Semiconductors & AI Hardware',
        status: 'Overheated',
        representative_tickers: ['NVDA', 'AMD', 'TSM']
    },
    {
        name: 'Cloud & SaaS',
        status: 'Emerging',
        representative_tickers: ['MSFT', 'CRM', 'NOW']
    },
    {
        name: 'Cybersecurity',
        status: 'Emerging',
        representative_tickers: ['CRWD', 'ZS', 'PANW']
    },
    {
        name: 'Digital Payments & Fintech',
        status: 'Emerging',
        representative_tickers: ['SQ', 'PYPL', 'ADYEY']
    },
    {
        name: 'E-commerce & Platforms',
        status: 'Mature',
        representative_tickers: ['AMZN', 'SHOP', 'ETSY']
    },
    {
        name: 'Renewables & Grid',
        status: 'Emerging',
        representative_tickers: ['ENPH', 'SEDG', 'NEE']
    },
    {
        name: 'Metals & Mining',
        status: 'Under Pressure',
        representative_tickers: ['FCX', 'NEM', 'LTHM']
    },
    {
        name: 'Luxury & Premium Brands',
        status: 'Mature',
        representative_tickers: ['LVMH', 'HERM', 'TIF']
    },
    {
        name: 'Travel & Leisure',
        status: 'Under Pressure',
        representative_tickers: ['BKNG', 'MAR', 'ABNB']
    },
    {
        name: 'Space & Satellite',
        status: 'Emerging',
        representative_tickers: ['SPCE', 'MAXR', 'IRDM']
    },
    {
        name: 'Genomics & Precision Medicine',
        status: 'Emerging',
        representative_tickers: ['ILMN', 'PACB', 'NVTA']
    },
    {
        name: 'Climate Tech / Carbon Solutions',
        status: 'Emerging',
        representative_tickers: ['CLH', 'CWCO', 'BLNK']
    },
    {
        name: 'Defense & Dual-Use Tech',
        status: 'Mature',
        representative_tickers: ['LMT', 'RTX', 'NOC']
    }
];
