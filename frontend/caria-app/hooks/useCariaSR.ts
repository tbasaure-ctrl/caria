
import { useEffect, useState } from "react";

type SrPoint = {
    date: string;
    e4: number;
    sync: number;
    sr: number;
    regime: number;
};

type SrStatus = {
    ticker: string;
    auc: number;
    mean_normal: number;
    mean_fragile: number;
    last_date: string;
    last_sr: number;
    last_regime: number;
    last_updated: string;
};

// Assuming API_BASE_URL is defined in environment or context, 
// for Next.js it's often /api proxy or absolute URL. 
// We'll use relative path assuming standard proxy setup or baseURL handling.
const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export function useCariaSR(ticker: string) {
    const [series, setSeries] = useState<SrPoint[]>([]);
    const [status, setStatus] = useState<SrStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        async function fetchData() {
            if (!ticker) return;
            setLoading(true);
            setError(null);
            try {
                // Adjust URL fetch logic as per project standards (axios vs fetch)
                // Using native fetch for simplicity as requested
                const [sRes, stRes] = await Promise.all([
                    fetch(`${API_BASE}/caria-sr/series/${ticker}`),
                    fetch(`${API_BASE}/caria-sr/status/${ticker}`),
                ]);

                if (sRes.ok) {
                    const seriesData = await sRes.json();
                    setSeries(seriesData);
                } else {
                    // If 404, maybe just empty series
                    if (sRes.status !== 404) console.error("Error fetching series");
                }

                if (stRes.ok) {
                    const statusData = await stRes.json();
                    setStatus(statusData);
                } else {
                    if (stRes.status !== 404) console.error("Error fetching status");
                }

            } catch (err) {
                console.error(err);
                setError("Failed to fetch CARIA-SR data");
            } finally {
                setLoading(false);
            }
        }
        fetchData();
    }, [ticker]);

    return { series, status, loading, error };
}
