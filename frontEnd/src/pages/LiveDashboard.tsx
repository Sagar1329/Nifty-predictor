import { useEffect, useState, type JSX } from "react";
import { Box, Typography, Paper } from "@mui/material";

import type { LiveState } from "../types/state";
import { fetchLiveState } from "../api/stateApi";
import SignalCard from "../components/SignalCard";

export default function LiveDashboard(): JSX.Element {
    const [state, setState] = useState<LiveState | null>(null);
    const REFRESH_INTERVAL = 6000; // 5 seconds


    useEffect(() => {
       
            const fetchState = async () => {
                console.log(`Fetching live state... ${new Date().toLocaleTimeString()}`);                const data = await fetchLiveState();
                setState(data);
            };

            // Initial fetch
            fetchState();

            // Polling
            const interval = setInterval(fetchState, REFRESH_INTERVAL);

            // Cleanup
            return () => clearInterval(interval);
        
    }, []);

    if (!state) {
        return <Typography>Loading…</Typography>;
    }

    return <SignalCard state={state} refresh_interval={REFRESH_INTERVAL} />;
}
