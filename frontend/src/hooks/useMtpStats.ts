import { useCallback, useEffect, useRef, useState } from 'react';

export interface MtpStatsSnapshot {
  acceptance_rate?: number;
  tokens_accepted?: number;
  tokens_drafted?: number;
  source?: string;
  raw?: string;
  updated_at: string;
}

function parseMtpFromLogLine(message: string): Partial<MtpStatsSnapshot> | null {
  const acceptance = message.match(
    /draft\s+acceptance\s+rate\s*=\s*([0-9.]+)\s*\(\s*(\d+)\s+accepted\s*\/\s*(\d+)\s+generated\s*\)/i,
  );
  if (acceptance) {
    return {
      acceptance_rate: parseFloat(acceptance[1]),
      tokens_accepted: parseInt(acceptance[2], 10),
      tokens_drafted: parseInt(acceptance[3], 10),
      source: 'draft_acceptance_rate',
      raw: message,
    };
  }
  const aggregate = message.match(/aggregate_accept_rate\s*[=:]\s*([0-9.]+)/i);
  if (aggregate) {
    return {
      acceptance_rate: parseFloat(aggregate[1]),
      source: 'aggregate_accept_rate',
      raw: message,
    };
  }
  return null;
}

export function useMtpStats(options: {
  enabled?: boolean;
  backend?: 'llamacpp' | 'vllm';
}) {
  const { enabled = true, backend = 'llamacpp' } = options;
  const [stats, setStats] = useState<MtpStatsSnapshot | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const applyStats = useCallback((partial: Partial<MtpStatsSnapshot>) => {
    setStats({
      ...partial,
      updated_at: new Date().toISOString(),
    } as MtpStatsSnapshot);
  }, []);

  useEffect(() => {
    if (!enabled || backend !== 'llamacpp') {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      setConnected(false);
      return;
    }

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    const url = `${protocol}//${host}/ws/logs?backend=${backend}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onerror = () => setConnected(false);

    ws.onmessage = (event) => {
      const text = event.data;
      if (typeof text !== 'string') return;

      try {
        const msg = JSON.parse(text);
        if (msg?.type === 'mtp_stats' && msg.data) {
          applyStats({ ...msg.data, source: msg.data.source || 'websocket' });
          return;
        }
        if (msg?.type === 'log') {
          const line = typeof msg.data === 'string' ? msg.data : msg.message;
          if (line) {
            const parsed = parseMtpFromLogLine(line);
            if (parsed) applyStats(parsed);
          }
          return;
        }
      } catch {
        // plain text log line from legacy replay
        const parsed = parseMtpFromLogLine(text);
        if (parsed) applyStats(parsed);
      }
    };

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [enabled, backend, applyStats]);

  return { stats, connected, clearStats: () => setStats(null) };
}
