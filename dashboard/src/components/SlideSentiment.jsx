import { useMemo } from "react";
import { useData } from "../DataContext";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine
} from "recharts";

export default function SlideSentiment() {
  const { comparison, enhancedRows, foldResults, fiEnhanced } = useData();
  const acc = (comparison.enhanced.accuracy * 100).toFixed(1);

  const topFeatures = fiEnhanced ? fiEnhanced.slice(0, 5) : [];

  const tickerStats = useMemo(() => {
    if (enhancedRows.length === 0) return [];
    
    const grouped = {};
    enhancedRows.forEach(r => {
      if (!grouped[r.ticker]) grouped[r.ticker] = [];
      grouped[r.ticker].push(r);
    });

    const stats = Object.keys(grouped).map(ticker => {
      const rows = grouped[ticker];
      return {
        ticker,
        sentiment_score: rows[0].transcript_sentiment || rows[0].sentiment_score || 0,
      };
    });

    return stats.sort((a, b) => b.sentiment_score - a.sentiment_score);
  }, [enhancedRows]);

  return (
    <div className="slide">
      <div className="slide-layout-split">
        <div className="main-pane">
          <h2 className="slide-title">The NLP Enhanced Model</h2>
          <p className="slide-subtitle">Applying FinBERT sentiment scoring to Executive Transcripts & Headlines.</p>

          <div className="big-metric-container">
            <div className="big-metric text-green">{acc}%</div>
            <div className="metric-label">Accuracy</div>
          </div>

          <div className="chart-wrapper">
            <ResponsiveContainer>
              <BarChart data={tickerStats} margin={{ top: 10, right: 0, left: 0, bottom: 20 }}>
                <XAxis dataKey="ticker" tick={{ fontSize: 14, fontWeight: 500 }} dy={10} axisLine={false} tickLine={false} />
                <YAxis hide />
                <Tooltip cursor={{ fill: "#f4f4f4" }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                <ReferenceLine y={0} stroke="#eaeaea" />
                <Bar dataKey="sentiment_score" radius={[4, 4, 4, 4]} isAnimationActive={true}>
                  {tickerStats.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.sentiment_score > 0 ? "#2d7d46" : "#c0392b"} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="side-pane">
          <div className="secondary-module">
            <div className="module-title">Top Impact Features</div>
            <table className="micro-table">
              <tbody>
                {topFeatures.map((fi, idx) => (
                  <tr key={idx}>
                    <td style={{ fontWeight: fi.feature.includes("sentiment") ? 600 : 400 }}>
                      {fi.feature}
                    </td>
                    <td>{fi.importance.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="secondary-module" style={{ flex: 1 }}>
            <div className="module-title">Time-Series Validation Check</div>
            <table className="micro-table">
              <thead>
                <tr>
                  <th>Fold</th>
                  <th>Accuracy</th>
                </tr>
              </thead>
              <tbody>
                {foldResults.map((fold, idx) => (
                  <tr key={idx}>
                    <td>Fold {fold.fold}</td>
                    <td className="text-green">
                      {fold.enhanced_accuracy ? (fold.enhanced_accuracy * 100).toFixed(1) + "%" : "N/A"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
