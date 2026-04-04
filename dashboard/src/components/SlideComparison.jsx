import { useData } from "../DataContext";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LabelList
} from "recharts";

export default function SlideComparison() {
  const { comparison, enhancedRows } = useData();

  const baseAcc = comparison.baseline.accuracy;
  const enhAcc = comparison.enhanced.accuracy;
  const baseF1 = comparison.baseline.f1;
  const enhF1 = comparison.enhanced.f1;
  const baseAuc = comparison.baseline.auc;
  const enhAuc = comparison.enhanced.auc;

  const accDelta = (enhAcc - baseAcc) * 100;

  const chartData = [
    {
      metric: "Accuracy",
      Baseline: baseAcc ?? 0,
      Enhanced: enhAcc ?? 0,
    },
    {
      metric: "F1 Score",
      Baseline: baseF1 ?? 0,
      Enhanced: enhF1 ?? 0,
    },
    {
      metric: "AUC",
      Baseline: baseAuc ?? 0,
      Enhanced: enhAuc ?? 0,
    },
  ];

  // Get the last 15 rows for the audit trail snippet
  const auditRows = [...enhancedRows].reverse().slice(0, 15);

  return (
    <div className="slide">
      <div className="slide-layout-split">
        <div className="main-pane">
          <h2 className="slide-title">Insights</h2>
          <p className="slide-subtitle">Applying Executive Transcript Sentiment universally lifts model prediction capability across 5 Folds.</p>

          <div className="big-metric-container">
            <div className={`big-metric ${accDelta > 0 ? "text-green" : "text-red"}`}>
              {accDelta > 0 ? "+" : ""}{accDelta.toFixed(1)}%
            </div>
            <div className="metric-label">Absolute Accuracy Gain</div>
          </div>

          <div className="chart-wrapper" style={{ minHeight: "250px", flex: "none" }}>
            <ResponsiveContainer height={250}>
              <BarChart
                data={chartData}
                margin={{ top: 20, right: 0, left: 0, bottom: 5 }}
                barGap={8}
              >
                <XAxis dataKey="metric" axisLine={false} tickLine={false} tick={{ fontSize: 14, fontWeight: 500 }} />
                <YAxis hide domain={[0, 1]} />
                <Tooltip cursor={{ fill: "transparent" }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '20px' }} />
                <Bar dataKey="Baseline" fill="#dddddd" radius={[4, 4, 0, 0]} isAnimationActive={true}>
                  <LabelList dataKey="Baseline" position="top" fill="#888" fontSize={12} formatter={(val) => val === 0 ? "" : val} />
                </Bar>
                <Bar dataKey="Enhanced" fill="#111111" radius={[4, 4, 0, 0]} isAnimationActive={true}>
                  <LabelList dataKey="Enhanced" position="top" fill="#111" fontSize={12} formatter={(val) => val === 0 ? "" : val} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="side-pane">
          <div className="secondary-module" style={{ flex: 1 }}>
            <div className="module-title">Raw Prediction Audit</div>
            <div className="log-container">
              <table className="log-table">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>Ticker</th>
                    <th>Actual</th>
                    <th>Pred</th>
                  </tr>
                </thead>
                <tbody>
                  {auditRows.map((r, i) => (
                    <tr key={i}>
                      <td>{r.earnings_date}</td>
                      <td style={{ fontWeight: 600 }}>{r.ticker}</td>
                      <td>{r.actual_return > 0 ? "UP" : "DOWN"}</td>
                      <td className={r.correct === "True" ? "text-green" : "text-red"}>
                        {r.predicted > 0 ? "UP" : "DOWN"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
