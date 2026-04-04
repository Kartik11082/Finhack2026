import { useData } from "../DataContext";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer
} from "recharts";

export default function SlideBaseline() {
  const { comparison, fiBaseline, foldResults } = useData();
  const acc = (comparison.baseline.accuracy * 100).toFixed(1);

  return (
    <div className="slide">
      <div className="slide-layout-split">
        <div className="main-pane">
          <h2 className="slide-title">The Baseline Model</h2>
          <p className="slide-subtitle">Momentum, Volatility, and Volume features (Price-only).</p>

          <div className="big-metric-container">
            <div className="big-metric">{acc}%</div>
            <div className="metric-label">Accuracy</div>
          </div>

          <div className="chart-wrapper">
            <ResponsiveContainer>
              <BarChart
                layout="vertical"
                data={[...fiBaseline].reverse()}
                margin={{ top: 0, right: 30, left: 100, bottom: 0 }}
              >
                <XAxis type="number" hide />
                <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} tick={{ fill: "#666", fontSize: 13 }} />
                <Tooltip cursor={{ fill: "transparent" }} contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}/>
                <Bar dataKey="importance" fill="#dddddd" radius={[0, 4, 4, 0]} isAnimationActive={true} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="side-pane">
          <div className="secondary-module">
            <div className="module-title">Model Metrics</div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "1rem" }}>
              <div>
                <div style={{ fontSize: "1.5rem", fontWeight: "600" }}>{comparison.baseline.auc.toFixed(3)}</div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>ROC-AUC</div>
              </div>
              <div style={{ textAlign: "right" }}>
                <div style={{ fontSize: "1.5rem", fontWeight: "600" }}>{comparison.baseline.f1.toFixed(3)}</div>
                <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>F1 Score</div>
              </div>
            </div>
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
                    <td>{fold.baseline_accuracy ? (fold.baseline_accuracy * 100).toFixed(1) + "%" : "N/A"}</td>
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
