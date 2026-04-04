import { useData } from "../DataContext";

export default function SlidePrologue() {
  const { comparison } = useData();
  const nTickers = comparison.baseline?.n_tickers || 9;
  const nRows = comparison.baseline?.n_rows || 214;

  return (
    <div className="slide" style={{ justifyContent: "center" }}>
      <div style={{ textAlign: "center", marginBottom: "4rem" }}>
        <h1 className="hero-title">AI Hype Decoded</h1>
        <p className="hero-subtitle" style={{ margin: "0 auto" }}>
          The Prediction Project: Does the structure of executive AI-speak meaningfully predict future equity return anomalies? 
          We tested conventional momentum variables against raw temporal NLP transcript scoring.
        </p>
      </div>

      <div className="footer-grid">
        <div className="footer-stat">
          <div className="footer-stat-val">{nTickers}</div>
          <div className="footer-stat-label">Mega-Cap Tech Equities</div>
        </div>
        <div className="footer-stat">
          <div className="footer-stat-val">{nRows}</div>
          <div className="footer-stat-label">Earnings Events Analyzed</div>
        </div>
        <div className="footer-stat">
          <div className="footer-stat-val">FinBERT</div>
          <div className="footer-stat-label">Sentiment NLP Engine</div>
        </div>
      </div>
    </div>
  );
}
