import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const scorecards = [
  { label: "Epicenter", value: "NVDA 91", note: "Highest spillover score in the basket" },
  { label: "Fastest channel", value: "Semis", note: "Pressure spreads through suppliers first" },
  { label: "Median impact", value: "67", note: "Broad market transmission remains elevated" },
];

const sectorImpact = [
  { sector: "Semis", impact: 89, color: "#111111" },
  { sector: "Software", impact: 77, color: "#383838" },
  { sector: "Internet", impact: 74, color: "#5b5b5b" },
  { sector: "Banks", impact: 66, color: "#7c7c7c" },
  { sector: "Industrials", impact: 54, color: "#9d9d9d" },
  { sector: "Defensives", impact: 46, color: "#c0c0c0" },
];

const transmission = [
  { session: "T0", leaders: -4.9, secondWave: -2.0 },
  { session: "T+1", leaders: -4.1, secondWave: -2.4 },
  { session: "T+2", leaders: -3.2, secondWave: -2.1 },
  { session: "T+3", leaders: -2.4, secondWave: -1.6 },
  { session: "T+4", leaders: -1.7, secondWave: -0.9 },
];

const leadStocks = [
  { ticker: "NVDA", company: "NVIDIA", score: 91, move: "-4.7%", channel: "AI supply chain" },
  // { ticker: "MSFT", company: "Microsoft", score: 84, move: "-2.9%", channel: "Enterprise demand" },
  // { ticker: "AMZN", company: "Amazon", score: 80, move: "-2.4%", channel: "Cloud + retail" },
  // { ticker: "GOOGL", company: "Alphabet", score: 76, move: "-2.1%", channel: "Digital ads" },
  // { ticker: "META", company: "Meta", score: 72, move: "-1.9%", channel: "Ad sensitivity" },
];

const widerUniverse = [
  { ticker: "AMD", sector: "Semis", score: 86 },
  { ticker: "AVGO", sector: "Semis", score: 82 },
  { ticker: "TSM", sector: "Semis", score: 79 },
  { ticker: "CRM", sector: "Software", score: 73 },
  { ticker: "NOW", sector: "Software", score: 71 },
  { ticker: "BAC", sector: "Banks", score: 67 },
  { ticker: "GS", sector: "Banks", score: 65 },
  { ticker: "MA", sector: "Payments", score: 62 },
];

export default function SlideSpilloverStocks() {
  return (
    <div className="slide slide-stocks">
      <div className="stocks-hero">
        <div>
          <div className="module-title">Market Spillover</div>
          <h2 className="slide-title">How pressure in a few leaders bends the whole tape.</h2>
          <p className="slide-subtitle">
            The initial shock lands in crowded mega-caps, then broadens through software, financials, and lower-beta
            names with a slower second wave.
          </p>
        </div>
        <div className="stocks-score-strip">
          {scorecards.map((card) => (
            <div key={card.label} className="stocks-score-card">
              <span>{card.label}</span>
              <strong>{card.value}</strong>
              <small>{card.note}</small>
            </div>
          ))}
        </div>
      </div>

      <div className="stocks-stage">
        <div className="stocks-main">
          <div className="stock-chart-panel secondary-module stock-main-chart-panel">
            <div className="module-row">
              <div>
                <div className="module-title">Main Graph</div>
                <div className="stock-caption">Primary move in leaders, followed by the slower second-wave spillover</div>
              </div>
              <div className="stock-caption">5-session cascade</div>
            </div>
            <div className="stock-chart-area stock-chart-area-main">
              <ResponsiveContainer>
                <AreaChart data={transmission} margin={{ top: 8, right: 10, left: -12, bottom: 0 }}>
                  <defs>
                    <linearGradient id="leadersFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#111111" stopOpacity={0.2} />
                      <stop offset="95%" stopColor="#111111" stopOpacity={0.01} />
                    </linearGradient>
                    <linearGradient id="secondFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#808080" stopOpacity={0.22} />
                      <stop offset="95%" stopColor="#808080" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid stroke="#f1f1f1" vertical={false} />
                  <XAxis dataKey="session" axisLine={false} tickLine={false} tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 11, fill: "#8a8a8a" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: "10px", border: "1px solid #ececec", boxShadow: "0 8px 24px rgba(0,0,0,0.06)" }} />
                  <Area type="monotone" dataKey="leaders" name="Leaders" stroke="#111111" fill="url(#leadersFill)" strokeWidth={2.6} />
                  <Area type="monotone" dataKey="secondWave" name="Second wave" stroke="#6f6f6f" fill="url(#secondFill)" strokeWidth={2.2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
            <p className="stock-main-note">
              The chart to explain: the first leg lower is concentrated in mega-cap leaders, but the broader market pain
              persists because the second-wave cohort keeps repricing after the initial shock.
            </p>
          </div>

          <div className="secondary-module stock-universe-panel">
            <div className="module-row">
              <div className="module-title">Broader Basket</div>
              <div className="stock-caption">Additional names linked to the chain</div>
            </div>
            <div className="stock-chip-grid">
              {widerUniverse.map((stock) => (
                <div key={stock.ticker} className="stock-chip">
                  <strong>{stock.ticker}</strong>
                  <span>{stock.sector}</span>
                  <em>{stock.score}</em>
                </div>
              ))}
            </div>
          </div>
        </div>

        <aside className="stocks-sidebar">
          <div className="secondary-module stock-mini-chart-panel">
            <div className="module-row">
              <div className="module-title">Side Graph</div>
              <div className="stock-caption">Sector intensity</div>
            </div>
            <div className="stock-chart-area stock-chart-area-side">
              <ResponsiveContainer>
                <BarChart data={sectorImpact} margin={{ top: 8, right: 4, left: -26, bottom: 0 }}>
                  <CartesianGrid stroke="#f1f1f1" vertical={false} />
                  <XAxis dataKey="sector" axisLine={false} tickLine={false} tick={{ fontSize: 10 }} />
                  <YAxis hide domain={[0, 100]} />
                  <Tooltip cursor={{ fill: "#f7f7f7" }} contentStyle={{ borderRadius: "10px", border: "1px solid #ececec", boxShadow: "0 8px 24px rgba(0,0,0,0.06)" }} />
                  <Bar dataKey="impact" radius={[8, 8, 0, 0]}>
                    {sectorImpact.map((entry) => (
                      <Cell key={entry.sector} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="secondary-module stock-reading-panel">
            <div className="module-title">Reading The Tape</div>
            <p className="stock-side-copy">
              This is not a single-name story. Mega-cap weakness first hits suppliers and infrastructure, then spills into
              enterprise spend, payments, and credit-sensitive financials. The index feels broader pain because the second
              wave arrives after the headline shock.
            </p>
          </div>

          <div className="secondary-module stock-leaders-panel">
            <div className="module-row">
              <div className="module-title">Lead Stocks</div>              
            </div>
            <div className="stock-lead-list refined">
              {leadStocks.map((stock) => (
                <div key={stock.ticker} className="stock-lead-row refined">
                  <div>
                    <strong>{stock.ticker}</strong>
                    <div className="stock-role">{stock.company} | {stock.channel}</div>
                  </div>
                  <div className="stock-lead-metrics">
                    <div className="stock-score">{stock.score}</div>
                    <div className="stock-move">{stock.move}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
